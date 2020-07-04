# import packages
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import metrics
from tensorflow.keras import losses, optimizers
from tensorflow.keras import callbacks


class LstmMany2Many:
    def __init__(self, data_frame: pd.DataFrame, days_ahead=15):
        """
        Initializes LSTM
        :param days_ahead: Number of days in the future to predict
        :param data_frame: Pandas Data Frame containing stock data
        """
        self.days_ahead = days_ahead
        self.num_features = 30
        self.TRAIN_LEN = int(0.8 * data_frame.shape[0])
        self.in_size = 120
        self.out_size = days_ahead
        self.data_frame = self.preprocess(data_frame, 15)
        self._preprocess()
        self.model = Sequential()
        self.model.add(layers.LSTM(100, return_sequences=True, input_shape=(self.in_size, self.num_features)))
        self.model.add(layers.LSTM(200, return_sequences=True))
        self.model.add(layers.LSTM(200, return_sequences=True))
        self.model.add(layers.LSTM(100, return_sequences=False))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(50 * self.num_features))
        self.model.add(layers.Dense(self.out_size * self.num_features))
        self.model.add(layers.Reshape((self.out_size, self.num_features)))

    def _preprocess(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.dataset = self.data_frame.values
        self.dataset = self.dataset.reshape(-1, self.num_features)
        self.dataset = np.flip(self.dataset, axis=0)
        self.scaled_closed = self.scaler.fit_transform(self.dataset)
        self.train_data = self.scaled_closed[:self.TRAIN_LEN]
        self.test_data = self.scaled_closed[self.TRAIN_LEN:]

        self.x_train = []
        self.y_train = []
        for i in range(self.in_size, len(self.train_data) - self.out_size):
            self.x_train.append(self.train_data[i - self.in_size:i])
            self.y_train.append(self.train_data[i:i + self.out_size])
        self.x_train = np.array(self.x_train).reshape(-1, self.in_size, self.num_features)
        self.y_train = np.array(self.y_train).reshape(-1, self.out_size, self.num_features)

        self.x_test = []
        self.y_test = []
        for i in range(self.in_size, len(self.test_data) - self.out_size):
            self.x_test.append(self.test_data[i - self.in_size:i])
            self.y_test.append(self.test_data[i:i + self.out_size])
        self.x_test = np.array(self.x_test).reshape(-1, self.in_size, self.num_features)
        self.y_test = np.array(self.y_test).reshape(-1, self.out_size, self.num_features)

    def train(self):
        lr = 1e-3
        loss = losses.MeanSquaredError()
        optimizer = optimizers.Adam()
        scheduler = lambda epoch: lr if epoch < 80 else lr * tf.math.exp(0.5 * (80 - epoch))
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy', metrics.RootMeanSquaredError()],
        )
        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=50,
            callbacks=[callbacks.LearningRateScheduler(scheduler)],
        )

    def predict(self):
        predictions = self.model(self.x_test).numpy()
        x_temp = self.y_test.copy()
        x_temp[:] = predictions[:]

        predictions = self.scaler.inverse_transform(x_temp.reshape(-1, self.num_features))
        # predictions = predictions.reshape(-1, self.out_size, self.num_features)
        x_temp = None

        train = self.data_frame[:self.TRAIN_LEN]
        valid = self.data_frame[-(len(self.test_data) - self.out_size - self.in_size) + self.days_ahead:].copy(deep=True)
        last_date = valid.iloc[[-1]].index[0]  # + timedelta(days=15)
        i = pd.date_range(last_date, periods=self.days_ahead, freq='1D')
        future = pd.DataFrame({'predictions': predictions[-self.days_ahead:, 3]}, index=i)
        valid.append(pd.DataFrame(index=[last_date]))

        valid['predictions'] = predictions[-(len(self.test_data) - self.out_size - self.in_size):-self.days_ahead, 3]

        return train, valid, future

    def eval(self):
        return self._print_score()

    @staticmethod
    def append_technical_indicators(df):
        """
        Add technical indicator to input dataframe.
        :param data_frame: pandas data frame.
        :return: pandas data frame.
        """
        data_frame = df.copy()
        data_frame['ma7'] = data_frame['close'].rolling(window=7).mean()
        data_frame['ma21'] = data_frame['close'].rolling(window=21).mean()

        data_frame['26ema'] = data_frame['close'].ewm(span=26).mean()
        data_frame['12ema'] = data_frame['close'].ewm(span=12).mean()
        data_frame['MACD'] = data_frame['12ema'] - data_frame['26ema']

        data_frame['20sd'] = data_frame['close'].rolling(20).std()
        data_frame['upper_band'] = data_frame['ma21'] + 2 * data_frame['20sd']
        data_frame['lower_band'] = data_frame['ma21'] - 2 * data_frame['20sd']

        data_frame['ema'] = data_frame['close'].ewm(com=0.5).mean()

        data_frame['momentum'] = data_frame['close'] - 1

        return data_frame

    @staticmethod
    def append_pca(df, n_out):
        """perform PCA algorithm on data."""
        data_frame = df.copy()
        pca = PCA(n_out)
        components = pca.fit_transform(data_frame)
        components_df = pd.DataFrame(
            components,
            columns=[f"principal component {i}" for i in range(n_out)],
            index=data_frame.index)
        data_frame = data_frame.join(components_df)

        return data_frame

    def preprocess(self, df, n_pca):
        df = df.copy()
        df = self.append_technical_indicators(df)
        df = df.dropna()
        df.reset_index(drop=True)
        df = self.append_pca(df, n_pca)

        return df
