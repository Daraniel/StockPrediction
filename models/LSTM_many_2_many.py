# import packages
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import callbacks
from tensorflow.keras import layers
from tensorflow.keras import losses, optimizers
from tensorflow.keras import metrics
from tensorflow.keras.models import Sequential

from tools.data_tools import prepare_validation_data, add_technical_indicators, add_pca


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
        valid = self.data_frame[-(len(self.test_data) - self.out_size - self.in_size) + self.days_ahead:].copy(
            deep=True)
        future = prepare_validation_data(self.days_ahead, predictions, valid)

        valid['predictions'] = predictions[-(len(self.test_data) - self.out_size - self.in_size):-self.days_ahead, 3]

        return train, valid, future

    def eval(self):
        return self._print_score()

    @staticmethod
    def preprocess(data_frame, n_pca):
        data_frame = data_frame.copy()
        data_frame = add_technical_indicators(data_frame)
        data_frame = data_frame.dropna()
        data_frame.reset_index(drop=True)
        data_frame = add_pca(data_frame, n_pca)

        return data_frame
