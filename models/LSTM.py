import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras import Model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np


class RawLSTM(Model):
    def __init__(self, lstm_units=50, sequence_len=60):
        super(RawLSTM, self).__init__()
        self.LSTM_UNITS = lstm_units
        self.SEQ_LEN = sequence_len
        self.lstm = tf.keras.Sequential([
            layers.LSTM(lstm_units, return_sequences=True, input_shape=(sequence_len, 1)),
            layers.LSTM(lstm_units, return_sequences=False)
        ])

        self.dense = tf.keras.Sequential([
            layers.Dense(lstm_units // 2),
            layers.Dense(1)
        ])

    def call(self, x, **kwargs):
        x = self.lstm(x)
        return self.dense(x)


# TODO: config files
# TODO: cleanup
class SelfFeedLSTM:
    def __init__(self, data_frame: pd.DataFrame, **kwargs):
        self.model = RawLSTM()
        self.data_frame = data_frame
        self.__preprocess()

    def __preprocess(self, **kwargs):
        self.data = self.data_frame.filter(['close'])
        dataset = np.asarray(self.data, dtype=np.float32).reshape(-1, 1)

        self.training_data_len = int(len(dataset) * 0.8)
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_dataset = self.scaler.fit_transform(dataset)

        train_data = scaled_dataset[:self.training_data_len, :]
        self.x_train = []
        self.y_train = []
        for i in range(60, len(train_data)):
            self.x_train.append(train_data[i - 60:i, 0])
            self.y_train.append(train_data[i, 0])
        self.x_train, self.y_train = np.array(self.x_train), np.array(self.y_train)
        self.x_train = np.reshape(self.x_train, (self.x_train.shape[0], self.x_train.shape[1], 1))
        self.test_data = scaled_dataset[self.training_data_len - 60:, :]

        self.x_test = []
        self.y_test = dataset[self.training_data_len:, :]
        for i in range(60, len(self.test_data)):
            self.x_test.append(self.test_data[i - 60:i, 0])
        self.x_test = np.array(self.x_test)

        self.x_test = np.reshape(self.x_test, (self.x_test.shape[0], self.x_test.shape[1], 1))

    @tf.function
    def train_step(self):
        pass

    @tf.function
    def test_step(self):
        pass

    def train(self):
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3,
                                                  beta_1=0.9,
                                                  beta_2=0.999)

        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.model.fit(self.x_train, self.y_train, batch_size=1000, epochs=10)

    def predict(self):
        predictions = self.model(self.x_test)
        predictions = self.scaler.inverse_transform(predictions)

        train = self.data[:self.training_data_len]
        valid = self.data[self.training_data_len:]
        valid['predictions'] = predictions

        return train, valid
