import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras import Model


class RawLSTM(Model):
    def __init__(self, lstm_units=50,  sequence_len=60):
        super(RawLSTM, self).__init__()
        self.LSTM_UNITS = lstm_units
        self.SEQ_LEN = sequence_len
        self.lstm = tf.keras.Sequential([
            layers.LSTM(lstm_units, return_sequences=True, input_shape=(sequence_len, 1)),
            layers.LSTM(lstm_units, return_sequences=False)
        ])

        self.dense = tf.keras.Sequential([
            layers.Dense(lstm_units//2),
            layers.Dense(1)
        ])

    def call(self, x, **kwargs):
        x = self.lstm(x)
        return self.dense(x)
