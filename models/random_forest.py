# import packages
import math

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

from tools.data_tools import prepare_validation_data


class RandomForest:
    def __init__(self, data_frame: pd.DataFrame, days_ahead=15):
        """
        Initializes Random Forest
        :param days_ahead: Number of days in the future to predict
        :param data_frame: Pandas Data Frame containing stock data
        """
        self.days_ahead = days_ahead
        self.num_features = len(data_frame.columns)
        self.data_frame = data_frame
        self._preprocess()

        self.model = RandomForestRegressor(n_estimators=100, bootstrap=True, min_samples_split=5, min_samples_leaf=5,
                                           n_jobs=-1)

    def _preprocess(self):
        dataset = np.asarray(self.data_frame, dtype=np.float32).reshape(-1, self.num_features)

        self.training_data_len = int(len(dataset) * 0.8)
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_dataset = self.scaler.fit_transform(dataset)

        # train_data = scaled_dataset[:self.training_data_len, :]
        self.X_train = (scaled_dataset[:self.training_data_len, :])[:-self.days_ahead]
        self.y_train = (scaled_dataset[:self.training_data_len, :])[self.days_ahead:, 3]

        # test_data = scaled_dataset[self.training_data_len:, :]
        self.X_val = (scaled_dataset[self.training_data_len:, :])
        # self.X_val = (scaled_dataset[self.training_data_len:, :])[:-self.days_ahead]
        self.y_val = (scaled_dataset[self.training_data_len:, :])[self.days_ahead:, 3]

        dataset = None
        scaled_dataset = None

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        predictions = self.model.predict(self.X_val)
        x_temp = self.X_val.copy()

        fft = np.fft.fft(predictions)
        fft[30:-30] = 0
        predictions = np.fft.ifft(fft)
        predictions = np.abs(predictions)

        x_temp[:, 3] = predictions

        predictions = self.scaler.inverse_transform(x_temp)
        x_temp = None

        train = self.data_frame[:self.training_data_len]
        valid = self.data_frame[self.training_data_len + self.days_ahead:].copy(deep=True)
        future = prepare_validation_data(self.days_ahead, predictions, valid)

        valid['predictions'] = predictions[:-self.days_ahead, 3]

        return train, valid, future

    def eval(self):
        return self._print_score()

    @staticmethod
    def _rmse(x, y): return math.sqrt(((x - y) ** 2).mean())

    def _print_score(self):
        res = [self._rmse(self.model.predict(self.X_train), self.y_train), self._rmse(self.model.predict(self.X_val),
                                                                                      self.y_val),
               self.model.score(self.X_train, self.y_train), self.model.score(self.X_val, self.y_val)]
        return f"rmse train {res[0]}, rmse val {res[1]}, r^2 train {res[2]}, r^2 val {res[3]}"
