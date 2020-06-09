import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


class SelfFeedLSTM():
    def __init__(self, data):
        self.data = data.filter(['close'])
        dataset = np.asarray(self.data, dtype=np.float32).reshape(-1, 1)

        # get the number of rows to train the model
        self.training_data_len = math.ceil(len(dataset) * 0.8)

        # Scale the data
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_dataset = self.scaler.fit_transform(dataset)

        # Create the training dataset
        train_data = scaled_dataset[:self.training_data_len, :]

        # split the data into x_train and y_train
        self.x_train = []
        self.y_train = []

        for i in range(60, len(train_data)):
            self.x_train.append(train_data[i - 60:i, 0])
            self.y_train.append(train_data[i, 0])

        # convert x_train and y_train to numpy array
        self.x_train, self.y_train = np.array(self.x_train), np.array(self.y_train)

        # Reshape data
        self.x_train = np.reshape(self.x_train, (self.x_train.shape[0], self.x_train.shape[1], 1))

        # Create the testing dataset
        self.test_data = scaled_dataset[self.training_data_len - 60:, :]

        # create x_test and y_test
        self.x_test = []
        self.y_test = dataset[self.training_data_len:, :]

        for i in range(60, len(self.test_data)):
            self.x_test.append(self.test_data[i - 60:i, 0])

        # convert data to numpy array
        self.x_test = np.array(self.x_test)

        # Reshape data from 2d to 3d
        self.x_test = np.reshape(self.x_test, (self.x_test.shape[0], self.x_test.shape[1], 1))

        self.model = Sequential()
        self.model.add(LSTM(50, return_sequences=True, input_shape=(self.x_train.shape[1], 1)))
        self.model.add(LSTM(50, return_sequences=False))
        self.model.add(Dense(25))
        self.model.add(Dense(1))

        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def train(self):
        self.model.fit(self.x_train, self.y_train, batch_size=1000, epochs=10)

    def predict(self):
        # Get predictions
        predictions = self.model(self.x_test)
        predictions = self.scaler.inverse_transform(predictions)

        # compute rmse
        rmse = np.sqrt(np.mean((predictions - self.y_test) ** 2))
        print(rmse)

        # Plot the data
        train = self.data[:self.training_data_len]
        valid = self.data[self.training_data_len:]
        valid['predictions'] = predictions

        return train, valid

    def predict_days(self, days):
        last_seq = self.x_test[-1:]

        predicted_days = []
        for _ in range(days):
            pred = self.model(last_seq)
            predicted_days.append(pred.numpy()[0, 0])
            last_seq = np.append(last_seq, pred)[1:].reshape(1, 60, 1)  # append predicted item and remove fist item

        return np.array(predicted_days)
