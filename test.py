import matplotlib.pyplot as plt

from models.raw_LSTM import SelfFeedLSTM
from tools.data_tools import read_data_frame

plt.style.use('fivethirtyeight')

data = read_data_frame("dataSample/Bahman.Inv..csv")

print("Create model")
lstm = SelfFeedLSTM(data)
print("Train model")
lstm.train()
print("Predict with model")
train, valid = lstm.predict()
print("Finish")

plt.figure(figsize=(16, 8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['close'])
plt.plot(valid[['close', 'predictions']])
plt.legend(['Train', 'Valid', 'Predictions'], loc='lower right')
plt.show()
