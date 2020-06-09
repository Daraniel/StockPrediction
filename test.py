import matplotlib.pyplot as plt
import pandas as pd
from models.SelfFeedLSTM import SelfFeedLSTM
import os
print(os. getcwd())

plt.style.use('fivethirtyeight')

data = pd.read_csv("dataSample/Bahman.Inv..csv", index_col='<DTYYYYMMDD>', parse_dates=True, engine='python')
data = data.drop(['<TICKER>', '<PER>', '<OPENINT>', '<VALUE>', '<FIRST>', '<LAST>'], axis=1)
data = data.iloc[::-1]
data = data[['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<VOL>']]
data = data.rename_axis('timestamp')
data = data.rename(columns={'<OPEN>': 'open', '<HIGH>': 'high', '<LOW>': 'low', '<CLOSE>': 'close', '<VOL>': 'volume'})

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