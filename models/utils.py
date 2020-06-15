import pandas as pd
import numpy as np

def from_mofidtrader(data_path: str) -> pd.DataFrame:
    data = pd.read_csv(data_path, index_col='<DTYYYYMMDD>', parse_dates=True, engine='python')
    data = data.drop(['<TICKER>', '<PER>', '<OPENINT>', '<VALUE>', '<FIRST>', '<LAST>'], axis=1)
    data = data.iloc[::-1]
    data = data[['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<VOL>']]
    data = data.rename_axis('timestamp')
    data = data.rename(
        columns={'<OPEN>': 'open', '<HIGH>': 'high', '<LOW>': 'low', '<CLOSE>': 'close', '<VOL>': 'volume'})
    return data


def predict_ndays(model, x, days):
    last_seq = x[-1:]

    predicted_days = []
    for _ in range(days):
        pred = model(last_seq)
        predicted_days.append(pred.numpy()[0, 0])
        last_seq = np.append(last_seq, pred)[1:].reshape(1, 60, 1)

    return np.array(predicted_days)