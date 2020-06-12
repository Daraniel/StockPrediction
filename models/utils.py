import pandas as pd

def from_mofidtrader(data_path: str) -> pd.DataFrame:
    data = pd.read_csv(data_path, index_col='<DTYYYYMMDD>', parse_dates=True, engine='python')
    data = data.drop(['<TICKER>', '<PER>', '<OPENINT>', '<VALUE>', '<FIRST>', '<LAST>'], axis=1)
    data = data.iloc[::-1]
    data = data[['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<VOL>']]
    data = data.rename_axis('timestamp')
    data = data.rename(
        columns={'<OPEN>': 'open', '<HIGH>': 'high', '<LOW>': 'low', '<CLOSE>': 'close', '<VOL>': 'volume'})
    return data
