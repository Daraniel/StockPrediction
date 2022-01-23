import pandas as pd
from sklearn.decomposition import PCA


def read_data_frame(file_name: str) -> pd.DataFrame:
    data_frame = pd.read_csv(file_name, index_col='<DTYYYYMMDD>', parse_dates=True, engine='python')
    data_frame = data_frame.drop(['<TICKER>', '<PER>', '<OPENINT>', '<VALUE>', '<FIRST>', '<LAST>'], axis=1)
    data_frame = data_frame.iloc[::-1]
    data_frame = data_frame[['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<VOL>']]
    data_frame = data_frame.rename_axis('timestamp')
    data_frame = data_frame.rename(
        columns={'<OPEN>': 'open', '<HIGH>': 'high', '<LOW>': 'low', '<CLOSE>': 'close', '<VOL>': 'volume'})
    return data_frame


# def prepare_multi_day_prediction_data(model, x, days_count: int) -> np.ndarray:
#     last_seq = x[-1:]
#
#     predicted_days = []
#     for _ in range(days_count):
#         pred = model(last_seq)
#         predicted_days.append(pred.numpy()[0, 0])
#         last_seq = np.append(last_seq, pred)[1:].reshape(1, 60, 1)
#
#     return np.array(predicted_days)


def prepare_validation_data(days_ahead: int, predictions: pd.DataFrame, valid: pd.DataFrame) -> pd.DataFrame:
    last_date = valid.iloc[[-1]].index[0]  # + timedelta(days=15)
    i = pd.date_range(last_date, periods=days_ahead, freq='1D')
    future = pd.DataFrame({'predictions': predictions[-days_ahead:, 3]}, index=i)
    valid.append(pd.DataFrame(index=[last_date]))
    return future


def add_technical_indicators(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicator to input dataframe.
    :param data_frame: pandas data frame.
    :return: pandas data frame.
    """
    data_frame = data_frame.copy()
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


def add_pca(data_frame: pd.DataFrame, n_out: int) -> pd.DataFrame:
    """perform PCA algorithm on data."""
    data_frame = data_frame.copy()
    pca = PCA(n_out)
    components = pca.fit_transform(data_frame)
    components_df = pd.DataFrame(
        components,
        columns=[f"principal component {i}" for i in range(n_out)],
        index=data_frame.index)
    data_frame = data_frame.join(components_df)

    return data_frame
