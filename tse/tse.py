from tse import get_tse_data


# written by nimilioum

def parse_symbols():
    stock = open("tse/stocks.txt", 'r', encoding="utf-8")
    symbols = []
    for line in stock.readlines():
        data = line.split()
        symbols.append(data[0])
    return symbols


def get_csv(symbol):  # gets symbol and calls a function to download the related csv file
    stock = open("tse/stocks.txt", 'r', encoding="utf-8")
    data = []
    for line in stock.readlines():
        if symbol in line:
            data = line.split()
            break
    get_tse_data.get_csv(data[1], data[0])
    return
