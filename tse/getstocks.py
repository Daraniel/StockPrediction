import re

import requests


# written by nimilioum

def main():
    req = requests.post(r'http://www.tsetmc.com/Loader.aspx?ParTree=111C1417')
    source = req.text
    symbol = re.compile(r'">')
    resultFile = open("stocks.txt", 'w', encoding="utf-8")
    isource = iter(source.splitlines())
    for line in isource:
        if re.search('<a href', line):  # InsCode and symbol both reside in a line containing <a href ...>
            data = data_comp(line)
            sym = data[1]
            InsCode = data[0]
            if sym[-1] != "ح":
                resultFile.write(sym + " " + InsCode + "\n")
            line = next(isource)
            line = next(isource)  # since there are 2 lines containing <a href...> and the second is useless,
    resultFile.close()  # I made an iterator to ignore the second one :)
    print("done...")


# gets InsCode and symbol from the specified line
def data_comp(line):
    line = line.strip()
    data = line.replace("""<td><a href='loader.aspx?ParTree=111C1412&inscode=""", "")
    data = data.replace("'", "")
    data = data.replace("""target="_blank">""", "")
    data = data.replace("""</a></td>""", "")
    data = data.split()
    return data


main()
