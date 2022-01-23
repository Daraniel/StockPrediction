import requests


# written by nimilioum

def get_csv(Inscode, name):  # downloads the csv file from tse site
    data = requests.get('http://www.tsetmc.com/tsev2/data/Export-txt.aspx?t=i&a=1&b=0&i=' + Inscode)
    file = open('data/{0}.csv'.format(name), 'wb')
    file.write(data.content)
    file.close()
    return
