import multiprocessing
from datetime import datetime, timedelta

from PyQt5 import uic, QtWidgets
import sys

from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.dates as mdates
from nibabel.tests.test_viewers import matplotlib

from models.MetaTrader5Interface import MetaTrader5Interface as mt5i
from models.UITools import MplCanvas, PandasModel, UITools
from models.RandomForest import RandomForest
from models.LstmMany2Many import LstmMany2Many

import pandas_datareader.data as web

matplotlib.use('Qt5Agg')


class UI:
    def __init__(self, dialog):
        """
        Initializes the gui on input window had to use dialog for its name because pycharm won't let me use window
        variable name for it because I used it in another scoop!
        :param dialog: parent window (Gui main window)
        """
        self.isIran = True
        self.window = dialog
        self.topGraph = MplCanvas(self.window, width=5, height=4, dpi=100)
        self.topGraph.axes.plot([0, 1, 2, 3, 4], [10, 1, 20, 3, 40])

        self.topToolbar = NavigationToolbar(self.topGraph, self.window)
        self.topLayout = QtWidgets.QVBoxLayout()
        self.topLayout.addWidget(self.topToolbar)
        self.topLayout.addWidget(self.topGraph)

        self.topWidget = self.window.findChild(QtWidgets.QWidget, 'topGraph')
        self.topWidget.setLayout(self.topLayout)

        # self.bottomGraph = MplCanvas(self.window, width=5, height=4, dpi=100)
        # self.bottomGraph.axes.plot([0, 1, 2, 3, 4], [10, 1, 20, 3, 40])

        # self.bottomToolbar = NavigationToolbar(self.bottomGraph, self.window)
        # self.bottomLayout = QtWidgets.QVBoxLayout()
        # self.bottomLayout.addWidget(self.bottomToolbar)
        # self.bottomLayout.addWidget(self.bottomGraph)

        # self.bottomWidget = self.window.findChild(QtWidgets.QWidget, 'bottomGraph')
        # self.bottomWidget.setLayout(self.bottomLayout)

        self.searchField = self.window.findChild(QtWidgets.QLineEdit, 'searchField')
        self.searchField.textChanged.connect(self.search)

        self.stockList = self.window.findChild(QtWidgets.QListWidget, 'stockList')
        self.stockList.itemClicked.connect(self.stock_list_item_click)

        self.stockName = self.window.findChild(QtWidgets.QLabel, 'stockName')

        self.daysComboBox = self.window.findChild(QtWidgets.QComboBox, 'daysComboBox')
        self.daysComboBox.addItems(['1', '5', '15', '30'])
        self.daysComboBox.currentTextChanged.connect(self.set_stock_date)
        # self.tradeHistoryTable = self.window.findChild(QtWidgets.QTableView, 'tradeHistoryTable')

        self.algorithmComboBox = self.window.findChild(QtWidgets.QComboBox, 'algorithmComboBox')
        self.algorithmComboBox.addItems(['Random Forest', 'LSTM Many2Many'])
        self.algorithmComboBox.currentTextChanged.connect(self.set_stock_date)

        self.stockNameLineEdit = self.window.findChild(QtWidgets.QLineEdit, 'stockNameLineEdit')

        self.searchStockButton = self.window.findChild(QtWidgets.QPushButton, 'searchStockButton')
        self.searchStockButton.clicked.connect(self.search_stock_sutton_clicked)

        self.symbols = mt5i.get_symbols(self.searchField.text())
        if self.symbols is not None and len(self.symbols) != 0:
            self.stockList.addItems(self.symbols)

    def search_stock_sutton_clicked(self):
        self.isIran = False
        self.stockName.setText(str(self.stockNameLineEdit.text()))
        self.set_stock_date()

    def search(self):
        """
        Searches for symbol in mql5 and adds similar symbols to gui list
        This function activates on text change
        """
        self.stockList.clear()
        self.stockList.addItems([s for s in self.symbols if self.searchField.text() in s])

    def stock_list_item_click(self, item):
        """
        Gets information for the selected stock from the list and draws its graphs
        :param item: selected stock
        """
        self.isIran = True
        self.stockName.setText(str(item.text()))
        self.set_stock_date()

    def set_stock_date(self):
        # draw price history and prediction
        data = ''
        if str(self.stockName.text()) == 'لطفا نمادی را انتخاب کنید':
            return

        self.topGraph.axes.cla()

        if self.isIran:
            manager = multiprocessing.Manager()
            return_dict = manager.dict()
            p = multiprocessing.Process(target=mt5i.get_symbol_data, args=(self.stockName.text(), return_dict))
            p.start()
            p.join(120)
            if p.is_alive():
                print("running... let's kill it...")
                # Terminate
                p.terminate()
                p.join()

            if 'stock_data' not in return_dict:
                UITools.popup("Server Timeout!")
                return

            data = return_dict['stock_data']
            data = data.set_index('time')
            data = data.drop(['tick_volume', 'spread'], 1)

        else:
            time_to = datetime.now()
            time_from = time_to - timedelta(weeks=300)
            try:
                data = web.DataReader(str(self.stockNameLineEdit.text()), data_source='yahoo', start=time_from,
                                      end=time_to)
            except:
                UITools.popup('Stock not found!')
                return
            data = data.drop(['Adj Close'], 1)
            data = data.rename(columns={'High': 'high', 'Low': 'low', 'Close': 'close', 'Open': 'open',
                                        'Volume': 'real_volume'})
            data = data[['open', 'high', 'low', 'close', 'real_volume']]
            data.index.names = ['time']

        model = ''
        if self.algorithmComboBox.currentText() == 'Random Forest':
            model = RandomForest(data, days_ahead=int(self.daysComboBox.currentText()))
        elif self.algorithmComboBox.currentText() == 'LSTM Many2Many':
            model = LstmMany2Many(data, days_ahead=int(self.daysComboBox.currentText()))
        model.train()
        train, valid, predictions = model.predict()
        model = None

        self.topGraph.axes.plot(train['close'])
        self.topGraph.axes.plot(valid[['close', 'predictions']])
        self.topGraph.axes.plot(predictions[['predictions']])
        self.topGraph.axes.legend(['Train', 'Valid', 'Predictions', 'Future'], loc='upper left')

        self.topGraph.axes.xaxis.set_major_locator(mdates.DayLocator(interval=10))
        self.topGraph.axes.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        self.topGraph.draw()

        # # draw today's last 100 exchange bid and ask graph
        # data = data.iloc[-300:]
        # data = data.drop(columns=['open', 'high', 'low'])
        # self.bottomGraph.axes.cla()
        # self.bottomGraph.axes.plot(data.index, data['close'], label='bids')
        # self.bottomGraph.axes.xaxis.set_major_locator(mdates.MinuteLocator(interval=15))
        # self.bottomGraph.axes.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        # self.bottomGraph.axes.legend()
        # self.bottomGraph.draw()
        #
        # data['time'] = str(data.index)[26:35]
        # data = data[['time', 'close', 'real_volume']]
        # model = PandasModel(data)
        # self.tradeHistoryTable.setModel(model)
        #
        # header = self.tradeHistoryTable.horizontalHeader()
        # header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        # header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        # header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        # data = None


if __name__ == '__main__':
    # good old function!
    Form, Window = uic.loadUiType('ui/mini_gui.ui')
    app = QtWidgets.QApplication([])

    window = Window()
    form = Form()
    form.setupUi(window)
    ui = UI(window)

    window.setFixedSize(window.size())
    window.show()
    window.actionQuit = QtWidgets.QAction("Quit")
    window.actionQuit.triggered.connect(QtWidgets.QApplication.quit)

    app.exec_()
    sys.exit(app.exec_())
