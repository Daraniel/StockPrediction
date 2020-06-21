import multiprocessing

import matplotlib
import pandas as pd
from PyQt5 import uic, QtWidgets, QtCore
import sys

from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.dates as mdates

from models.MetaTrader5Interface import MetaTrader5Interface as mt5i
from models.UITools import MplCanvas, PandasModel, UITools
from models.RandomForest import RandomForest

matplotlib.use('Qt5Agg')


class UI:
    def __init__(self, dialog):
        """
        Initializes the gui on input window had to use dialog for its name because pycharm won't let me use window
        variable name for it because I used it in another scoop!
        :param dialog: parent window (Gui main window)
        """
        self.window = dialog
        self.topGraph = MplCanvas(self.window, width=5, height=4, dpi=100)
        self.topGraph.axes.plot([0, 1, 2, 3, 4], [10, 1, 20, 3, 40])

        self.topToolbar = NavigationToolbar(self.topGraph, self.window)
        self.topLayout = QtWidgets.QVBoxLayout()
        self.topLayout.addWidget(self.topToolbar)
        self.topLayout.addWidget(self.topGraph)

        self.topWidget = self.window.findChild(QtWidgets.QWidget, 'topGraph')
        self.topWidget.setLayout(self.topLayout)

        self.bottomGraph = MplCanvas(self.window, width=5, height=4, dpi=100)
        self.bottomGraph.axes.plot([0, 1, 2, 3, 4], [10, 1, 20, 3, 40])

        self.bottomToolbar = NavigationToolbar(self.bottomGraph, self.window)
        self.bottomLayout = QtWidgets.QVBoxLayout()
        self.bottomLayout.addWidget(self.bottomToolbar)
        self.bottomLayout.addWidget(self.bottomGraph)

        self.bottomWidget = self.window.findChild(QtWidgets.QWidget, 'bottomGraph')
        self.bottomWidget.setLayout(self.bottomLayout)

        self.searchField = self.window.findChild(QtWidgets.QLineEdit, 'searchField')
        self.searchField.textChanged.connect(self.search)

        self.stockList = self.window.findChild(QtWidgets.QListWidget, 'stockList')
        self.stockList.itemClicked.connect(self.stock_list_item_click)

        self.stockName = self.window.findChild(QtWidgets.QLabel, 'stockName')

        self.tradeHistoryTable = self.window.findChild(QtWidgets.QTableView, 'tradeHistoryTable')

        self.search()

    def search(self):
        """
        Searches for symbol in mql5 and adds similar symbols to gui list
        This function activates on text change
        """
        self.stockList.clear()
        symbols = mt5i.get_symbols(self.searchField.text())
        if symbols is not None and len(symbols) != 0:
            self.stockList.addItems(symbols)

    def stock_list_item_click(self, item):
        """
        Gets information for the selected stock from the list and draws its graphs
        :param item: selected stock
        """
        print("click!")
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        # draw price history and prediction
        self.stockName.setText(str(item.text()))
        self.topGraph.axes.cla()
        p = multiprocessing.Process(target=mt5i.get_symbol_data, args=(item.text(), return_dict))
        p.start()
        p.join(60)
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

        random_forest = RandomForest(data)
        random_forest.train()
        train, valid = random_forest.predict()

        self.topGraph.axes.plot(train['close'])
        self.topGraph.axes.plot(valid[['close', 'predictions']])
        self.topGraph.axes.legend(['Train', 'Valid', 'Predictions'], loc='lower right')

        self.topGraph.axes.xaxis.set_major_locator(mdates.DayLocator(interval=15))
        self.topGraph.axes.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        self.topGraph.draw()

        print("tick!")
        # draw today's last 100 exchange bid and ask graph
        # data2 = mt5i.get_symbol_current_orders(item.text())
        # self.bottomGraph.axes.cla()
        # if data2 is not None and len(data2) > 0:
        data = data.iloc[-300:]
        data = data.drop(columns=['open', 'high', 'low'])
        self.bottomGraph.axes.cla()
        self.bottomGraph.axes.plot(data.index, data['close'], label='bids')
        #self.bottomGraph.axes.plot(data2['time'], data['ask'], color='orange', label='asks')
        self.bottomGraph.axes.xaxis.set_major_locator(mdates.MinuteLocator(interval=15))
        self.bottomGraph.axes.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        self.bottomGraph.axes.legend()
        self.bottomGraph.draw()

        data['time'] = str(data.index)[18:26]
        model = PandasModel(data)
        self.tradeHistoryTable.setModel(model)

        header = self.tradeHistoryTable.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeToContents)


if __name__ == '__main__':
    # good old function!
    Form, Window = uic.loadUiType('ui/gui.ui')
    app = QtWidgets.QApplication([])

    window = Window()
    form = Form()
    form.setupUi(window)
    ui = UI(window)

    window.setFixedSize(window.size())
    window.show()
    app.exec_()
    sys.exit(app.exec_())
