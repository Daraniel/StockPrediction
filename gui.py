import matplotlib
from PyQt5 import uic, QtWidgets
import sys

from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.dates as mdates


from models.MetaTrader5Interface import MetaTrader5Interface as mt5i

matplotlib.use('Qt5Agg')


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        """
        Creates a canvas used to draw matplotlib.pyplot (or plt) plots in pyqt5 gui
        :param parent: parent gui element which houses the plot
        :param width: plot width
        :param height: plot height
        :param dpi: plot resolution
        """
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


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
        # draw price history and prediction
        self.stockName.setText(str(item.text()))
        data = mt5i.get_symbol_data(item.text())
        self.topGraph.axes.cla()
        self.topGraph.axes.plot(data['time'], data['close'])
        self.topGraph.axes.xaxis.set_major_locator(mdates.DayLocator(interval=15))
        self.topGraph.axes.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        self.topGraph.draw()

        # draw today's last 100 exchange bid and ask graph
        data2 = mt5i.get_symbol_current_orders(item.text())
        print(data2)
        data2 = data2.iloc[-100:]
        self.bottomGraph.axes.cla()
        self.bottomGraph.axes.plot(data2['time'], data2['bid'], label='bids')
        self.bottomGraph.axes.plot(data2['time'], data2['ask'], color='orange', label='asks')
        self.bottomGraph.axes.xaxis.set_major_locator(mdates.MinuteLocator(interval=15))
        self.bottomGraph.axes.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        self.bottomGraph.axes.legend()
        self.bottomGraph.draw()


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
