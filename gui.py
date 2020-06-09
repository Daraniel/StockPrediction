import matplotlib
from PyQt5 import uic, QtWidgets

from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

matplotlib.use('Qt5Agg')


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


Form, Window = uic.loadUiType("ui/gui.ui")
app = QtWidgets.QApplication([])
window = Window()
form = Form()

sc = MplCanvas(window, width=5, height=4, dpi=100)
sc.axes.plot([0,1,2,3,4], [10,1,20,3,40])

# Create toolbar, passing canvas as first parament, parent (self, the MainWindow) as second.
toolbar = NavigationToolbar(sc, window)
layout = QtWidgets.QVBoxLayout()
layout.addWidget(toolbar)
layout.addWidget(sc)
form.setupUi(window)

widget = window.findChild(QtWidgets.QWidget, "widget")
widget.setLayout(layout)

window.show()
app.exec_()
