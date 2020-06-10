from PyQt5.QtWidgets import QMessageBox


class UITools:
    @staticmethod
    def popup(message, title="Info"):
        msg = QMessageBox()
        msg.setWindowTitle(title)
        msg.setText(message)
        return msg.exec_()  # this will show our messagebox

