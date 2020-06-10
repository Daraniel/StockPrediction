from PyQt5.QtWidgets import QMessageBox


class UITools:
    @staticmethod
    def popup(message, title="Info"):
        """
        displays a popup message with custom message and title
        :param message: message text
        :param title: optional message title (header)
        :return: executed message
        """
        msg = QMessageBox()
        msg.setWindowTitle(title)
        msg.setText(message)
        return msg.exec_()  # this will show our messagebox

