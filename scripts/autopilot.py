from PyQt6 import QtWidgets

from data_collector import DataCollectionUI

class ExampleNNMsgProcessor:
    def __init__(self):
        self.always_forward = True

    def nn_infer(self, message):
        #   Do smart NN inference here


        return [("forward", True)]

    def process_message(self, message, data_collector):

        commands = self.nn_infer(message)

        for command, start in commands:
            data_collector.onCarControlled(command, start)

if  __name__ == "__main__":
    import sys
    def except_hook(cls, exception, traceback):
        sys.__excepthook__(cls, exception, traceback)
    sys.excepthook = except_hook

    app = QtWidgets.QApplication(sys.argv)

    nn_brain = ExampleNNMsgProcessor()
    data_window = DataCollectionUI(nn_brain.process_message)
    data_window.show()

    app.exec()