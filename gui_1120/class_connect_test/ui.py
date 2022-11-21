import main
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore  import *


class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('test')
        self.move(300, 300)
        self.resize(300, 500)

        btn1 = QPushButton('&Button1', self)
        btn1.setCheckable(True)
        btn1.move(100, 300)
        btn1.clicked.connect(main.testtest)

        btn2 = QPushButton('&Button2', self)
        btn2.setCheckable(True)
        btn2.move(100, 350)
        btn2.clicked.connect(self.changeUIui)

        self.label1 = QLineEdit(self)
        self.label1.setGeometry(QRect(0, 0, 200, 40))
        self.label1.setText(str(main.num_all))

        self.show()

        # self는 main
        print(self)

    def changeUIui(self) :
        print('ChangeUiui')
        main.num_all +=1
        self.label1.setText(str(main.num_all))

def chageUI(self) :
    self.label1.setText(str(main.num_all))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())

