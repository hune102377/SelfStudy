import ui 
import deftest

from PyQt5.QtWidgets import *
from PyQt5.QtCore  import *

import time

num_all = 3

def testtest():
    global num_all

    print('main.py의 testtest 함수 실행')
    time.sleep(1)

    num_all = deftest.plus(num_all)
    time.sleep(1)

    print('다시 main.py의 파일의 testtest 함수')
    time.sleep(1)

    print('testtest :',num_all)
    time.sleep(1)

    ui.chageUI(ui.MyApp)