from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import threading
import cv2
from uiwindow import Ui_window
from hrnet.tools import pose
from mmaction2.demo import webcam_demo
from PyQt5 import QtCore
import os
class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        # 从文件中加载UI定义
        # 从 UI 定义中动态 创建一个相应的窗口对象
        flag=0
        self.ui = Ui_window()
        self.ui.setupUi(self)  #画出窗口
        styleFile = os.getcwd() + '\style.qss'
        self.setStyleSheet(str(self.LoadStyleFromQss(styleFile)))
        self.setWindowFlags(QtCore.Qt.CustomizeWindowHint)
        # self.ui.pose.clicked.connect(self.play_video_pose) #拍照识别
        # self.ui.action.clicked.connect(self.play_video_activity)  # 动作识别

        self.ui.pose.clicked.connect(self.modechange(1))  # 拍照识别
        self.thread = None  # 创建线程

    def play_video_pose(self):  #创建线程不断调用摄像头捕捉画面
        self.thread = threading.Thread(target=self.detect)  #线程调用的是detect
        self.thread.start()
        print("线程开启")

    def play_video_activity(self):  #创建线程不断调用摄像头捕捉画面
        self.thread = threading.Thread(target=self.react)  #线程调用的是detect
        self.thread.start()
        print("线程开启")

    def LoadStyleFromQss(self, f):
        file = open(f)
        lines = file.readlines()
        file.close()
        res = ''
        for line in lines:
            res += line
        return res
    # def windowmode(self):

    # def modechange(self):


    def detect(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 拿到摄像头的流
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            img = pose.main(frame)

            convert_to_qt_format = QImage(img.data, img.shape[1], img.shape[0],
                                          QImage.Format_RGB888)  #转成qt格式！！！！！
            self.ui.show.setPixmap(QPixmap.fromImage(convert_to_qt_format))  #显示
        # self.cap.release()
        cv2.destroyAllWindows()

    def react(self):
        webcam_demo.main()


# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助

app = QApplication([])
mainwindow = MainWindow()
mainwindow.show()
app.exec_()
