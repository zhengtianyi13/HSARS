from PyQt5 import QtCore,QtGui,QtWidgets
import sys
import qtawesome

class MainUi(QtWidgets.QMainWindow) :
    def __init__(self):
        super().__init__()  #类的初始化
        self.init_ui() #画ui的类实例化

    def init_ui(self):
        self.setFixedSize(960,700) #大小
        self.main_widget=QtWidgets.QWidget() #窗口主部件
        self.main_layout=QtWidgets.QGridLayout() #创建主部件的网络布局
        self.main_widget.setLayout(self.main_layout)  #设置窗口部件为网络布局

        self.left_widget=QtWidgets.QWidget() #创建左侧布局
        self.left_widget.setObjectName('left_widget') #设名字
        self.left_layout = QtWidgets.QGridLayout()   # 创建左侧部件的网格布局层
        self.left_widget.setLayout(self.left_layout)  # 设置左侧部件布局为网格

        self.right_widget = QtWidgets.QWidget()  # 创建左侧布局
        self.right_widget.setObjectName('right_widget')  # 设名字
        self.right_layout = QtWidgets.QGridLayout()  # 创建左侧部件的网格布局层
        self.right_widget.setLayout(self.right_layout)  # 设置左侧部件布局为网格

        self.main_layout.addWidget(self.left_widget,0,0,12,2)# 左侧部件在第0行第0列，占8行3列)
        self.main_layout.addWidget(self.right_widget, 0, 2, 12, 10)  # 左侧部件在第0行第3列，占8行3列)
        self.setCentralWidget(self.main_widget)  #设置窗口主部件


#生成左侧按钮，左侧布局——————————————————————————————————————————————————————————
        self.left_close=QtWidgets.QPushButton("")# 关闭按钮
        self.left_visit=QtWidgets.QPushButton("")# 空白按钮
        self.left_mini=QtWidgets.QPushButton("")# 最小化按钮

        self.left_label_1=QtWidgets.QPushButton("骨架生成")
        self.left_label_1.setObjectName('left_label')
        self.left_label_2 = QtWidgets.QPushButton("动作识别")
        self.left_label_2.setObjectName('left_label')
        self.left_label_3 = QtWidgets.QPushButton("联系与帮助")
        self.left_label_3.setObjectName('left_label')

        self.left_button_1 = QtWidgets.QPushButton(qtawesome.icon('fa.music',color='white'),"HRnet")
        self.left_button_1.setObjectName('left_button')
        self.left_button_2 = QtWidgets.QPushButton(qtawesome.icon('fa.music', color='white'), "Alphapose")
        self.left_button_2.setObjectName('left_button')
        self.left_button_3 = QtWidgets.QPushButton(qtawesome.icon('fa.music', color='white'), "TaNet")
        self.left_button_3.setObjectName('left_button')
        self.left_button_4 = QtWidgets.QPushButton(qtawesome.icon('fa.music', color='white'), "TSN")
        self.left_button_4.setObjectName('left_button')
        self.left_button_5 = QtWidgets.QPushButton(qtawesome.icon('fa.music', color='white'), "QQ123456")
        self.left_button_5.setObjectName('left_button')
        self.left_button_6 = QtWidgets.QPushButton(qtawesome.icon('fa.music', color='white'), "华语流行")
        self.left_button_6.setObjectName('left_button')

        self.left_layout.addWidget(self.left_mini,0,0,1,1)
        self.left_layout.addWidget(self.left_close, 0, 2, 1, 1)
        self.left_layout.addWidget(self.left_visit, 0, 1, 1, 1)
        self.left_layout.addWidget(self.left_label_1, 1, 0, 1, 3)
        self.left_layout.addWidget(self.left_button_1, 2, 0, 1, 3)
        self.left_layout.addWidget(self.left_button_2, 3, 0, 1, 3)
        self.left_layout.addWidget(self.left_label_2, 4, 0, 1, 3)
        self.left_layout.addWidget(self.left_button_3, 5, 0, 1, 3)
        self.left_layout.addWidget(self.left_button_4, 6, 0, 1, 3)
        self.left_layout.addWidget(self.left_label_3, 7, 0, 1, 3)
        self.left_layout.addWidget(self.left_button_5, 8, 0, 1, 3)


#生成右侧画框，右侧布局————————————————————————————————————————————————————
        self.showlabel=QtWidgets.QLabel()
        self.showlabel.setObjectName('showlabel')  #显示框
       # self.showlabel.setStyleSheet('QLabel{background:#F76677}')

        self.labeltext = QtWidgets.QLabel()
        self.labeltext.setObjectName('labeltext')  #识别动作
        #self.labeltext.setStyleSheet('QLabel{background:#F76677}')

        self.right_layout.addWidget(self.showlabel, 0, 0, 8, 7)
        self.right_layout.addWidget(self.labeltext, 11, 0, 1, 7)



#设置样式qss----------------------------------------------
        self.left_mini.setFixedSize(15,15)  #设置大小
        self.left_visit.setFixedSize(15, 15)
        self.left_close.setFixedSize(15, 15)

        self.left_close.setStyleSheet('''QPushButton{background:#F76677;border-radius:5px;}QPushButton:hover{background:red;}''')
        self.left_visit.setStyleSheet('''QPushButton{background:#F7D674;border-radius:5px;}QPushButton:hover{background:yellow;}''')
        self.left_mini.setStyleSheet('''QPushButton{background:#6DDF6D;border-radius:5px;}QPushButton:hover{background:green;}''')
        self.setWindowOpacity(0.9)  # 设置窗口透明度
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground) # 设置窗口背景透明

        self.left_widget.setStyleSheet(
'''
    QPushButton{border:none;color:white;}
    QPushButton#left_label{
        border:none;
        border-bottom:1px solid white;
        font-size:18px;
        font-weight:700;
        font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    }
    QPushButton#left_button:hover{border-left:4px solid red;font-weight:700;}
    QWidget#left_widget{
        background:black;
        border-top:1px solid white;
        border-bottom:1px solid white;
        border-left:1px solid white;
        border-top-left-radius:10px;
        border-bottom-left-radius:10px;
        }
 
    
'''

)

        self.right_widget.setStyleSheet(
'''

    QWidget#right_widget{
        color:#232C51;
        background:white;
        border-top:1px solid darkGray;
        border-bottom:1px solid darkGray;
        border-right:1px solid darkGray;
        border-top-right-radius:10px;
        border-bottom-right-radius:10px;
    }

    QLabel#right_lable{
        border:none;
        font-size:16px;
        font-weight:700;
        font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    }

'''
)


        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)# 隐藏边框
        self.main_layout.setSpacing(0)





# 设置窗口背景透明

def main():
        app=QtWidgets.QApplication(sys.argv)
        gui=MainUi()
        gui.show()
        sys.exit(app.exec_())

if __name__ =='__main__':
    main()


