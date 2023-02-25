# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'project.ui'
##
## Created by: Qt User Interface Compiler version 6.4.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QLayout,
    QMainWindow, QMenuBar, QPushButton, QSizePolicy,
    QStatusBar, QVBoxLayout, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(800, 628)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.horizontalLayout_2 = QHBoxLayout(self.centralwidget)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setSizeConstraint(QLayout.SetNoConstraint)
        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setSpacing(80)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(-1, -1, 0, -1)
        self.pushButton_img = QPushButton(self.centralwidget)
        self.pushButton_img.setObjectName(u"pushButton_img")
        sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_img.sizePolicy().hasHeightForWidth())
        self.pushButton_img.setSizePolicy(sizePolicy)
        self.pushButton_img.setMinimumSize(QSize(150, 100))
        self.pushButton_img.setMaximumSize(QSize(150, 100))
        font = QFont()
        font.setFamilies([u"Agency FB"])
        font.setPointSize(12)
        self.pushButton_img.setFont(font)

        self.verticalLayout.addWidget(self.pushButton_img, 0, Qt.AlignHCenter)

        self.pushButton_camera = QPushButton(self.centralwidget)
        self.pushButton_camera.setObjectName(u"pushButton_camera")
        sizePolicy1 = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.pushButton_camera.sizePolicy().hasHeightForWidth())
        self.pushButton_camera.setSizePolicy(sizePolicy1)
        self.pushButton_camera.setMinimumSize(QSize(150, 100))
        self.pushButton_camera.setMaximumSize(QSize(150, 100))
        self.pushButton_camera.setFont(font)

        self.verticalLayout.addWidget(self.pushButton_camera, 0, Qt.AlignHCenter)

        self.pushButton_waring = QPushButton(self.centralwidget)
        self.pushButton_waring.setObjectName(u"pushButton_waring")

        self.verticalLayout.addWidget(self.pushButton_waring)

        self.pushButton_video = QPushButton(self.centralwidget)
        self.pushButton_video.setObjectName(u"pushButton_video")
        sizePolicy1.setHeightForWidth(self.pushButton_video.sizePolicy().hasHeightForWidth())
        self.pushButton_video.setSizePolicy(sizePolicy1)
        self.pushButton_video.setMinimumSize(QSize(150, 100))
        self.pushButton_video.setMaximumSize(QSize(150, 100))
        self.pushButton_video.setFont(font)

        self.verticalLayout.addWidget(self.pushButton_video, 0, Qt.AlignHCenter)

        self.verticalLayout.setStretch(3, 1)

        self.horizontalLayout.addLayout(self.verticalLayout)

        self.label = QLabel(self.centralwidget)
        self.label.setObjectName(u"label")

        self.horizontalLayout.addWidget(self.label)

        self.horizontalLayout.setStretch(1, 3)

        self.horizontalLayout_2.addLayout(self.horizontalLayout)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 800, 22))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"\u667a\u80fd\u53a8\u623f\u7cfb\u7edfdemo", None))
        self.pushButton_img.setText(QCoreApplication.translate("MainWindow", u"\u56fe\u7247\u68c0\u6d4b", None))
        self.pushButton_camera.setText(QCoreApplication.translate("MainWindow", u"\u6444\u50cf\u5934\u68c0\u6d4b", None))
        self.pushButton_waring.setText(QCoreApplication.translate("MainWindow", u"\u8b66\u544a\u91cd\u7f6e", None))
        self.pushButton_video.setText(QCoreApplication.translate("MainWindow", u"\u89c6\u9891\u68c0\u6d4b", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"TextLabel", None))
    # retranslateUi

