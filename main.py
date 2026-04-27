import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QFileDialog, QTableWidgetItem, QProgressBar
from PyQt6.QtCore import pyqtSignal, QObject, QAbstractTableModel, Qt,QThread
from PyQt6 import QtGui as qg
from gui.form_ui import Ui_MainWindow
from model.train import train_model
from model.predict import predict_model
from PyQt6.QtCore import QSettings
import pandas as pd
from PyQt6.QtGui import QTextCursor

import ctypes
ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 0)

from warnings import simplefilter
simplefilter(action="ignore",category=FutureWarning)

class Stream(QObject):
    """【输出重定向】重定向控制台输出到文本框控件"""
    newText = pyqtSignal(str)
    
    def write(self, text):
        self.newText.emit(str(text))
        QApplication.processEvents()


class TrainThread(QThread):
    def __init__(self,uiObj):
        super().__init__()
        self.uiObj = uiObj

    # 自定义信号
    finished = pyqtSignal()
    
    progress = pyqtSignal(int)  # 进度信号
 
    # 线程任务
    def run(self):
        # 读取输入文件
        data_geo = pd.read_csv(self.uiObj.lineEdit_7.text(), header= 0).iloc[:,1:]
        label_geo = pd.read_csv(self.uiObj.lineEdit_8.text(),header=0).iloc[:,1]
        anchor_list = pd.read_csv(self.uiObj.lineEdit_9.text(), header= 0)
        data_x = pd.read_csv(self.uiObj.lineEdit_10.text(),header=0).iloc[:,1:]
        data_ppi_link_index = pd.read_csv(self.uiObj.lineEdit_11.text(),header=0)
        data_homolog_index = pd.read_csv(self.uiObj.lineEdit_12.text(),header=0)
        # 调用模型训练方法并传递进度信号
        train_model(data_geo,label_geo,anchor_list,data_x,data_ppi_link_index,data_homolog_index,self.progress)

class predictThread(QThread):
    def __init__(self,uiObj):
        super().__init__()
        self.uiObj = uiObj

    # 自定义信号
    finished = pyqtSignal()
    
    progress = pyqtSignal(int)  # 进度信号
    tableWidget = pyqtSignal(dict)
    item_v = pyqtSignal(dict)
 
    # 线程run方法
    def run(self):
        model_path = self.uiObj.lineEdit.text()
        data_sample = pd.read_csv(self.uiObj.lineEdit_2.text(), header= 0)
        sample_name = data_sample.iloc[:,0]
        
        data_geo = data_sample.iloc[:,1:]

        pw_id = pd.read_csv(r"gui/pw_id.csv", header= 0).iloc[:,1]
        anchor_list = pd.read_csv(self.uiObj.lineEdit_3.text(), header= 0)
        data_x = pd.read_csv(self.uiObj.lineEdit_4.text(),header=0).iloc[:,1:]
        data_ppi_link_index = pd.read_csv(self.uiObj.lineEdit_5.text(),header=0)
        data_homolog_index = pd.read_csv(self.uiObj.lineEdit_6.text(),header=0)
        result = predict_model(model_path,data_geo,anchor_list,data_x,data_ppi_link_index,data_homolog_index,self.progress)
        
        df = pd.concat([pd.DataFrame({"sample_name":sample_name.values}),pd.DataFrame({"predict":result["out"]})],axis=1)
        data = df.values.tolist()
        headers = list(df.columns)
        
        
        self.tableWidget.emit({"id":0,"ColumnCount":len(headers),"HorizontalHeaderLabels":headers,"RowCount":len(data)})
        for i,row in enumerate(data):
            for j,val in enumerate(row):
                self.item_v.emit({"id":0,"i":i,"j":j,"val":val})

        df_pw = pd.concat([pw_id,pd.DataFrame({"pw_w":result["pw_w"]})],axis=1)
        data_pw = df_pw.values.tolist()
        headers_pw = list(df_pw.columns)

        self.tableWidget.emit({"id":1,"ColumnCount":len(headers_pw),"HorizontalHeaderLabels":headers_pw,"RowCount":len(data_pw)})
        for i,row in enumerate(data_pw):
            for j,val in enumerate(row):
                self.item_v.emit({"id":1,"i":i,"j":j,"val":val})


class MyApp(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        
        
        
        self.setupUi(self)
        self.retranslateUi(self)

        self.setWindowTitle("GC-PGE")
        self.setWindowIcon(qg.QIcon("./gui/icon.ico"))
        
        ##################

        self.console_obj = self.textEdit
       
        ##################
        #这里定义按钮消息
        """ 1.定义选择文件消息 """
        self.toolButton.clicked.connect(lambda: self.select_file(self.lineEdit))
        self.toolButton_2.clicked.connect(lambda: self.select_file(self.lineEdit_2))
        self.toolButton_3.clicked.connect(lambda: self.select_file(self.lineEdit_3))
        self.toolButton_4.clicked.connect(lambda: self.select_file(self.lineEdit_4))
        self.toolButton_5.clicked.connect(lambda: self.select_file(self.lineEdit_5))
        self.toolButton_6.clicked.connect(lambda: self.select_file(self.lineEdit_6))
        self.toolButton_7.clicked.connect(lambda: self.select_file(self.lineEdit_7))
        self.toolButton_8.clicked.connect(lambda: self.select_file(self.lineEdit_8))
        self.toolButton_9.clicked.connect(lambda: self.select_file(self.lineEdit_9))
        self.toolButton_10.clicked.connect(lambda: self.select_file(self.lineEdit_10))
        self.toolButton_11.clicked.connect(lambda: self.select_file(self.lineEdit_11))
        self.toolButton_12.clicked.connect(lambda: self.select_file(self.lineEdit_12))

        ##################

        #定义 模型训练-运算（按钮）
        self.pushButton_6.clicked.connect(self.train)
        self.pushButton.clicked.connect(self.predict)

        sys.stdout = Stream()
        sys.stdout.newText.connect(self.onUpdateText)


    ####################
    #下面定义相关方法，对应于上面的按钮消息
    
    """ 1.定义选择文件方法 """
    def select_file(self,obj):
        filename, _ = QFileDialog.getOpenFileName(self, 'Open file', '/path/to/initial/directory')
        if filename:
            obj.setText(filename)

    """ 2.定义日志输出 """
    def closeEvent(self, event):
        """【输出重定向】重写closeEvent,程序结束时将stdout恢复默认"""
        sys.stdout = sys.__stdout__
        super().closeEvent(event)
 
    def onUpdateText(self, text):
        """【输出重定向】重定向控制台输出到文本框控件"""
        cursor = self.console_obj.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        self.console_obj.setTextCursor(cursor)
        self.console_obj.ensureCursorVisible()


    """ 3.定义模型训练 """
    def train(self):
        # 禁用按钮，防止多次启动线程
        self.pushButton_6.setEnabled(False)

        self.console_obj = self.textEdit
 
        # 创建和启动工作线程
        self.trainThread = TrainThread(self)
        self.trainThread.finished.connect(self.trainFinished)  # 连接任务完成信号
        self.trainThread.progress.connect(self.updateProgress)    # 连接进度信号
 
        self.trainThread.start()  # 启动线程
    def trainFinished(self):
        self.pushButton_6.setEnabled(True)
    def updateProgress(self, value):
        self.progressBar_3.setValue(value)

    """ 4.定义模型预测 """
    def predict(self):
        self.pushButton.setEnabled(False)
        self.console_obj = self.textEdit_2

        # 创建和启动工作线程
        self.predictThread = predictThread(self)
        self.predictThread.finished.connect(self.predictFinished)  # 连接任务完成信号
        self.predictThread.progress.connect(self.updateProgress2)    # 连接进度信号
        self.predictThread.tableWidget.connect(self.setTable)    # 连接表格信号
        self.predictThread.item_v.connect(self.updateItem)    # 连接表格信号
        self.predictThread.start()  # 启动线程
        
    def predictFinished(self):
        self.pushButton.setEnabled(True)
    def updateProgress2(self, value):
        self.progressBar.setValue(value)
        
    def setTable(self, tableData):
        
        if tableData["id"] == 0:
            tableObj = self.tableWidget
        elif tableData["id"] == 1:
            tableObj = self.tableWidget_2
        tableObj.setColumnCount(tableData["ColumnCount"])
        tableObj.setHorizontalHeaderLabels(tableData["HorizontalHeaderLabels"])

        tableObj.setRowCount(tableData["RowCount"])
        


    def updateItem(self, itemData):
        if itemData["id"] == 0:
            tableObj = self.tableWidget
        elif itemData["id"] == 1:
            tableObj = self.tableWidget_2
        tableObj.setItem(itemData["i"],itemData["j"],QTableWidgetItem(str(itemData["val"])))


        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    mainWin = MyApp()
    mainWin.show()
    sys.exit(app.exec())