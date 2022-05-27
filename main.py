# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'untitled.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
import csv
import pandas
import matplotlib.pyplot as plt
import film_info_form
import predict_review_form
df = pandas.read_csv('metacritic.csv', sep='|')
review_df = pandas.read_csv('reviews.csv', sep='|')
years = list(df.year.unique())
years.remove('TBA')
years.sort()
means, counts, positive, mixed, negative = [], [], [], [], []
for year in years:
        means.append(df.loc[df['year'] == year]['metascore'].mean())
        counts.append(len(df.loc[df['year'] == year]))
        positive.append(len(df.loc[df['year'] == year].loc[df['metascore'] > 60]))
        negative.append(len(df.loc[df['year'] == year].loc[df['metascore'] <= 40]))
        mixed.append(len(df.loc[df['year'] == year].loc[(df['metascore'] > 40) & (df['metascore'] <= 60)]))

class Ui_MainWindow(object):
    def __init__(self):
        self.secondWin = None
        self.secondForm = None
        
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(720, 520)
        MainWindow.setStyleSheet("background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 rgba(20, 131, 193, 255), stop:1 rgba(75, 196, 113, 255));")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.Button_Number = QtWidgets.QPushButton(self.centralwidget)
        self.Button_Number.setGeometry(QtCore.QRect(540, 60, 171, 51))
        self.line = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Segoe UI Historic")
        font.setPointSize(12)
        self.line.setFont(font)
        self.line.resize(441, 25)
        self.line.move(10, 10)
        self.Button_search = QtWidgets.QPushButton(self.centralwidget)
        self.Button_search.setFont(font)
        self.Button_search.setCheckable(False)
        self.Button_search.setObjectName("Button_Search")
        self.Button_search.setGeometry(QtCore.QRect(540, 10, 171, 25))
        font = QtGui.QFont()
        font.setFamily("Segoe UI Historic")
        font.setPointSize(12)
        self.Button_Number.setFont(font)
        self.Button_Number.setCheckable(False)
        self.Button_Number.setObjectName("Button_Number")
        self.Button_Avg = QtWidgets.QPushButton(self.centralwidget)
        self.Button_Avg.setGeometry(QtCore.QRect(540, 130, 171, 51))
        font = QtGui.QFont()
        font.setFamily("Segoe UI Historic")
        font.setPointSize(12)
        self.Button_Avg.setFont(font)
        self.Button_Avg.setObjectName("Button_Avg")
        self.Button_Quality = QtWidgets.QPushButton(self.centralwidget)
        self.Button_Quality.setGeometry(QtCore.QRect(540, 200, 171, 51))
        self.Button_Predict = QtWidgets.QPushButton(self.centralwidget)
        self.Button_Predict.setGeometry(QtCore.QRect(540, 270, 171, 51))
        self.Button_Predict.setFont(font)
        self.Button_Predict.setObjectName("Button_Predict")
        font = QtGui.QFont()
        font.setFamily("Segoe UI Historic")
        font.setPointSize(12)
        self.Button_Quality.setFont(font)
        self.Button_Quality.setObjectName("Button_Quality")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 40, 81, 16))
        font = QtGui.QFont()
        font.setFamily("Segoe UI Historic")
        font.setPointSize(12)
        self.label.setFont(font)
        self.label.setStyleSheet("background-color: rgb(24, 135, 188);")
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(230, 38, 61, 20))

        font = QtGui.QFont()
        font.setFamily("Segoe UI Historic")
        font.setPointSize(12)
        self.label_2.setFont(font)
        self.label_2.setStyleSheet("background-color: rgb(38, 153, 166);")
        self.label_2.setObjectName("label_2")
        self.listWidget = QtWidgets.QListWidget(self.centralwidget)
        self.listWidget.setGeometry(QtCore.QRect(0, 60, 451, 461))
        self.listWidget.setObjectName("listWidget")
        self.listWidget.itemClicked.connect(self.selectionChanged)
        MainWindow.setCentralWidget(self.centralwidget)
        self.Button_Number.clicked.connect(self.number)
        self.Button_Avg.clicked.connect(self.avg)
        self.Button_Quality.clicked.connect(self.stats)
        self.Button_search.clicked.connect(self.search)
        self.Button_Predict.clicked.connect(self.predict_form)
        self.getFilmsList('')
        
        self.retranslateUi(MainWindow)
        
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def predict_form(self):
        self.secondForm = QtWidgets.QDialog()
        self.secondWin = predict_review_form.Ui_Form()
        self.secondWin.setupUi(self.secondForm)
        self.secondForm.show()
            
    def getFilmsList(self, search):
        items = []
        if search == "":
                with open('metacritic.csv', encoding='utf-8') as f:
                    reader = csv.DictReader(f, delimiter = '|')
                    for row in reader:
                        score = '{:^24}'.format(row['metascore'])
                        items.append( score + ' | ' + row['place'].replace(' ', '') + '. ' + row['title'])
        else:
                with open('metacritic.csv', encoding='utf-8') as f:
                    reader = csv.DictReader(f, delimiter = '|')
                    for row in reader:
                        score = '{:^24}'.format(row['metascore'])
                        text = score + ' | ' + row['place'].replace(' ', '') + '. ' + row['title']
                        if search.lower() in text.lower():
                                items.append(text)
        self.listWidget.clear()
        self.listWidget.addItems(items)

    def selectionChanged(self, item):
        title = item.text()
        number = title[title.find('|') + 2: title.find('.')]
        self.secondForm = QtWidgets.QDialog()
        self.secondWin = film_info_form.Ui_Form(df, number, review_df)
        self.secondWin.setupUi(self.secondForm)
        self.secondForm.show()
        
    def avg(self):
        plt.bar(years,means)
        plt.xticks(rotation=90)
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()

    def number(self):
        plt.bar(years, counts)
        plt.xticks(rotation=90)
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()

    def stats(self):
        plt_df = pandas.DataFrame({'years' : years, 'positive' : positive, 'mixed' : mixed, 'negative' : negative})
        plt_df = plt_df[['years','positive','mixed','negative']]
        plt_df.set_index(['years'],inplace=True)
        plt_df.plot(kind='bar', rot=90, color=['g','orange','r'], width=0.9)
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        
    def search(self, e):
        self.getFilmsList(self.line.text())
        
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Metacritic analysis"))
        self.Button_search.setText(_translate("MainWindow", "Search"))
        self.Button_Number.setText(_translate("MainWindow", "Number by years"))
        self.Button_Avg.setText(_translate("MainWindow", "Average Metascore"))
        self.Button_Quality.setText(_translate("MainWindow", "Metascore stats"))
        self.Button_Predict.setText(_translate("MainWindow", "Review predict"))
        self.label.setText(_translate("MainWindow", "Metascore"))
        self.label_2.setText(_translate("MainWindow", "Title"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())