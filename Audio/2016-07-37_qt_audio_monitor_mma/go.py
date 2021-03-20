from PyQt5 import QtGui,QtCore

import sys
import ui_main
import numpy as np
import pyqtgraph
import SWHear

class ExampleApp(QtGui.QMainWindow, ui_main.Ui_MainWindow):
    def __init__(self, parent=None):
        pyqtgraph.setConfigOption('background', 'w') #before loading widget
        super(ExampleApp, self).__init__(parent)
        self.setupUi(self)
        self.grFFT.plotItem.showGrid(True, True, 0.7)
        self.grPCM.plotItem.showGrid(True, True, 0.7)
        self.maxFFT=0
        self.maxPCM=0
        self.ear = SWHear.SWHear(rate=50,updatesPerSecond=10)#rate=770   357
        self.ear.stream_start()

    def update(self):
        if not self.ear.data is None and not self.ear.fft is None:
            pcmMax=np.max(np.abs(self.ear.data))
            
            if pcmMax>self.maxPCM:
                pass
                self.maxPCM=pcmMax
                self.grPCM.plotItem.setRange(yRange=[-pcmMax,pcmMax])
                
            if np.max(self.ear.fft)>self.maxFFT:
                self.maxFFT=np.max(np.abs(self.ear.fft))
                self.grFFT.plotItem.setRange(yRange=[0,self.maxFFT])
                #self.grFFT.plotItem.setRange(yRange=[0,1])
                
           # self.pbLevel.setValue(pcmMax/self.maxPCM)
            pen=pyqtgraph.mkPen(color='b')
            self.grPCM.plot(self.ear.data,pen=pen,clear=True)
            pen=pyqtgraph.mkPen(color='r')
            #self.grFFT.plot(self.ear.fftx,self.ear.fft/self.maxFFT,pen=pen,clear=True)
            self.grFFT.plot(self.ear.fftx,self.ear.fft,pen=pen,clear=True)
            self.ear.data = None
            self.ear.fft = None
        QtCore.QTimer.singleShot(1, self.update) # QUICKLY repeat

if __name__=="__main__":
    app = QtGui.QApplication(sys.argv)
    form = ExampleApp()
    form.show()
    form.update() #start with something
    app.exec_()
    print("DONE")
