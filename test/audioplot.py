import numpy as np
import sys

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

import pyaudio

sample_rate = 16000
frame_length = 1024
frame_shift = 80


class PlotWindow:
    def __init__(self):

        self.win = pg.GraphicsWindow()
        self.win.setWindowTitle(u"波形のリアルタイムプロット")
        self.win.resize(1100, 800)
        self.plt = self.win.addPlot()  # プロットのビジュアル関係
        self.ymin = -100
        self.ymax = 80
        self.plt.setYRange(-1.0, 1.0)  # y軸の上限、下限の設定
        self.curve = self.plt.plot()  # プロットデータを入れる場所

        # マイク設定
        self.CHUNK = frame_length  # 1度に読み取る音声のデータ幅
        self.RATE = sample_rate  # サンプリング周波数
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=pyaudio.paInt16,
                                      channels=1,
                                      rate=self.RATE,
                                      input=True,
                                      frames_per_buffer=self.CHUNK)
        self.canvas = np.zeros((self.CHUNK*10,))

        # アップデート時間設定
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(5) 

        self.data = np.zeros(self.CHUNK)

    def update(self):
        self.data = self.AudioInput()
        self.canvas[:-self.CHUNK] = self.canvas[self.CHUNK:].copy()
        self.canvas[-self.CHUNK:] = self.data
        self.curve.setData(self.canvas)

    def AudioInput(self):
        ret = self.stream.read(self.CHUNK)
        ret = np.frombuffer(ret, dtype="int16") / 32768
        return ret


if __name__ == "__main__":
    plotwin = PlotWindow()

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()