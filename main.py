import sys
from PyQt6 import QtGui, QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QVBoxLayout, QWidget, QHBoxLayout, QPushButton
from torchvision.transforms import v2
import torch
from MNISTmodel import MNIST_model
from PIL import ImageQt, Image
import torch.nn as nn
import numpy as np


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.label = QtWidgets.QLabel()
        canvas = QtGui.QPixmap(300, 300)
        canvas.fill(Qt.GlobalColor.black)
        self.label.setPixmap(canvas)

        button = QPushButton("Обновить")
        button.clicked.connect(self.reset)

        self.layout = QHBoxLayout()
        paint_layout = QVBoxLayout()
        paint_layout.addWidget(self.label)
        paint_layout.addWidget(button)
        layout1 = QVBoxLayout()
        self.layout.addLayout(paint_layout)

        for i in range(0, 10):
            layout2 = QHBoxLayout()
            lab = QtWidgets.QLabel(str(i))
            layout2.addWidget(lab)

            canv = QtGui.QPixmap(30, 15)
            canv.fill(Qt.GlobalColor.red)

            st = QtWidgets.QLabel()
            st.setPixmap(canv)
            layout2.addWidget(st)
            layout1.addLayout(layout2)


        self.layout.addLayout(layout1)

        widget = QWidget()
        widget.setLayout(self.layout)
        self.setCentralWidget(widget)

        self.last_x, self.last_y = None, None

    def mouseMoveEvent(self, e):
        if self.last_x is None:
            self.last_x = e.position().x()
            self.last_y = e.position().y()
            return

        canvas = self.label.pixmap()
        painter = QtGui.QPainter(canvas)
        pen = QtGui.QPen()
        pen.setWidth(20)
        pen.setColor(QtGui.QColor('white'))
        painter.setPen(pen)
        painter.drawLine(int(self.last_x), int(self.last_y), int(e.position().x()), int(e.position().y()))
        painter.end()
        self.label.setPixmap(canvas)
        self.makeguess()

        # Update the origin for next time.
        self.last_x = e.position().x()
        self.last_y = e.position().y()

    def mouseReleaseEvent(self, e):
        self.last_x = None
        self.last_y = None

    def reset(self):
        canvas = QtGui.QPixmap(300, 300)
        canvas.fill(Qt.GlobalColor.black)
        self.label.setPixmap(canvas)

    def makeguess(self):
        pixmap = self.label.pixmap()
        img = ImageQt.fromqpixmap(pixmap)
        img = img.resize((28, 28))
        img = img.convert('L')
        # img.show()

        transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.5,), std=(0.5,))
        ])
        picture = transform(img)
        # plt.imshow(picture.numpy, cmap='gray')
        # plt.show()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        params = torch.load('MNIST-model-params.pt')
        MNIST_model.load_state_dict(params)
        MNIST_model.eval()
        picture = picture.reshape(-1, 28 * 28).to(device)
        prediction = MNIST_model(picture)
        SMpred = nn.Softmax(dim=1)(prediction)
        stData = SMpred[0].cpu().detach().numpy()
        layout1 = self.layout.findChildren(QVBoxLayout)[1]
        layout = layout1.findChildren(QHBoxLayout)

        for i, box in enumerate(layout):
            px = QtGui.QPixmap(int(30 * stData[i]), 15)
            px.fill(Qt.GlobalColor.red)
            box.itemAt(1).widget().setPixmap(px)


app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()