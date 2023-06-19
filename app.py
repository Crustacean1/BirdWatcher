from PyQt5.QtWidgets import QApplication, QPushButton, QLabel, QFileDialog, QVBoxLayout, QWidget, QMessageBox, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage
import torch
from bird_watcher import BirdWatcher
import cv2
import os
import matplotlib.pyplot as plt

app = QApplication([])
window = QWidget()
layout = QVBoxLayout()
image_layout = QHBoxLayout()
base_image = QLabel()
mask_image = QLabel()

model = BirdWatcher()
model.load_state_dict(torch.load("./model_weights,pth"))


def predict(image):
    with torch.no_grad():
        model.eval()
        print(f"Tensor shape: {image.shape}")
        image = torch.from_numpy(image)
        logits = model(image)
        prediction = logits.sigmoid()
        print(f"Prediction shape: {prediction.shape}")
        return prediction.numpy().squeeze()


def load_image(filename):
    try:
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, dsize=(256, 256))
        image = image.transpose(2, 0, 1)
        return image
    except:
        print(f"Failed to read {filename}")


def to_qimage(image):
    image = (image * 255).astype(int)
    # print(f"Converting image: {image.shape} {image.min()} {image.max()}")
    return QImage(image, 256, 256, 256, QImage.Format_Grayscale8)


def on_select_image():
    dlg = QFileDialog()
    if dlg.exec_():
        filename = dlg.selectedFiles()[0]
        if filename != None:
            image = load_image(filename)
            mask = to_qimage(predict(image))

            base_image.setPixmap(QPixmap(filename))
            mask_image.setPixmap(QPixmap(mask))
            base_image.resize(256, 256)
            mask_image.resize(256, 256)


image_selection_button = QPushButton('Select Image')
image_selection_button.clicked.connect(on_select_image)

image_layout.addWidget(base_image)
image_layout.addWidget(mask_image)

layout.addLayout(image_layout)
layout.addWidget(image_selection_button)

window.setLayout(layout)
window.resize(800, 600)
window.setWindowTitle("BirdWatcher")
window.show()

app.exec_()
