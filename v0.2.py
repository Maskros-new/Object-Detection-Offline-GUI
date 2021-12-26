from torchvision import transforms
from utils import *
from model import SSD300
from PIL import Image, ImageDraw, ImageFont, ImageQt
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from PyQt5.QtWidgets import QFileDialog, QApplication, QWidget
import os

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
n_classes = len(label_map)
model = SSD300(n_classes=n_classes)

# pyinstaller need
if getattr(sys, 'frozen', None):
    basedir = sys._MEIPASS
else:
    basedir = os.path.dirname(__file__)
pthpath = os.path.join(basedir, 'model_state_dict.pth')
# pthpath = 'model_state_dict.pth'
model.load_state_dict(torch.load(pthpath, map_location="cpu"))

model = model.to(device)
model.eval()

# Transforms
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def detect(original_image, min_score, max_overlap, top_k, suppress=None):
    # Transform
    image = normalize(to_tensor(resize(original_image)))

    # Move to default device
    image = image.to(device)

    # Forward prop.
    predicted_locs, predicted_scores = model(image.unsqueeze(0))

    # Detect objects in SSD output
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)

    # Move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    # Decode class integer labels
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

    # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
    if det_labels == ['background']:
        # Just return original image
        return original_image

    # Annotate
    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.truetype("./calibril.ttf", 15)

    # Suppress specific classes, if needed
    for i in range(det_boxes.size(0)):
        if suppress is not None:
            if det_labels[i] in suppress:
                continue

        # Boxes
        box_location = det_boxes[i].tolist()
        draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
        draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[
            det_labels[i]])

        # Text
        text_size = font.getsize(det_labels[i].upper())
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                            box_location[1]]
        draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
        draw.text(xy=text_location, text=det_labels[i].upper(), fill='white',
                  font=font)
    del draw

    return annotated_image


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1038, 657)
        Form.setWindowOpacity(0.9)
        self.FuckWidget = QtWidgets.QWidget(Form)
        self.gridLayout = QtWidgets.QGridLayout(Form)
        self.gridLayout.setObjectName("gridLayout")
        self.picshowLabel = QtWidgets.QLabel(Form)
        self.picshowLabel.setText("")
        self.picshowLabel.setObjectName("picshowLabel")
        self.gridLayout.addWidget(self.picshowLabel, 1, 0, 1, 1)
        self.pushButton = QtWidgets.QPushButton(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton.sizePolicy().hasHeightForWidth())
        self.pushButton.setSizePolicy(sizePolicy)
        self.pushButton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton.setMouseTracking(False)
        self.pushButton.setObjectName("pushButton")
        self.gridLayout.addWidget(self.pushButton, 0, 0, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Object_Detection v0.2.0 ---presented by edicius & maskros"))
        self.pushButton.setText(_translate("Form", "Click Here to Load a Picture"))
        self.pushButton.clicked.connect(self.openImage)

    def openImage(self):
        self.imageName, imgType = QFileDialog.getOpenFileName(self.FuckWidget, "openImage", "",
                                                              "*.jpg;*.png;*.jpeg")
        if self.imageName != "":
            img_path = self.imageName
            self.loadImage(img_path)

    def loadImage(self, path):
        original_image = Image.open(path, mode='r')
        original_image = original_image.convert('RGB')
        jpg = ImageQt.toqpixmap(original_image)
        self.picshowLabel.setPixmap(jpg)  # 在label控件上显示选择的图片
        # resize
        height = original_image.height
        width = original_image.width
        print("before:" + str(height), str(width))
        maxwidth = 1440
        maxheight = 900
        if width > maxwidth or height > maxheight:
            modw = maxwidth/width
            modh = maxheight/height
            mod = min(modw, modh)
            height = int(height * mod)
            width = int(width * mod)
            original_image = original_image.resize((width, height), Image.ANTIALIAS)
        height = original_image.height
        width = original_image.width
        print("after:" + str(height), str(width))

        finimg = detect(original_image, min_score=0.2, max_overlap=0.5, top_k=200)
        jpg = ImageQt.toqpixmap(finimg)
        self.picshowLabel.setPixmap(jpg)  # 在label控件上显示选择的图片
        self.picshowLabel.setScaledContents(True)  # 让图片自适应label大小

if __name__ == '__main__':
    app = QApplication(sys.argv)
    Qwidget = QWidget()
    ui = Ui_Form()
    ui.setupUi(Qwidget)
    Qwidget.show()
    sys.exit(app.exec_())