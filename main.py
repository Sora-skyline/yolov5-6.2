# UI

import sys

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtGui import QImage, QPixmap

from project import Ui_MainWindow
from PySide6.QtWidgets import QFileDialog, QApplication, QMainWindow, QWidget, QPushButton, QLabel, QLineEdit
from PySide6 import QtGui, QtWidgets

# yolov5

import argparse
import os
import platform
import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

from utils.augmentations import letterbox

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode, time_sync

class MainWindow(QMainWindow):
    def __init__(self):
        # 初始化
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        # 视频流
        self.cap = cv2.VideoCapture()
        self.timer_video = QTimer()
        # 槽函数绑定



        self.band()
        # 输入参数
        self.waring = False
        self.cnt = 0
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
        # parser.add_argument('--weights', nargs='+', type=str, default=ROOT/'runs/train/exp29/weights/best.pt', help='model path(s)')
        # parser.add_argument('--source', type=str, default=ROOT / 'data/test', help='file/dir/URL/glob, 0 for webcam')
        parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
        # parser.add_argument('--source', type=str, default=ROOT / '0', help='file/dir/URL/glob, 0 for webcam')
        parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
        # parser.add_argument('--data', type=str, default=ROOT / 'data/fixedbike.yaml', help='(optional) dataset.yaml path')
        parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640],
                            help='inference size h,w')
        parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
        parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='show results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
        parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--visualize', action='store_true', help='visualize features')
        parser.add_argument('--update', action='store_true', help='update all models')
        parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
        parser.add_argument('--name', default='exp', help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
        parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
        parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
        parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
        parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
        self.opt = parser.parse_args()
        # 如果len(imgsz) ==1 则 乘2
        self.opt.imgsz *= 2 if len(self.opt.imgsz) == 1 else 1  # expand
        print_args(vars(self.opt))  # 打印参数信息
        #
        # # 命令行参数转为变量
        # device, weights, dnn, data, half, imgsz = self.opt.device, self.opt.weights, self.opt.dnn, self.opt.data, self.opt.half, self.opt.imgsz
        #
        # # 加载模型权重
        # # Load model
        # device = select_device(device)
        # model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)  # 选择模型的后端框架
        # stride, names, pt = model.stride, model.names, model.pt
        # imgsz = check_img_size(imgsz, s=stride)  # check image size

    def __del__(self):
        self.cap.release()

    def band(self):
        self.ui.pushButton_img.clicked.connect(self.imgload)
        self.ui.pushButton_video.clicked.connect(self.videoload)
        self.ui.pushButton_camera.clicked.connect(self.cameraload)
        self.ui.pushButton_waring.clicked.connect(self.reset)
    def reset(self):
        self.waring = False
        self.cnt = 0
    def imgload(self):
        fname, _ = QFileDialog.getOpenFileNames(self, '选择你需要的图片', '.', '图片类型 (*.png *.jpg *.bmp)')
        # getOpenFileNames 返回一个list
        # getOpenFileName 返回一个str
        path = fname[0]
        if not path:
            return
        # 命令行参数转为变量
        source, weights, view_img, save_txt, imgsz, device = path, self.opt.weights, self.opt.view_img, self.opt.save_txt, self.opt.imgsz, self.opt.device
        project, name, exist_ok, dnn, data, half, nosave = self.opt.project, self.opt.name, self.opt.exist_ok, self.opt.dnn, self.opt.data, self.opt.half, self.opt.nosave
        augment, conf_thres, iou_thres, classes, agnostic_nms, max_det = self.opt.augment, self.opt.conf_thres, self.opt.iou_thres, self.opt.classes, self.opt.agnostic_nms, self.opt.max_det
        line_thickness, hide_labels, hide_conf, visualize = self.opt.line_thickness, self.opt.hide_labels, self.opt.hide_conf, self.opt.visualize
        # Load model
        device = select_device(device)
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)  # 选择模型的后端框架
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        # 加载待预测的图片
        # Dataloader
        save_dir = ROOT
        # 5 执行模型推理过程，并画检测框
        # Run inference
        im = cv2.imread(source)
        im0 = im

        with torch.no_grad():
            im = letterbox(im0, imgsz, stride)[0]  # 改变图片尺寸， 模型只能输入640
            # 640 *640            32                 true
            # Convert
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB , [::-1] 代表顺序相反操作
            im = np.ascontiguousarray(im)
            im = torch.from_numpy(im).to(device)  # 把img转成pytorch支持的格式
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32 ,有没有用到半精度
            im /= 255  # 0 - 255 to 0.0 - 1.0 归一化
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            # Inference
            visualize = increment_path(save_dir / Path(path).stem,
                                       mkdir=True) if visualize else False  # visualize 是否保存特征图
            pred = model(im, augment=augment, visualize=visualize)  # 模型所有检测框
            # [tensor([[3.98096e+02, 2.34295e+02, 4.80189e+02, 5.20510e+02, 8.96172e-01, 0.00000e+00],

            # NMS 非极大值过滤
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms,
                                       max_det=max_det)  # 根据置信度做非极大值过滤
            # Process predictions
            for i, det in enumerate(pred):  # per image, torch.Size([5, 6])
                save_path = str(save_dir / 'predict.jpg')  # im.jpg  图片保存路径
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4],
                                              im0.shape).round()  # 画的box是640*640的，坐标不能直接画在img0

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        # Add bbox to image 画到原图上
                        c = int(cls)  # integer class
                        if c == 0:
                            QtWidgets.QMessageBox.warning(
                                self, u"Warning", u"person警告", buttons=QtWidgets.QMessageBox.Ok,
                                defaultButton=QtWidgets.QMessageBox.Ok)
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))

                im0 = annotator.result()
                image = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)
                print(image.data)
                jpg = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
                print(jpg)
                self.ui.label.setPixmap(QPixmap(jpg))
                # Save results (image with detections)
                cv2.imwrite(save_path, image)

    def videoload(self):
        if not self.timer_video.isActive():
            file_name, _ = QFileDialog.getOpenFileName(self, "打开视频", "", "*.mp4;;*.avi;;All Files(*)")  # 打开文件选择框选择文件
            # 不加, _ =
            # ('G:/SteamLibrary/steamapps/workshop/content/431960/2726715713/Thumbs up.mp4', '*.mp4')
            # class 'tuple'>
            # print(file_name)
            # print(type(file_name))
            if not file_name:
                return
            flag = self.cap.open(file_name)
            if flag == False:
                QtWidgets.QMessageBox.warning(
                    self, u"Warning", u"打开视频失败", buttons=QtWidgets.QMessageBox.Ok,
                    defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                self.out = cv2.VideoWriter('predict.avi', cv2.VideoWriter_fourcc(
                    *'MJPG'), 20, (int(self.cap.get(3)), int(self.cap.get(4))))
                self.ui.pushButton_camera.setDisabled(True)
                self.ui.pushButton_img.setDisabled(True)
                self.ui.pushButton_video.setText(u"关闭视频")
                self.timer_video.start(30)  # 定时器读取视频流
                self.timer_video.timeout.connect(self.openframe)
        else:
            self.timer_video.stop()
            self.out.release()
            self.cap.release()
            self.ui.label.clear()
            self.ui.pushButton_camera.setDisabled(False)
            self.ui.pushButton_img.setDisabled(False)
            self.ui.pushButton_video.setText(u"视频检测")

    def cameraload(self):
        if not self.timer_video.isActive():
            flag = self.cap.open(0)
            if flag == False:
                QtWidgets.QMessageBox.warning(  # 摄像头打开失败弹窗
                    self, u"Warning", u"打开摄像头失败", buttons=QtWidgets.QMessageBox.ok,
                    defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                self.out = cv2.VideoWriter('prediction.avi', cv2.VideoWriter_fourcc(  # 录制摄像头视频
                    *'MJPG'), 20, (int(self.cap.get(3)), int(self.cap.get(4))))

                self.ui.pushButton_video.setDisabled(True)
                self.ui.pushButton_img.setDisabled(True)
                self.ui.pushButton_camera.setText(u"关闭摄像头")
                self.timer_video.start(30)
                self.timer_video.timeout.connect(self.openframe)
        else:
            self.timer_video.stop()
            self.cap.release()
            self.out.release()
            self.ui.label.clear()
            self.ui.pushButton_video.setDisabled(False)
            self.ui.pushButton_img.setDisabled(False)
            self.ui.pushButton_camera.setText(u"摄像头检测")

    def openframe(self):
        # 命令行参数转为变量
        weights, view_img, save_txt, imgsz, device = self.opt.weights, self.opt.view_img, self.opt.save_txt, self.opt.imgsz, self.opt.device
        project, name, exist_ok, dnn, data, half, nosave = self.opt.project, self.opt.name, self.opt.exist_ok, self.opt.dnn, self.opt.data, self.opt.half, self.opt.nosave
        augment, conf_thres, iou_thres, classes, agnostic_nms, max_det = self.opt.augment, self.opt.conf_thres, self.opt.iou_thres, self.opt.classes, self.opt.agnostic_nms, self.opt.max_det
        line_thickness, hide_labels, hide_conf, visualize = self.opt.line_thickness, self.opt.hide_labels, self.opt.hide_conf, self.opt.visualize
        # Load model
        device = select_device(device)
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)  # 选择模型的后端框架
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        save_path = str(ROOT / 'warn_img' / 'waring.jpg')
        ret, im = self.cap.read()
        im0 = im
        # Run inference
        # 执行模型推理过程，并画检测框
        if ret:
            im = letterbox(im0, new_shape=imgsz)[0]
            im = im[:, :, ::-1].transpose(2, 0, 1)
            im = np.ascontiguousarray(im)
            im = torch.from_numpy(im).to(device)  # 把img转成pytorch支持的格式
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32 ,有没有用到半精度
            im /= 255  # 0 - 255 to 0.0 - 1.0 归一化
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            # Inference
            pred = model(im, augment=augment, visualize=visualize)  # 模型所有检测框
            # [tensor([[3.98096e+02, 2.34295e+02, 4.80189e+02, 5.20510e+02, 8.96172e-01, 0.00000e+00],
            # NMS 非极大值过滤
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms,
                                       max_det=max_det)  # 根据置信度做非极大值过滤
            for i, det in enumerate(pred):  # per image, torch.Size([5, 6])
                # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh 获得原图宽高大小
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4],
                                              im0.shape).round()  # 画的box是640*640的，坐标不能直接画在img0

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        # Add bbox to image 画到原图上
                        c = int(cls)  # integer class

                        if c == 0:
                            self.waring = True
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))

                # Stream results
                im0 = annotator.result()
                image = im0


                self.out.write(im0)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                video_img = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
                self.ui.label.setPixmap(QPixmap(video_img))
                self.ui.label.setScaledContents(True)  # 自适应窗口
            if self.waring == True and  self.cnt == 0:
                QtWidgets.QMessageBox.warning(
                    self, u"Warning", u"person警告", buttons=QtWidgets.QMessageBox.Ok,
                    defaultButton=QtWidgets.QMessageBox.Ok)
                self.cnt += 1
                cv2.imwrite(save_path, im0)

        else:
            self.cap.release()
            self.timer_video.stop()


if __name__ == '__main__':
    app = QApplication([])  # 启动一个应用
    window = MainWindow()  # 实例化主窗口
    window.show()  # 展示主窗口
    app.exec()  # 避免程序执行到这一行后直接退出
