######## This script is intended to be run within the YOLO directory ########
######## It will take a directory of images and run them through the YOLO ########
######## model to detect objects in the images. ########
######## For further information visit the github in https://github.com/ultralytics/yolov5 ########


import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                            increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

from utils.augmentations import letterbox
import numpy as np

class detector:
    def __init__(self,
        weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):

        self.weights=weights
        self.source=source
        self.data=data
        self.imgsz=imgsz
        self.conf_thres=conf_thres
        self.iou_thres=iou_thres
        self.max_det=max_det
        self.device=device
        self.view_img=view_img
        self.save_txt=save_txt
        self.save_conf=save_conf
        self.save_crop=save_crop
        self.nosave=nosave
        self.classes=classes
        self.agnostic_nms=agnostic_nms
        self.augment=augment
        self.visualize=visualize
        self.update=update
        self.project=project
        self.name=name
        self.exist_ok=exist_ok
        self.line_thickness=line_thickness
        self.hide_labels=hide_labels
        self.hide_conf=hide_conf
        self.half=half
        self.dnn=dnn

        """source = str(source)
        self.save_img = not nosave and not source.endswith('.txt')  # save inference images
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
        if is_url and is_file:
            source = check_file(source)  # download"""

        # Load model
        self.device = select_device(device)
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=self.dnn, data=self.data)
        self.stride, self.names, self.pt, jit, onnx, engine = self.model.stride, self.model.names, self.model.pt, self.model.jit, self.model.onnx, self.model.engine
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check image size

        # Half
        half &= (self.pt or jit or onnx or engine) and self.device.type != 'cpu'  # FP16 supported on limited backends with CUDA
        if self.pt or jit:
            self.model.model.half() if half else self.model.model.float()

        

        

        
    @torch.no_grad()
    def detect(self,data):

        """# Dataloader
        if self.webcam:
            self.view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(self.source, img_size=self.imgsz, stride=self.stride, auto=self.pt)
            bs = len(dataset)  # batch_size
        else:
            dataset = LoadImages(self.source, img_size=self.imgsz, stride=self.stride, auto=self.pt)
            bs = 1  # batch_size"""
        
        #####
        im = data


        img = letterbox(im, self.imgsz, stride=self.stride, auto=self.pt)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(img)

        bs = 1  # batch_size


        self.vid_path, self.vid_writer = [None] * bs, [None] * bs

        # Run inference
        self.model.warmup(imgsz=(1 if self.pt else self.bs, 3, *self.imgsz), half=self.half)  # warmup
        dt, seen = [0.0, 0.0, 0.0], 0
        #for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = False #increment_path(self.save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = self.model(im, augment=self.augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            #im0, frame = im0s.copy()

            im0 = data

            #p = Path(p)  # to Path
            #save_path = str(self.save_dir / p.name)  # im.jpg
            #txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            #s += '%gx%g ' % im.shape[2:]  # print string
            #gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            #imc = im0.copy() if self.save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                #for c in det[:, -1].unique():
                #    n = (det[:, -1] == c).sum()  # detections per class
                #    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    """if self.save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if self.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')"""

                    # Add bbox to image
                    c = int(cls)  # integer class
                    label = None if self.hide_labels else (self.names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))
                    #if self.save_crop:
                    #    save_one_box(xyxy, imc, file=self.save_dir / 'crops' / self.names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
        return im0

            






