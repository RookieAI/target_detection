# !/usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : yolo_detector.py
#   Author      : JiangPan
#   Created date: 2019-09-20 22:37
#   Description :
#   Contact     : jsyxjp@163.com
#
# ================================================================
import time

import cv2 as cv
import numpy as np

import config as cfg


class YOLODetector:
    def __init__(self):
        self.config_path = cfg.MODEL_CONFIG
        self.weights_path = cfg.MODEL_WEIGHTS
        self.model = self.load_model()

    def load_model(self):
        print("[INFO] loading YOLO from disk ...")
        model = cv.dnn.readNetFromDarknet(self.config_path, self.weights_path)
        return model

    def get_layers(self):
        ln = self.model.getLayerNames()
        ln = [ln[i[0] - 1] for i in self.model.getUnconnectedOutLayers()]
        return ln

    def inference(self, image):
        layer_names = self.get_layers()
        blob = cv.dnn.blobFromImage(image, 1/255.0, (416, 416),
                                    swapRB=True, crop=False)
        self.model.setInput(blob)
        start = time.time()
        output_layers = self.model.forward(layer_names)
        end = time.time()
        print("[INFO] cost {:.3f}s to inference".format(end-start))
        return output_layers







