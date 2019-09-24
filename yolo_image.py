# !/usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : yolo_image.py
#   Author      : JiangPan
#   Created date: 2019-09-21 07:08
#   Description :
#   Contact     : jsyxjp@163.com
#
# ================================================================
import os

import numpy as np
import cv2 as cv

import core.config as cfg
from core.yolo_detector import YOLODetector


class ImageDetection:
    def __init__(self, detector):
        self.detector = detector

        self.labels = open(cfg.COCO_NAMES).read().strip().split('\n')
        self.colors = self.set_colors()

        self.confidence_thre = cfg.CONFIDENCE_THRE
        self.nms_thre = cfg.NMS_THRE

    def set_colors(self):
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(len(self.labels), 3), dtype="uint8")
        return colors

    def detect_single_image(self, image):
        boxes = []
        confidences = []
        class_ids = []

        image_height = image.shape[0]
        image_width = image.shape[1]
        outputs_layers = self.detector.inference(image)
        for outputs in outputs_layers:
            for detection in outputs:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > self.confidence_thre:
                    box = detection[0:4] * np.array([image_width, image_height,
                                                     image_width, image_height])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        idxes = cv.dnn.NMSBoxes(boxes, confidences, self.confidence_thre, self.nms_thre)

        if len(idxes) > 0:
            for i in idxes.flatten():
                # 提取边界框的坐标
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # 绘制边界框以及在左上角添加类别标签和置信度
                color = [int(c) for c in self.colors[class_ids[i]]]
                cv.rectangle(image, (x, y), (x + w, y + h), color, 2)
                text = '{}: {:.3f}'.format(self.labels[class_ids[i]], confidences[i])
                (text_w, text_h), baseline = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv.rectangle(image, (x, y - text_h - baseline), (x + text_w, y), color, -1)
                cv.putText(image, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        return image


if __name__ == '__main__':
    yolo_detector = YOLODetector()
    for filename in os.listdir(cfg.IMAGE_DIR):
        print("[INFO] detecting {}".format(filename))
        filepath = os.path.join(cfg.IMAGE_DIR, filename)
        try:
            img = cv.imread(filepath)
            detection_result = ImageDetection(yolo_detector).detect_single_image(img)

            cv.imwrite(os.path.join(cfg.SAVE_DIR, filename), img)

        except:
            print('load image failed')
