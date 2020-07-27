import tensorflow as tf
from utils import load_class_names, output_boxes, draw_outputs, resize_image
import cv2
import numpy as np
from yolov3 import YOLOv3Net

model_size = (416, 416,3)
num_classes = 4
class_name = './data/coco.names'
max_output_size = 40
max_output_size_per_class= 20
iou_threshold = 0.5
confidence_threshold = 0.5
cfgfile = 'cfg/yolov3.cfg'
weightfile = 'weights/yolov3_weights.tf'
img_path = "data/images/test.jpg"

model = YOLOv3Net(cfgfile,model_size,num_classes)
model.load_weights(weightfile)
model.summary()