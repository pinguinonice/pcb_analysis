import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import tensorflow as tf
from yolov3.utils import detect_image, detect_realtime, detect_video, Load_Yolo_model, detect_video_realtime_mp
from yolov3.configs import *

image_folder = "/Users/philippschneider/Documents/Code/Repositories/pcb_analysis/dataset/pcb_dataset/image_test/"


yolo = Load_Yolo_model()



    
#detect_image(yolo, image_path, "pcb_prediction.jpg", input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))

# jpg files in image_folder, predict and save in output_folder

for filename in os.listdir(image_folder):
    if filename.endswith(".jpg"):
        image_path = os.path.join(image_folder, filename)
        print(image_path)
        detect_image(yolo, image_path, os.path.join('model_data/pcb/prediction', filename), input_size=YOLO_INPUT_SIZE, show=False, CLASSES=TRAIN_CLASSES, score_threshold=0.4, iou_threshold=0.45, rectangle_colors=(255,0,0))
        continue
    else:
        continue