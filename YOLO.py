  
import cv2
import argparse
import os
import matplotlib.pyplot as plt
from os.path import isfile, join
from os import listdir
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes
from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body

#%matplotlib inline

def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=.6):
    box_scores = np.multiply(box_confidence, box_class_probs)
    box_classes = K.argmax(box_scores, axis=-1)
    box_class_scores = K.max(box_scores, axis=-1)

    filtering_mask = box_class_scores >= threshold

   
    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    boxes = tf.boolean_mask(boxes, filtering_mask)
    classes = tf.boolean_mask(box_classes, filtering_mask)

    return scores, boxes, classes


def iou(box1, box2):
    
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = (yi2 - yi1) * (xi2 - xi1)

    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area

    return iou

def yolo_non_max_suppression(scores, boxes, classes, max_boxes=10, iou_threshold=0.5):
    

    max_boxes_tensor = K.variable(max_boxes, dtype='int32')  
    K.get_session().run(tf.variables_initializer([max_boxes_tensor])) 

    
    nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes_tensor, iou_threshold)

  
    scores = K.gather(scores, nms_indices)
    boxes = K.gather(boxes, nms_indices)
    classes = K.gather(classes, nms_indices)

    return scores, boxes, classes


def yolo_eval(yolo_outputs, image_shape=(720., 1280.), max_boxes=10, score_threshold=.6, iou_threshold=.5):
    
    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs
    boxes = yolo_boxes_to_corners(box_xy, box_wh)

    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=score_threshold)
    
    boxes = scale_boxes(boxes, image_shape)

   
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes=max_boxes, iou_threshold=iou_threshold)

    return scores, boxes, classes

sess = K.get_session()
class_names = read_classes("model_data/coco_classes.txt")
anchors = read_anchors("model_data/yolo_anchors.txt")
image_shape = (540., 960.)     
yolo_model = load_model("model_data/yolo.h5")

yolo_model.summary()

yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))

scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)

def predict(sess, image_file):
    
    image, image_data = preprocess_image("in2/" + image_file, model_image_size=(416, 416))

    # Run the yolo model
    
    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes], feed_dict={yolo_model.input: image_data,
                                                                                       K.learning_phase(): 0})

    colors = generate_colors(class_names)
    
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    
    image.save(os.path.join("out8", image_file), quality=90)
    
    output_image = scipy.misc.imread(os.path.join("out8", image_file))
    return out_scores, out_boxes, out_classes
all_outputs = [predict(sess,f) for f in listdir('in2') if isfile(join('in2/', f))]  






# Construct the argument parser and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-ext", "--extension", required=False, default='png', help="extension name. default is 'png'.")
#ap.add_argument("-o", "--output", required=False, default='output.mp4', help="output video file")
#args = vars(ap.parse_args())

## Arguments
#dir_path = 'out2/'
#ext = args['extension']
#output = args['output']
#
#images = []
#for f in os.listdir(dir_path):
#    if f.endswith(ext):
#        images.append(f)
#
## Determine the width and height from the first image
#image_path = os.path.join(dir_path, images[0])
#frame = cv2.imread(image_path)
#cv2.imshow('video',frame)
#height, width, channels = frame.shape
#
## Define the codec and create VideoWriter object
#fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
#out = cv2.VideoWriter(output, fourcc, 20.0, (width, height))
#
#for image in images:
#
#    image_path = os.path.join(dir_path, image)
#    frame = cv2.imread(image_path)
#
#    out.write(frame) # Write out frame to video
#
#    cv2.imshow('video',frame)
#    if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
#        break
#
## Release everything if job is finished
#out.release()
#cv2.destroyAllWindows()
#
#print("The output video is {}".format(output))

#Alternative
#image_folder = 'out2'
#video_name = 'video.avi'
#
#images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
#frame = cv2.imread(os.path.join(image_folder, images[0]))
#height, width, layers = frame.shape
#
#video = cv2.VideoWriter(video_name, -1, 1, (width,height))
#
#for image in images:
#    video.write(cv2.imread(os.path.join(image_folder, image)))
#
#cv2.destroyAllWindows()
#video.release()