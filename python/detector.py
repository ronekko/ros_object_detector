#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 21:40:56 2018

@author: sakurai
"""

import sys
sys.path.append('/usr/lib/python2.7/dist-packages')

import chainercv
import cv2
import matplotlib.pyplot as plt
import numpy as np

import cv_bridge
import message_filters
import rospy
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from std_msgs.msg import String
from object_detector.msg import BoundingBox, BoundingBoxes


def callback_raw(image, camera_info, net, publisher):
    hwc_bgr = cv_bridge.CvBridge().imgmsg_to_cv2(image)
    hwc_rgb = hwc_bgr[:, :, ::-1]
#    print(camera_info)
#    print(hwc_rgb.dtype, hwc_rgb.shape)

    chw_rgb = hwc_rgb.transpose(2, 0, 1)
    bboxes, labels, scores = net.predict([chw_rgb])
    bboxes, labels, scores = bboxes[0], labels[0], scores[0]

    classes = []
    for label in labels:
        classes.append(chainercv.datasets.voc_bbox_label_names[label])

    detection = BoundingBoxes()
    for name, bbox, score in zip(classes, bboxes, scores):
        y_min, x_min, y_max, x_max = bbox
        bounding_box = BoundingBox(
            class_name=name, score=score,
            y_min=y_min, x_min=x_min, y_max=y_max, x_max=x_max)
        detection.bounding_boxes.append(bounding_box)
    detection.num_detection = len(detection.bounding_boxes)

    publisher.publish(detection)


if __name__ == '__main__':
    net = chainercv.links.model.ssd.SSD300(pretrained_model='voc0712')

    rospy.init_node('image_receiver', anonymous=True)

    sub_image = message_filters.Subscriber('/camera/image_raw', Image)
    sub_info = message_filters.Subscriber('/camera/camera_info', CameraInfo)

    pub_objects = rospy.Publisher(
        'object_classes', BoundingBoxes, queue_size=10)

    ts = message_filters.TimeSynchronizer([sub_image, sub_info], 2)
    ts.registerCallback(callback_raw, net, pub_objects)

    rospy.spin()
