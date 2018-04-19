#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 21:40:56 2018

@author: sakurai
"""

import chainercv
import cv2
import matplotlib.pyplot as plt
import numpy as np

import cv_bridge
import message_filters
import rospy
from sensor_msgs.msg import Image, CameraInfo, CompressedImage


def callback_raw(image, camera_info, net):
    hwc_bgr = cv_bridge.CvBridge().imgmsg_to_cv2(image)
#    hwc_rgb = cv2.cvtColor(hwc_bgr, cv2.COLOR_BGR2RGB)
    hwc_rgb = hwc_bgr[:, :, ::-1]
#    print(camera_info)
#    print(hwc_rgb.dtype, hwc_rgb.shape)

    chw_rgb = hwc_rgb.transpose(2, 0, 1)
    bboxes, labels, scores = net.predict([chw_rgb])
    bboxes, labels, scores = bboxes[0], labels[0], scores[0]
    chainercv.visualizations.vis_bbox(
        chw_rgb, bboxes, labels, scores,
        chainercv.datasets.voc_bbox_label_names)
    plt.show()


def callback_comp(comp_image, camera_info, net):
    hwc_bgr = cv_bridge.CvBridge().compressed_imgmsg_to_cv2(comp_image)
#    hwc_rgb = cv2.cvtColor(hwc_bgr, cv2.COLOR_BGR2RGB)
    hwc_rgb = hwc_bgr[:, :, ::-1]
#    print(camera_info)
#    print(hwc_rgb.dtype, hwc_rgb.shape)

    chw_rgb = hwc_rgb.transpose(2, 0, 1)
    bboxes, labels, scores = net.predict([chw_rgb])
    bboxes, labels, scores = bboxes[0], labels[0], scores[0]
    chainercv.visualizations.vis_bbox(
        chw_rgb, bboxes, labels, scores,
        chainercv.datasets.voc_bbox_label_names)
    plt.show()


if __name__ == '__main__':
    net = chainercv.links.model.ssd.SSD300(pretrained_model='voc0712')

    rospy.init_node('image_receiver', anonymous=True)

    sub_image = message_filters.Subscriber('/camera/image_raw', Image)
#    sub_image = message_filters.Subscriber('/camera/image_raw//compressed',
#                                           CompressedImage)
    sub_info = message_filters.Subscriber('/camera/camera_info', CameraInfo)

    ts = message_filters.TimeSynchronizer([sub_image, sub_info], 2)
    ts.registerCallback(callback_raw, net)
#    ts.registerCallback(callback_comp, net)

    rospy.spin()
