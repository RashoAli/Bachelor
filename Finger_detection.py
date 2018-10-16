#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 16:29:05 2018

@author: ali
"""
import cv2
import sys
import os
import PIL
from PIL import Image, ImageDraw, PILLOW_VERSION
import numpy as np
from matplotlib import pyplot as plt


def finger_convex(img):  # gibt alle qurven_punkte auf dem img

    img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (thresh, im_bw) = cv2.threshold(img_g, 150, 200, 0)
    _, contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print('contours',contours[0])
    cnt = contours[0]
    hull = cv2.convexHull(cnt, returnPoints=False)
    # print('cnt',cnt)
    # print('hull',hull)
    defects = cv2.convexityDefects(cnt, hull)
    # print('defects',defects)
    far_list = list()
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        # start = tuple(cnt[s][0])
        # end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        # cv2.line(img_g, start, end, [255, 0, 0], 2)
        # cv2.circle(img_g, far, 2, [100, 0, 0], -1)
        far_list.append(far)
    return far_list


def nachbaren(img, x, y, r):  # gibt wie vie protzen von die benachbarte pix's nicht 0 sind
    c_img = img.copy()
    c_img[:x - r, :] = 0
    c_img[x + r:, :] = 0
    c_img[:, :y - r] = 0
    c_img[:, y + r:] = 0
    no_zero = np.count_nonzero(c_img)
    no_zero_prot = no_zero / (4 * r * r)
    return no_zero_prot


def finger_color(img, start, line_end, img2, color):
    '''
    lienie zwichen jeweilige_nacheinandere finger_convex pixel duchziehen
    so entstehen zwei Objekten (ein finger und rest des handes),
    mit die funktion flooFill der hand objekt = 0 ein setzen 
    '''

    img = img.copy()  # der finger isoliert
    # img2 = np.asarray(img2)  # hand mit dem finger markiert
    img2.setflags(write=1)
    cv2.line(img, start, line_end, 0, 1)

    h, w = img.shape[:2]
    # mask = np.zeros((h + 2, w + 2), np.uint8)
    # start_pixel = (abs(int((start[0] + line_end[0]) / 2)) - 0, abs(int((start[1] + line_end[1]) / 2)) - 5)
    # cv2.floodFill(img, mask, start_pixel, 0)
    ''' 
    start pixel für foolFill vestlegen ,in dem 5 pixel unten die rand pixelen annehmen
    '''

    mask1 = np.zeros((h + 2, w + 2), np.uint8)
    mask2 = np.zeros((h + 2, w + 2), np.uint8)
    start_pixel1 = (start[0], start[1] - 5)
    start_pixel2 = (line_end[0], line_end[1] - 5)
    cv2.floodFill(img, mask1, start_pixel1, 0)
    cv2.floodFill(img, mask2, start_pixel2, 0)

    img2[:, :, 0][img == 255] = color[0]
    img2[:, :, 1][img == 255] = color[1]
    img2[:, :, 2][img == 255] = color[2]

    return img, img2


def finger(img):
    img2 = np.asarray(img.copy())
    img_array = np.asarray(img).copy()

    far_list = finger_convex(img_array)
    far_array = np.asarray(far_list)

    y = np.asarray(far_array[:, 0])
    x = np.asarray(far_array[:, 1])

    img_gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    finger_bot = list()
    for i in range(x.shape[0]):

        no_zero = nachbaren(img_gray, x[i], y[i], 6)
        # print(no_zero,'no be',i)
        if no_zero > 0.85:
            position = ((no_zero, x[i], y[i]))
            finger_bot.append(position)

    finger_bot = np.asarray(finger_bot)

    y = np.int_(finger_bot[:, 1])
    x = np.int_(finger_bot[:, 2])

    np.asarray(img_gray)
    img_gray[img_gray < 50] = 0  # convertieren in bienere format
    img_gray[img_gray != 0] = 255

    fingers = np.zeros((len(img_array), len(img_array[0]), 5))  # 5 bynere bilder speicheren 5 finger

    for i in range(3):
        if i == 0: color = (255, 0, 0)
        if i == 1: color = (0, 255, 0)
        if i == 2: color = (0, 0, 255)

        ff, img22 = finger_color(img_gray, (x[i], y[i]), (x[i + 1], y[i + 1]), img2, color)
        fingers[:, :, i + 1]

    length = 30
    angle = np.arctan((y[0] - y[1]) / (x[0] - x[1])) + 3
    x1 = x[0]
    y1 = y[0]
    x2 = x1 + int(np.cos(angle) * length)
    y2 = y1 + int(np.sin(angle) * length)
    # cv2.line(img22, (x1,y1), (x2,y2), [255, 255, 255], 4)
    temp, img22 = finger_color(img_gray, (x1, y1), (x2, y2), img22, (255, 255, 0))
    fingers[:, :, 0] = temp  # das erste finger ist der Daumen

    length = 20
    angle = np.arctan((y[-1] - y[-2]) / (x[-1] - x[-2])) + 0
    x1 = x[-1]
    y1 = y[-1]
    x2 = x1 + int(np.cos(angle) * length)
    y2 = y1 + int(np.sin(angle) * length)
    # cv2.line(img2, (x1,y1), (x2,y2), [255, 255, 255], 4)
    temp, img22 = finger_color(img_gray, (x1, y1), (x2, y2), img22, (255, 0, 255))
    fingers[:, :, 4] = temp

    return img22


img = Image.open('hand.jpg')
img2 = finger(img)

plt.imshow(img2)
plt.show()