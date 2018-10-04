import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt


def finger(img):  # gibt alle qurven_punkte auf dem img
    img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    (thresh, im_bw) = cv2.threshold(img_g, 2, 200, 0)
    _, contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    hull = cv2.convexHull(cnt, returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull)

    far_list = list()
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        # cv2.line(img_g, start, end, [255, 0, 0], 2)
        cv2.circle(img_g, far, 10, [255, 0, 0], -1)
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


def finger_color(img, start, line_end, img2, color): # gib jeder finger eine farbe
    img = img.copy()  # der finger isoliert
    img2 = np.asarray(img2)  # hand mit dem finger markiert
    img2.setflags(write=1)

    cv2.line(img, start, line_end, 0, 2)
    h, w = img.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    start_pixel = (abs(int((start[0] + line_end[0]) / 2)) - 10, abs(int((start[1] + line_end[1]) / 2)) + 10)
    cv2.circle(img2, start_pixel, 10, [0, 0, 0], -1)
    cv2.floodFill(img, mask, start_pixel, 0)

    img2[:, :, 0][img == 1] = color[0]
    img2[:, :, 1][img == 1] = color[1]
    img2[:, :, 2][img == 1] = color[2]

    return img, img2 # img gibt der figer, img2 gibt gefäbte finger auf der hand


img = Image.open('Hand.jpg')
#plt.imshow(img)
#plt.show()
img2 = img.copy()
img_array = np.asarray(img).copy()
img_array[img_array > 224] = 0
far_list = finger(img_array)
far_array = np.asarray(far_list)

y = np.asarray(far_array[:, 0])
x = np.asarray(far_array[:, 1])

img_gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
finger_bot = list()
for i in range(x.shape[0]):
    # print(x[i],y[i])
    no_zero = nachbaren(img_gray, x[i], y[i], 20)
    if no_zero > 0.6:
        position = ((no_zero, x[i], y[i]))
        finger_bot.append(position)

finger_bot = np.asarray(finger_bot)

y = np.int_(finger_bot[:, 1])
x = np.int_(finger_bot[:, 2])

np.asarray(img_gray)
img_gray[img_gray != 0] = 1  # convertieren in bienere format
for i in range(3):
    if i == 0: color = (255, 0, 0)
    if i == 1: color = (0, 255, 0)
    if i == 2: color = (0, 0, 255)

    ff, img2 = finger_color(img_gray, (x[i], y[i]), (x[i + 1], y[i + 1]), img2, color)


length = 300
angle = np.arctan((y[0]-y[1])/(x[0]-x[1]))+0.2
x1 = x[0]
y1 = y[0]
x2 = x1 + int(np.cos(angle) * length)
y2 = y1 + int(np.sin(angle) * length)
#cv2.line(img2, (x1,y1), (x2,y2), [255, 255, 255], 4)
_, img2 = finger_color(img_gray, (x1, y1), (x2, y2), img2, (255,255,0))

length = 200
angle = np.arctan((y[-1]-y[-2])/(x[-1]-x[-2]))+3
x1 = x[-1]
y1 = y[-1]
x2 = x1 + int(np.cos(angle) * length)
y2 = y1 + int(np.sin(angle) * length)
#cv2.line(img2, (x1,y1), (x2,y2), [255, 255, 255], 4)
_, img2 = finger_color(img_gray, (x1, y1), (x2, y2), img2, (255,0,255))

plt.imshow(img2)
plt.show()
