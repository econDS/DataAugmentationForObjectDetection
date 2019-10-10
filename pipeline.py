from data_aug.data_aug import *
from data_aug.bbox_util import *
import numpy as np 
import cv2 
import matplotlib.pyplot as plt 

sample_img = "/home/kevin/ascent/dataset/apolloScape/road02_ins/ColorImage/Record001/Camera 5/170927_063847913_Camera_5.jpg"
sample_lbl = "/home/kevin/ascent/dataset/apolloScape/road02_ins/ColorImage/Record001/Camera 5/170927_063847913_Camera_5.txt"
img = cv2.imread(sample_img)[:,:,::-1]   #opencv loads images in bgr. the [:,:,::-1] does bgr -> rgb
bboxes = pkl.load(open("messi_ann.pkl", "rb"))

file_reader = open(sample_lbl, "r")
width = img.shape[1]
height = img.shape[0]
bboxes = []
for row in file_reader:
    tmp = []
    row = row.strip("\n").split(" ")
    x_c = float(row[1])
    y_c = float(row[2])
    w = float(row[3])*width
    h = float(row[4])*height
    cls_id = float(row[0])
    x1 = x_c*width-w/2
    y1 = y_c*height-h/2
    x2 = x_c*width+w/2
    y2 = y_c*height+h/2
    bboxes.append([x1, y1, x2, y2, cls_id])

#inspect the bounding boxes
bboxes = np.array(bboxes, dtype="float64")




seq = Sequence([RandomHSV(40, 40, 30), RandomHorizontalFlip(0.5), conditionalZoomIn(3)])
img_, bboxes_ = seq(img.copy(), bboxes.copy())
plotted_img = draw_rect(img_, bboxes_)
plt.imshow(plotted_img)
plt.show()


