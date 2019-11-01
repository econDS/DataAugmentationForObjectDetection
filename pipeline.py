from data_aug.data_aug import *
from data_aug.bbox_util import *
import numpy as np 
import cv2 
import matplotlib.pyplot as plt 
import glob
import argparse

def fromYoloLabel(img, yolo_label):
    file_reader = open(yolo_label, "r")
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
    if len(bboxes) == 0:
        return bboxes
    bboxes = clip_box(bboxes, [0,0,1 + width, height], 0.25)
    return bboxes

def toYoloLabel(img, bboxes):
    ret = []
    for each_bbox in bboxes:
        cls_id = str(int(each_bbox[-1]))
        [x1,y1,x2,y2] = each_bbox[:4]
        w = x2 - x1
        h = y2 - y1
        x_c = str(((x1 + w/2)/img.shape[1]))
        y_c = str((y1 + h/2)/img.shape[0])
        w /= img.shape[1]
        h /= img.shape[0]
        ret.append(" ".join([cls_id, x_c, y_c, str(w), str(h)]))
    return "\n".join(ret)

# a folder is assumed to have jpg images and yolo label txts  
def process_one_folder(new_folder, target_folder):
    if not os.path.exists(new_folder):
        os.mkdir(new_folder)

    for each_txt in glob.glob(os.path.join(target_folder, "*.txt")):
        each_img = each_txt.replace(".txt", ".png")

        img = cv2.imread(each_img)[:,:,::-1]   #opencv loads images in bgr. the [:,:,::-1] does bgr -> rgb
        bboxes = fromYoloLabel(img, each_txt)
        
        if len(bboxes) == 0:
            print("Empty label for {}".format(each_img))
            continue
        #seq = Sequence([AdjustBrightnessAndContrast(64, 32), RandomHorizontalFlip(0.5), conditionalZoomIn(3)])
        seq = Sequence([RandomHorizontalFlip(0.5)])
        img_, bboxes_ = seq(img.copy(), bboxes.copy())

        yolo_bboxes = toYoloLabel(img_, bboxes_)
        # write new img
        img_processed = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(new_folder, os.path.basename(each_img)), img_processed)
        # write new txt
        txt_writter = open(os.path.join(new_folder, os.path.basename(each_txt)), "w+")
        txt_writter.write(yolo_bboxes)
        txt_writter.close()

        '''
        for i, each_pair in enumerate(res):
            (img_, bboxes_) = each_pair
            img_processed, bboxes_processed = RandomHorizontalFlip(0.5)(img_.copy(), bboxes_.copy())
            yolo_bboxes = toYoloLabel(img_processed, bboxes_processed)
            
            # write new img
            img_processed = cv2.cvtColor(img_processed, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(new_folder, str(i) + os.path.basename(each_img)), img_processed)
            # write new txt
            txt_writter = open(os.path.join(new_folder, str(i) + os.path.basename(each_txt)), "w+")
            txt_writter.write(yolo_bboxes)
            txt_writter.close()
        '''

# main function for tsinghua cyclists 
def main(args):
    print(args)
    new_folder = os.path.join(args.dataset_folder, "augmented")    
    process_one_folder(new_folder, target_folder=args.dataset_folder)


'''
# main function for apollo

def main(args):
    print(args)
    for PATH in glob.glob(os.path.join(args.dataset_folder, "*_ins")):
        color_img_path = os.path.join(PATH, "ColorImage")
        for each_record in glob.glob(os.path.join(color_img_path, "Record*")):
            new_folder = os.path.join(each_record, "augmented")
            for target_folder in glob.glob(os.path.join(each_record, "Camera*")):
                print(target_folder)
                process_one_folder(new_folder, target_folder)
'''

# generate yolo format train.txt or val.txt
def generateTxt(dataset_folder, dst):
    counter = 0
    for PATH in glob.glob(os.path.join(dataset_folder, "*_ins")):
        color_img_path = os.path.join(PATH, "ColorImage")
        for each_record in glob.glob(os.path.join(color_img_path, "Record*")):
            new_folder = os.path.join(each_record, "augmented")
            temp_txt_lst = []
            for each_txt in glob.glob(os.path.join(new_folder, "*.txt")):
                jpg_loc = each_txt.replace(".txt", ".jpg")
                temp_txt_lst.append(jpg_loc)
                counter += 1
            dst_writter =  open(dst, "a+")
            dst_writter.write("\n".join(temp_txt_lst) + "\n")
            dst_writter.close()
                   
    print("Total " + str(counter))

parser = argparse.ArgumentParser()
#args.add_argument('--target_folder', type=str, default="/home/kevin/ascent/dataset/apolloScape/road02_ins/ColorImage/Record001/Camera\ 5/")
#args.add_argument('--new_folder', type=str, default="/home/kevin/ascent/dataset/apolloScape/road02_ins/ColorImage/Record001/augmented_Camera\ 5/") 
parser.add_argument('--dataset_folder', type=str, default="/home/kevin/ascent/dataset/apolloScape/")
args = parser.parse_args() 
main(args)
#generateTxt(args.dataset_folder, os.path.join(args.dataset_folder, "train.txt"))
#print(yolo_bboxes)
#plotted_img = draw_rect(img_, bboxes_)
#plt.imshow(plotted_img)
#plt.show()


