from data_aug.data_aug import *
from data_aug.bbox_util import *
import numpy as np 
import cv2 
import glob
import argparse
import os

def fromYoloLabel(img, yolo_label):
    file_reader = open(yolo_label, "r")
    width = img.shape[1]
    height = img.shape[0]
    bboxes = []
    for row in file_reader:
        # tmp = []
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

def transform(meta_data, img, bboxes, by):
    new_folder, each_img, each_txt = meta_data
    if by == "HSV":
        img_, bboxes_ = RandomHSV(15, 15, 15)(img.copy(), bboxes.copy())
    if by == "HorizontalFlip":
        img_, bboxes_ = RandomHorizontalFlip(1)(img.copy(), bboxes.copy())
    if by == "Scale":
        img_, bboxes_ = RandomScale(0.3, diff = True)(img.copy(), bboxes.copy())
    if by == "Translate":
        img_, bboxes_ = RandomTranslate(0.3, diff = True)(img.copy(), bboxes.copy())
    if by == "Rotate":
        img_, bboxes_ = RandomRotate(30)(img.copy(), bboxes.copy())
    if by == "Shear":
        img_, bboxes_ = RandomShear(0.3)(img.copy(), bboxes.copy())
    
    yolo_bboxes = toYoloLabel(img_, bboxes_)
    # write new img
    img_processed = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(new_folder, os.path.basename(each_img).split(".")[0]+'_'+by+'.jpg'), img_processed)
    # write new txt
    txt_writter = open(os.path.join(new_folder, os.path.basename(each_txt).split(".")[0]+'_'+by+'.txt'), "w+")
    txt_writter.write(yolo_bboxes)
    txt_writter.close()
    

# a folder is assumed to have jpg images and yolo label txts  
def process_one_folder(new_folder, target_folder):
    if not os.path.exists(new_folder):
        os.mkdir(new_folder)

    for each_txt in glob.glob(os.path.join(target_folder, "*.txt")):
        each_img = each_txt.replace(".txt", ".jpg")

        img = cv2.imread(each_img)[:,:,::-1]   #opencv loads images in bgr. the [:,:,::-1] does bgr -> rgb
        bboxes = fromYoloLabel(img, each_txt)
        
        if len(bboxes) == 0:
            print("Empty label for {}".format(each_img))
            continue
        meta_data = (new_folder, each_img, each_txt)
        if args.augment == 'all':
            trans_list = ["HSV", "HorizontalFlip", "Scale", "Translate",
                          "Rotate", "Shear"]
            for trans in trans_list:
                transform(meta_data, img, bboxes, trans)
        else:
            trans_list = args.augment.split("+")
            for trans in trans_list:
                transform(meta_data, img, bboxes, trans)
        

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
def generateTxt(dataset_folder, with_val=1):
    counter = 0
    jpg_lst = []
    txt_lst = []
    for PATH in glob.glob(os.path.join(dataset_folder, "*_ins")):
        color_img_path = os.path.join(PATH, "ColorImage")
        for each_record in glob.glob(os.path.join(color_img_path, "Record*")):
            new_folder = os.path.join(each_record, "augmented")
            for each_txt in glob.glob(os.path.join(new_folder, "*.txt")):
                jpg_loc = each_txt.replace(".txt", ".jpg")
                jpg_lst.append(jpg_loc)
                txt_lst.append(each_txt)
                counter += 1
    X_train = jpg_lst

    if with_val:
        X_train, X_test, y_train, y_test = train_test_split(jpg_lst, txt_lst, test_size=0.1, random_state=42)

        val_writter =  open(os.path.join(dataset_folder, "val.txt"), "w+")
        val_writter.write("\n".join(X_test))
        val_writter.close()
        print("val: " + str(len(X_test)))

    train_writter =  open(os.path.join(dataset_folder, "train.txt"), "w+")
    train_writter.write("\n".join(X_train))
    train_writter.close()
    print("train: " + str(len(X_train)))

    print("Total " + str(counter))


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset_folder', type=str, default="imglabel")
parser.add_argument('-a', '--augment', type=str, default="all")
args = parser.parse_args() 
main(args)
