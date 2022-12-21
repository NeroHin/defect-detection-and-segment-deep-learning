import json
from tqdm import tqdm
from PIL import Image
import os
import pandas as pd
import shutil
import pathlib
import argparse

argparser = argparse.ArgumentParser(description="Preprocessing for YOLOv7")
argparser.add_argument("-r", "--rename", help="rename the image, label and mask files", action="store_true", default=False)
argparser.add_argument("-s", "--save", help="save the image, label and mask files", action="store_true", default=False)
argparser.add_argument("-c", "--copy", help="copy the image, label and mask files", action="store_true", default=False)

args = argparser.parse_args()
args.rename = False
args.save = False
args.copy = False

yolo_train_image_path = "../defect-detection-and-segment-deep-learning/yolov7/defect/images/train/"
yolo_val_image_path = "../defect-detection-and-segment-deep-learning/yolov7/defect/images/val/"
yolo_image_path = "../defect-detection-and-segment-deep-learning/yolov7/defect/"

pathlib.Path(yolo_train_image_path).mkdir(parents=True, exist_ok=True)
pathlib.Path(yolo_val_image_path).mkdir(parents=True, exist_ok=True)

train_set_dir = '../defect-detection-and-segment-deep-learning/class_data/Train'
test_set_dir = '../defect-detection-and-segment-deep-learning/class_data/Val'

class_names = ['powder_uncover', 'powder_uneven', 'scratch']
types = ['image']
yolo_csv = pd.DataFrame(columns=["category", "x", "y", "w", "h", "image_name", "set_type" ,"image_path"])


# convert the label to yolo format
def yolo_format(convert_img_file:str, save_img_file_name:str, save:bool=False):
    
    label_path = convert_img_file.replace("image", "label").replace(".png", ".json")
    image_name = convert_img_file.split("/")[-1]
    
    with open(label_path, "r") as file:
        json_file = json.load(file)

    width, height = Image.open(convert_img_file).size
    
    for annotation in json_file["shapes"]:
        if annotation["label"] == "powder_uncover":
            category_id = 0
        elif annotation["label"] == "powder_uneven":
            category_id = 1
        else:
            category_id = 2
        points = annotation["points"]
        
        # point[0] is left top of the bounding box
        # point[1] is right bottom of the bounding box
        
        x_min, y_min = points[0]
        x_max, y_max = points[1]
        
        
        # convert the bounding box to yolo format
        x = (x_min + (x_max-x_min)/2) * 1.0 / width
        y = (y_min + (y_max-y_min)/2) * 1.0 / height
        w = (x_max-x_min) * 1.0 / width
        h = abs((y_max-y_min) * 1.0 / height)
  
        
        yolo_format_data = str(category_id) + " " + str(x) + " " + str(y) + " " + str(w) + " " + str(h)
        
        yolo_csv.loc[len(yolo_csv)] = [category_id, x, y, w, h, image_name, save_img_file_name ,convert_img_file]
        
        # save the bounding box and category_id to a text file
        # name is the same as the image name
        
        
        if save == True:
        
            with open(yolo_train_image_path.replace("images", "labels").replace("train", save_img_file_name) + image_name.replace(".png", ".txt"), "a") as file:
                file.write(yolo_format_data)
                file.write("\n")
                
    if save == True:
    # save the image path to a text file
        with open(yolo_image_path + f"{ save_img_file_name }.txt", "a") as file:
            file.write(f"./images/{ save_img_file_name }/" + image_name)
            file.write("\n")

# rename the image, label and mask files
def rename_files(dataset_dir:str):
    
    for dataset in [dataset_dir]:
        for class_name in class_names:
            class_dir = os.path.join(dataset, class_name)
            # concatenate the image, label and mask directories
            for type_name in types:
                type_dir = os.path.join(class_dir, type_name)
                for filename in tqdm(os.listdir(type_dir)):
                    # rename the image name with class + image name

                    if rename == True:
                        # rename the image name with class + image name
                        os.rename(src=f"{type_dir}/{filename}", dst=f"{type_dir}/{class_name}_{filename}")
                        
                        # rename the label name with class + label name
                        os.rename(src=f"{type_dir.replace('image', 'label')}/{filename.replace('.png', '.json')}", dst=f"{type_dir.replace('image', 'label')}/{class_name}_{filename.replace('.png', '.json')}")
                        
                        # rename the mask name with class + mask name
                        os.rename(src=f"{type_dir.replace('image', 'mask')}/{filename}", dst=f"{type_dir.replace('image', 'mask')}/{class_name}_{filename}")
                    


if __name__ == "__main__":
    
    if rename == True:
    
        rename_files(dataset_dir=train_set_dir)
        rename_files(dataset_dir=test_set_dir)
    
    
    for dataset in [train_set_dir, test_set_dir]:
        for class_name in class_names:
            class_dir = os.path.join(dataset, class_name)
            # concatenate the image, label and mask directories
            for type_name in types:
                type_dir = os.path.join(class_dir, type_name)
                for filename in tqdm(os.listdir(type_dir)):

                    if dataset == train_set_dir:
                        yolo_format(convert_img_file=f"{type_dir}/{filename}", save_img_file_name='train',save=save)
                    else:
                        yolo_format(convert_img_file=f"{type_dir}/{filename}", save_img_file_name='val', save=save)
                        
    if copy == True:
        for dataset in ["train", "val"]:
            for image_paths in yolo_csv.query(f"set_type == '{dataset}'")["image_path"].unique():
                # print(image_paths)
                shutil.copy(image_paths, yolo_image_path + f"images/{dataset}/")
            print(f"copy {dataset} images done")
    
    
