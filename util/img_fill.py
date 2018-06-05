import os,shutil
from PIL import Image

img_path = '/home/ronglz/datasets/Face_train_mtcnn_160/'
src_img_path = '/home/ronglz/datasets/Face_train_raw/'
raw_img_list = os.listdir(img_path)
for raw_img in raw_img_list:
    if raw_img.endswith("txt"):
        continue
    sub_dir = os.listdir(img_path+raw_img)
    if len(sub_dir) == 0:
        print(raw_img)
        im = Image.open(src_img_path+raw_img+"/"+raw_img+"_1.jpg")
        out = im.resize((160, 160))
        out.save(img_path+raw_img+"/"+raw_img+"_1.png")
    else:
        img_filename = img_path + raw_img + "/" + raw_img + "_1.png"
        im = Image.open(img_filename)
        out = im.resize((160, 160))
        out.save(img_path + raw_img + "/" + raw_img + "_1.png")