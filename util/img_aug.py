# -*- coding:utf-8 -*-
# 增广数据集，将原目录下的每张图片扩充至16张，是否左右翻转，是否裁剪再扩充至原来大小，是否高斯模糊,是否旋转90度
import os
from PIL import Image,  ImageFilter


def aug_img(raw_img, img_name, out_dir, img_num, suffix, rotate_angle):
    im = Image.open(raw_img)
    width = im.size[0]
    height = im.size[1]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # print('save' + out_dir + img_name + '_:.jpg')
    index = 1
    im.save(out_dir + img_name + '_' + str(index) + suffix)
    index += 1

    # flip
    out_imgs = os.listdir(out_dir)
    for out_img_name in out_imgs:
        out_img = Image.open(out_dir+out_img_name)
        image_1 = out_img.transpose(Image.FLIP_LEFT_RIGHT)
        image_1.save(out_dir + img_name + '_' + str(index) + suffix)
        index += 1
    # crop
    out_imgs = os.listdir(out_dir)
    for out_img_name in out_imgs:
        out_img = Image.open(out_dir+out_img_name)
        box = (0, 0, int(0.8 * width), int(0.8 * height))
        image_2 = out_img.crop(box)
        image_2 = image_2.resize((160, 160))
        image_2.save(out_dir + img_name + '_' + str(index) + suffix)
        index += 1
    # # gaussian
    # out_imgs = os.listdir(out_dir)
    # for out_img_name in out_imgs:
    #     out_img = Image.open(out_dir+out_img_name)
    #     image_3 = out_img.filter(ImageFilter.GaussianBlur(radius=1.5))
    #     image_3.save(out_dir + img_name + '_' + str(index) + suffix)
    #     index += 1
    # rotate
    out_imgs = os.listdir(out_dir)
    for out_img_name in out_imgs:
        if len(os.listdir(out_dir)) >= img_num:
            break
        out_img = Image.open(out_dir+out_img_name)
        image_4 = out_img.rotate(rotate_angle)
        image_4.save(out_dir + img_name + '_' + str(index) + suffix)
        index += 1
        image_5 = out_img.rotate(-rotate_angle)
        image_5.save(out_dir + img_name + '_' + str(index) + suffix)
        index += 1
    # for flip_id in range(2):
    #     for crop_id in range(2):
    #         for blur_id in range(2):
    #             for rotate_id in range(2):
    #                 image_1 = im.transpose(Image.FLIP_LEFT_RIGHT) if flip_id % 2 == 0 else im
    #                 box = (0, 0, int(0.8 * width), int(0.8 * height))
    #                 image_2 = image_1.crop(box) if crop_id % 2 == 0 else image_1
    #                 image_3 = image_2.filter(ImageFilter.GaussianBlur(radius=1.5)) if blur_id % 2 == 0 else image_2
    #
    #                 image_3 = image_3.resize((160, 160))
    #
    #                 image_4 = image_3.rotate(90) if rotate_id % 2 == 0 else image_3
    #                 image_4.save(out_dir + img_name + '_' + str(index) + '.jpg')
    #                 index += 1


raw_base_dir = '/home/ronglz/datasets/Face_train_mtcnn_160/'
crop_rate = 0.8
gaussian_radius = 0.8
out_num = 6
rtt = 8
suffix = '.jpg'
aug_base_dir = '/home/ronglz/datasets/Face_train_mt_aug_' + str(crop_rate) + '_' + str(gaussian_radius)\
               + '_' + str(out_num) + '_' + str(rtt) + './'
raw_sub_dirs = os.listdir(raw_base_dir)
file_count = 0
for sub_dir in raw_sub_dirs:
    img_dir = os.path.join(raw_base_dir, sub_dir+'/')
    w = os.walk(img_dir)
    for path, d, filelist in w:
        for filename in filelist:
            if filename.endswith('.png'):
                img_path = os.path.join(path, filename)
                aug_img(img_path, filename.split('.')[0].split('_')[0], aug_base_dir+sub_dir+'/',
                        out_num, suffix, rtt)
                file_count += 1
                if file_count % 100 == 0:
                    print(str(file_count)+'/2000')
