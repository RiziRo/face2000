import os,shutil

img_path = '/home/ronglz/datasets/HW_1_Face/'
dst_img_path = '/home/ronglz/datasets/Face_test_raw/'
raw_img_list = os.listdir(img_path)
for raw_img in raw_img_list:
    if raw_img.split('_')[1][0] == '2':
        p_id = raw_img.split('_')[0]
        sub_path = os.path.join(dst_img_path, p_id) + '/'
        if not os.path.exists(sub_path):
            os.makedirs(sub_path)
        shutil.copyfile(img_path+raw_img, sub_path+raw_img)
