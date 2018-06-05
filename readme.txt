1.如何训练:
使用src/train_softmax.py文件,
在第507行设置pre_trained的模型参数文件
从链接: https://pan.baidu.com/s/1jdM22vvZwCCPkiblkwY5Zg 密码: 7eyf中下载vggface文件夹,包含了要用到的模型
在第510行设置默认使用的数据集，数据集预处理方式见3
第520-560行设置训练过程的参数,如learning_rate,weight_decay,epoch,epoch_size,batch_size,
keep_probability等
一组推荐的参数是learning_rate=0.005,weight_decay=5e-4,epoch=500,epoch_size=100,batch_size=100,
keep_probability=0.4
训练时,每隔10次输出一个记录,格式为[Epoch信息 Time Loss cross_entropy regularization_loss
train_accuracy learning_rate center_loss]其中Loss=cross_entropy+regularization_loss
2.如何测试:
可以进行两种文件结构的测试,输出的格式为:[图片名,真实类别,top5的预测类别(第一个是top1的结果)]
每个batch输出top1和top5准确率,最终输出总的平均准确率，测试集无需手动进行mtcnn人脸检测，程序内会自动检测
2.1文件结构为test_imgs/(0/0_2.jpg - x/x_2.jpg),即两层目录.使用src/dir_test.py文件,
在第52行的test_data_dir变量中设置测试文件目录,即test_data_dir=上面的test_imgs,
从链接: https://pan.baidu.com/s/1jdM22vvZwCCPkiblkwY5Zg 密码: 7eyf中下载20180605-171230-6-.4-5e-4-.005-500
文件夹,包含了训练好的模型
在第60行的meta_file变量中设置.meta文件的位置
在第61行的ckpt_dir变量中设置ckpt文件夹的位置,即.meta文件所在的文件夹
在第73行设置batch_size,默认为50
请注意相对路径和绝对路径,否则找不到文件
2.2文件结构为test_imgs/(0_2.jpg,x_2.jpg~y_2,jpg),即一层目录.使用src/img_test.py文件,
在第129行的img_path变量中设置测试文件目录,即img_path=上面的test_imgs,
在第135行的meta_file变量中设置.meta文件的位置
在第136行的ckpt_dir变量中设置ckpt文件夹的位置,即.meta文件所在的文件夹
在第83行设置batch_size,默认为50
请注意相对路径和绝对路径,否则找不到文件
3.训练数据集预处理方式
3.1使用mtcnn进行人脸检测
执行 export PYTHONPATH=[...]/facenet/src
 [...] should be replaced with the directory where the cloned facenet repo resides.
 python src/align/align_dataset_mtcnn.py \
src_dir \
dst_dir \
--image_size 182 \
--margin 44
3.2数据增广
使用util/img_aug.py文件，
在第69行设置原图目录
在第70行设置裁剪的倍数
在第71行设置高斯模糊的参数
在第72行设置输出图像个数
在第73行设置旋转角度
在第74行设置增广后图片的后缀（'.jpg'或'.png'）
执行的顺序是，先进行反转，数量×2，在进行裁剪和尺寸还原，数量再×2，再进行高斯模糊，数量再×2，再进行旋转。
以上步骤中检测到图片数目达到输出图片个数时，即停止。
