import tensorflow as tf
import copy
import align.detect_face
import os
from scipy import misc
import numpy as np
import facenet

# return top n max index of each row
def top_n_index(arr, n):
    top_list = dict()
    arr_copy = arr
    for n_loop in range(n):
        indices = np.argmax(arr_copy, axis=1)
        arr_copy[np.arange(len(indices)), indices] = -10000
        top_list[n_loop] = indices
    return top_list

def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction):
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    print('Creating align networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

    tmp_image_paths = copy.copy(image_paths)
    img_list = []
    cannot_detect = 0
    for image in tmp_image_paths:
        img = misc.imread(os.path.expanduser(image), mode='RGB')
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        if len(bounding_boxes) < 1:
            cannot_detect += 1
            print(str(cannot_detect)+':'+str(image.split('/')[-1])+",can't detect face, use raw image")
            not_aligned = misc.imresize(img, (image_size, image_size), interp='bilinear')
            na_prewhitened = facenet.prewhiten(not_aligned)
            img_list.append(na_prewhitened)
        else:
            det = np.squeeze(bounding_boxes[0, 0:4])
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0]-margin/2, 0)
            bb[1] = np.maximum(det[1]-margin/2, 0)
            bb[2] = np.minimum(det[2]+margin/2, img_size[1])
            bb[3] = np.minimum(det[3]+margin/2, img_size[0])
            cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
            aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
            prewhitened = facenet.prewhiten(aligned)
            img_list.append(prewhitened)
    images = np.stack(img_list)
    return images


def get_image_paths_and_labels(dataset):
    img_sub_names = os.listdir(dataset)
    image_paths_flat = list()
    labels_flat = list()
    for i in range(len(img_sub_names)):
        image_paths_flat.append(dataset+img_sub_names[i])
        labels_flat.append(int(img_sub_names[i].split('_')[0]))
    return image_paths_flat, labels_flat


def do_test(test_data_dir, meta_file, ckpt_dir):
    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

    saver = tf.train.import_meta_graph(meta_file)
    saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))
    graph = tf.get_default_graph()
    # embeddings = graph.get_tensor_by_name("embeddings:0")
    images_placeholder = graph.get_tensor_by_name("input:0")
    phase_train_placeholder = graph.get_tensor_by_name("phase_train:0")

    paths, labels = get_image_paths_and_labels(test_data_dir)
    test_num = len(paths)
    batch_size = 50
    accuracy = 0
    top_5_acc = 0
    epoch_num = int(test_num / batch_size)
    for epoch in range(epoch_num):
        print('%d of epoch %d:' % (epoch, epoch_num))
        paths_batch = paths[epoch * batch_size: (epoch + 1) * batch_size]
        labels_batch = labels[epoch * batch_size: (epoch + 1) * batch_size]
        images = load_and_align_data(paths_batch, 160, 44, 0.05)

        feed_dict = {images_placeholder: images, phase_train_placeholder: False}
        # emb = sess.run(embeddings, feed_dict=feed_dict)
        BiasAdd = graph.get_tensor_by_name('Logits/BiasAdd:0')
        output = sess.run(BiasAdd, feed_dict=feed_dict)
        best_class_indices = np.argmax(output, axis=1)
        batch_accuracy = np.mean(np.equal(best_class_indices, labels_batch))
        accuracy += batch_accuracy

        top_dic = top_n_index(output, 5)
        top_5_equal_count = 0
        for i in range(5):
            top_list = top_dic[i]
            top_5_equal_count += np.sum(np.equal(top_list, labels_batch))
        batch_top_5_acc = top_5_equal_count / batch_size
        top_5_acc += batch_top_5_acc
        for i in range(len(labels_batch)):
            img_name = paths_batch[i].split('/')[-1]
            predict_str = img_name + ',gt label:'
            predict_str += str(labels_batch[i])
            predict_str += ',top1:'
            predict_str += str(best_class_indices[i])
            predict_str += ', top5:'
            for j in range(5):
                predict_str += str(top_dic[j][i]) + ','
            print(predict_str)
        print('top1 accuracy:%.3f' % batch_accuracy)
        print('top5 accuracy:%.3f' % batch_top_5_acc)

    accuracy /= (test_num / batch_size)
    top_5_acc /= (test_num / batch_size)
    print('--------------------------------')
    print('average top1 accuracy:%.3f' % accuracy)
    print('average top5 accuracy:%.3f' % top_5_acc)


if __name__ == '__main__':
    img_path = '/home/ronglz/datasets/HW_1_Face/'
    img_list = os.listdir(img_path)
    image_size = 160
    margin = 44
    gpu_memory_fraction = 0.8
    model_name = '20180604-162515-6-.4-.4-.002-50'
    meta_file = '/home/ronglz/models/facenet/' + model_name + '/model-' + model_name + '.meta'
    ckpt_dir = '/home/ronglz/models/facenet/' + model_name
    test_img_list = list()
    for i in range(5):
        test_img_list.append(img_path + img_list[i])
    # images = load_and_align_data(test_img_list, image_size, margin, gpu_memory_fraction)
    do_test(img_path, meta_file, ckpt_dir)
