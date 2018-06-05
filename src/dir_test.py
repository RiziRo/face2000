# test new imgs to evaluate accuracy
# input: *_2.png output:*/[1,2,3,4,5]
import tensorflow as tf
import sys
import argparse
import os
import facenet
import numpy as np
import img_test


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--lfw_dir', type=str,
                        help='Path to the data directory containing aligned LFW face patches.',
                        default='~/datasets/lfw/lfw_mtcnnpy_160/')
    parser.add_argument('--lfw_batch_size', type=int,
                        help='Number of images to process in a batch in the LFW test set.', default=100)
    parser.add_argument('--model', type=str,
                        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file',
                        default='~/models/vggface/')
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--lfw_pairs', type=str,
                        help='The file containing the pairs to use for validation.',
                        default='~/PycharmProjects/facenet/data/pairs.txt')
    parser.add_argument('--lfw_nrof_folds', type=int,
                        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    parser.add_argument('--distance_metric', type=int,
                        help='Distance metric  0:euclidian, 1:cosine similarity.', default=0)
    parser.add_argument('--use_flipped_images',
                        help='Concatenates embeddings for the image and its horizontally flipped counterpart.',
                        action='store_true')
    return parser.parse_args(argv)


# return top n max index of each row
def top_n_index(arr, n):
    top_list = dict()
    arr_copy = arr
    for n_loop in range(n):
        indices = np.argmax(arr_copy, axis=1)
        arr_copy[np.arange(len(indices)), indices] = -10000
        top_list[n_loop] = indices
    return top_list


if __name__ == '__main__':
    # main(parse_arguments(sys.argv[1:]))
    # main(filename,ckpt_path)
    test_data_dir = '/home/ronglz/datasets/Face_test_raw/'

    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    # model_name = '20180604-162515-6-.4-.4-.002-50'
    # meta_file = '/home/ronglz/models/facenet/'+model_name+'/model-'+model_name+'.meta'
    # ckpt_dir = '/home/ronglz/models/facenet/'+model_name
    meta_file = './20180605-171230-6-.4-5e-4-.005-500/model-20180605-171230-6-.4-5e-4-.005-500.meta'
    ckpt_dir = './20180605-171230-6-.4-5e-4-.005-500/'
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(meta_file)
        saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))
        graph = tf.get_default_graph()
        # embeddings = graph.get_tensor_by_name("embeddings:0")
        images_placeholder = graph.get_tensor_by_name("input:0")
        phase_train_placeholder = graph.get_tensor_by_name("phase_train:0")
        dataset = facenet.get_dataset(test_data_dir)
        paths, labels = facenet.get_image_paths_and_labels(dataset)
        images = img_test.load_and_align_data(paths, 160, 44, 0.05)
        test_num = len(paths)
        batch_size = 50
        accuracy = 0
        top_5_acc = 0
        epoch_num = int(test_num/batch_size)
        for epoch in range(epoch_num):
            print('%d of epoch %d:' % (epoch, epoch_num))
            images_batch = images[epoch * batch_size: (epoch + 1) * batch_size]
            paths_batch = paths[epoch * batch_size: (epoch + 1) * batch_size]
            labels_batch = labels[epoch * batch_size: (epoch + 1) * batch_size]

            # images = facenet.load_data(paths_batch, do_random_crop=False, do_random_flip=False, image_size=160)
            feed_dict = {images_placeholder: images_batch, phase_train_placeholder: False}
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
            batch_top_5_acc = top_5_equal_count/batch_size
            top_5_acc += batch_top_5_acc
            for i in range(len(labels_batch)):
                img_name = paths_batch[i].split('/')[-1]
                predict_str = img_name + ',gt label:'
                predict_str += str(labels_batch[i])
                # predict_str += ',top1:'
                # predict_str += str(best_class_indices[i])
                predict_str += ', top5:'
                for j in range(5):
                    predict_str += str(top_dic[j][i]) + ','
                print(predict_str)
            print('top1 accuracy:%.3f' % batch_accuracy)
            print('top5 accuracy:%.3f' % batch_top_5_acc)

        accuracy /= (test_num/batch_size)
        top_5_acc /= (test_num/batch_size)
        print('--------------------------------')
        print('average top1 accuracy:%.3f' % accuracy)
        print('average top5 accuracy:%.3f' % top_5_acc)
