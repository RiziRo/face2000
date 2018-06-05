import tensorflow as tf
import os
import facenet
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=True))
meta_file_fc = '/home/ronglz/models/facenet/20180602-094439/model-20180602-094439.meta'
ckpt_dir_fc = '/home/ronglz/models/facenet/20180602-094439'
meta_file_original = '/home/ronglz/models/facenet/20180602-093529/model-20180602-093529.meta'
ckpt_dir_original = '/home/ronglz/models/facenet/20180602-093529'
meta_file_all = '/home/ronglz/models/facenet/20180601-223222/model-20180601-223222.meta'
ckpt_dir_all = '/home/ronglz/models/facenet/20180601-223222'
dataset = facenet.get_dataset('/home/ronglz/datasets/Face_train_raw/')
paths, labels = facenet.get_image_paths_and_labels(dataset)
paths_batch = paths[0: 1]
labels_batch = labels[0: 1]
images = facenet.load_data(paths_batch, do_random_crop=False, do_random_flip=False, image_size=160)
with tf.Session() as sess:
    saver_fc = tf.train.import_meta_graph(meta_file_fc)
    saver_fc.restore(sess, tf.train.latest_checkpoint(ckpt_dir_fc))
    graph_fc = tf.get_default_graph()
    images_placeholder_fc = graph_fc.get_tensor_by_name("input:0")
    phase_train_placeholder_fc = graph_fc.get_tensor_by_name("phase_train:0")
    embeddings_fc = graph_fc.get_tensor_by_name("embeddings:0")
    feed_dict_fc = {images_placeholder_fc: images, phase_train_placeholder_fc: False}
    emb_fc = sess.run(embeddings_fc, feed_dict=feed_dict_fc)
    print('emb_fc size:'+str(emb_fc.size))
    # original
    saver_original = tf.train.import_meta_graph(meta_file_original)
    saver_original.restore(sess, tf.train.latest_checkpoint(ckpt_dir_original))
    graph_original = tf.get_default_graph()
    images_placeholder_original = graph_original.get_tensor_by_name("input:0")
    phase_train_placeholder_original = graph_original.get_tensor_by_name("phase_train:0")
    embeddings_original = graph_original.get_tensor_by_name("embeddings:0")
    feed_dict_original = {images_placeholder_original: images, phase_train_placeholder_original: False}
    emb_original = sess.run(embeddings_original, feed_dict=feed_dict_original)
    print('emb_original size:' + str(emb_original.size))
    # all
    saver_all = tf.train.import_meta_graph(meta_file_all)
    saver_all.restore(sess, tf.train.latest_checkpoint(ckpt_dir_all))
    graph_all = tf.get_default_graph()
    images_placeholder_all = graph_all.get_tensor_by_name("input:0")
    phase_train_placeholder_all = graph_all.get_tensor_by_name("phase_train:0")
    embeddings_all = graph_all.get_tensor_by_name("embeddings:0")
    feed_dict_all = {images_placeholder_all: images, phase_train_placeholder_all: False}
    emb_all = sess.run(embeddings_all, feed_dict=feed_dict_all)
    print('emb_all size:' + str(emb_all.size))

    print(np.sum(np.equal(emb_original, emb_all)))
    print(np.sum(np.equal(emb_original, emb_fc)))
