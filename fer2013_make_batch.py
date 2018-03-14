import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


anger_0 = []
anger_0_labels = []
disgust_1 = []
disgust_1_label = []
fear_2 = []
fear_2_label = []
happy_3 = []
happy_3_label = []
sad_4 = []
sad_4_label = []
surprised_5 = []
surprised_5_label = []
normal_6 = []
normal_6_label = []


def get_file(file_dir):
    for file in os.listdir(file_dir + '0'):
        anger_0.append(file_dir + '0' + '/' + file)
        anger_0_labels.append(0)
    for file in os.listdir(file_dir + '1'):
        disgust_1.append(file_dir + '1' + '/' + file)
        disgust_1_label.append(1)
    for file in os.listdir(file_dir + '2'):
        fear_2.append(file_dir + '2' + '/' + file)
        fear_2_label.append(2)
    for file in os.listdir(file_dir + '3'):
        happy_3.append(file_dir + '3' + '/' + file)
        happy_3_label.append(3)
    for file in os.listdir(file_dir + '4'):
        sad_4.append(file_dir + '4' + '/' + file)
        sad_4_label.append(4)
    for file in os.listdir(file_dir + '5'):
        surprised_5.append(file_dir + '5' + '/' + file)
        surprised_5_label.append(5)
    for file in os.listdir(file_dir + '6'):
        normal_6.append(file_dir + '6' + '/' + file)
        normal_6_label.append(6)

    # hstack and shuffle
    image_list = np.hstack((anger_0, disgust_1, fear_2, happy_3, sad_4, surprised_5, normal_6))
    label_list = np.hstack((anger_0_labels, disgust_1_label, fear_2_label, happy_3_label, sad_4_label, surprised_5_label, normal_6_label))

    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    # convert all shuffled_temp into list
    all_image_list = list(temp[:, 0])
    all_label_list = list(temp[:, 1])
    all_label_list = [int(i) for i in all_label_list]

    return all_image_list, all_label_list


def get_batch(image, label, image_W, image_H, batch_size, capacity):
    # To change data type
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # To make an input queue
    input_queue = tf.train.slice_input_producer([image, label])

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])  # read image from a queue

    # To decode image
    image = tf.image.decode_jpeg(image_contents, channels=1)

    # To pre_process image
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    image = tf.image.per_image_standardization(image)

    # To create batch
    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=32,
                                              capacity=capacity)
    # To reshape label
    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
    return image_batch, label_batch

