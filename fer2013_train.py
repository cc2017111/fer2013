import os
import numpy as np
import tensorflow as tf
import fer2013_make_batch as input_data
import fer2013_model as model
from tensorflow.contrib.layers import l2_regularizer


N_CLASSES = 7
IMG_W = 48
IMG_H = 48
TRAIN_BATCH_SIZE = 32
VALIDATION_BATCH_SIZE = 100
CAPACITY = 256
MAX_STEP = 50000
LEARNING_RATE = 0.0001
REGULARITION_RATE = 0.0001

train_dir = "D:/fer2013/train/"
logs_train_dir = "C:/Users/PVer/Desktop/train/log/train/"
logs_validation_dir = "D:/fer2013/val/"

train, train_label = input_data.get_file(file_dir=train_dir)
validation, validation_label = input_data.get_file(file_dir=logs_validation_dir)

train_batch, train_label_batch = input_data.get_batch(train, train_label,
                                                      IMG_W, IMG_H,
                                                      TRAIN_BATCH_SIZE, CAPACITY)
validation_batch, validation_label_batch = input_data.get_batch(validation, validation_label,
                                                                IMG_W, IMG_H,
                                                                VALIDATION_BATCH_SIZE, CAPACITY)

regularizer = l2_regularizer(REGULARITION_RATE)

train_logits_op = model.inference(images=train_batch, batch_size=TRAIN_BATCH_SIZE, n_classes=N_CLASSES,
                                  regularizer=regularizer, reuse=False)

validation_logits_op = model.inference(images=validation_batch, batch_size=VALIDATION_BATCH_SIZE, n_classes=N_CLASSES,
                                       regularizer=None, reuse=True)

train_losses_op = model.losses(logits=train_logits_op, labels=train_label_batch)

validation_losses_op = model.losses(logits=validation_logits_op, labels=validation_label_batch)

train_op = model.training(train_losses_op, learning_rate=LEARNING_RATE)

train_accuracy_op = model.evaluation(logits=train_logits_op, labels=train_label_batch)

validation_accuracy_op = model.evaluation(logits=validation_logits_op, labels=validation_label_batch)

summary_op = tf.summary.merge_all()

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph, max_queue=3)
    val_writer = tf.summary.FileWriter(logs_validation_dir, sess.graph, max_queue=3)
    Saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            _, train_loss, train_accuracy = sess.run([train_op, train_losses_op, train_accuracy_op])
            if step % 100 == 0:
                print('Step %d, train loss = %.2f, train accuracy = %.2f' % (step, train_loss, train_accuracy * 100.0))
                summery_str = sess.run(summary_op)
                train_writer.add_summary(summery_str, step)
            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                Saver.save(sess, checkpoint_path, global_step=step)
            if step % 500 == 0 or (step + 1) == MAX_STEP:
                val_loss, val_accuracy = sess.run([validation_losses_op, validation_accuracy_op])
                print('** step %d, val loss = %.2f, val accuracy = %.2f' % (step, val_loss, val_accuracy * 100.0))
                summery_str = sess.run(summary_op)
                val_writer.add_summary(summery_str, step)

    except tf.errors.OutOfRangeError:
        print("Done training -- epoch limit reached")
    finally:
        coord.request_stop()
