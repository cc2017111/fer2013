import tensorflow as tf


def inference(images, batch_size, n_classes, regularizer, reuse):
    # input 48x48x1
    # output 48x48x32
    with tf.variable_scope('conv1', reuse=reuse) as scope:
        conv1_weights = tf.get_variable("weights", shape=[3, 3, 1, 16], dtype=tf.float32,
                                        initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))

        conv1_biases = tf.get_variable("biases", shape=[16], dtype=tf.float32,
                                       initializer=tf.constant_initializer(0.1))

        conv1 = tf.nn.conv2d(images, conv1_weights, strides=[1, 1, 1, 1], padding="SAME")
        pre_activation = tf.nn.bias_add(conv1, conv1_biases)
        activation = tf.nn.relu(pre_activation, name=scope.name)

    # input 48x48x1
    # output 24x24x16
    with tf.variable_scope('pool1') as scope:
        pool1 = tf.nn.max_pool(activation, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name=scope.name)

    # input 24x24x32
    # output 24x24x64
    with tf.variable_scope('conv2', reuse=reuse) as scope:
        conv2_weights = tf.get_variable("weights", shape=[3, 3, 16, 32], dtype=tf.float32,
                                        initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))

        conv2_biases = tf.get_variable("biases", shape=[32], dtype=tf.float32,
                                       initializer=tf.constant_initializer(0.1))

        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding="SAME", name=scope.name)
        pre_activation = tf.nn.bias_add(conv2, conv2_biases)
        activation = tf.nn.relu(pre_activation)
    # input 24x24x16
    # output 12x12x32
    with tf.variable_scope('pool2') as scope:
        pool2 = tf.nn.max_pool(activation, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name=scope.name)

    with tf.variable_scope('fc1', reuse=reuse) as scope:
        reshaped = tf.reshape(pool2, shape=[batch_size, -1])
        dim = reshaped.get_shape()[1].value
        fc1_weights = tf.get_variable("weights", shape=[dim, 2048], dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        if regularizer is not None:
            tf.add_to_collection("losses", regularizer(fc1_weights))

        fc1_biases = tf.get_variable("biases", shape=[2048], dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))

        fc1 = tf.matmul(reshaped, fc1_weights) + fc1_biases
        activation = tf.nn.relu(fc1, name=scope.name)
        if not reuse:
            activation = tf.nn.dropout(activation, keep_prob=0.5)

    with tf.variable_scope('fc2', reuse=reuse) as scope:
        fc2_weights = tf.get_variable("weights", shape=[2048, 512], dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        if regularizer is not None:
            tf.add_to_collection("losses", regularizer(fc2_weights))

        fc2_biases = tf.get_variable("biases", shape=[512], dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.0))

        fc2 = tf.matmul(activation, fc2_weights) + fc2_biases
        activation = tf.nn.relu(fc2, name=scope.name)
        if not reuse:
            activation = tf.nn.dropout(activation, keep_prob=0.5)

    with tf.variable_scope('softmax', reuse=reuse) as scope:
        softmax_weights = tf.get_variable("weights", shape=[512, n_classes], dtype=tf.float32,
                                          initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))

        softmax_biases = tf.get_variable("biases", shape=[n_classes], dtype=tf.float32,
                                         initializer=tf.constant_initializer(0.1))

        softmax_linear = tf.add(tf.matmul(activation, softmax_weights), softmax_biases, name=scope.name)

    return softmax_linear


def losses(logits, labels):
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels,
                                                                       name='entropy_per_example')

        cross_entropy_mean = tf.reduce_mean(cross_entropy, name=scope.name)
        loss = tf.add_n(tf.get_collection("losses")) + cross_entropy_mean
        tf.summary.scalar(scope.name + '/loss', cross_entropy_mean)
    return loss


def training(loss, learning_rate):
    with tf.variable_scope('optimizer') as scope:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, trainable=False, name='global_step')
        train_op = optimizer.minimize(loss, global_step=global_step, name=scope.name)
    return train_op


def evaluation(logits, labels):
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name + '/accuracy', accuracy)
    return accuracy
