import tensorflow as tf

from imports.layers import conv, instance_norm


def segmentation_block(x):
    x_shape = tf.shape(x)
    out_w, out_h = x_shape[1], x_shape[2]
    with tf.variable_scope("segmentation_block", reuse=tf.AUTO_REUSE):
        conv1 = conv(x, name="conv1", filters=13, strides=2)
        pool1 = tf.layers.max_pooling2d(x, pool_size=2, strides=2)
        conv1_concat = tf.concat([conv1, pool1], axis=3)
        conv2 = tf.nn.relu(conv(conv1_concat, name="conv2", filters=16, dilation=2))
        conv2_concat = tf.concat([conv1_concat, conv2], axis=3)
        conv3 = tf.nn.relu(conv(conv2_concat, name="conv3", filters=16, dilation=4))
        conv3_concat = tf.concat([conv2_concat, conv3], axis=3)
        conv4 = tf.nn.relu(conv(conv3_concat, name="conv4", filters=16, dilation=6))
        conv4_concat = tf.concat([conv3_concat, conv4], axis=3)
        conv5 = tf.nn.relu(conv(conv4_concat, name="conv5", filters=16, dilation=8))
        conv5_concat = tf.concat([conv2, conv3, conv4, conv5], axis=3)
        conv6 = tf.nn.relu(conv(conv5_concat, name="conv6", filters=2))
        pred = tf.image.resize_images(conv6, size=[out_w, out_h])
    return pred


def feathering_block(x, coarse_mask):
    with tf.variable_scope("feathering_block", reuse=tf.AUTO_REUSE):
        foreground, background = tf.split(coarse_mask, axis=3, num_or_size_splits=2)
        x_square = tf.square(x)
        x_masked = x * tf.tile(foreground, multiples=(1,1,1,3))

        x = tf.concat([x, coarse_mask, x_square, x_masked], axis=3)

        conv1 = tf.nn.relu(instance_norm(conv(x, name="conv1", filters=32), name="norm1"))
        conv4 = conv(conv1, name="conv4", filters=3)

        a, b, c = tf.split(conv4, axis=3, num_or_size_splits=3)

        output = a * foreground + b * background + c
    output = tf.nn.sigmoid(output)
    return output
