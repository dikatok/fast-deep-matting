import tensorflow as tf

from imports.layers import conv, instance_norm


def segmentation_block(x):
    x_shape = tf.shape(x)
    out_w, out_h = x_shape[1], x_shape[2]
    with tf.variable_scope("segmentation_block", reuse=tf.AUTO_REUSE):
        conv1 = conv(x, name="conv1", filters=13, strides=2)
        pool1 = tf.layers.max_pooling2d(x, pool_size=2, strides=2)
        conv1_concat = tf.concat([conv1, pool1], axis=3)
        conv2 = conv(conv1_concat, name="conv2", filters=16, dilation=2)
        conv2_concat = tf.concat([conv1_concat, conv2], axis=3)
        conv3 = conv(conv2_concat, name="conv3", filters=16, dilation=4)
        conv3_concat = tf.concat([conv2_concat, conv3], axis=3)
        conv4 = conv(conv3_concat, name="conv4", filters=16, dilation=6)
        conv4_concat = tf.concat([conv3_concat, conv4], axis=3)
        conv5 = conv(conv4_concat, name="conv5", filters=16, dilation=8)
        conv5_concat = tf.concat([conv2, conv3, conv4, conv5], axis=3)
        conv6 = conv(conv5_concat, name="conv6", filters=2)
        pred = tf.image.resize_images(conv6, size=[out_w, out_h])
    return pred


def feathering_block(x, coarse_mask):
    with tf.variable_scope("feathering_block", reuse=tf.AUTO_REUSE):
        x_square = tf.square(x)
        mask_soft = tf.nn.softmax(coarse_mask, dim=3)
        mask_soft_split = tf.split(mask_soft, axis=3, num_or_size_splits=2)
        mask_tiled = tf.concat([tf.tile(mask_soft_split[0], multiples=(1, 1, 1, 3)),
                                tf.tile(mask_soft_split[1], multiples=(1, 1, 1, 3))], axis=3)
        x_masked = tf.tile(x, multiples=(1, 1, 1, 2)) * mask_tiled
        x = tf.concat([x, coarse_mask, x_square, x_masked], axis=3)
        conv1 = tf.nn.relu(instance_norm(conv(x, name="conv1", filters=10), name="norm1"))
        conv2 = conv(conv1, name="conv2", filters=3)
        filter_split = tf.split(conv2, axis=3, num_or_size_splits=3)
        mask_split = tf.split(coarse_mask, axis=3, num_or_size_splits=2)
        output = tf.sigmoid(filter_split[0] * mask_split[0] + filter_split[1] * mask_split[1] + filter_split[2])
    return output
