import tensorflow as tf


def iou(real_B, fake_B):
    real_B_ones = tf.greater_equal(real_B, 0.5)
    fake_B_ones = tf.greater_equal(fake_B, 0.5)
    i = tf.cast(tf.logical_and(real_B_ones, fake_B_ones), dtype=tf.float32)
    u = tf.cast(tf.logical_or(real_B_ones, fake_B_ones), dtype=tf.float32)
    iou = tf.reduce_mean(tf.reduce_sum(i, axis=[1, 2, 3]) / tf.reduce_sum(u, axis=[1, 2, 3]))
    return iou
