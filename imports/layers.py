import tensorflow as tf


def conv(x, name, filters, kernel_size=3, strides=1, dilation=1):
    with tf.variable_scope(name):
        x = tf.layers.conv2d(x, filters=filters, kernel_size=kernel_size, strides=strides,
                             dilation_rate=dilation, padding="same")
    return x


def instance_norm(x, name, epsilon=1e-5):
    with tf.variable_scope(name):
        gamma = tf.get_variable(initializer=tf.ones([x.shape[-1]]), name="gamma")
        beta = tf.get_variable(initializer=tf.zeros([x.shape[-1]]), name="beta")
        mean, var = tf.nn.moments(x, axes=[1,2], keep_dims=True)
        x = tf.nn.batch_normalization(x, mean, var, beta, gamma, epsilon, name="norm",)
    return x
