import tensorflow as tf


def loss_fun(images, gt_masks, alpha_mattes, epsilon=1e-6):
    la = tf.reduce_sum(tf.sqrt(tf.square(gt_masks - alpha_mattes) + epsilon))
    lcolor = tf.reduce_sum(tf.sqrt(tf.square(tf.tile(gt_masks, multiples=(1,1,1,3)) * images
                                             - tf.tile(alpha_mattes, multiples=(1,1,1,3)) * images) + epsilon))
    return la + lcolor
