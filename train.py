import tensorflow as tf
from tensorflow.python.lib.io import file_io

from imports.data_utils import create_one_shot_iterator, augment_dataset, create_initializable_iterator
from imports.losses import loss_fun
from imports.models import segmentation_block, feathering_block
import os
import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_files', nargs='+', required=False, default="train-00001-of-00001")
    parser.add_argument('--test_files', nargs='+', required=False, default="val-00001-of-00001")
    parser.add_argument('--log_dir', default='./logs', type=str)
    parser.add_argument('--ckpt_dir', default='./ckpts', type=str)
    parser.add_argument('--train_batch_size', default=256, type=int)
    parser.add_argument('--test_batch_size', default=256, type=int)
    parser.add_argument('--resume', default=None, type=int)
    return parser.parse_args()


if __name__ == '__main__':
    args = args_parser()

    num_iter_first_state = 20000
    num_iter_second_stage = 20000
    num_iter_third_stage = 20000

    lr_first_1 = 1e-3
    lr_first_2 = 1e-4
    lr_second = 1e-6
    lr_third = 1e-7

    train_files = args.train_files
    test_files = args.test_files
    
    train_batch_size = args.train_batch_size
    test_batch_size = args.test_batch_size

    num_train_samples = sum(1 for f in file_io.get_matching_files(train_files) 
                            for n in tf.python_io.tf_record_iterator(f))
    num_test_samples = sum(1 for f in file_io.get_matching_files(test_files) 
                           for n in tf.python_io.tf_record_iterator(f))
    
    num_epochs = int((num_iter_first_state + num_iter_second_stage + num_iter_third_stage) 
                     // (num_train_samples / train_batch_size))

    train_iterator = create_one_shot_iterator(train_files, train_batch_size, num_epoch=num_epochs)
    test_iterator = create_initializable_iterator(test_files, batch_size=num_test_samples)

    next_images, next_masks = train_iterator.get_next()
    next_images, next_masks = augment_dataset(next_images, next_masks, size=[128, 128])
    coarse_masks = segmentation_block(next_images)
    alpha_mattes = feathering_block(next_images, coarse_masks)
    loss = loss_fun(next_images, next_masks, alpha_mattes)

    test_images, test_masks = test_iterator.get_next()
    test_images, test_masks = augment_dataset(test_images, test_masks, size=[128, 128], augment=False)
    test_coarse_masks = segmentation_block(test_images)
    test_alpha_mattes = feathering_block(test_images, test_coarse_masks)
    test_loss = loss_fun(test_images, test_masks, test_alpha_mattes)

    segmentation_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "segmentation_block/")
    feathering_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "feathering_block/")
    all_trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    train_op_first_stage_1 = tf.train.AdamOptimizer(learning_rate=lr_first_1).minimize(loss, var_list=segmentation_vars)
    train_op_first_stage_2 = tf.train.AdamOptimizer(learning_rate=lr_first_2).minimize(loss, var_list=segmentation_vars)
    train_op_second_stage = tf.train.AdamOptimizer(learning_rate=lr_second).minimize(loss, var_list=feathering_vars)
    train_op_third_stage = tf.train.AdamOptimizer(learning_rate=lr_third).minimize(loss, var_list=all_trainable_vars)

    summary = tf.summary.FileWriter(logdir=args.log_dir)
    image_summary = tf.summary.image("image", next_images)
    gt_summary = tf.summary.image("gt", next_masks * 255.)
    coarse_summary = tf.summary.image("coarse", tf.cast(tf.expand_dims(tf.argmax(coarse_masks, axis=3), axis=3),
                                                        dtype=tf.float32))
    result_summary = tf.summary.image("result", alpha_mattes * 255.)
    images_summary = tf.summary.merge([image_summary, gt_summary, coarse_summary, result_summary])

    loss_summary = tf.summary.scalar("loss", loss)
    test_loss_summary = tf.summary.scalar("test_loss", test_loss)

    saver = tf.train.Saver(var_list=tf.trainable_variables())

    resume = args.resume

    def get_session(sess):
        session = sess
        while type(session).__name__ != 'Session':
            session = session._sess
        return session


    with tf.train.MonitoredTrainingSession() as sess:
        it = 0
        if resume is not None and resume > 0:
            saver.restore(sess, os.path.join(args.ckpt_dir, "ckpt") + "-{it}".format(it=resume))
            it = resume + 1

        while not sess.should_stop():
            train_op = train_op_first_stage_1

            if it > 40000:
                train_op = train_op_third_stage
            elif it > 20000:
                train_op = train_op_second_stage
            elif it > 10000:
                train_op = train_op_first_stage_2

            _, cur_loss, cur_images_summary, cur_loss_summary = sess.run([train_op, loss, images_summary, loss_summary])
            summary.add_summary(cur_loss_summary, it)

            if it % 100 == 0:
                summary.add_summary(cur_images_summary, it)

            if it % 1000 == 0:
                sess.run(test_iterator.initializer)
                cur_test_loss_summary = sess.run(test_loss_summary)
                summary.add_summary(cur_test_loss_summary, it)

            summary.flush()

            if it % 5000 == 0:
                ckpt_path = saver.save(get_session(sess), save_path=os.path.join(args.ckpt_dir, "ckpt"),
                                       write_meta_graph=False, global_step=it)
                print("Checkpoint saved as: {ckpt_path}".format(ckpt_path=ckpt_path))

            it += 1

        sess.run(test_iterator.initializer)
        cur_test_loss_summary = sess.run(test_loss_summary)
        summary.add_summary(cur_test_loss_summary, it)

        ckpt_path = saver.save(get_session(sess), save_path=os.path.join(args.ckpt_dir, "ckpt"), write_meta_graph=False,
                               global_step=it)
        print("Checkpoint saved as: {ckpt_path}".format(ckpt_path=ckpt_path))
