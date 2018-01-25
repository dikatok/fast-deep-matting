import tensorflow as tf


def _extract_features(example):
    features = {
        "image": tf.FixedLenFeature((), tf.string),
        "mask": tf.FixedLenFeature((), tf.string)
    }
    parsed_example = tf.parse_single_example(example, features)
    images = tf.cast(tf.image.decode_jpeg(parsed_example["image"]), dtype=tf.float32)
    images.set_shape([800, 600, 3])
    masks = tf.cast(tf.image.decode_jpeg(parsed_example["mask"]), dtype=tf.float32) / 255.
    masks.set_shape([800, 600, 1])
    return images, masks


def create_one_shot_iterator(filenames, batch_size, num_epoch):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_extract_features)
    dataset = dataset.shuffle(buffer_size=batch_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epoch)
    return dataset.make_one_shot_iterator()


def create_initializable_iterator(filenames, batch_size):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_extract_features)
    dataset = dataset.shuffle(buffer_size=batch_size)
    dataset = dataset.batch(batch_size)
    return dataset.make_initializable_iterator()


def augment_dataset(images, masks, size=None, augment=True):
    if augment:
        cond_flip_lr = tf.cast(tf.random_uniform([], maxval=2, dtype=tf.int32), tf.bool)

        def flip(images, masks):
            return tf.map_fn(tf.image.flip_left_right, images), tf.map_fn(tf.image.flip_left_right, masks)

        images, masks = tf.cond(cond_flip_lr, lambda: flip(images, masks), lambda: (images, masks))

    if size is not None:
        images = tf.image.resize_images(images, 1 * size)
        masks = tf.image.resize_images(masks, 1 * size)

    return images, masks
