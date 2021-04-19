import os

import tensorflow as tf

from src.core.train import Trainer
from src.models.elg import ELGBuilder


class LDMKSTrainer(Trainer):
    def __init__(self,
                 model,
                 epochs=100,
                 global_batch_size=32,
                 strategy=tf.distribute.MirroredStrategy(),
                 initial_learning_rate=0.0001,
                 version='0.0.1',
                 start_epoch=1,
                 tensorboard_dir='./logs/elg_ldmks'):
        super().__init__(
                epochs=epochs,
                global_batch_size=global_batch_size,
                strategy=strategy,
                initial_learning_rate=initial_learning_rate,
                version=version,
                start_epoch=start_epoch,
                tensorboard_dir=tensorboard_dir)
        self.model = model
        self.model_name = 'elg_ldmks'

    def train_step(self, inputs):
        input_tensor = inputs['eye']
        label = inputs['landmarks']

        with tf.GradientTape() as tape:
            predict = self.model(input_tensor)
            loss = self.compute_loss(label, predict)
        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_weights)
        )

        return loss

    def val_step(self, inputs):
        input_tensor = inputs['eye']
        label = inputs['landmarks']

        predict = self.model(input_tensor)
        loss = self.compute_loss(label, predict)

        return loss


class RadiusTrainer(Trainer):
    def __init__(self,
                 model,
                 epochs=100,
                 global_batch_size=32,
                 strategy=tf.distribute.MirroredStrategy(),
                 initial_learning_rate=0.0001,
                 version='0.0.1',
                 start_epoch=1,
                 tensorboard_dir='./logs/elg_radius'):
        super().__init__(
            epochs=epochs,
            global_batch_size=global_batch_size,
            strategy=strategy,
            initial_learning_rate=initial_learning_rate,
            version=version,
            start_epoch=start_epoch,
            tensorboard_dir=tensorboard_dir)
        self.model = model
        self.model_name = 'elg_radius'

    def train_step(self, inputs):
        input_tensor = inputs['landmarks']
        label = inputs['radius']

        with tf.GradientTape() as tape:
            predict = self.model(input_tensor)
            loss = self.compute_loss(label, predict)
        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_weights)
        )

        return loss

    def val_step(self, inputs):
        input_tensor = inputs['landmarks']
        label = inputs['radius']

        predict = self.model(input_tensor)
        loss = self.compute_loss(label, predict)

        return loss


class GazeTrainer(Trainer):
    def __init__(self,
                 model,
                 epochs=100,
                 global_batch_size=32,
                 strategy=tf.distribute.MirroredStrategy(),
                 initial_learning_rate=0.0001,
                 version='0.0.1',
                 start_epoch=1,
                 tensorboard_dir='./logs/elg_gaze'):
        super().__init__(
            epochs=epochs,
            global_batch_size=global_batch_size,
            strategy=strategy,
            initial_learning_rate=initial_learning_rate,
            version=version,
            start_epoch=start_epoch,
            tensorboard_dir=tensorboard_dir)
        self.model = model
        self.model_name = 'elg_gaze'

    def train_step(self, inputs):
        landmarks = inputs['landmarks']
        radius = inputs['radius']
        label = inputs['gaze']

        # Concatenate landmarks and radius as input
        landmarks = tf.transpose(landmarks, perm=[0, 2, 1])
        landmarks = tf.keras.layers.Flatten()(landmarks)
        radius = tf.expand_dims(radius, axis=-1)
        input_tensor = tf.keras.layers.Concatenate()([landmarks, radius])

        with tf.GradientTape() as tape:
            predict = self.model(input_tensor)
            loss = self.compute_loss(label, predict)
        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_weights)
        )

        return loss

    def val_step(self, inputs):
        landmarks = inputs['landmarks']
        radius = inputs['radius']
        label = inputs['gaze']

        # Concatenate landmarks and radius as input
        landmarks = tf.transpose(landmarks, perm=[0, 2, 1])
        landmarks = tf.keras.layers.Flatten()(landmarks)
        radius = tf.expand_dims(radius, axis=-1)
        input_tensor = tf.keras.layers.Concatenate()([landmarks, radius])

        predict = self.model(input_tensor)
        loss = self.compute_loss(label, predict)

        return loss


def parse_tfexample(example_proto):
    image_feature_description = {
        'radius': tf.io.FixedLenFeature([], tf.float32),
        'gaze': tf.io.FixedLenFeature([], tf.string),
        'landmarks': tf.io.FixedLenFeature([], tf.string),
        'eye': tf.io.FixedLenFeature([], tf.string),
    }
    return tf.io.parse_single_example(example_proto,
                                      image_feature_description)


def parse_array_element(features):
    """Parse array byte_string to tensor"""
    features['gaze'] = tf.io.parse_tensor(features['gaze'], tf.float32)
    features['landmarks'] = tf.io.parse_tensor(features['landmarks'], tf.float32)
    features['eye'] = tf.io.parse_tensor(features['eye'], tf.float32)
    return features


def create_dataset(tfrecords, batch_size, is_train):
    dataset = tf.data.Dataset.list_files(tfrecords)
    dataset = tf.data.TFRecordDataset(dataset)
    dataset = dataset.map(parse_tfexample, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(parse_array_element, num_parallel_calls=tf.data.AUTOTUNE)

    if is_train:
        dataset = dataset.shuffle(batch_size)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


def train(model_builder: ELGBuilder, model_name: str, epochs, start_epoch, learning_rate, tensorboard_dir,
          batch_size, train_tfrecords, val_tfrecords, version, checkpoint=None):

    strategy = tf.distribute.MirroredStrategy()
    global_batch_size = strategy.num_replicas_in_sync * batch_size

    train_dataset = create_dataset(
        train_tfrecords, global_batch_size, is_train=True)
    val_dataset = create_dataset(
        val_tfrecords, global_batch_size, is_train=False)

    if not os.path.exists(os.path.join('./models')):
        os.makedirs(os.path.join('./models'))

    with strategy.scope():
        if model_name == 'elg_ldmks':
            _, model, _, _ = model_builder.build_model()
            if checkpoint and os.path.exists(checkpoint):
                model.load_weights(checkpoint)
            trainer = LDMKSTrainer(
                model=model,
                epochs=epochs,
                global_batch_size=global_batch_size,
                strategy=strategy,
                initial_learning_rate=learning_rate,
                start_epoch=start_epoch,
                version=version,
                tensorboard_dir=tensorboard_dir)
        elif model_name == 'elg_radius':
            _, _, model, _ = model_builder.build_model()
            if checkpoint and os.path.exists(checkpoint):
                model.load_weights(checkpoint)
            trainer = RadiusTrainer(
                model=model,
                epochs=epochs,
                global_batch_size=global_batch_size,
                strategy=strategy,
                initial_learning_rate=learning_rate,
                start_epoch=start_epoch,
                version=version,
                tensorboard_dir=tensorboard_dir)
        elif model_name == 'elg_gaze':
            _, _, _, model = model_builder.build_model()
            if checkpoint and os.path.exists(checkpoint):
                model.load_weights(checkpoint)
            trainer = GazeTrainer(
                model=model,
                epochs=epochs,
                global_batch_size=global_batch_size,
                strategy=strategy,
                initial_learning_rate=learning_rate,
                start_epoch=start_epoch,
                version=version,
                tensorboard_dir=tensorboard_dir)

        train_dist_dataset = strategy.experimental_distribute_dataset(
            train_dataset)
        val_dist_dataset = strategy.experimental_distribute_dataset(
            val_dataset)

        print('Start training...')
        return trainer.run(train_dist_dataset, val_dist_dataset)