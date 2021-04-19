from abc import abstractmethod
import math
import os

import tensorflow as tf


class Trainer(object):
    def __init__(self,
                 epochs=100,
                 global_batch_size=32,
                 strategy=tf.distribute.MirroredStrategy(),
                 initial_learning_rate=0.0001,
                 version='0.0.1',
                 start_epoch=1,
                 tensorboard_dir='./logs/'):
        self.model = None
        self.model_name = 'None'

        self.start_epoch = start_epoch
        self.epochs = epochs
        self.strategy = strategy
        self.global_batch_size = global_batch_size
        self.loss_object = tf.keras.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.NONE)
        # "we use Adam with a learning rate of 2.5e-4.""
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=initial_learning_rate)

        self.current_learning_rate = initial_learning_rate
        self.last_val_loss = math.inf
        self.lowest_val_loss = math.inf

        self.patience_count = 0
        self.max_patience = 10

        self.tensorboard_dir = tensorboard_dir
        self.best_model = None
        self.version = version

        self.is_train = False

    def lr_decay(self):
        """
        This effectively simulate ReduceOnPlateau learning rate schedule. Learning rate
        will be reduced by a factor of 5 if there's no improvement over [max_patience] epochs
        """
        if self.patience_count >= self.max_patience:
            self.current_learning_rate /= 10.0
            self.patience_count = 0
        elif self.last_val_loss == self.lowest_val_loss:
            self.patience_count = 0
        self.patience_count += 1

        self.optimizer.learning_rate = self.current_learning_rate

    def lr_decay_step(self, epoch):
        if epoch == 25 or epoch == 50 or epoch == 75:
            self.current_learning_rate /= 10.0
        self.optimizer.learning_rate = self.current_learning_rate

    def compute_loss(self, label, predict):
        loss = self.loss_object(label, predict)
        return tf.reduce_sum(loss) * (1. / self.global_batch_size)

    @abstractmethod
    def train_step(self, inputs):
        pass

    @abstractmethod
    def val_step(self, inputs):
        pass

    def run(self, train_dist_dataset, val_dist_dataset):
        @tf.function
        def distributed_train_epoch(dataset):
            tf.print('Start distributed training...')
            total_loss = 0.0
            num_train_batches = 0.0
            for one_batch in dataset:
                # tf.autograph.experimental.set_loop_options(
                #     shape_invariants=[(total_loss, tf.TensorShape([None]))]
                # )
                per_replica_loss = self.strategy.run(
                    self.train_step, args=(one_batch,))
                batch_loss = self.strategy.reduce(
                    tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)
                total_loss += batch_loss
                num_train_batches += 1
                if num_train_batches % 500 == 0:
                    tf.print('Trained batch', num_train_batches, 'batch loss',
                             batch_loss, 'epoch total loss', total_loss)
            return total_loss, num_train_batches

        @tf.function
        def distributed_val_epoch(dataset):
            total_loss = 0.0
            num_val_batches = 0.0
            for one_batch in dataset:
                # tf.autograph.experimental.set_loop_options(
                #     shape_invariants=[(total_loss, tf.TensorShape([None]))]
                # )
                per_replica_loss = self.strategy.run(
                    self.val_step, args=(one_batch,))
                batch_loss = self.strategy.reduce(
                    tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)
                total_loss += batch_loss
                num_val_batches += 1
                if num_val_batches % 500 == 0:
                    tf.print('Validation batch', num_val_batches, 'batch loss',
                             batch_loss, 'epoch total loss', total_loss)
            return total_loss, num_val_batches

        if not os.path.exists(os.path.join('./logs/{}'.format(self.model_name))):
            os.makedirs(os.path.join('./logs/{}'.format(self.model_name)))
        summary_writer = tf.summary.create_file_writer(self.tensorboard_dir)
        summary_writer.set_as_default()

        for epoch in range(self.start_epoch, self.epochs + 1):
            tf.summary.experimental.set_step(epoch)

            self.lr_decay()
            tf.summary.scalar('epoch learning rate',
                              self.current_learning_rate)

            print('Start epoch {} with learning rate {}'.format(
                epoch, self.current_learning_rate))

            train_total_loss, num_train_batches = distributed_train_epoch(
                train_dist_dataset)
            train_loss = train_total_loss / num_train_batches
            print('Epoch {} train loss {}'.format(epoch, train_loss))
            tf.summary.scalar('epoch train', train_loss)

            val_total_loss, num_val_batches = distributed_val_epoch(
                val_dist_dataset)
            val_loss = val_total_loss / num_val_batches
            print('Epoch {} val loss {}'.format(epoch, val_loss))
            tf.summary.scalar('epoch train', val_loss)

            # save model when reach a new lowest validation loss
            if val_loss < self.lowest_val_loss:
                self.save_model(epoch, val_loss)
                self.lowest_val_loss = val_loss
            self.last_val_loss = val_loss

        return self.best_model

    def save_model(self, epoch, loss):
        if not os.path.exists(os.path.join('./models/{}'.format(self.model_name))):
            os.makedirs(os.path.join('./models/{}'.format(self.model_name)))
        model_name = './models/{}/model-v{}-epoch-{}-loss-{:.4f}.h5'.format(
            self.model_name, self.version, epoch, loss)
        self.model.save_weights(model_name)
        self.best_model = model_name
        print("Model {} saved.".format(model_name))
