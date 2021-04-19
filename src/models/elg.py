"""ELG architecture."""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Add,
    Conv2D,
    Input,
    ReLU,
    MaxPool2D,
    UpSampling2D,
    BatchNormalization,
    Softmax,
    Lambda,
    Flatten,
    Dense,
    Concatenate,
    Reshape,
)


class ELGBuilder(object):
    """ELG architecture as introduced in [Park et al. ETRA'18]."""

    def __init__(self, first_layer_stride=1, num_modules=3, num_feature_maps=32, input_shape=(36, 60), **kwargs):
        """Specify ELG-specific parameters."""
        self._hg_first_layer_stride = first_layer_stride
        self._hg_num_modules = num_modules
        self._hg_num_feature_maps = num_feature_maps
        self._input_shape = input_shape

        # Call parent class constructor
        super().__init__(**kwargs)

    _hg_first_layer_stride = 1
    _hg_num_modules = 3
    _hg_num_feature_maps = 32
    _hg_num_landmarks = 18
    _hg_num_residual_blocks = 1
    _kernel_initializer = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01)
    _kernel_regularizer = keras.regularizers.L2(1e-4)
    _bias_initializer = keras.initializers.Zeros()

    def build_model(self):
        """Build model."""
        # Preprocess before hourglass
        pre_inputs = Input(shape=self._input_shape+(1,), name='pre_inputs')
        n = self._hg_num_feature_maps
        x = Conv2D(
            filters=n,
            kernel_size=7,
            strides=self._hg_first_layer_stride,
            padding='same',
            kernel_initializer=self._kernel_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_initializer=self._bias_initializer
        )(pre_inputs)
        x = ReLU()(x)
        x = self.build_rb(x, n, 2*n)
        pre_outputs = self.build_rb(x, 2*n, n)
        pre_img = keras.Model(pre_inputs, pre_outputs, name='pre_img')

        # Hourglass blocks
        hg_inputs = Input(shape=self._input_shape+(32,), name='hg_inputs')
        x_prev = hg_inputs
        for i in range(self._hg_num_modules):
            x = self.build_hg(x_prev, steps_to_go=4, num_features=self._hg_num_feature_maps)
            # At last output do not merge.
            x, h = self.build_hg_after(
                x_prev, x, do_merge=(i < (self._hg_num_modules - 1)),
            )
            x_prev = x
        hg_outputs = h
        stacked_hg_heatmaps = keras.Model(hg_inputs, hg_outputs, name='stacked_hg_heatmaps')

        # Soft-argmax for landmarks calculation
        ldmks_layer = Lambda(self.cal_landmarks, name='ldmks_layer')

        # Fully-connected layers for radius regression
        fc_radius_inputs = Input(shape=(18, 2), name='fc_radius')
        x = tf.transpose(fc_radius_inputs, perm=[0, 2, 1])
        x = Flatten()(x)
        for i in range(3):
            x = Dense(
                units=100,
                kernel_initializer=self._kernel_initializer,
                kernel_regularizer=self._kernel_regularizer,
                bias_initializer=self._bias_initializer
            )(x)
            x = BatchNormalization(scale=True, center=True, trainable=True)(x)
            x = ReLU()(x)
        fc_radius_outputs = Dense(
            units=1,
            kernel_initializer=self._kernel_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_initializer=self._bias_initializer
        )(x)
        fc_radius = keras.Model(fc_radius_inputs, fc_radius_outputs, name='fc_radius')

        # # Fully-connected layers for gaze regression
        fc_gaze_inputs = Input(shape=(37,), name='fc_gaze_inputs_ldmks')
        x = Dense(
            units=50,
            kernel_initializer=self._kernel_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_initializer=self._bias_initializer
        )(fc_gaze_inputs)
        for i in range(3):
            x = Dense(
                units=100,
                kernel_initializer=self._kernel_initializer,
                kernel_regularizer=self._kernel_regularizer,
                bias_initializer=self._bias_initializer
            )(x)
            x = BatchNormalization(scale=True, center=True, trainable=True)(x)
            x = ReLU()(x)
        fc_gaze_outputs = Dense(
            units=2,
            kernel_initializer=self._kernel_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_initializer=self._bias_initializer
        )(x)
        fc_gaze = keras.Model(fc_gaze_inputs, fc_gaze_outputs, name='fc_radius')

        elg_inputs = Input(shape=self._input_shape+(1,), name='elg_inputs')
        pre_input = pre_img(elg_inputs)
        heatmaps = stacked_hg_heatmaps(pre_input)
        ldmks = ldmks_layer(heatmaps)
        elg_heatmaps = keras.Model(elg_inputs, heatmaps, name='elg_heatmaps')
        elg_ldmks = keras.Model(elg_inputs, ldmks, name='elg_ldmks')

        # Return model
        return elg_heatmaps, elg_ldmks, fc_radius, fc_gaze

    def build_rb(self, x, num_in, num_out):
        half_num_out = max(int(num_out/2), 1)

        # Lower branch
        c = x
        # Conv1
        c = BatchNormalization(scale=True, center=True, trainable=True)(c)
        c = ReLU()(c)
        c = Conv2D(
            filters=half_num_out,
            kernel_size=1,
            strides=1,
            padding='same',
            kernel_initializer=self._kernel_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_initializer=self._bias_initializer
        )(c)
    
        # Conv2
        c = BatchNormalization(scale=True, center=True, trainable=True)(c)
        c = ReLU()(c)
        c = Conv2D(
            filters=half_num_out,
            kernel_size=3,
            strides=1,
            padding='same',
            kernel_initializer=self._kernel_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_initializer=self._bias_initializer
        )(c)
    
        # Conv3
        c = BatchNormalization(scale=True, center=True, trainable=True)(c)
        c = ReLU()(c)
        c = Conv2D(
            filters=num_out,
            kernel_size=1,
            strides=1,
            padding='same',
            kernel_initializer=self._kernel_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_initializer=self._bias_initializer
        )(c)
    
        # Upper branch
        s = x
        # Skip
        if num_in == num_out:
            s = tf.identity(s)
        else:
            s = Conv2D(
                filters=num_out,
                kernel_size=1,
                strides=1,
                padding='same',
                kernel_initializer=self._kernel_initializer,
                kernel_regularizer=self._kernel_regularizer,
                bias_initializer=self._bias_initializer
            )(s)
        x = Add()([c, s])
        return x

    def build_hg(self, x, steps_to_go, num_features):
        # Upper branch
        up1 = x
        for i in range(self._hg_num_residual_blocks):
            up1 = self.build_rb(up1, num_features, num_features)
    
        # Lower branch
        low1 = MaxPool2D(pool_size=2)(x)
        for i in range(self._hg_num_residual_blocks):
            low1 = self.build_rb(low1, num_features, num_features)
    
        # Recursive
        if steps_to_go > 1:
            low2 = self.build_hg(low1, steps_to_go=steps_to_go-1, num_features=num_features)
        else:
            low2 = low1
            for i in range(self._hg_num_residual_blocks):
                low2 = self.build_rb(low2, num_features, num_features)
    
        # Additional rb
        low3 = low2
        for i in range(self._hg_num_residual_blocks):
            low3 = self.build_rb(low3, num_features, num_features)
    
        # Upsampling
        # up2 = UpSampling2D(size=2, interpolation='bilinear')(low3)
        up2 = tf.image.resize(images=low3, size=up1.shape[1:3])
    
        x = Add()([up1, up2])
        return x

    def build_hg_after(self, x_prev, x_now, do_merge=True):
        # After
        for i in range(self._hg_num_residual_blocks):
            x_now = self.build_rb(x_now, self._hg_num_feature_maps, self._hg_num_feature_maps)
    
        # A linear layer to predict each channel like a fc layer
        # Create the last feature_maps for heatmaps' creation
        x_now = Conv2D(
            filters=self._hg_num_feature_maps,
            kernel_size=1,
            strides=1,
            padding='same',
            kernel_initializer=self._kernel_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_initializer=self._bias_initializer
        )(x_now)
        x_now = BatchNormalization(scale=True, center=True, trainable=True)(x_now)
        x_now = ReLU()(x_now)
    
        # Heatmaps, the num of heatmaps is also num of landmarks
        h = Conv2D(
            filters=self._hg_num_landmarks,
            kernel_size=1,
            strides=1,
            padding='same',
            kernel_initializer=self._kernel_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_initializer=self._bias_initializer
        )(x_now)
    
        # Save feature_maps for next stack of hg
        x_next = x_now
    
        # Merge heatmaps and feature_maps
        # First, do conv for heatmaps and feature_maps, then merge them
        if do_merge:
            h_merge_1 = Conv2D(
                filters=self._hg_num_feature_maps,
                kernel_size=1,
                strides=1,
                padding='same',
                kernel_initializer=self._kernel_initializer,
                kernel_regularizer=self._kernel_regularizer,
                bias_initializer=self._bias_initializer
            )(h)
            h_merge_2 = Conv2D(
                filters=self._hg_num_feature_maps,
                kernel_size=1,
                strides=1,
                padding='same',
                kernel_initializer=self._kernel_initializer,
                kernel_regularizer=self._kernel_regularizer,
                bias_initializer=self._bias_initializer
            )(x_now)
    
            # Save the information of feature_maps and merged heat_maps
            h_merged = Add()([h_merge_1, h_merge_2])
            x_next = Add()([x_prev, h_merged])
        return x_next, h

    """
    After cal_landmarks() we can get positions of different landmarks(in different channels).
    Each channel has one pair of coordinates.
    
    Without variables, can be wrapped by Lambda layer.
    """
    def cal_landmarks(self, x):
        _, h, w, _ = x.shape.as_list()

        # Assume normalized coordinate [0, 1] for numeric stability
        ref_xs, ref_ys = np.meshgrid(
            np.linspace(0, 1.0, num=w, endpoint=True),
            np.linspace(0, 1.0, num=h, endpoint=True),
            indexing='xy'
        )

        ref_xs = np.reshape(ref_xs, [-1, h*w])
        ref_ys = np.reshape(ref_ys, [-1, h*w])

        # Assuming NHWC
        beta = 1e2
        # Transpose x from NHWC to NCHW
        x = tf.transpose(x, (0, 3, 1, 2))
        x = tf.reshape(x, [-1, self._hg_num_landmarks, h*w])
        x = Softmax(axis=-1)(beta*x)
        lmrk_xs = tf.math.reduce_sum(ref_xs * x, axis=[2])
        lmrk_ys = tf.math.reduce_sum(ref_ys * x, axis=[2])

        # Return to actual coordinates ranges
        return tf.stack([
            lmrk_xs * (w - 1.0) + 0.5,
            lmrk_ys * (h - 1.0) + 0.5
        ], axis=2) # N x 18 x 2

# class ELG(keras.Model):
#     """ELG Model"""
#     def __init__(self, elg_builder: ELGBuilder):
#         super(ELG, self).__init__()
#         _, self.elg_ldmks, self.elg_radius = elg_builder.build_model()
#
#     def compile(self, global_optimizer, loss_fn):
#         super(ELG, self).compile()
#         self.global_optimizer = global_optimizer
#         self.loss_fn = loss_fn
#
#     def train_step(self, inputs):
#         eyes, label_ldmks, label_radius = inputs
#         with tf.GradientTape() as tape:
#             predict_ldmks = self.elg_ldmks(eyes)
#             loss_ldmks = self.loss_fn(label_ldmks, predict_ldmks)
#         grads_ldmks = tape.gradient(loss_ldmks, self.elg_ldmks.trainable_weights)
#         with tf.GradientTape() as tape:
#             predict_radius = self.elg_radius(eyes)
#             loss_radius = self.loss_fn(label_radius, predict_radius)
#         grads_radius = tape.gradient(loss_radius, self.elg_radius.trainable_weights)
#         self.global_optimizer.apply_gradients(
#             zip(grads_ldmks, self.elg_ldmks.trainable_weights)
#         )
#         self.global_optimizer.apply_gradients(
#             zip(grads_radius, self.elg_radius.trainable_weights)
#         )
#         return {"loss_ldmks": loss_ldmks, "loss_radius": loss_radius}
