# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Box Head.

Contains Box prediction head classes for different meta architectures.
All the box prediction heads have a _predict function that receives the
`features` as the first argument and returns `box_encodings`.
"""
import tensorflow.compat.v1 as tf

from object_detection.predictors.heads import head


class ConvolutionalEmbeddingHead(head.KerasHead):
  """Convolutional Embedding prediction head."""

  def __init__(self,
               is_training,
               embedding_size,
               kernel_size,
               num_predictions_per_location,
               conv_hyperparams,
               freeze_batchnorm,
               use_depthwise=False,
               name=None):
    """Constructor.
    Args:
      is_training: Indicates whether the BoxPredictor is in training mode.
      embedding_size: Size of encoding for each box.
      kernel_size: Size of final convolution kernel.  If the
        spatial resolution of the feature map is smaller than the kernel size,
        then the kernel size is automatically set to be
        min(feature_width, feature_height).
      use_depthwise: Whether to use depthwise convolutions for prediction
        steps. Default is False.
      box_encodings_clip_range: Min and max values for clipping box_encodings.
    Raises:
      ValueError: if min_depth > max_depth.
      ValueError: if use_depthwise is True and kernel_size is 1.
    """
    if use_depthwise and (kernel_size == 1):
      raise ValueError('Should not use 1x1 kernel when using depthwise conv')

    super(ConvolutionalEmbeddingHead, self).__init__(name=name)
    self._is_training = is_training
    self._embedding_size = embedding_size
    self._kernel_size = kernel_size
    self._use_depthwise = True
    self._embedding_predictor_layers = []

    if self._use_depthwise:
      self._embedding_predictor_layers.append(
          tf.keras.layers.DepthwiseConv2D(
              [self._kernel_size, self._kernel_size],
              padding='SAME',
              depth_multiplier=1,
              strides=1,
              dilation_rate=1,
              name='EmbeddingEncodingPredictor1_depthwise',
              **conv_hyperparams.params()))
      self._embedding_predictor_layers.append(
          conv_hyperparams.build_batch_norm(
              training=(is_training),
              name='EmbeddingEncodingPredictor1_depthwise_batchnorm'))
      self._embedding_predictor_layers.append(
          conv_hyperparams.build_activation_layer(
              name='EmbeddingEncodingPredictor1_depthwise_activation'))
      self._embedding_predictor_layers.append(tf.keras.layers.Dropout(0.3))
      self._embedding_predictor_layers.append(
          tf.keras.layers.Conv2D(
              num_predictions_per_location * self._embedding_size, [1, 1],
              name='EmbeddingEncodingPredictor',
              **conv_hyperparams.params(use_bias=True)))
      self._embedding_predictor_layers.append(
          tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)) # L2 normalize embeddings
      )
    else:
      self._embedding_predictor_layers.append(
          tf.keras.layers.Conv2D(
              num_predictions_per_location * self._embedding_size,
              [self._kernel_size, self._kernel_size],
              padding='SAME',
              name='EmbeddingEncodingPredictor',
              **conv_hyperparams.params(use_bias=True)))

  def _predict(self, features):
    """Predicts boxes.

    Args:
      features: A float tensor of shape [batch_size, height, width, channels]
        containing image features.

    Returns:
      box_encodings: A float tensor of shape
        [batch_size, num_anchors, q, code_size] representing the location of
        the objects, where q is 1 or the number of classes.
    """
    embedding_encodings = features
    for layer in self._embedding_predictor_layers:
      embedding_encodings = layer(embedding_encodings)
    batch_size = features.get_shape().as_list()[0]
    if batch_size is None:
      batch_size = tf.shape(features)[0]

    embedding_encodings = tf.reshape(embedding_encodings,
                               [batch_size, -1, 1, self._embedding_size])
    return embedding_encodings


class WeightSharedConvolutionalEmbeddingHead(head.KerasHead):
  """Weight shared convolutional box prediction head based on Keras.

  This head allows sharing the same set of parameters (weights) when called more
  then once on different feature maps.
  """

  def __init__(self,
               embedding_size,
               num_predictions_per_location,
               conv_hyperparams,
               kernel_size=3,
               use_depthwise=False,
               apply_conv_hyperparams_to_heads=False,
               box_encodings_clip_range=None,
               return_flat_predictions=True,
               name=None):
    """Constructor.

    Args:
      box_code_size: Size of encoding for each box.
      num_predictions_per_location: Number of box predictions to be made per
        spatial location. Int specifying number of boxes per location.
      conv_hyperparams: A `hyperparams_builder.KerasLayerHyperparams` object
        containing hyperparameters for convolution ops.
      kernel_size: Size of final convolution kernel.
      use_depthwise: Whether to use depthwise convolutions for prediction steps.
        Default is False.
      apply_conv_hyperparams_to_heads: Whether to apply conv_hyperparams to
        depthwise seperable convolution layers in the box and class heads. By
        default, the conv_hyperparams are only applied to layers in the
        predictor tower when using depthwise separable convolutions.
      box_encodings_clip_range: Min and max values for clipping box_encodings.
      return_flat_predictions: If true, returns flattened prediction tensor
        of shape [batch, height * width * num_predictions_per_location,
        box_coder]. Otherwise returns the prediction tensor before reshaping,
        whose shape is [batch, height, width, num_predictions_per_location *
        num_class_slots].
      name: A string name scope to assign to the model. If `None`, Keras
        will auto-generate one from the class name.

    Raises:
      ValueError: if use_depthwise is True and kernel_size is 1.
    """
    if use_depthwise and (kernel_size == 1):
      raise ValueError('Should not use 1x1 kernel when using depthwise conv')

    super(WeightSharedConvolutionalEmbeddingHead, self).__init__(name=name)
    self._embedding_size = embedding_size
    self._kernel_size = kernel_size
    self._num_predictions_per_location = num_predictions_per_location
    self._use_depthwise = use_depthwise
    self._apply_conv_hyperparams_to_heads = apply_conv_hyperparams_to_heads
    self._box_encodings_clip_range = box_encodings_clip_range
    self._return_flat_predictions = return_flat_predictions

    self._embedding_predictor_layers = []
    if self._use_depthwise:
      kwargs = conv_hyperparams.params(use_bias=True)
      if self._apply_conv_hyperparams_to_heads:
        kwargs['depthwise_regularizer'] = kwargs['kernel_regularizer']
        kwargs['depthwise_initializer'] = kwargs['kernel_initializer']
        kwargs['pointwise_regularizer'] = kwargs['kernel_regularizer']
        kwargs['pointwise_initializer'] = kwargs['kernel_initializer']
      self._embedding_predictor_layers.append(
          tf.keras.layers.SeparableConv2D(
              num_predictions_per_location * self._embedding_size,
              [self._kernel_size, self._kernel_size],
              padding='SAME',
              name='EmbeddingBoxPredictor',
              **kwargs))
      self._embedding_predictor_layers.append(
          tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)) # L2 normalize embeddings
      )
    else:
      self._embedding_predictor_layers.append(
          tf.keras.layers.Conv2D(
              num_predictions_per_location * self._embedding_size,
              [self._kernel_size, self._kernel_size],
              padding='SAME',
              name='EmbeddingEncodingPredictor',
              **conv_hyperparams.params(use_bias=True)))
      self._embedding_predictor_layers.append(
          tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)) # L2 normalize embeddings
      )

  def _predict(self, features):
    """Predicts boxes.

    Args:
      features: A float tensor of shape [batch_size, height, width, channels]
        containing image features.

    Returns:
      box_encodings: A float tensor of shape
        [batch_size, num_anchors, q, code_size] representing the location of
        the objects, where q is 1 or the number of classes.
    """
    embedding_encodings = features
    for layer in self._embedding_predictor_layers:
      embedding_encodings = layer(embedding_encodings)
    batch_size = features.get_shape().as_list()[0]
    if batch_size is None:
      batch_size = tf.shape(features)[0]
    if self._return_flat_predictions:
      embedding_encodings = tf.reshape(embedding_encodings,
                                 [batch_size, -1, self._embedding_size])
    return embedding_encodings
