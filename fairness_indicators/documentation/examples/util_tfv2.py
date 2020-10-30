# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Util methods for the example colabs."""

import tensorflow.compat.v2 as tf
import tensorflow_model_analysis as tfma

TEXT_FEATURE = 'comment_text'
LABEL = 'toxicity'

SEXUAL_ORIENTATION_COLUMNS = [
    'heterosexual', 'homosexual_gay_or_lesbian', 'bisexual',
    'other_sexual_orientation'
]
GENDER_COLUMNS = ['male', 'female', 'transgender', 'other_gender']
RELIGION_COLUMNS = [
    'christian', 'jewish', 'muslim', 'hindu', 'buddhist', 'atheist',
    'other_religion'
]
RACE_COLUMNS = ['black', 'white', 'asian', 'latino', 'other_race_or_ethnicity']
DISABILITY_COLUMNS = [
    'physical_disability', 'intellectual_or_learning_disability',
    'psychiatric_or_mental_illness', 'other_disability'
]

IDENTITY_COLUMNS = {
    'gender': GENDER_COLUMNS,
    'sexual_orientation': SEXUAL_ORIENTATION_COLUMNS,
    'religion': RELIGION_COLUMNS,
    'race': RACE_COLUMNS,
    'disability': DISABILITY_COLUMNS
}
FEATURE_MAP = {
    # Label:
    LABEL: tf.io.FixedLenFeature([], tf.float32),
    # Text:
    TEXT_FEATURE: tf.io.FixedLenFeature([], tf.string),

    # Identities:
    'sexual_orientation': tf.io.VarLenFeature(tf.string),
    'gender': tf.io.VarLenFeature(tf.string),
    'religion': tf.io.VarLenFeature(tf.string),
    'race': tf.io.VarLenFeature(tf.string),
    'disability': tf.io.VarLenFeature(tf.string),
}

_THRESHOLD = 0.5


def eval_input_receiver_fn() -> tfma.export.EvalInputReceiver:
  """EvalInputReceiver for Fairness_Indicators_Example_Colab."""
  serialized_tf_example = tf.compat.v1.placeholder(
      dtype=tf.string, shape=[None], name='input_example_placeholder')

  # This *must* be a dictionary containing a single key 'examples', which
  # points to the input placeholder.
  receiver_tensors = {'examples': serialized_tf_example}

  features = tf.io.parse_example(serialized_tf_example, FEATURE_MAP)
  features['weight'] = tf.ones_like(features[LABEL])

  return tfma.export.EvalInputReceiver(
      features=features,
      receiver_tensors=receiver_tensors,
      labels=features[LABEL])
