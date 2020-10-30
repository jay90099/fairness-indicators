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
"""Tests for fairness_indicators.examples.util_tfv2."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from fairness_indicators.examples import util_tfv2
import tensorflow.compat.v2 as tf


class UtilTFV2Test(tf.test.TestCase):
  """Tests for fairness_indicators.examples.util_tfv2."""

  def test_eval_input_receiver_fn(self):
    input_receiver = util_tfv2.eval_input_receiver_fn()
    with self.subTest('input_receiver.features'):
      self.assertSameElements(
          input_receiver.features, {
              'gender', 'sexual_orientation', 'religion', 'race', 'disability',
              'comment_text', 'toxicity', 'weight'
          })
    with self.subTest('input_receiver.labels'):
      self.assertEqual(input_receiver.labels.shape.as_list(), [None])
      self.assertEqual(input_receiver.labels.dtype, tf.float32)


if __name__ == '__main__':
  tf.test.main()
