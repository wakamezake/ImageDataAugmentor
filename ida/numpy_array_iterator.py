"""Utilities for real-time data augmentation on image data.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
from keras_preprocessing.image.numpy_array_iterator import NumpyArrayIterator
from keras_preprocessing.image.utils import array_to_img


class CustomNumpyArrayIterator(NumpyArrayIterator):
    def _get_batches_of_transformed_samples(self, index_array):

        # build batch of image data
        batch_x = np.array([self.x[j] for j in index_array])

        # transform the image data
        batch_x = np.array(
            [self.image_data_generator.transform_image(x) for x in batch_x])

        if self.y is not None:
            batch_y = np.array([self.y[j] for j in index_array])

        else:
            batch_y = np.array([])

        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
                    index=j,
                    hash=np.random.randint(1e4),
                    format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        batch_x_miscs = [xx[index_array] for xx in self.x_misc]
        output = (batch_x if batch_x_miscs == []
                  else [batch_x] + batch_x_miscs,)
        if self.y is None:
            return output[0]

        output += (batch_y,)
        if self.sample_weight is not None:
            output += (self.sample_weight[index_array],)
        return output
