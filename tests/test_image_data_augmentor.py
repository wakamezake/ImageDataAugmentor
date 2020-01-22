"""
Reference:
https://github.com/keras-team/keras-preprocessing/blob/master/tests/image/image_data_generator_test.py
"""

import numpy as np
import pytest
from PIL import Image
from keras_preprocessing.image.utils import img_to_array

from ida.image_data_augmentor import ImageDataAugmentor


@pytest.fixture(scope='module')
def all_test_images():
    img_w = img_h = 20
    rgb_images = []
    rgba_images = []
    gray_images = []
    for n in range(8):
        bias = np.random.rand(img_w, img_h, 1) * 64
        variance = np.random.rand(img_w, img_h, 1) * (255 - 64)
        imarray = np.random.rand(img_w, img_h, 3) * variance + bias
        im = Image.fromarray(imarray.astype('uint8')).convert('RGB')
        rgb_images.append(im)

        imarray = np.random.rand(img_w, img_h, 4) * variance + bias
        im = Image.fromarray(imarray.astype('uint8')).convert('RGBA')
        rgba_images.append(im)

        imarray = np.random.rand(img_w, img_h, 1) * variance + bias
        im = Image.fromarray(
            imarray.astype('uint8').squeeze()).convert('L')
        gray_images.append(im)

    return [rgb_images, rgba_images, gray_images]


def test_image_data_augmentor(all_test_images):
    for test_images in all_test_images:
        img_list = []
        for img in test_images:
            img_list.append(img_to_array(img)[None, ...])

        ImageDataAugmentor(augment=None)


def test_image_data_generator_with_split_value_error():
    with pytest.raises(ValueError):
        ImageDataAugmentor(validation_split=5)
