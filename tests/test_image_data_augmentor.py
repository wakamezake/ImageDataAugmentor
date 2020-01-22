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


@pytest.fixture(scope='module')
def alb_compose():
    from albumentations import Compose
    from albumentations.augmentations.transforms import Flip, RandomRotate90, \
        ShiftScaleRotate, CoarseDropout
    return Compose([
        Flip(),
        RandomRotate90(),
        ShiftScaleRotate(),
        CoarseDropout()
    ])


def test_image_data_augmentor(all_test_images):
    for test_images in all_test_images:
        img_list = []
        for img in test_images:
            img_list.append(img_to_array(img)[None, ...])

        ImageDataAugmentor(featurewise_center=True,
                           samplewise_center=True,
                           featurewise_std_normalization=True,
                           samplewise_std_normalization=True,
                           zca_whitening=True)


def test_image_data_augmentor_with_albumentations(all_test_images,
                                                  alb_compose):
    for test_images in all_test_images:
        img_list = []
        for img in test_images:
            img_list.append(img_to_array(img)[None, ...])

        ImageDataAugmentor(augment=alb_compose)


def test_image_data_augmentor_with_validation_split(all_test_images):
    for test_images in all_test_images:
        img_list = []
        for im in test_images:
            img_list.append(img_to_array(im)[None, ...])

        images = np.vstack(img_list)
        labels = np.concatenate([
            np.zeros((int(len(images) / 2),)),
            np.ones((int(len(images) / 2),))])
        generator = ImageDataAugmentor(validation_split=0.5)

        # training and validation sets would have different
        # number of classes, because labels are sorted
        with pytest.raises(ValueError,
                           match='Training and validation subsets '
                                 'have different number of classes after '
                                 'the split.*'):
            generator.flow(images, labels,
                           shuffle=False, batch_size=10,
                           subset='validation')

        labels = np.concatenate([
            np.zeros((int(len(images) / 4),)),
            np.ones((int(len(images) / 4),)),
            np.zeros((int(len(images) / 4),)),
            np.ones((int(len(images) / 4),))
        ])

        seq = generator.flow(images, labels,
                             shuffle=False, batch_size=10,
                             subset='validation')

        x, y = seq[0]
        assert 2 == len(np.unique(y))

        seq = generator.flow(images, labels,
                             shuffle=False, batch_size=10,
                             subset='training')
        x2, y2 = seq[0]
        assert 2 == len(np.unique(y2))

        with pytest.raises(ValueError):
            generator.flow(images, np.arange(images.shape[0]),
                           shuffle=False, batch_size=3,
                           subset='foo')


def test_image_data_augmentor_with_split_value_error():
    with pytest.raises(ValueError):
        ImageDataAugmentor(validation_split=5)


def test_image_data_augmentor_invalid_data():
    generator = ImageDataAugmentor(
        featurewise_center=True,
        samplewise_center=True,
        featurewise_std_normalization=True,
        samplewise_std_normalization=True,
        zca_whitening=True,
        data_format='channels_last'
    )
    # Test fit with invalid data
    with pytest.raises(ValueError):
        x = np.random.random((3, 10, 10))
        generator.fit(x)

    # Test flow with invalid data
    with pytest.raises(ValueError):
        x = np.random.random((32, 10, 10))
        generator.flow(np.arange(x.shape[0]))


def test_image_data_augmentor_flow(all_test_images, tmpdir):
    for test_images in all_test_images:
        img_list = []
        for im in test_images:
            img_list.append(img_to_array(im)[None, ...])

        images = np.vstack(img_list)
        dsize = images.shape[0]
        generator = ImageDataAugmentor(
            featurewise_center=True,
            samplewise_center=True,
            featurewise_std_normalization=True,
            samplewise_std_normalization=True,
            zca_whitening=True
        )

        generator.flow(
            images,
            np.arange(images.shape[0]),
            shuffle=False,
            save_to_dir=str(tmpdir),
            batch_size=3
        )

        generator.flow(
            images,
            np.arange(images.shape[0]),
            shuffle=False,
            sample_weight=np.arange(images.shape[0]) + 1,
            save_to_dir=str(tmpdir),
            batch_size=3
        )

        # Test with `shuffle=True`
        generator.flow(
            images, np.arange(images.shape[0]),
            shuffle=True,
            save_to_dir=str(tmpdir),
            batch_size=3,
            seed=42
        )

        # Test without y
        generator.flow(
            images,
            None,
            shuffle=True,
            save_to_dir=str(tmpdir),
            batch_size=3
        )

        # Test with a single miscellaneous input data array
        x_misc1 = np.random.random(dsize)
        generator.flow(
            (images, x_misc1),
            np.arange(dsize),
            shuffle=False,
            batch_size=2
        )

        # Test with two miscellaneous inputs
        x_misc2 = np.random.random((dsize, 3, 3))
        generator.flow(
            (images, [x_misc1, x_misc2]),
            np.arange(dsize),
            shuffle=False,
            batch_size=2
        )

        # Test cases with `y = None`
        generator.flow(images, None, batch_size=3)
        generator.flow((images, x_misc1), None, batch_size=3, shuffle=False)
        generator.flow(
            (images, [x_misc1, x_misc2]),
            None,
            batch_size=3,
            shuffle=False
        )
        generator = ImageDataAugmentor(validation_split=0.2)
        generator.flow(images, batch_size=3)

        # Test some failure cases:
        x_misc_err = np.random.random((dsize + 1, 3, 3))
        with pytest.raises(ValueError) as e_info:
            generator.flow((images, x_misc_err), np.arange(dsize),
                           batch_size=3)
        assert str(e_info.value).find('All of the arrays in') != -1

        with pytest.raises(ValueError) as e_info:
            generator.flow((images, x_misc1), np.arange(dsize + 1),
                           batch_size=3)
        assert str(e_info.value).find(
            '`x` (images tensor) and `y` (labels) ') != -1

        # Test `flow` behavior as Sequence
        generator.flow(
            images,
            np.arange(images.shape[0]),
            shuffle=False,
            save_to_dir=str(tmpdir),
            batch_size=3
        )

        # Test with `shuffle=True`
        generator.flow(
            images,
            np.arange(images.shape[0]),
            shuffle=True, save_to_dir=str(tmpdir),
            batch_size=3, seed=123
        )

    # test order_interpolation
    labels = np.array([[2, 2, 0, 2, 2],
                       [1, 3, 2, 3, 1],
                       [2, 1, 0, 1, 2],
                       [3, 1, 0, 2, 0],
                       [3, 1, 3, 2, 1]])

    label_generator = ImageDataAugmentor(
    )
    label_generator.flow(
        x=labels[np.newaxis, ..., np.newaxis],
        seed=123
    )
