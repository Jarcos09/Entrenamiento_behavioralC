import numpy as np

# Compatibilidad con NumPy 2.0: np.sctypes fue eliminado
if not hasattr(np, 'sctypes'):
    np.sctypes = {
        'int': [np.int8, np.int16, np.int32, np.int64],
        'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
        'float': [np.float16, np.float32, np.float64],
        'complex': [np.complex64, np.complex128],
        'others': [bool, object, bytes, str, np.void]
    }

import cv2
import random
import os
from imgaug import augmenters as iaa


def load_image(path):
    # cv2 devuelve uint8 BGR; convertimos a RGB para imgaug y matplotlib
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def zoom(image):
    aug = iaa.Affine(scale=(1, 1.3))
    return aug.augment_image(image)


def pan(image):
    aug = iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
    return aug.augment_image(image)


def img_random_flip(image, steering_angle):
    image = cv2.flip(image, 1)
    steering_angle = -steering_angle
    return image, steering_angle


def img_random_brightness(image):
    aug = iaa.Multiply((0.2, 1.2))
    return aug.augment_image(image)


def random_augment(image_path, steering_angle):
    image = load_image(image_path)
    if np.random.rand() < 0.5:
        image = pan(image)
    if np.random.rand() < 0.5:
        image = zoom(image)
    if np.random.rand() < 0.5:
        image = img_random_brightness(image)
    if np.random.rand() < 0.5:
        image, steering_angle = img_random_flip(image, steering_angle)
    return image, steering_angle


def img_preprocess(img):
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = img / 127.5 - 1.0
    return img


def batch_generator(image_paths, steering_ang, batch_size, istraining):
    while True:
        batch_img      = []
        batch_steering = []
        for _ in range(batch_size):
            idx = random.randint(0, len(image_paths) - 1)
            if istraining:
                im, steering = random_augment(image_paths[idx], steering_ang[idx])
            else:
                im       = load_image(image_paths[idx])
                steering = steering_ang[idx]
            batch_img.append(img_preprocess(im))
            batch_steering.append(steering)
        yield np.asarray(batch_img), np.asarray(batch_steering)


def load_img_steering(df):
    image_paths = df['image_path'].str.strip().to_numpy()
    steerings   = df['steering'].astype(float).to_numpy()
    return image_paths, steerings
