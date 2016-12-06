#!/usr/bin/env python3
import cv2
import numpy as np
from deform import GaussianDeform
from common import config

rng = np.random.RandomState()

def rand_range(low, high):
    return low + rng.rand() * (high - low)

def rand_int(low, high):
    return int(rand_range(low, high))


h, w = config.img_size[:2]

gaussian_deformer = GaussianDeform(
    [(0, 0), (0, 1), (1, 1), (1, 0)], (h, w), .5, randrange=20, rng=rng
)

def augment_img(img):
    img = img.copy().astype(np.float32)

    h, w = img.shape[:2]
    row_ones = np.ones((1, w))
    bs = rand_int(0, bw-200)

    # got time range
    ts, te = 0, w
    for t in range(w):
        if img[:, t].sum() > 10:
            ts = t
            break
    for t in range(w):
        if img[:, w-t-1].sum() > 10:
            te = w-t-1
            break

    # RankOneBrightness
    if rng.rand() < .5:
        black = (rng.rand(h, 1) < .2)
        black = black.dot(row_ones).astype(np.bool)
        img[black] = 0

    # euqalizer
    if rng.rand() < .5:
        equalizer = np.sin(np.linspace(
            rand_range(-np.pi, np.pi), rand_range(0, 2*np.pi), h, )) + 1
        equalizer = equalizer.reshape((h, 1)).dot(row_ones) * rand_range(0, .5) + rand_range(.5, 1)
        img *= equalizer

    # Gaussian distortion
    if rng.rand() < .2:
        img = gaussian_deformer.augment(img[:, :, np.newaxis])
        img = img.max(axis=2)

    # Contrast
    if rng.rand() < .5:
        r = rand_range(.7, 5)
        mean = np.mean(img, axis=(0, 1), keepdims=True)
        img = (img-mean) * r + mean

    # Brightness
    if rng.rand() < .5:
        v = rand_range(-16, 64)
        img[img>2] += v

    # Gaussian noise
    if rng.rand() < .5:
        noise = rng.randn(*img.shape) * 10
        img = img + noise

    # Salt and Pepper
    if rng.rand() < .5:
        noise = rng.uniform(low=0, high=1, size=img.shape)
        img[noise > .999] = 255
        img[noise < .01] = 0

    sigma = rand_range(0, 2)
    img = cv2.GaussianBlur(img, (0, 0), sigma, sigma)

    return np.clip(img, 0, 255)
