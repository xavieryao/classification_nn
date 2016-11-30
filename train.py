#!/usr/bin/env python3

import model
from keras.preprocessing.image import ImageDataGenerator

def main():
    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1./255)

    # this is a generator that will read pictures found in
    # subfolers of 'data/train', and indefinitely generate
    # batches of augmented image data
    train_generator = train_datagen.flow_from_directory(
            'data/train',  # this is the target directory
            target_size=(150, 150),  # all images will be resized to 150x150
            batch_size=16,
            class_mode='categorical')

    # this is a similar generator, for validation data
    validation_generator = test_datagen.flow_from_directory(
            'data/validation',
            target_size=(150, 150),
            batch_size=16,
            class_mode='categorical')

    net = model.get_model()
    net.fit_generator(
        train_generator,
        samples_per_epoch=512,
        nb_epoch=30,
        validation_data=validation_generator,
        nb_val_samples=117)
    net.save_weights('first_try.h5')


if __name__ == '__main__':
    main()
