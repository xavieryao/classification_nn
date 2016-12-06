from common import config
from image import ImageDataGenerator

def get_dataset_generator(type='train'):
    if type=='train':
        # this is the augmentation configuration we will use for training
        train_datagen = ImageDataGenerator(
                rescale=1./255,
                shear_range=0.2,
                zoom_range=0.2,
                advanced_augment=True,
                horizontal_flip=True)
        # this is a generator that will read pictures found in
        # subfolers of 'data/train', and indefinitely generate
        # batches of augmented image data
        train_generator = train_datagen.flow_from_directory(
                'data/train',  # this is the target directory
                target_size=config.img_size[:2],  # all images will be resized to 150x150
                batch_size=config.batch_size,
                class_mode='categorical')

        return train_generator

    else :
        # this is the augmentation configuration we will use for testing:
        # only rescaling
        test_datagen = ImageDataGenerator(rescale=1./255)

        # this is a similar generator, for validation data
        validation_generator = test_datagen.flow_from_directory(
                'data/validation',
                target_size=config.img_size[:2],
                batch_size=config.batch_size,
                class_mode='categorical')
