#!/usr/bin/env python3

import model
from keras.preprocessing.image import ImageDataGenerator

def get_log_limited(interval=1):
    last = 0
    def log(s):
        nonlocal last
        now = time.time()
        if now - last < interval:
            return
        print(s)
        last = now
    return log

def main(args):
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

    if args.load_model:
        print("loading weights from {}".format(args.load_model))
        net.load_weights(args.load_model)

    net.save_weights("models/start.hdf5")

    print("training starts")
    log = get_log_limited(interval=1)
    clock = [0, 0]
    while True:
        clock[1] += 1
        if clock[1] > 500:
            checkpoint = 'models/latest'.format(clock[0])
            if clock[0] % 5 == 0:
                checkpoint = 'models/epoch-{}.h5'.format(clock[0])
                print("checkpoint: {}".format(checkpoint))
                net.save_weights(checkpoint)
                clock[1] = 0
                clock[0] += 1
                epoch, mb = clock
                if epoch > 10:
                    break
                    Xtrain, Ytrain = next(train_generator)
                    loss, accuracy = net.train_on_batch(Xtrain, Ytrain)
                    log("clock: {}:{}, loss: {:.2f}, accuracy: {:.2f}".format(*clock, loss, accuracy))
                    if mb % 100 == 0:
                        Xtest, Ytest = next(validation_generator)
                        loss, accuracy = net.evaluate(Xtest, Ytest, verbose=False)
                        print("TEST: clock: {}:{}, loss: {:.2f}, accuracy: {:.2f}".format(*clock, loss, accuracy))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', "--load-model", default=None)
    args = parser.parse_args()
    main(args)
