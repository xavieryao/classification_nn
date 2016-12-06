#!/usr/bin/env python3

import model
import dataset

def main():


    net = model.get_model()
    net.fit_generator(
        dataset.get_dataset_generator('train'),
        samples_per_epoch=512,
        nb_epoch=30,
        validation_data=dataset.get_dataset_generator('validation'),
        nb_val_samples=117)
    net.save_weights('first_try.h5')


if __name__ == '__main__':
    main()
