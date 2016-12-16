#!/usr/bin/env python3
import model
import numpy as np
from keras.preprocessing.image import img_to_array, load_img

net = model.get_model()
net.load_weights('models/latest')

class_map = ['dalitang', 'erxiaomen', 'xuetang', 'zhulou']

def classify(filename):
    img = load_img(filename, target_size=(150, 150))
    arr = img_to_array(img)
    x = np.zeros((1, 3, 150, 150))
    x[0] = arr
    r = net.predict(x, batch_size=1)
    return np.argmax(r)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', "--picture")
    args = parser.parse_args()
    clz = classify(args.picture)
    print("Belongs to {}".format(class_map[clz]))
