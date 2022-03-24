from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os


def count_img_emotion(dataDir):
    img_counter = dict()
    for folder in os.listdir(dataDir):
        path = os.path.join(dataDir, folder)
        classname = os.path.basename(path)
        img_counter[classname] = len(os.listdir(path))
    return img_counter


def load_data(path, labels):
    x = []
    y = []
    for label in labels:
        print("[INFO] Loading class " + label)
        for filename in os.listdir(path + label):
            if filename.split(".").pop().lower() == "jpg" or \
                    filename.split(".").pop().lower() == "png" or \
                    filename.split(".").pop().lower() == "jpeg":
                img = image.load_img(path + label + "/" + filename, target_size=(48, 48, 1), color_mode='grayscale')
                img = np.array(img)
                img = img / 255.0

                x.append(img)
                y.append(labels[label])

    return np.array(x), y


def plot_data(total_train, total_test):
    # Histogram:
    f = plt.figure(figsize=(14, 5))
    ax1 = f.add_subplot(121)
    ax2 = f.add_subplot(122)

    # Plot training dataset
    names_train = list(total_train.keys())
    values_train = list(total_train.values())
    ax1.bar(range(len(total_train)), values_train, tick_label=names_train)
    ax1.set_title("Training dataset", fontsize=14)

    # Plot testing dataset
    names_test = list(total_test.keys())
    values_test = list(total_test.values())
    ax2.bar(range(len(total_test)), values_test, tick_label=names_test)
    ax2.set_title("Testing dataset", fontsize=14)
    plt.show()
