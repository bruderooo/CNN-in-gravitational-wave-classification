import csv
from collections import defaultdict

import tensorflow as tf
from matplotlib import pyplot as plt


def plot_acc(history):
    plt.plot(history.history['binary_accuracy'])
    plt.plot(history.history['val_binary_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('acc.png')
    plt.clf()


def plot_auc(history):
    plt.plot(history.history['recall'], history.history['precision'])
    plt.plot(history.history['val_recall'], history.history['val_precision'])
    plt.ylabel('precision')
    plt.xlabel('recall')
    plt.legend(['train', 'val'], loc='upper right')
    plt.savefig('auc.png')
    plt.clf()


def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.savefig('loss.png')
    plt.clf()
    plt.clf()


def plotting_file(checkpoint_path):
    with open(f"{checkpoint_path}/log.csv", 'r') as csvfile:
        file = csv.reader(csvfile, delimiter=',')
        history = defaultdict(list)
        keys = next(file)

        for row in file:
            for key, el in zip(keys, row):
                history[key].append(float(el))

        return dict(history)


if __name__ == '__main__':
    history = tf.keras.callbacks.History()
    history.history = plotting_file('..')

    plot_acc(history)
    plot_loss(history)
    plot_auc(history)
