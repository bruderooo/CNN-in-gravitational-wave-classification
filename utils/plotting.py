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
