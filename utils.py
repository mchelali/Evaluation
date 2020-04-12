from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


def plot_confusion_matrix(cm, classes, savepath, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    title = 'Confusion matrix'

    # Compute confusion matrix
    #cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = range(cm.shape[0])

    print('Confusion matrix')

    print(cm)
    #plt.rcParams.update({'font.size': 12})
    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True

    # normalization de la matrice
    for i in range(cm.shape[0]):
        cm[i, :] = cm[i, :]/np.sum(cm[i, :])

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    #ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           #title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="center")#, rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    #fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, np.around(cm[i, j], 2),# format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    fig.savefig(path_res + "Poly" + str(polyID) + ".png", bbox_inches='tight', pad_inches=0)

if __name__ == "__main__":
    class_names = ["Meadow","vines", "Trad. \nOrchards", "Inten.\norchard"]

    cm = np.array([[91, 9, 0, 0],
               [1, 96, 6, 4],
               [1, 0, 26, 4],
               [1, 9, 6, 50]], dtype=float)

    

    colors = ["white", "yellow", "lightseagreen"]
    cmap1 = LinearSegmentedColormap.from_list("mycmap", colors)
    plot_confusion_matrix(cm, class_names, cmap=cmap1)
    plt.show()