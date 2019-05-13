import numpy as np
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues,
                          target_names=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    #cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    classes = target_names
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    
    plt.show()
    
    return ax


np.set_printoptions(precision=2)

#temp = np.array([[36, 32, 11, 4, 2],[105, 558, 324, 62, 8],[22, 388, 1061, 436, 23],[3, 48, 359, 1045, 160],[1, 4, 30, 135, 148]])
#temp = np.array([[1,28,18,0,0],[35,276,221,36,0],[17,311,901,317,7],[0,59,708,1052,179],[0,3,99,448,290]])

temp = np.array([[16,31,   6,   0,   0],[86, 394, 175,  22,   0],[59, 498, 971, 398,  21],[
6, 106, 599, 968, 174],[0,   1,  35, 294, 146]])

temp = temp.T

print(temp)

plot_confusion_matrix(temp, title="Confusion Matrix", target_names =['1','2','3','4','5'], normalize=True)
