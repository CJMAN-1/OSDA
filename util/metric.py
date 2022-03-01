import numpy as np

def conf_mat(GT, Pred, num_class=20):
    """
    GT : BHW
    Pred : BCHW
    -> CC
    """
    confusion_matrix = np.zeros((num_class,) * 2)
    mask = (GT >= 0) & (GT < num_class)
    label = num_class * GT[mask].astype('int') + Pred[mask]
    count = np.bincount(label, minlength=num_class ** 2)
    confusion_matrix += count.reshape(num_class, num_class)
    return confusion_matrix

def iou(conf_matrix):
    iou = 100 * np.diag(conf_matrix)/ (np.sum(conf_matrix, axis=0) + np.sum(conf_matrix, axis=1) - np.diag(conf_matrix))
    if np.isnan(iou[-1]):
        iou[-1] = 0

    return iou