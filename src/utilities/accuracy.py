import numpy


class AccuracyTracker(object):
    """Tracks accuracy metrics for classification tasks.

    This class calculates and tracks various accuracy metrics for classification tasks using a confusion matrix.
    It supports overall accuracy, mean accuracy, frequency-weighted accuracy, and class-wise Dice scores.

    Attributes:
        n_classes (int): Number of classes in the classification task.
        confusion_matrix (ndarray): Confusion matrix to track predictions.
    """

    def __init__(self, n_classes):
        """Initializes the AccuracyTracker with a given number of classes."""
        self.n_classes = n_classes
        self.confusion_matrix = numpy.zeros((n_classes, n_classes))

    def reset(self):
        """Resets the confusion matrix to all zeros."""
        self.confusion_matrix = numpy.zeros((self.n_classes, self.n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        """Computes a histogram of the true and predicted labels for a single class.

        This private method is used internally to update the confusion matrix.

        Args:
            label_true (ndarray): Ground truth labels.
            label_pred (ndarray): Predicted labels.
            n_class (int): The class to compute the histogram for.

        Returns:
            ndarray: The updated histogram for the given class.
        """
        mask = (label_true >= 0) & (label_true < n_class)
        hist = numpy.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask],
            minlength=n_class ** 2,
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        """Updates the confusion matrix with new batches of true and predicted labels.

        Args:
            label_trues (list of ndarray): A list of ground truth label batches.
            label_preds (list of ndarray): A list of predicted label batches.
        """
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(
                lt.flatten(), lp.flatten(), self.n_classes
            )

    def _calculate_overall_accuracy(self):
        """Calculates the overall accuracy.

        Returns:
            float: The overall accuracy calculated from the confusion matrix.
        """
        hist = self.confusion_matrix
        return numpy.diag(hist).sum() / hist.sum()

    def _calculate_mean_accuracy(self):
        """Calculates the mean accuracy.

        Returns:
            float: The mean accuracy calculated for each class.
        """
        hist = self.confusion_matrix
        acc_cls = numpy.diag(hist) / hist.sum(axis=1)
        return numpy.nanmean(acc_cls)

    def _calculate_freq_weighted_accuracy(self):
        """Calculates the frequency-weighted accuracy.

        Returns:
            float: The frequency-weighted accuracy calculated for each class.
        """
        hist = self.confusion_matrix
        freq = hist.sum(axis=1) / hist.sum()
        dice = 2 * numpy.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) + numpy.finfo(numpy.float32).eps)
        fwavacc = (freq[freq > 0] * dice[freq > 0]).sum()
        return fwavacc

    def _calculate_mean_dice(self):
        """Calculates the mean Dice score.

        Returns:
            float: The mean Dice score calculated for each class.
        """
        hist = self.confusion_matrix
        with numpy.errstate(invalid='ignore'):
            dice = 2 * numpy.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) + numpy.finfo(numpy.float32).eps)
        return numpy.nanmean(dice)

    def _calculate_cls_dice(self):
        """Calculates the Dice score for each class.

        Returns:
            dict: A dictionary with classes as keys and their Dice scores as values.
        """
        hist = self.confusion_matrix
        with numpy.errstate(invalid='ignore'):
            dice = 2 * numpy.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) + numpy.finfo(numpy.float32).eps)
        return dict(zip(range(self.n_classes), dice))

    def get_accuracy(self):
        """Returns the overall accuracy."""
        return self._calculate_overall_accuracy()

    def get_mean_dice(self):
        """Returns the mean Dice score."""
        return self._calculate_mean_dice()

    def get_mean_accuracy(self):
        """Returns the mean accuracy."""
        return self._calculate_mean_accuracy()

    def get_freq_weighted_accuracy(self):
        """Returns the frequency weighted accuracy."""
        return self._calculate_freq_weighted_accuracy()

    def get_cls_dice(self):
        """Returns the class-wise dice scores."""
        return self._calculate_cls_dice()