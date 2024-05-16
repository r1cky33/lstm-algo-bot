from keras.callbacks import Callback
from sklearn.metrics import recall_score, accuracy_score
import numpy as np

class CustomEarlyStoppingForOnes(Callback):
    def __init__(self, validation_data, conf_threshold, patience=0):
        super().__init__()
        self.validation_data = validation_data
        self.patience = patience
        self.best_weights = None
        self.best_accuracy = 0
        self.conf_threshold = conf_threshold
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        X_val, y_val = self.validation_data
        y_pred = self.model.predict(X_val)
        y_pred_classes = (y_pred > np.percentile(y_pred, self.conf_threshold)).astype(int)

        # Filter out only the instances where true class is 1
        true_ones = y_val == 1
        predicted_ones = y_pred_classes[true_ones]

        # Calculate accuracy only for class 1
        accuracy_for_ones = accuracy_score(y_val[true_ones], predicted_ones)

        # Check if the current accuracy is better than the best observed so far
        if accuracy_for_ones > self.best_accuracy:
            self.best_accuracy = accuracy_for_ones
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
                self.model.set_weights(self.best_weights)
                print(f"Restoring model weights from the end of the best epoch. Epoch: {epoch + 1}")
                print(f"Early stopping triggered after {epoch + 1} epochs.")

    def on_train_end(self, logs=None):
        print(f"Best '1' class accuracy: {self.best_accuracy:.4f}")
