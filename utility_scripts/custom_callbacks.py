import numpy as np
import tensorflow as tf
from scipy.stats import spearmanr

def compute_spearmanr(trues, preds):
    rhos = []
    for col_trues, col_pred in zip(trues.T, preds.T):
        rhos.append(spearmanr(col_trues, col_pred).correlation)
    return np.nanmean(rhos)

class EarlyStopping(tf.keras.callbacks.Callback):

    def __init__(self,
               validation_data,
               batch_size=16,
               min_delta=0,
               patience=0,
               verbose=1,
               mode='auto',
               restore_best_weights=False):

        self.valid_inputs = validation_data[0]
        self.valid_targets = validation_data[1]
        self.batch_size = batch_size
        self.patience = patience
        self.verbose = verbose
        self.min_delta = abs(min_delta)
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None

        if mode not in ['auto', 'min', 'max']:
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value()
        if current is None:
            return
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights:
                    if self.verbose > 0:
                        print('Restoring model weights from the end of the best epoch.')
                    self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

    def get_monitor_value(self):
        valid_predictions = self.model.predict(self.valid_inputs, batch_size=self.batch_size)
        valid_rho = compute_spearmanr(self.valid_targets, valid_predictions)
        if self.verbose > 0:
            print(f" - valid_spearman_rho: {valid_rho}")
        return valid_rho
