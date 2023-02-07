import tensorflow as tf
import numpy as np
import sklearn
from transformers import TrainerCallback
from copy import deepcopy


def scheduler(epoch, lr=1e-3):
    """
    Scheduler that reduces exponentialy after 100 epoch
    """
    # Scheduler reduces lr after the first 5 epochs
    if epoch < 100:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


def model_sequential(dict_layers={},
                     loss=tf.keras.losses.categorical_crossentropy,
                     optimizer=tf.keras.optimizers.Adam(1e-4),
                     metrics=[tf.keras.metrics.CategoricalAccuracy(name='cat_accuracy'),
                              tf.keras.metrics.Precision(name='prec')]):

    layers=list(map(lambda i: getattr(tf.keras.layers, i[0])(**i[1]), dict_layers.items()))
    model = tf.keras.Sequential(layers)
    model.compile(loss=loss,
                optimizer=optimizer,
                metrics=metrics)
    return model


def compute_metrics(eval_pred, metrics = [("accuracy_score", {}), ("recall_score", {'average': 'macro'}),\
                                          ("f1_score", {'average': 'macro'}),\
                                          ("precision_score", {'average': 'macro'}), \
                                          ("confusion_matrix", {})]):
    """
    This function uses sklearn to calc all the metrics that has been passed into metrics argument.
    Metrics should be a list with tuples. First index of the tuple should be the name of the metric
    and the second a dictionary of the kwargs for this metric
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    dict_eval = dict(map(lambda i: (i[0].split('-')[0], getattr(sklearn.metrics, i[0])(y_pred=predictions, y_true=labels, **i[1])), metrics))
    return dict_eval


class CustomCallback(TrainerCallback):
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy

def sigopt_hp_space(trial):
    return [
        {"bounds": {"min": 1e-6, "max": 1e-4}, "name": "learning_rate", "type": "double"},
        {"categorical_values": ["30", "50", "70", "80"], "name": "num_train_epochs", "type": "categorical"},
        {"categorical_values": ["16", "32", "64", "128", "256"], "name": "per_device_train_batch_size", "type": "categorical"}
        ]