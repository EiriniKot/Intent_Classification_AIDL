import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

def plot_metric_graph(history, metric=''):
    """
    Pass a tensorflow history object and the metric you want to plot during
    epochs.
    """
    # summarize history for loss
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+ metric])
    plt.title('model '+metric)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.show()


def plot_label_count(data, target_col = ''):
    """
    Plot for labels counting. Good tool for understand label distribution
    :param data: pd.Dataframe Data dataframe
    :param target_col: str . Column name of dataframe that contains the labels.
    """
    ax = sns.countplot(x=target_col, data = data, palette = 'Set3')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    plt.tight_layout()
    plt.show()


def plot_results(results, confusion, label_names, plot_title='', save=False):
    """
    Two sided Plot for  results.
    :param results: dict Dictionary with metric names and values of them
    :param confusion: confusion matrix from sklearn
    :param label_names: list , Label names
    :param plot_title: str Name of the title of image
    :param save : bool Save png or not
    """
    num = len(results)
    percentages = list(results.values())
    percentages = np.array(percentages) * 100
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=label_names)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8))
    disp.plot(ax=ax1)
    ax1.set_title('Confusion Matrix')
    ax1.set_xticklabels(label_names, rotation=45)

    # make color of the bar will be dependent on the value of the percentage.
    colours = plt.cm.Reds(np.linspace(0, 1, num))

    # Add the percentage bars to ax2
    ax2.bar(range(num), percentages, color=colours)

    for i, value in enumerate(percentages):
        ax2.text(i, round(value, 2), f"{round(value, 2)}%", ha="center", fontsize=16, va="bottom")

    ax2.set_title("Metrics Plot")
    ax2.axhline(100, color='gray', ls='--', lw=1)
    ax2.set_xticks(range(num))
    ax2.set_xticklabels(results.keys(), rotation=45)
    ax2.set_ylim(10, 110)
    plt.suptitle(plot_title, fontsize=15)

    if save:
        plt.savefig(plot_title + '.png')
    plt.show()
    print('')
    return plt