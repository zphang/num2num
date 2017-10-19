import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns


def plot_attn_weights(attn_weights_mat, x, pred_y, dataset,
                      show=False, save_to=None):

    x_len = len(x[0])
    y_len = len(pred_y[0])

    if x_len > y_len:
        fig = plt.figure(figsize=(16, 16. * y_len / x_len))
    else:
        fig = plt.figure(figsize=(16. * x_len / y_len, 16))

    ax = fig.add_subplot(111)
    cax = ax.matshow(attn_weights_mat, cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels(
        [''] + list(map(dataset.input_lang.i2t.get, x[0])) + [''])
    ax.set_yticklabels(
        [''] + list(map(dataset.output_lang.i2t.get, pred_y[0])))

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.grid(False)
    if show:
        plt.show()

    if save_to:
        plt.savefig(save_to)
    plt.close()
