import matplotlib as mpl
import matplotlib.pyplot as mpl
import numpy as np
from TFANN import ANNR

from DataGathering import get_currency_data


def main():
    hp = 16  # holdout period i.e. number of time units in period
    psn = 256  # number of past samples
    fsn = 16  # number of future samples
    sample_matrix, target_matrix, stacked_data_holdout, stacked_data, scale_vector = get_currency_data(
        past_samples_n=psn, future_samples_n=fsn,
        holdout_period=hp)

    filter_width = sample_matrix.shape[2]
    # 2 1-D conv layers with relu followed by 1-d conv output layer
    network_model = [('C1d', [8, filter_width, filter_width * 2], 4), ('AF', 'relu'),
                     ('C1d', [8, filter_width * 2, filter_width * 2], 2), ('AF', 'relu'),
                     ('C1d', [8, filter_width * 2, filter_width], 2)]

    # Create the neural network in TensorFlow
    cnnr = ANNR(sample_matrix[0].shape, network_model, batchSize=32, learnRate=2e-5,
                maxIter=64, reg=1e-5, tol=1e-2, verbose=True)
    cnnr.fit(sample_matrix, target_matrix)

    pts = []  # Predicted time sequences
    latest_data = sample_matrix[[-1]]  # Most recent time sequence
    for i in range(hp // fsn + 1):  # Repeat prediction
        prediction = cnnr.predict(latest_data)
        latest_data = np.concatenate([latest_data[:, fsn:], prediction], axis=1)
        pts.append(prediction)
    pts = np.hstack(pts).transpose((1, 0, 2))
    stacked_data_holdout = np.vstack([stacked_data_holdout, pts])  # Combine predictions with original data
    stacked_data_holdout = np.squeeze(stacked_data_holdout) * scale_vector  # Remove unittime dimension and rescale
    stacked_data = np.squeeze(stacked_data) * scale_vector

    plot_layers(cnnr, sample_matrix)


def plot_layers(cnnr, sample_matrix):
    nt = 4
    prediction = cnnr.PredictFull(sample_matrix[:nt])
    for i in range(nt):
        fig, ax = mpl.subplots(1, 4, figsize=(16 / 1.24, 10 / 1.25))
        ax[0].plot(prediction[0][i])
        ax[0].set_title('Input')
        ax[1].plot(prediction[2][i])
        ax[1].set_title('Layer 1')
        ax[2].plot(prediction[4][i])
        ax[2].set_title('Layer 2')
        ax[3].plot(prediction[5][i])
        ax[3].set_title('Output')
        fig.text(0.5, 0.06, 'Time', ha='center')
        fig.text(0.06, 0.5, 'Activation', va='center', rotation='vertical')
        mpl.show()


def plot_result(stacked_data, stacked_data_holdout, holdout_period, pts):
    ci = list(range(stacked_data.shape[0]))
    ai = list(range(stacked_data.shape[0] + pts.shape[0] - holdout_period))
    npd = pts.shape[0]  # Number of days predicted
    for i, cli in enumerate(cl):
        fig, ax = mpl.subplots(figsize=(16 / 1.5, 10 / 1.5))
        hind = i * len(CN) + CN.index('high')
        ax.plot(ci[-4 * holdout_period:], stacked_data[-4 * holdout_period:, hind], label='Actual')
        ax.plot(ai[-(npd + 1):], stacked_data_holdout[-(npd + 1):, hind], '--', label='Prediction')
        ax.legend(loc='upper left')
        ax.set_title(cli + ' (High)')
        ax.set_ylabel('USD')
        ax.set_xlabel('Time')
        ax.axes.xaxis.set_ticklabels([])
        mpl.show()


if __name__ == '__main__':
    main()
