from keras.layers import Conv2D, LSTM, TimeDistributed, Conv1D, Dropout, Flatten, Dense, GRU
from keras.models import Sequential
from vis.visualization import visualize_cam, get_num_filters, visualize_activation, visualize_saliency
from vis.utils import utils

import matplotlib

matplotlib.use("Agg")  # only comment this out when you are in pycharm and want to do visualiziation debugging
import matplotlib.pyplot as plt

from numpy.fft import fft, fftfreq, ifft
from random import sample
import numpy as np
import random
import h5py
import pandas as pd
import os
from sklearn.metrics import mean_squared_error, r2_score

MODE_FREQUENCY = True
FREQUENCY_FS_RATE = 2048


def calculate_percentile(respType, percentile=75, p=-1):
    if p == -1:
        p = np.percentile(respType, percentile)
    target_raw = []
    for t in respType:
        if t < p:
            target_raw.append(0)
        else:
            target_raw.append(1)

    target_raw = np.array(target_raw)
    return target_raw, p


def upsample(data, target, target_i, sampling_limit):
    for i in range(max(target_i) + 1):
        indices = np.where(target_i == i)[0]

        data_to_replicate = data[indices]
        target_to_replicate = target[indices]
        target_i_to_replicate = target_i[indices]

        for _ in range(int(sampling_limit / len(indices))):
            data = np.concatenate((data, data_to_replicate))
            target = np.concatenate((target, target_to_replicate))
            target_i = np.concatenate((target_i, target_i_to_replicate))

        indices_to_keep = np.concatenate((np.where(target_i == i)[0][:sampling_limit], np.where(target_i != i)[0]))

        data = data[indices_to_keep]
        target = target[indices_to_keep]
        target_i = target_i[indices_to_keep]

    return data, target


def upsample_bins(data, target, bins):
    sampling_limit = 0

    # range of values for each bin
    ranges = np.arange(0, target.max() + np.finfo(float).eps, target.max() / bins)

    for i in range(1, len(ranges)):
        indices = np.where((target >= ranges[i - 1]) & (target <= ranges[i]))[0]
        if len(indices) > sampling_limit:
            sampling_limit = len(indices)

    for i in range(1, len(ranges)):
        indices = np.where((target >= ranges[i - 1]) & (target <= ranges[i]))[0]

        if len(indices):
            data_to_replicate = data[indices]
            target_to_replicate = target[indices]

            for _ in range(int(sampling_limit / len(indices))):
                data = np.concatenate((data, data_to_replicate))
                target = np.concatenate((target, target_to_replicate))
        else:
            # no data belonging to this bin range
            print('No data in bin ', ranges[i - 1], ranges[i])
            pass

        indices_to_keep = np.concatenate((np.where((target >= ranges[i - 1]) & (target <= ranges[i]))[0][:sampling_limit], np.where((target < ranges[i - 1]) | (target > ranges[i]))[0]))

        data = data[indices_to_keep]
        target = target[indices_to_keep]

    return data, target


'''
since our target is between 0-1, we are basically splitting into x bins and seeing how many numbers belong to each
'''


def downsample(data, target, sampling_limit):
    for i in range(max(target) + 1):
        indices = np.where(target == i)[0][sampling_limit:]

        data = np.delete(data, indices, axis=0)
        target = np.delete(target, indices, axis=0)

    return data, target


def augment_dataset_fft(data, target):
    data_noisy_wga = induce_noise(data, 50, fs=2000, mode='whiteGaussianAdditive', max_signal_fq=1000)
    data_noisy_wra = induce_noise(data, 50, fs=2000, mode='whiteRandomAdditive', max_signal_fq=1000)

    data, target = \
        np.concatenate((data, data_noisy_wga, data_noisy_wra), axis=0), \
        np.concatenate((target, target, target), axis=0)

    return data, target


def augment_dataset_shift(data, target, percentile, data_range, data_avg_points, shift_by=5, left_shifts=2, right_shifts=2, response_shift=False, y_min=None, y_max=None):
    # Shift by shift_by points to the left and right
    interval_range = [data_range[0] - (left_shifts * shift_by), data_range[0] + (right_shifts * shift_by)]
    # Starting range must be a positive time point
    interval_range[0] = max(interval_range[0], 0)

    new_data = []
    new_target = []
    for sample in range(data.shape[0]):

        # By default, no shifting
        curr_intervals = [data_range[0]]

        if target[sample] >= percentile or random.random() <= 0.1:
            # Update intervals to contain the various shifts
            curr_intervals = [interval for interval in range(interval_range[0], interval_range[-1] + 1, shift_by)]

        for interval in curr_intervals:

            data_slice = data[sample, :, interval:interval + data_range[1] - data_range[0]]
            data_slice_avg = data_slice.reshape(-1, data_slice.shape[1] // data_avg_points, data_avg_points).mean(axis=2)

            # Change range of data to be non-negative (>= 0)
            # np.sign(np.amin(data_slice_avg, axis=1)) * np.abs(np.amin(data_slice_avg, axis=1)) is the same as np.amin(data_slice_avg, axis=1) (?)
            data_slice_avg -= np.amin(data_slice_avg, axis=1)[:, np.newaxis]  # gets the amin for each contact

            new_data.append(data_slice_avg)

            if response_shift:
                # Shifting to left will make this negative; to right is positive
                shift_time_by = (interval - data_range[0]) / 2048

                # Unscale the response and add (or subtract) time adjusted. Then re-scale
                y_shifted = ((target[sample] * y_max) + y_min) + shift_time_by
                y_shifted -= y_min
                y_shifted /= y_max
                y_shifted = np.clip(y_shifted, 0, 1)
            else:
                y_shifted = target[sample]

            new_target.append(y_shifted)

    data = np.asarray(new_data)
    target = np.asarray(new_target)

    return data, target


def generate_hyperparameters():
    percentile = random.choice([75, 80, 85, 95, 95, 97, 97])
    model_num = 1
    learning_rate = round(random.random() * (0.005 - 0.001) + 0.001, 10)
    dropout_rate = random.choice([0.4, 0.5, 0.6])
    epochs = random.choice([400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1100, 1200]) + 1500
    epochs = 500
    loss = random.choice(['mse', 'mae', 'binary_crossentropy'])
    num_graphs = random.choice([1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])

    L1_units = random.choice([8, 16, 32, 64, 128])
    L2_units = L1_units // 2
    L3_units = L2_units // 2
    L4_units = L3_units // 2

    # TODO: default is 256 and 256//2
    stft_nperseg = random.choice([64, 128, 256, 512, 1024])
    # NOVERLAP_RANDOM = random.choice([NPERSEG_RANDOM // 2, NPERSEG_RANDOM // 4, NPERSEG_RANDOM // 8])
    stft_noverlap = stft_nperseg // 2

    return percentile, model_num, learning_rate, dropout_rate, epochs, loss, num_graphs, L1_units, L2_units, L3_units, L4_units, stft_nperseg, stft_noverlap


def load_hyperparameters(results_path, run_id):
    df = pd.read_csv(results_path)
    row = pd.DataFrame(df.loc[df['Run ID'] == run_id])
    row.reset_index(inplace=True)
    percentile = row['Percentile'][0]
    model_num = 1
    learning_rate = row['Learning Rate'][0]
    dropout_rate = row['Dropout Rate'][0]

    # TODO: temporary:
    # row['ES Epochs'] = row['ES Epochs'].astype(str)
    epochs = list(map(int, row['ES Epochs'][0].split(':')))

    loss = row['Loss'][0]
    num_graphs = row['# Graphs'][0]
    model_params = row['Model'][0].split('_')
    regions_to_use = row['Regions To Use'][0].split(':')
    L1_units = int(model_params[0][4:])
    L2_units = int(model_params[1][2:])
    L3_units = int(model_params[2][2:])
    L4_units = 0

    stft_nperseg, stft_noverlap = row['STFT Nperseg'][0], row['STFT Noverlap'][0]

    return percentile, model_num, learning_rate, dropout_rate, epochs, loss, num_graphs, L1_units, L2_units, L3_units, L4_units, regions_to_use, stft_nperseg, stft_noverlap

def load_hyperparameters2(results_path, run_id):
    df = pd.read_csv(results_path)
    row = pd.DataFrame(df.loc[df['Run ID'] == run_id])
    row.reset_index(inplace=True)
    percentile = row['Percentile'][0]
    model_num = 1
    learning_rate = row['Learning Rate'][0]
    dropout_rate = row['Dropout Rate'][0]

    # TODO: temporary:
    # row['ES Epochs'] = row['ES Epochs'].astype(str)
    epochs = list(map(int, row['ES Epochs'][0].split(':')))

    loss = row['Loss'][0]
    num_graphs = row['# Graphs'][0]
    model_params = row['Model'][0].split('_')
    contacts_to_use = row['Contacts To Use'][0].split(':')
    L1_units = int(model_params[0][4:])
    L2_units = int(model_params[1][2:])
    L3_units = int(model_params[2][2:])
    L4_units = 0

    stft_nperseg, stft_noverlap = row['STFT Nperseg'][0], row['STFT Noverlap'][0]

    return percentile, model_num, learning_rate, dropout_rate, epochs, loss, num_graphs, L1_units, L2_units, L3_units, L4_units, contacts_to_use, stft_nperseg, stft_noverlap


def get_best_model(results_svr_path):
    df = pd.read_csv(results_svr_path)
    df.sort_values(by=['Accuracy 1 STDDEV'], ascending=False, inplace=True)
    run_id = int(df.iloc[0]['Run ID'])
    # run_id = int(df.iloc[2]['Run ID'])  # TODO
    return run_id


def get_fft(data, fs, column_wise=False):
    data_copy = data.copy()
    if column_wise:
        data_copy = data_copy.transpose()

    n = len(data_copy)
    if data_copy.ndim == 2:
        n = data_copy.shape[1]
    t = n / fs

    freqs = fftfreq(n) * fs
    mask = freqs > 0

    fft_vals = fft(data_copy)
    fft_theo = 2.0 * np.abs(fft_vals / n)

    if not column_wise:
        return freqs[mask], fft_theo.transpose()[mask].transpose()
    else:
        return freqs[mask], fft_theo.transpose()[mask]


def induce_noise(all_data, snr, fs, mode='random', max_signal_fq=100, column_wise=False):
    modes = ['whiteGaussianMultiplicative', 'whiteRandomMultiplicative', 'whiteGaussianAdditive',
             'whiteRandomAdditive', 'highFrequencyAdditive']
    if mode == 'random':
        mode = random.choice(modes)

    induced_noise = []
    for sample in all_data:
        curr_induced_noise = []
        for data in sample:
            fft_vals = fft(data)
            # print('fft_vals.shape:', fft_vals.shape)
            if mode == 'whiteGaussianMultiplicative':
                # 99% of the gaussian numbers will be in the correct SNR
                noise = np.random.normal(1, (1.0 / snr) / 3.0, size=fft_vals.shape)
                noisy = ifft(fft_vals * noise)

            elif mode == 'whiteRandomMultiplicative':
                noise = np.random.uniform(1 - (1.0 / snr), 1 + (1.0 / snr), size=fft_vals.shape)
                noisy = ifft(fft_vals * noise)

            elif mode == 'whiteGaussianAdditive':
                _, fft_theo = get_fft(data, fs)
                sigStrength = np.max(fft_theo)
                noiseStrength = float(sigStrength / snr)
                noise = np.random.normal(0, noiseStrength / 3, size=fft_vals.shape)
                noise = noise / 2 * len(data)
                noisy = ifft(fft_vals + noise)

            elif mode == 'whiteRandomAdditive':
                _, fft_theo = get_fft(data, fs)
                sigStrength = np.max(fft_theo)
                noiseStrength = float(sigStrength / snr)
                noise = np.random.uniform(-1 * noiseStrength, noiseStrength, size=fft_vals.shape)
                noise = noise / 2 * len(data)
                noisy = ifft(fft_vals + noise)

            elif mode == 'highFrequencyAdditive':
                _, fft_theo = get_fft(data, fs)
                sigStrength = np.max(fft_theo)
                noiseStrength = float(sigStrength / snr)
                noise = np.zeros(len(fft_vals))
                half_length = len(fft_vals) // 2
                noise[max_signal_fq:half_length] = np.random.random((half_length - max_signal_fq,)) * noiseStrength
                noise[half_length + max_signal_fq:len(fft_vals)] = np.random.random((len(fft_vals) - (half_length + max_signal_fq),)) * noiseStrength
                noise = noise / 2 * len(data)
                noisy = ifft(fft_vals + noise)
            else:
                raise Exception('ERROR: mode not recognized!')
            curr_induced_noise.append(noisy)
        induced_noise.append(curr_induced_noise)

    induced_noise = np.asarray(induced_noise, dtype=float)

    return induced_noise


def calculate_top_n_graphs(X, num_graphs, additional_X=[]):
    # Limit dataset to num_graphs indices only, in descending order of magnitude
    X = X[:, :num_graphs]
    additional_X = [X_array[:, :num_graphs] for X_array in additional_X]

    return X, additional_X


# NOTE: NOT USED
# def calculate_abs_difference(X, target_classes, num_classes=1):
#     X_c = np.copy(X)
#
#     # Generate two arrays, one for each outcome
#     if num_classes == 1:
#         outcome_array = [[[] for _ in range(X_c.shape[1])] for _ in range(num_classes + 1)]
#     else:
#         outcome_array = [[[] for _ in range(X_c.shape[1])] for _ in range(num_classes)]
#
#     # Populate different outcome arrays
#     for trial in range(X_c.shape[0]):
#         for contact in range(X_c.shape[1]):
#             Ra = X_c[trial, contact]
#             outcome_array[target_classes[trial]][contact].append(Ra.tolist())
#
#     outcome_array = np.asarray(outcome_array)
#
#     # Squeeze array and calculate averages for each outcome
#     outcome_array_avg = [[] for _ in range(outcome_array.shape[0])]
#     for class_num in range(outcome_array.shape[0]):
#         for contact in range(outcome_array.shape[1]):
#             outcome_array_avg[class_num].append(np.mean(outcome_array[class_num][contact], axis=0))
#
#     outcome_array_avg = np.asarray(outcome_array_avg)
#
#     outcome_array_diffs_o = np.sum(abs(abs(outcome_array_avg[0]) - abs(outcome_array_avg[1])), axis=1)  # Get the difference of each EEG plot between the two outcomes
#
#     X_c = np.swapaxes(X_c, 0, 1)
#     # for i in range(X_c.shape[1]):
#     # X_c[[58, 59, 69, 133]] = 0
#     # outcome_array_diffs = np.sum(np.sum(X_c, axis=1), axis=1)
#     outcome_array_diffs = np.sum(np.mean(X_c, axis=1), axis=1)
#
#     # Average across all trials, then take 'derivative' and sum separately for each contact. Contact with the highest value has experienced the most change
#     outcome_array_diffs = np.sum(abs(np.diff(np.mean(X_c, axis=1), axis=1)), axis=1)
#
#     return outcome_array_avg, outcome_array_diffs
# NOTE: condensed version
def calculate_abs_difference(X):
    X = X.copy()
    # Group trials by contact. Average over all the trials per contact and sort by the contacts with the most change in their EEG plot
    X = np.swapaxes(X, 0, 1)
    return np.sum(abs(np.diff(np.mean(X, axis=1), axis=1)), axis=1)


def sort_by_abs_difference(X, contact_info, additional_X=[]):
    outcome_array_diffs = calculate_abs_difference(X)

    indices = outcome_array_diffs.argsort()[::-1]

    outcome_array_diffs = outcome_array_diffs[indices]
    contact_info = contact_info.iloc[indices, :]
    contact_info = contact_info.reset_index(drop=True)

    # Limit dataset to num_graphs indices only, in descending order of magnitude
    additional_X = [X_array[:, indices] for X_array in additional_X]

    # # Normalize everything to between 0-1
    # X = X - np.min(np.reshape(X, (X.shape[0], -1)))
    # X = X / np.max(np.reshape(X, (X.shape[0], -1)))
    # Standardize X
    # X -= np.mean(X)
    # X /= np.std(X)

    return contact_info, additional_X


# Remove nan regions/regions we want to exclude from both the dataframe and from out data
def remove_regions(region, contact_info, X_values=[]):
    region_to_remove = contact_info.index[contact_info['aal_label'] == region].tolist()
    regions_to_exclude = contact_info.index[contact_info['exclude'] == '1'].tolist()
    indices_to_remove = region_to_remove + regions_to_exclude
    indices_to_remove = list(set(indices_to_remove))

    X_values = [np.delete(X_array, indices_to_remove, axis=1) for X_array in X_values]
    contact_info = contact_info.drop(indices_to_remove)
    contact_info = contact_info.reset_index(drop=True)

    return contact_info, X_values


def set_data_range(data, data_range, data_avg_points):
    data = data[:, :, data_range[0]:data_range[1]]
    data_new = np.empty((data.shape[0], data.shape[1], data.shape[2] // data_avg_points))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data_new[i, j] = np.mean(data[i, j].reshape(-1, data_avg_points), axis=1)
            data_new[i, j] -= np.sign(np.amin(data_new[i, j])) * np.abs(np.amin(data_new[i, j]))
    return data_new


# validation, test -> set aside data for these.
# When not on final run, we set aside these but only use train and cross validate on val.
# When on final run, we set aside only test and train on everything else.
# When creating our final model, we dont set any aside and just train on all data.
def calculate_batch(X_full, Y, B, step, num_batches, data_range, data_avg_points, validation=True, test=True):
    def set_data_range(data, data_range, data_avg_points):
        data = data[:, :, data_range[0]:data_range[1]]
        data_new = np.empty((data.shape[0], data.shape[1], data.shape[2] // data_avg_points))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                data_new[i, j] = np.mean(data[i, j].reshape(-1, data_avg_points), axis=1)
                data_new[i, j] -= np.sign(np.amin(data_new[i, j])) * np.abs(np.amin(data_new[i, j]))
        return data_new


    i_batch = step * B

    validData, validTarget, testData, testTarget = None, None, None, None

    # Indices of all the data not part of our validation/testing batch (so we can use for training)
    train_indices = np.ones(X_full.shape[0], dtype=bool)

    if test:
        # On the final step, we will have to loop around the data
        if step == num_batches - 1:
            testData, testTarget = X_full[i_batch:], Y[i_batch:]
            train_indices[i_batch:] = 0
        else:
            testData, testTarget = X_full[i_batch:i_batch + B], Y[i_batch:i_batch + B]
            train_indices[i_batch:i_batch + B] = 0

        testData = set_data_range(testData, data_range, data_avg_points)

    if validation:
        if step == num_batches - 1:
            validData, validTarget = X_full[0:B], Y[0:B]
            train_indices[0:B] = 0
        else:
            validData, validTarget = X_full[i_batch + B:i_batch + (2 * B)], Y[i_batch + B:i_batch + (2 * B)]
            train_indices[i_batch + B:i_batch + (2 * B)] = 0

        validData = set_data_range(validData, data_range, data_avg_points)

    trainData, trainTarget = X_full[train_indices], Y[train_indices]

    return trainData, trainTarget, validData, validTarget, testData, testTarget


def create_model(input_shape, dropout_rate, num_classes, L1_units, L2_units, L3_units, activation_function):
    model = Sequential()

    if not MODE_FREQUENCY:
        model.add(Conv2D(L1_units, padding='valid', kernel_size=(1, input_shape[1] // 5), dilation_rate=2, activation='relu', input_shape=input_shape))
        model.add(Conv2D(L2_units, padding='valid', kernel_size=(input_shape[0], 10), dilation_rate=1, activation='relu'))
    else:
        from keras.layers import Conv3D
        model.add(Conv3D(L1_units, padding='valid', kernel_size=(1, 3, 3), activation='relu', input_shape=input_shape))
        model.add(Conv3D(L2_units, padding='valid', kernel_size=(input_shape[0], 3, 3), activation='relu'))

    model.add(Flatten())
    model.add(Dense(L2_units, activation='relu'))
    model.add(Dense(L3_units, activation='relu'))
    model_name = 'C2NN' + str(L1_units) + '_NN' + str(L2_units) + '_NN' + str(L3_units)

    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation=activation_function))

    return model, model_name


def plot_gradcams(model, testData, preds, targets, step, path_to_save, xmean, xstd, nperseg, noverlap, attn_map_cutoff=0.7):
    if MODE_FREQUENCY:
        penultimate_layer_idx = utils.find_layer_idx(model, "conv3d_1")
    else:
        penultimate_layer_idx = utils.find_layer_idx(model, "conv2d_1")

    # Sorted test data indices (test data predictions closest to target to farthest)
    sorted_indices = np.argsort(np.abs(preds - targets))
    best_indices = sorted_indices[:5]
    worst_indices = sorted_indices[-5:][::-1]

    # Go through the best indices and then the worst indices; so we can name accordingly
    for category, indices in zip(['best', 'worst'], [best_indices, worst_indices]):

        for i in range(indices.shape[0]):

            attn_map = visualize_cam(model, layer_idx=-1, filter_indices=None, seed_input=testData[indices[i]],
                                     penultimate_layer_idx=penultimate_layer_idx, backprop_modifier=None,
                                     grad_modifier=None)
            img = np.squeeze(testData[indices[i]])

            if MODE_FREQUENCY:
                from scipy import signal
                from sklearn.preprocessing import MinMaxScaler

                # TODO:
                # raise Exception('On best run, type in the nperseg and noverlap params below and then comment out this exception.')

                NUM_DATA_POINTS = 4500  # TODO:
                try:
                    img = signal.istft(img * xstd + xmean, FREQUENCY_FS_RATE, nperseg=nperseg, noverlap=noverlap)[1][:, :NUM_DATA_POINTS]
                except:
                    # if just 1 contact was chosen
                    img = signal.istft(img * xstd + xmean, FREQUENCY_FS_RATE, nperseg=nperseg, noverlap=noverlap)[1][:NUM_DATA_POINTS]
                    img = np.expand_dims(img, axis=0)

                # NOTE: scaling or not scaling results in same shape. and then when we scale 0-1, they will be indentical
                attn_map_inverted = signal.istft(attn_map * xstd + xmean, FREQUENCY_FS_RATE, nperseg=nperseg, noverlap=noverlap)[1][:, :NUM_DATA_POINTS]

                attn_map_inverted = MinMaxScaler().fit_transform(attn_map_inverted.T).T

                # img = signal.resample(img, 4500, axis=1)
                # attn_map_inverted = signal.resample(attn_map_inverted, 4500, axis=1)

                '''
                inverted = signal.istft(attn_map, 2048)[1]
                upsampled = signal.resample(inverted, 4500, axis=1)
                attn_map = upsampled # back to signal form
                '''

                plt.close('all')

                num_contacts = img.shape[0]
                # for plotting. we will put all contacts in one plot
                num_rows = np.ceil(num_contacts / 2).astype(int)
                num_cols = 2

                # TODO: EXPERIMENTAL STUFF
                # plotting the current raw CAMs (all its contacts)
                fig, axes = plt.subplots(num_rows, num_cols, tight_layout=True)
                axes = axes.flatten()
                for contact_ind in range(num_contacts):
                    axes[contact_ind].set_title(f'Contact {contact_ind}')
                    axes[contact_ind].axis('off')
                    axes[contact_ind].imshow(attn_map[contact_ind])
                fig.savefig(
                    os.path.join(path_to_save, 'attn_maps', 'test_b{}_CAM_{}_{}.png'.format(step, category, i)), dpi=300
                )

                # plotting the frequency CAMs back to signal
                fig, axes = plt.subplots(num_rows, num_cols, tight_layout=True)
                axes = axes.flatten()
                for contact_ind in range(num_contacts):
                    axes[contact_ind].set_title(f'Contact {contact_ind}')
                    axes[contact_ind].axhline(attn_map_cutoff, 0, 1, color='r')
                    axes[contact_ind].plot(attn_map_inverted[contact_ind])
                fig.savefig(
                    os.path.join(path_to_save, 'attn_maps', 'test_b{}_INVERTED_FREQ_CAM_{}_{}.png'.format(step, category, i)), dpi=300
                )

                # plotting the frequency eeg data back to signal
                fig, axes = plt.subplots(num_rows, num_cols, tight_layout=True)
                axes = axes.flatten()
                for contact_ind in range(num_contacts):
                    axes[contact_ind].set_title(f'Contact {contact_ind}')
                    axes[contact_ind].plot(img[contact_ind])
                fig.savefig(
                    os.path.join(path_to_save, 'attn_maps', 'test_b{}_INVERTED_FREQ_{}_{}.png'.format(step, category, i)), dpi=300
                )

                # how the original test signal looks like
                # fig, axes = plt.subplots(num_rows, num_cols, tight_layout=True)
                # axes = axes.flatten()
                # for contact_ind in range(num_contacts):
                #     axes[contact_ind].set_title(f'Contact {contact_ind}')
                #     axes[contact_ind].plot(orig_test_data[indices[i]][contact_ind])
                # fig.savefig(
                #     os.path.join(path_to_save, 'attn_maps', 'test_b{}_ORIG_SIGNAL_{}_{}.png'.format(step, category, i)), dpi=300
                # )

                # overlay CAM
                attn_map_inverted_clipped = np.where(attn_map_inverted >= attn_map_cutoff, attn_map_inverted, 0)

                attn_map_colors = plt.cm.get_cmap('jet')(attn_map_inverted_clipped)[:, :, :-1]
                fig, axes = plt.subplots(num_rows, num_cols, tight_layout=True)
                axes = axes.flatten()
                for contact_ind in range(num_contacts):
                    axes[contact_ind].set_title(f'Contact {contact_ind}')
                    axes[contact_ind].plot(img[contact_ind])
                    for ind in range(len(attn_map_inverted_clipped[contact_ind])):
                        # axes[contact_ind].axvline(ind, 0, 1, alpha=0.2, color=attn_map_colors[contact_ind, ind])
                        axes[contact_ind].axvline(ind, 0, 1, alpha=0.01, color=attn_map_colors[contact_ind, ind])
                fig.savefig(
                    os.path.join(path_to_save, 'attn_maps', 'test_b{}_INVERTED_FREQ_CAM_ON_INVERTED_FREQ_{}_{}.png'.format(step, category, i)), dpi=300
                )

            else:

                # EEG plot as a 2d image
                if 0:
                    plt.close('all')
                    fig = plt.figure(figsize=(180, 6))
                    plt.imshow(img, cmap='gray')
                    plt.savefig(os.path.join(path_to_save, 'attn_maps', 'test_b{}_raw_{}_{}.png'.format(step, category, i)))

                # EEG plot for each contact
                if 0:
                    for contact_ind in range(testData.shape[1]):
                        plt.close('all')
                        fig, ax = plt.subplots()
                        ax.plot(img[contact_ind])
                        fig.savefig(
                            os.path.join(path_to_save, 'attn_maps', 'test_b{}_raw_{}_{}_c{}.png'.format(step, category, i, contact_ind))
                        )

                # ATTN MAP OUTPUT AS A 2D IMAGE
                if 0:
                    plt.close('all')
                    fig = plt.figure(figsize=(180, 6))
                    plt.imshow(img, cmap='gray')
                    attn_map_img = plt.imshow(attn_map, cmap="jet", alpha=0.8)
                    fig.colorbar(attn_map_img)
                    plt.savefig(os.path.join(path_to_save, 'attn_maps', 'test_b{}_gc_{}_{}.png'.format(step, category, i)))

                # ATTN MAP OUTPUT OVER EACH EEG PLOT
                attn_map_colors = plt.cm.get_cmap('jet')(attn_map)[:, :, :-1]
                for contact_ind in range(testData.shape[1]):
                    plt.close('all')
                    fig, ax = plt.subplots()
                    ax.plot(img[contact_ind])
                    for ind in range(len(attn_map[contact_ind])):
                        ax.axvline(ind, 0, 1, alpha=0.2, color=attn_map_colors[contact_ind, ind])
                    fig.savefig(
                        os.path.join(path_to_save, 'attn_maps', 'test_b{}_gc_{}_{}_c{}.png'.format(step, category, i, contact_ind))
                    )


def plot_loss_graphs(dataset, run_id, step, path_to_save):
    training_loss = dataset['loss']
    validation_loss = dataset['val_loss']

    plt.close('all')
    plt.plot(training_loss, '-b', label='training')
    plt.plot(validation_loss, '-g', label='validation')

    plt.legend(loc="upper right")
    plt.xlabel('epochs')
    plt.ylabel('loss')

    plt.savefig(os.path.join(path_to_save, 'losses', 'loss_{}_{}.jpg'.format(run_id, step)), dpi=500)


def normalize_output(Y):
    Y[Y >= 1.5] = 1.5  # Don't bother predicting higher than 1.5s, not enough data
    y_min = np.amin(Y).astype(float)
    Y -= y_min  # Zero-center output
    y_max = np.amax(Y).astype(float)
    Y /= y_max  # Normalize output

    return Y, y_min, y_max


def select_regions(num_regions, max_regions_limit, top_regions, top_region_indices, data, region_nums_to_use=None):
    if region_nums_to_use is None:
        region_nums_to_use = sample([x for x in range(num_regions)], random.randint(1, min(num_regions, max_regions_limit)))

    # all the indices of the contacts that we will use (all belonging to a total of `region_nums_to_use` regions)
    indices_to_use = [top_region_indices[ind] for ind in region_nums_to_use]
    indices_to_use = [item for sublist in indices_to_use for item in sublist]

    # names of the regions we will be using
    regions_to_use = ":".join([top_regions[ind] for ind in region_nums_to_use])
    num_graphs = len(indices_to_use)

    data = [data_X[:, indices_to_use] for data_X in data]

    return data, regions_to_use, num_graphs, region_nums_to_use, indices_to_use


def select_contacts2(data, contacts, regions, top_x_contacts=20, max_contact_limit=5, indices_to_use=None):
    if indices_to_use is None:
        indices_to_use = np.random.choice(np.arange(top_x_contacts), size=np.random.randint(1, max_contact_limit + 1), replace=False)
    else:
        # already being passed in. ie when we are on final run and we know which contacts to use
        pass

    # names
    contacts_to_use = ':'.join(contacts[indices_to_use].tolist())
    regions_to_use = ':'.join(regions[indices_to_use].tolist())

    data = data[:, indices_to_use]

    return data, contacts_to_use, regions_to_use


def load_data(path_to_load):
    dataset = h5py.File(os.path.join(path_to_load, 'processed_data.h5'), 'r')
    X_full, X_spec, respTimes, shifts = dataset.get('data_full')[()], \
                                        dataset.get('data_spec')[()], dataset.get('respTimes')[()], \
                                        dataset.get('shifts')[()]
    num_regions, top_regions = dataset.get('num_regions')[()], np.asarray(dataset.get('top_regions')[()], dtype='str')
    top_region_indices = []
    for i in range(num_regions):
        top_region_indices.append(dataset.get('top_region_indices_' + str(i))[()].tolist())
    # contact_info, contact_info_headers = dataset.get('contact_info')[()], dataset.get('contact_info_headers')[()]
    dataset.close()

    X_spec = np.expand_dims(X_spec, axis=-1)
    # contact_info = pd.DataFrame(contact_info, columns=contact_info_headers)

    return [X_full, X_spec], respTimes, num_regions, top_regions, top_region_indices


# Loads the contact based data
def load_data2(path_to_load):
    dataset = h5py.File(os.path.join(path_to_load, 'processed_data.h5'), 'r')
    X_full, respTimes, shifts = dataset.get('data_full')[()], dataset.get('respTimes')[()], dataset.get('shifts')[()]

    contacts, regions = dataset.attrs['contacts'], dataset.attrs['regions']

    dataset.close()

    return X_full, respTimes, contacts, regions


def plot_svr_plot(target, classification, accuracy_window, title, run_id, path_to_save, batch_mode=False):
    # Mask of which indices are within our accuracy window
    mask = (classification + accuracy_window >= target) & (target >= classification - accuracy_window)

    plt.close('all')
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    if accuracy_window == 0.0:
        ax.scatter(x=target, y=classification, c='b', alpha=0.5)

    elif batch_mode:
        # each batch will have a different color. useful for seeing if some batches are more wrong than others
        num_batches = 5
        batch_size = target.shape[0] // num_batches

        for i in range(num_batches):
            curr_target = target[i * batch_size:(i + 1) * batch_size]
            curr_classification = classification[i * batch_size:(i + 1) * batch_size]

            ax.scatter(x=curr_target, y=curr_classification, alpha=0.5)
    else:

        # green for dots inside acc window. red for outside
        ax.scatter(x=target[mask], y=classification[mask], c='g', alpha=0.5)
        ax.scatter(x=target[~mask], y=classification[~mask], c='r', alpha=0.5)

    upper_limit_1 = max(target) + 0.05
    upper_limit_2 = max(classification) + 0.05
    upper_limit = max(upper_limit_1, upper_limit_2)

    lower_limit_1 = min(target) - 0.05
    lower_limit_2 = min(classification) - 0.05
    lower_limit = 0  # min(lower_limit_1, lower_limit_2)

    x = np.linspace(lower_limit, upper_limit, 1000)
    ax.plot(x, x, '-k')
    ax.plot(x, x + accuracy_window, ':k')
    ax.plot(x, x - accuracy_window, ':k')

    ax.axis([lower_limit, upper_limit, lower_limit, upper_limit])
    ax.grid(which='major', alpha=0.1)

    accuracy = sum(mask) * 100.0 / len(target)

    if accuracy_window == 0.0:
        fig.suptitle("Scatter Plot - Raw")
    else:
        fig.suptitle("Scatter Plot - SVR {}. Acc: {:.2f}".format(title, accuracy))
    ax.set(xlabel="True value")
    ax.set(ylabel="Predicted value")
    plt.tight_layout()
    fig.savefig(os.path.join(path_to_save, 'plots', '{}plots_scatter_{}_{}.png'.format('BATCH_' if batch_mode else '', accuracy_window, run_id)))

    mse_loss = mean_squared_error(classification[~mask], target[~mask])
    r2_score_val = r2_score(classification[~mask], target[~mask])

    print("MSE Loss for AW-" + str(accuracy_window) + " :" + str(round(mse_loss, 5)))
    print("R2 score for AW-" + str(accuracy_window) + " :" + str(round(r2_score_val, 5)))
    print("Accuracy for AW-" + str(accuracy_window) + " :" + str(round(accuracy, 5)))
    print("")

    return mse_loss, r2_score_val, accuracy
