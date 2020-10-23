from numpy.fft import fft, fftfreq, ifft
from random import sample
import numpy as np
import random
import h5py
import os


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


def upsample_bins(data, target, shift, bins):
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
            shift_to_replicate = shift[indices]

            for _ in range(int(sampling_limit / len(indices))):
                data = np.concatenate((data, data_to_replicate))
                target = np.concatenate((target, target_to_replicate))
                shift = np.concatenate((shift, shift_to_replicate))
        else:
            # no data belonging to this bin range
            print('No data in bin ', ranges[i - 1], ranges[i])
            pass

        indices_to_keep = np.concatenate((np.where((target >= ranges[i - 1]) & (target <= ranges[i]))[0][:sampling_limit], np.where((target < ranges[i - 1]) | (target > ranges[i]))[0]))

        data = data[indices_to_keep]
        target = target[indices_to_keep]
        shift = shift[indices_to_keep]

    return data, target, shift


'''
since our target is between 0-1, we are basically splitting into x bins and seeing how many numbers belong to each
'''


def downsample(data, target, sampling_limit):
    for i in range(max(target) + 1):
        indices = np.where(target == i)[0][sampling_limit:]

        data = np.delete(data, indices, axis=0)
        target = np.delete(target, indices, axis=0)

    return data, target


# TODO: AUGMENT FFT FALSE IN SPECTOGRAM
######
# def augment_dataset(data, target, target_p, data_range, data_avg_points, augment_fft=False):
#     # Augment using FFT method
#     if not MODE_FREQUENCY or augment_fft:
#         data, target = augment_dataset_fft(data, target)
#
#     # Augment using shifting method
#     # TODO: for frequency, we are not averaging so doing 1.5 * data_avg_points doesnt make sense. we will hard code 5 for small shift and 100 for big
#     points_small_shift = 5
#     augment_range = (data_range[0] - (2 * points_small_shift), data_range[0] + (2 * points_small_shift))
#
#     augment_interval_points = int(1.5 * data_avg_points)
#     augment_range = (data_range[0] - (2 * augment_interval_points), data_range[0] + (2 * augment_interval_points))
#     data, target = augment_dataset_shifting(data, target, target_p, augment_range, augment_interval_points, data_range, data_avg_points)
#
#
#     np.random.seed(0)
#
#     return data, target
########


def augment_dataset_fft(data, target):
    data_noisy_wga = induce_noise(data, 50, fs=2000, mode='whiteGaussianAdditive', max_signal_fq=1000)
    data_noisy_wra = induce_noise(data, 50, fs=2000, mode='whiteRandomAdditive', max_signal_fq=1000)

    data, target = \
        np.concatenate((data, data_noisy_wga, data_noisy_wra), axis=0), \
        np.concatenate((target, target, target), axis=0)

    return data, target


def augment_dataset_shift(data, target, shift, percentile, data_range, data_avg_points, shift_by=5, left_shifts=2, right_shifts=2, response_shift=False, y_min=None, y_max=None):
    # Shift by shift_by points to the left and right
    interval_range = [data_range[0] - (left_shifts * shift_by), data_range[0] + (right_shifts * shift_by)]
    # Starting range must be a positive time point
    interval_range[0] = max(interval_range[0], 0)

    new_data = []
    new_target = []
    new_shift = []
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
            new_shift.append(shift[sample])

    data = np.asarray(new_data)
    target = np.asarray(new_target)
    shift = np.asarray(new_shift)

    return data, target, shift


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


def calculate_abs_difference(X, target_classes, num_classes=1):
    X_c = np.copy(X)

    # Generate two arrays, one for each outcome
    if num_classes == 1:
        outcome_array = [[[] for _ in range(X_c.shape[1])] for _ in range(num_classes + 1)]
    else:
        outcome_array = [[[] for _ in range(X_c.shape[1])] for _ in range(num_classes)]

    # Populate different outcome arrays
    for trial in range(X_c.shape[0]):
        for contact in range(X_c.shape[1]):
            Ra = X_c[trial, contact]
            outcome_array[target_classes[trial]][contact].append(Ra.tolist())

    outcome_array = np.asarray(outcome_array)

    # Squeeze array and calculate averages for each outcome
    outcome_array_avg = [[] for _ in range(outcome_array.shape[0])]
    for class_num in range(outcome_array.shape[0]):
        for contact in range(outcome_array.shape[1]):
            outcome_array_avg[class_num].append(np.mean(outcome_array[class_num][contact], axis=0))

    outcome_array_avg = np.asarray(outcome_array_avg)

    outcome_array_diffs_o = np.sum(abs(abs(outcome_array_avg[0]) - abs(outcome_array_avg[1])), axis=1)  # Get the difference of each EEG plot between the two outcomes

    X_c = np.swapaxes(X_c, 0, 1)
    # for i in range(X_c.shape[1]):
    # X_c[[58, 59, 69, 133]] = 0
    # outcome_array_diffs = np.sum(np.sum(X_c, axis=1), axis=1)
    outcome_array_diffs = np.sum(np.mean(X_c, axis=1), axis=1)

    # Average across all trials, then take 'derivative' and sum separately for each contact. Contact with the highest value has experienced the most change
    outcome_array_diffs = np.sum(abs(np.diff(np.mean(X_c, axis=1), axis=1)), axis=1)

    return outcome_array_avg, outcome_array_diffs


def sort_by_abs_difference(X, contact_info, target_classes, additional_X=[], num_classes=1):
    outcome_array_avg, outcome_array_diffs = calculate_abs_difference(X, target_classes, num_classes)

    indices = outcome_array_diffs.argsort()[::-1]
    outcome_array_diffs = outcome_array_diffs[indices]
    outcome_array_avg[0] = outcome_array_avg[0][indices]
    outcome_array_avg[1] = outcome_array_avg[1][indices]
    contact_info = contact_info.iloc[indices, :]
    contact_info = contact_info.reset_index(drop=True)

    # Limit dataset to num_graphs indices only, in descending order of magnitude
    X = X[:, indices]
    additional_X = [X_array[:, indices] for X_array in additional_X]

    # # Normalize everything to between 0-1
    # X = X - np.min(np.reshape(X, (X.shape[0], -1)))
    # X = X / np.max(np.reshape(X, (X.shape[0], -1)))
    # Standardize X
    # X -= np.mean(X)
    # X /= np.std(X)

    return X, contact_info, additional_X


def remove_regions(region, contact_info, regions, X_values=[]):
    region_to_remove = contact_info.index[contact_info['aal_label'] == region].tolist()
    regions_to_exclude = contact_info.index[contact_info['exclude'] == '1'].tolist()
    indices_to_remove = region_to_remove + regions_to_exclude
    indices_to_remove = list(set(indices_to_remove))

    X_values = [np.delete(X_array, indices_to_remove, axis=1) for X_array in X_values]
    contact_info = contact_info.drop(indices_to_remove)
    contact_info = contact_info.reset_index(drop=True)
    regions = np.delete(regions, indices_to_remove, axis=0)

    return contact_info, regions, X_values


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

    # TODO:
    # region_nums_to_use = [9, 7]

    # all the indices of the contacts that we will use (all belonging to a total of `region_nums_to_use` regions)
    indices_to_use = [top_region_indices[ind] for ind in region_nums_to_use]
    indices_to_use = [item for sublist in indices_to_use for item in sublist]

    # names of the regions we will be using
    regions_to_use = ":".join([top_regions[ind] for ind in region_nums_to_use])
    num_graphs = len(indices_to_use)

    data = [data_X[:, indices_to_use] for data_X in data]

    return data, regions_to_use, num_graphs, region_nums_to_use, indices_to_use


def select_contacts2(data, contacts, regions, top_x_contacts=None, max_contact_limit=5, indices_to_use=None):
    if indices_to_use is None:
        if top_x_contacts is None:
            top_x_contacts = len(contacts)
        # out of top_x_contacts contacts, choose up to max_contact_limit contacts
        indices_to_use = np.random.choice(np.arange(top_x_contacts), size=np.random.randint(1, max_contact_limit + 1), replace=False)
    else:
        # already being passed in. ie we know which contacts to use
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

    return X_full, respTimes, contacts, regions, shifts
