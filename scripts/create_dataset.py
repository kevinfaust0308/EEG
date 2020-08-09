# Extracts data for EEG random forest and saves into numpy text file.


# AREA UNDER THE CURVE?

import os
import sys
import h5py
import math
import socket
import numpy as np
import pandas as pd
from scipy import signal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.utils import *

machine_hostname = socket.gethostname()

if machine_hostname == 'mpc':
    path = "/hpf/largeprojects/ccm_home/masgouri/eeg/data/"
else:
    path = "/hpf/largeprojects/ccm/masgouri/eeg/data/"

filename = sys.argv[1]
path_to_load = path + filename + "/raw/"
path_to_save = path + filename + "/processed/"

num_regions = 10
percentile = 75


def calculate_abs_differences(data, respType, is_normalized):
    X_c = np.copy(data)

    target_raw, p = calculate_percentile(respType, percentile=percentile)

    L = [[], []]
    L[0] = [[] for _ in range(X_c.shape[1])]
    L[1] = [[] for _ in range(X_c.shape[1])]

    for ind in range(X_c.shape[0]):
        for a in range(X_c.shape[1]):
            Ra = X_c[ind, a]
            L[target_raw[ind]][a].append(Ra.tolist())

    L = np.asarray(L)

    LL = [[], []]
    for aa in range(L.shape[0]):
        for a in range(L.shape[1]):
            LL[aa].append(np.mean(L[aa][a], axis=0))

    LL = np.asarray(LL)

    if is_normalized:
        LLx = np.copy(np.concatenate((LL[0], LL[1]), axis=1))
        LL[0] = LL[0] + abs(np.min(LLx, axis=1)[:, np.newaxis])
        LL[1] = LL[1] + abs(np.min(LLx, axis=1)[:, np.newaxis])

        LLx = np.copy(np.concatenate((LL[0], LL[1]), axis=1))
        LL[0] = LL[0] / np.max(LLx, axis=1)[:, np.newaxis]
        LL[1] = LL[1] / np.max(LLx, axis=1)[:, np.newaxis]

    LLL = np.sum(abs(abs(LL[0]) - abs(LL[1])), axis=1)

    return L, LL, LLL

def plot_responseTimes(respTimes, path_to_save):
    p = np.percentile(respTimes, percentile)  # <- adjust here to change percentile classification

    bins = np.linspace(math.ceil(min(respTimes)), math.floor(max(respTimes)), 10)  # fixed number of bins
    plt.xlim([min(respTimes) - 0.5, max(respTimes) + 0.5])
    plt.hist(respTimes, color='c', edgecolor='k', alpha=0.65)
    plt.axvline(p, color='r')
    plt.savefig(path_to_save + "responseTimes.png")
    plt.clf()

def plot_top10(data, target_classes, contact_info, path_to_save):
    LL, LLL = calculate_abs_difference(data, target_classes)
	
    data_avg = np.mean(np.swapaxes(data, 0, 1), axis=1)

    fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(12, 16))
    a = 0
    for row in ax:
        for col in row:
            col.plot(LL[0][a], 'r', alpha=0.5)
            col.plot(LL[1][a], 'b', alpha=0.5)
            col.plot(data_avg[a], 'k', alpha=0.5)

            col.set_title("Electrode: " + str(contact_info['electrode'][a]) + ", Contact: " + str(contact_info['contact'][a]))
            a += 1

    filename = path_to_save + "/top10_contacts.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=800)
    plt.clf()

def plot_n_regions(data, target_classes, contact_info, path_to_save, n=1):
    LL, LLL = calculate_abs_difference(data, target_classes)

    data_avg = np.mean(np.swapaxes(data, 0, 1), axis=1)
    all_regions = contact_info['aal_label'].unique()
    region_vals = {region: {'value': 0, 'count': 0, 'indices': []} for region in all_regions}

    for index, diff in enumerate(LLL):
        region = contact_info['aal_label'][index]

        region_vals[region]['value'] += diff
        region_vals[region]['count'] += 1
        region_vals[region]['indices'].append(index)

    # Get the averages of each region, and sort regions from highest-lowest average. Then, get top region
    # region_avgs = []
    # for region in all_regions:
    #     region_avg = region_vals[region]['value'] / region_vals[region]['count']
    #
    #     print region + " - " + str(round(region_avg, 2))
    #
    #     region_avgs.append(region_avg)
    #
    # region_indices = np.asarray(region_avgs).argsort()[::-1]
    # all_regions = all_regions[region_indices]
    # top_region = all_regions[0]

    top_region_indices = []
    for region_num in range(n):
        top_region = all_regions[region_num]

        indices = region_vals[top_region]['indices']
        top_region_indices.append(indices)

        print("Top Region " + str(region_num+1) + ": " + top_region + " - " + str(indices))

        fig_cols = 2
        fig_rows = (len(indices)/2) + (len(indices)%2)
        fig, ax = plt.subplots(nrows=fig_rows, ncols=fig_cols, figsize=(16, 12*fig_rows/fig_cols))
        a = 0
        if fig_rows > 1 and fig_cols > 1:
            for row in ax:
                for col in row:
                    col.plot(LL[0][indices[a]], 'r', alpha=0.5)
                    col.plot(LL[1][indices[a]], 'b', alpha=0.5)
                    col.plot(data_avg[indices[a]], 'k', alpha=0.5)

                    col.set_title("Electrode: " + str(contact_info['electrode'][indices[a]]) + ", Contact: " + str(contact_info['contact'][indices[a]]))
                    a += 1

                    if a == len(indices):
                        break
        else:
            for col in ax:
                col.plot(LL[0][indices[a]], 'r', alpha=0.5)
                col.plot(LL[1][indices[a]], 'b', alpha=0.5)
                col.plot(data_avg[indices[a]], 'k', alpha=0.5)

                col.set_title("Electrode: " + str(contact_info['electrode'][indices[a]]) + ", Contact: " + str(contact_info['contact'][indices[a]]))
                a += 1

                if a == len(indices):
                    break

        filename = path_to_save + "/top_region" + str(region_num+1) + ".png"

        plt.suptitle(top_region)
        plt.savefig(filename, dpi=800)
        plt.clf()

    top_regions = all_regions[:n]
    #indices = np.asarray([region_vals[top_region]['indices'] for top_region in top_regions])
    print(top_region_indices)
    #indices = [item for sublist in indices for item in sublist]
    return top_regions, top_region_indices

def get_regions(path_to_load):
    labels = []
    ft_structure = h5py.File(path_to_load + "ft_structure.mat", 'r')
    for row in range(ft_structure['label'].shape[0]):
        obj = ft_structure[ft_structure['label'][row, 0]]
        curr_label = ''.join(chr(val) for val in obj[:])
        labels.append(curr_label)
    labels = np.asarray(labels)

    contact_mappings = pd.read_csv(path_to_load + "contact_mapping.csv")
    contacts = contact_mappings['contact'].values

    indices = [np.where(labels==elem)[0][0] for elem in contacts]

    contact_mappings['indices'] = indices
    contact_mappings = contact_mappings.sort_values(['indices'])
    contact_mappings = contact_mappings.reset_index(drop=True)

    regions = contact_mappings['aal_label'].values.astype(str)

    contact_info = contact_mappings[['electrode', 'contact', 'aal_label', 'ba_label', 'exclude']].astype(str)

    if len(contact_info) != len(labels):
        print("ERROR, missing contacts! Exiting now.")
        sys.exit(1)

    return contact_info, regions

def plot_spectrogram(data, path_to_save):
    f, t, Sxx = data
    print(f)
    print(t)
    print(Sxx)
    plt.pcolormesh(t, f, Sxx, cmap='Greys')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.savefig(path_to_save + "test1.png")

def get_spectrogram(data, fs, nperseg=100, noverlap=50, fs_cap=None):
    f, t, Sxx = signal.spectrogram(data, fs)#, nperseg=nperseg, noverlap=noverlap)
    fs_ind = None
    if fs_cap is not None:
        fs_ind = next(i for i, res in enumerate(f > fs_cap) if res) - 1
        #print('max frequency:', f[fs_ind])
    #print(f.shape)
    #print(t.shape)
    #print(Sxx.shape)
    return f[:fs_ind], t, Sxx[:fs_ind, :]

def load_data(path_to_load):
    dataset = h5py.File(path_to_load + "trl.mat", 'r')
    shifted_respTimes = dataset[dataset['misc']['behavstats_accepted'][0, 0]]['timeToCorrect'][0]
    unshifted_respTimes = dataset[dataset['misc']['behavstats_accepted'][1, 0]]['timeToCorrect'][0]
    shifts = dataset['trl'][3].astype(int)
    fs = dataset['misc']['Fs'][0, 0].astype(float)

    shift1_trial_index = 0
    shift2_trial_index = 0

    respTimes = []
    for i in range(shifts.shape[0]):
        shift = shifts[i]
        if shift == 1:
            respTime = shifted_respTimes[shift1_trial_index]
            shift1_trial_index += 1
        else:
            respTime = unshifted_respTimes[shift2_trial_index]
            shift2_trial_index += 1

        respTimes.append(respTime/fs)
    respTimes = np.array(respTimes)
    shifts = 2 - shifts

    #Get regions and contact info
    contact_info, regions = get_regions(path_to_load)

    return respTimes, shifts, contact_info, regions


respTimes, shifts, contact_info, regions = load_data(path_to_load)

#pd.set_option('display.max_rows', 1000)
#print contact_info['contact']
#sys.exit(1)

print(respTimes.shape)

X = []
X_full = []
X_spec = []
Y = []
X_diff = []

for i in range(respTimes.shape[0]):
    trialindex = "%03d" % (i + 1)
    f = h5py.File(path_to_load + "trialdata_" + trialindex + ".mat", 'r')
    A = f['trialdata'][:]
    A = np.transpose(A)

    # Extract FFT Parameters
    trialData = []
    trialData_full = []
    trialData_spec = []

    for n in range(A.shape[0]):
        A_n = A[n]
        R = A_n[1500:3000]

        R_avg = np.mean(R.reshape(-1, 15), axis=1)

        # Change range of data to be non-negative (>= 0)
        R_avg -= np.sign(np.amin(R_avg))*np.abs(np.amin(R_avg))
        A_n -= np.sign(np.amin(A_n[1250:3250]))*np.abs(np.amin(A_n[1250:3250]))

        #R_spec = get_spectrogram(A_n[1250:3250], 2048, fs_cap=100)
        f, t, Sxx = get_spectrogram(A_n[1250:3250], 2048, fs_cap=100)
        R_spec = Sxx

        trialData.append(R_avg)
        trialData_full.append(A_n)
        trialData_spec.append(R_spec)

    trialData_full = np.asarray(trialData_full)
    #trialData_full /= np.amax(trialData_full[:, 1250:3250])
    curr_diff = np.amax(trialData_full[:, 1000:3500]) - np.amin(trialData_full[:, 1000:3500])
    print(str(trialindex) + " -- " + str(round(np.amin(trialData_full[:, 1000:3500]), 2)) + " - " + str(round(np.amax(trialData_full[:, 1000:3500]), 2)) + " -- " + str(round(np.average(trialData_full[:, 1000:3500]), 2)) + " -- " + str(round(curr_diff, 2)) + " -- " + str(round(respTimes[i], 2)))
    X_diff.append(curr_diff)

    X.append(trialData)
    X_full.append(trialData_full)
    X_spec.append(trialData_spec)
    Y.append(respTimes[i])

    # print len(trialData_full)
    # fig = plt.figure()
    # plt.imshow(trialData_full)
    # plt.show()
    # sys.exit(1)

# Change into numpy arrays
X = np.asarray(X)
X_full = np.asarray(X_full)
X_spec = np.asarray(X_spec)
Y = np.asarray(Y)

#X_full /= np.amax(X_full[:, :, 1250:3250])

print("")
print(X.shape)
print(X_full.shape)
print(X_spec.shape)
print(Y.shape)
print(shifts.shape)
print("")

if not os.path.isdir(path_to_save):
    os.makedirs(path_to_save)

contact_info, regions, [X, X_full, X_spec] = remove_regions('nan', contact_info, regions, X_values=[X, X_full, X_spec])


range_min = 1000
range_max = 4500

for trial_index in range(X_full.shape[0]):
    curr_diff = np.amax(X_full[trial_index, :, range_min:range_max]) - np.amin(X_full[trial_index, :, range_min:range_max])
    print(str(trial_index) + " -- " + str(round(np.amin(X_full[trial_index, :, range_min:range_max]), 2)) + " - " + str(round(np.amax(X_full[trial_index, :, range_min:range_max]), 2)) + " -- " + str(round(np.average(X_full[trial_index, :, range_min:range_max]), 2)) + " -- " + str(round(curr_diff, 2)) + " -- " + str(round(respTimes[trial_index], 2)))
# print ""
# for trial_index in range(X_full.shape[0]):
#     X_full[trial_index] /= np.amax(X_full[:, :, 1250:3250])
#     curr_diff = np.amax(X_full[trial_index, :, 1250:3250]) - np.amin(X_full[trial_index, :, 1250:3250])
#     print str(trial_index) + " -- " + str(round(np.amin(X_full[trial_index, :, 1250:3250]), 2)) + " - " + str(round(np.amax(X_full[trial_index, :, 1250:3250]), 2)) + " -- " + str(round(np.average(X_full[trial_index, :, 1250:3250]), 2)) + " -- " + str(round(curr_diff, 2)) + " -- " + str(round(respTimes[trial_index], 2))

X_full /= np.amax(X_full[:, :, range_min:range_max])
Y_i, p = calculate_percentile(Y, percentile=percentile)
#_, contact_info, [X, X_full, X_spec] = sort_by_abs_difference(X_full[:, :, 1250:3250], contact_info, Y_i, additional_X=[X, X_full, X_spec])
_, contact_info, [X, X_full, X_spec] = sort_by_abs_difference(X_full[:, :, range_min:range_max], contact_info, Y_i, additional_X=[X, X_full, X_spec])

pd.set_option('display.max_rows', 500)
print(contact_info)

print(X_full.shape)
print(contact_info.values.shape)

print(contact_info['aal_label'].unique()[:5])

trial_num=0
fig, axs = plt.subplots(15, 25, sharex='all')
for row in range(15):
    for col in range(25):
        axs[row, col].plot(X_full[trial_num, 0, range_min:range_max])
        axs[row, col].tick_params(labelsize=4)
        trial_num += 1
        if trial_num >= X_full.shape[0]:
            break
    if trial_num >= X_full.shape[0]:
        break
#plt.show()
plt.savefig(path_to_save + "all_images.png", dpi=800)


top_regions, top_region_indices = plot_n_regions(X_full[:, :, range_min:range_max], Y_i, contact_info, path_to_save, n=num_regions)
plot_top10(X_full[:, :, range_min:range_max], Y_i, contact_info, path_to_save)
plot_responseTimes(Y, path_to_save)

h5py_file = h5py.File(path_to_save + "processed_data.h5", 'w')
h5py_file.create_dataset('data', data=X)
h5py_file.create_dataset('data_full', data=X_full)
h5py_file.create_dataset('data_spec', data=X_spec)
h5py_file.create_dataset('respTimes', data=Y)
h5py_file.create_dataset('shifts', data=shifts)

h5py_file.create_dataset('contact_info', data=contact_info.values.astype(str))
h5py_file.create_dataset('contact_info_headers', data=contact_info.columns.values.astype(str))

h5py_file.create_dataset('regions', data=regions)
h5py_file.create_dataset('num_regions', data=num_regions)
h5py_file.create_dataset('top_regions', data=np.asarray(top_regions, dtype='str'))
for i in range(num_regions):
    h5py_file.create_dataset('top_region_indices_' + str(i), data=top_region_indices[i])
h5py_file.close()
