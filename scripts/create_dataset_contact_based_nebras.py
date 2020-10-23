## This script creates the data files that we can then run the regression ML on, in H5PY format

import sys
import socket
import matplotlib
from scipy import signal

matplotlib.use('Agg')

from scripts.utils import *

machine_hostname = socket.gethostname()

path = r'Z:\ml'

try:
    subject_name = sys.argv[1]
except:
    # local debugging parameters
    subject_name = 'SEEG-SK-04'

path_to_load = os.path.join(path, subject_name, 'raw')
path_to_save = os.path.join(path, subject_name, 'processed')

percentile = 75


# called in load_data
def get_contact_mapping(path_to_load):
    # This is the order of the contacts for each contact
    labels = []
    ft_structure = h5py.File(os.path.join(path_to_load, "ft_structure.mat"), 'r')

    # NOTE: in the old file, we had it as (num_contacts, 1). but now it is (1, num_contacts)
    for row in range(ft_structure['label'].shape[1]):
        obj = ft_structure[ft_structure['label'][0, row]]
        curr_label = ''.join(chr(val) for val in obj[:])
        labels.append(curr_label)
    labels = np.asarray(labels)

    # NOTE. STEP NOT NEEDED/PERFORMED: Want to make the corresponding info mapping file in the same order. The below will do the sorting
    contact_mappings = pd.read_csv(os.path.join(path_to_load, "contact_mapping.csv"))
    contacts = contact_mappings['contact'].values
    contact_info = contact_mappings[['electrode', 'contact', 'aal_label', 'ho_label', 'exclude']].astype(str)
    return contact_info


def load_data(path_to_load):
    dataset = h5py.File(os.path.join(path_to_load, "trl.mat"), 'r')
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

        respTimes.append(respTime / fs)
    respTimes = np.array(respTimes)
    shifts = 2 - shifts

    # Get contact info, ordered alongside the mat file format
    contact_info = get_contact_mapping(path_to_load)

    # Here we can now return an array of the response times, the contacts involved, and whether the trial was shift or non shift
    return shifted_respTimes, unshifted_respTimes, respTimes, shifts, contact_info


# Main body of the conversion
shifted_respTimes, unshifted_respTimes, respTimes, shifts, contact_info = load_data(path_to_load)
print(respTimes.shape)

X_full = []
Y = []

for i in range(respTimes.shape[0]):
    trialindex = "%03d" % (i + 1)
    f = h5py.File(os.path.join(path_to_load, "trialdata_" + trialindex + ".mat"), 'r')
    # Holds the trial data for each contact over the 4096 timepoints (num_contacts x time)
    trialData_full = f['trialdata'][:]
    trialData_full = np.transpose(trialData_full)
    X_full.append(trialData_full)
    Y.append(respTimes[i])

    # print(len(trialData_full))
    # fig = plt.figure()
    # plt.imshow(trialData_full)
    # plt.savefig('/Users/nebras/Desktop/hi.png')
    # sys.exit(1)

# Change into numpy arrays
# X_full is a 3D array formatted as follows: [Trials, contacts, timeseries]
X_full = np.asarray(X_full)
Y = np.asarray(Y)

print("")
print(X_full.shape)
print(Y.shape)
print(shifts.shape)
print("")

if not os.path.isdir(path_to_save):
    os.makedirs(path_to_save)

# Essentially removes regions that don't have a label, probably not necessary as EEG pipeline does a lot of this already
# Contact_info, [X_full] = remove_regions('nan', contact_info, X_values=[X_full])

# Gives some padding to data
range_min = 50
range_max = 4025

# Loops over the trials and prints the descriptive info about all of them to screen
for trial_index in range(X_full.shape[0]):
    print(str(trial_index) + " -- " + str(round(respTimes[trial_index], 2)) + " -- " + str(shifts[trial_index]))

Y_i, p = calculate_percentile(Y, percentile=percentile)

# Now we are ready to split shift and non shift trials

X_shift = []
Y_shift = []
X_nonshift = []
Y_nonshift = []

shift_idx = 0
for is_shift in shifts:
    if is_shift:
        X_shift.append(X_full[shift_idx, :, :])
        Y_shift.append(Y_i[shift_idx])
    else:
        X_nonshift.append(X_full[shift_idx, :, :])
        Y_nonshift.append(Y_i[shift_idx])
    shift_idx += 1

X_shift = np.asarray(X_shift)
Y_shift = np.asarray(Y_shift)
X_nonshift = np.asarray(X_nonshift)
Y_nonshift = np.asarray(Y_nonshift)

print(X_shift.shape)
print(X_nonshift.shape)
print(Y_shift.shape)
print(Y_nonshift.shape)

# This code performs the STFT transformations and plots the spectrograms
Fs = 2048
shift_speed = np.median(shifted_respTimes) / Fs
non_shift_speed = np.median(unshifted_respTimes) / Fs
print("Starting STFT...")
for contact_i in range(len(X_full[1])):
    # Store the STFT's
    contact_stft_shift = []
    contact_stft_shift_slow = []
    contact_stft_shift_fast = []

    contact_stft_nonshift = []
    contact_stft_nonshift_slow = []
    contact_stft_nonshift_fast = []

    for trial in range(X_full.shape[0]):
        f, t, dataSTFT = signal.stft(X_full[trial, contact_i, :], fs=Fs, nperseg=2048, noverlap=1800)
        dataSTFT = np.asarray(dataSTFT)
        if shifts[trial]:
            contact_stft_shift.append(dataSTFT)
            if respTimes[trial] < shift_speed:
                contact_stft_shift_fast.append(dataSTFT)
            else:
                contact_stft_shift_slow.append(dataSTFT)
        else:
            contact_stft_nonshift.append(dataSTFT)
            if respTimes[trial] < non_shift_speed:
                contact_stft_nonshift_fast.append(dataSTFT)
            else:
                contact_stft_nonshift_slow.append(dataSTFT)

    # Takes the average spectral values across shift and non-shift trials for the given contact, then does fast vs slow

    contact_dir = str('/Users/nebras/Desktop/figs/contact' + str(contact_i + 1) + '/stft/')

    if not os.path.isdir(contact_dir):
        os.makedirs(contact_dir)

    # Shifts
    average_shift = np.mean(contact_stft_shift, axis=0)
    average_shift = np.abs(average_shift)
    plt.pcolormesh(t, f, average_shift, shading='gouraud')
    plt.ylim(0, 100)
    plt.title('Shifts for contact: ' + str(contact_info['contact'][contact_i]) + '\n AAL Region: ' + str(contact_info['aal_label'][contact_i]) + '\n HO Region: ' + str(contact_info['ho_label'][contact_i]))
    figname = contact_dir + 'all_shift.png'
    plt.savefig(figname, dpi=300)
    plt.close()

    # Nonshifts
    average_nonshift = np.mean(contact_stft_nonshift, axis=0)
    average_nonshift = np.abs(average_nonshift)
    plt.pcolormesh(t, f, average_nonshift, shading='gouraud')
    plt.ylim(0, 100)
    plt.title('Non-shifts for contact: ' + str(contact_info['contact'][contact_i]) + '\n AAL Region: ' + str(contact_info['aal_label'][contact_i]) + '\n HO Region: ' + str(contact_info['ho_label'][contact_i]))
    figname = contact_dir + 'all_nonshift.png'
    plt.savefig(figname, dpi=300)
    plt.close()

    # Fast Shifts
    average_shift_fast = np.mean(contact_stft_shift_fast, axis=0)
    average_shift_fast = np.abs(average_shift_fast)
    plt.pcolormesh(t, f, average_shift_fast, shading='gouraud')
    plt.ylim(0, 100)
    plt.title('Fast shifts for contact: ' + str(contact_info['contact'][contact_i]) + '\n AAL Region: ' + str(contact_info['aal_label'][contact_i]) + '\n HO Region: ' + str(contact_info['ho_label'][contact_i]))
    figname = contact_dir + 'fast_shift.png'
    plt.savefig(figname, dpi=300)
    plt.close()

    # Slow Shifts
    average_shift_slow = np.mean(contact_stft_shift_slow, axis=0)
    average_shift_slow = np.abs(average_shift_slow)
    plt.pcolormesh(t, f, average_shift_slow, shading='gouraud')
    plt.ylim(0, 100)
    plt.title('Slow shifts for contact: ' + str(contact_info['contact'][contact_i]) + '\n AAL Region: ' + str(contact_info['aal_label'][contact_i]) + '\n HO Region: ' + str(contact_info['ho_label'][contact_i]))
    figname = contact_dir + 'slow_shift.png'
    plt.savefig(figname, dpi=300)
    plt.close()

    # Fast Non-Shifts
    average_nonshift_fast = np.mean(contact_stft_nonshift_fast, axis=0)
    average_nonshift_fast = np.abs(average_nonshift_fast)
    plt.pcolormesh(t, f, average_nonshift_fast, shading='gouraud')
    plt.ylim(0, 100)
    plt.title('Fast nonshifts for contact: ' + str(contact_info['contact'][contact_i]) + '\n AAL Region: ' + str(contact_info['aal_label'][contact_i]) + '\n HO Region: ' + str(contact_info['ho_label'][contact_i]))
    figname = contact_dir + 'fast_nonshift.png'
    plt.savefig(figname, dpi=300)
    plt.close()

    # Slow Non-Shifts
    average_nonshift_slow = np.mean(contact_stft_nonshift_slow, axis=0)
    average_nonshift_slow = np.abs(average_nonshift_slow)
    plt.pcolormesh(t, f, average_nonshift_slow, shading='gouraud')
    plt.ylim(0, 100)
    plt.title('Slow nonshifts for contact: ' + str(contact_info['contact'][contact_i]) + '\n AAL Region: ' + str(contact_info['aal_label'][contact_i]) + '\n HO Region: ' + str(contact_info['ho_label'][contact_i]))
    figname = contact_dir + 'slow_nonshift.png'
    plt.savefig(figname, dpi=300)
    plt.close()

print("Done STFT")
# CWT Parameters
t = np.arange(0, 2, 1 / Fs)
w = 7
freqs = np.linspace(0.5, 100, 200)
widths = w * Fs / (2 * freqs * np.pi)

# This function will output the continuous wavelet transforms of the data for shift and non-shift trials
for contact_i in range(len(X_full[1])):
    # Store the data
    contact_cwt_shift = []
    contact_cwt_shift_slow = []
    contact_cwt_shift_fast = []

    contact_cwt_nonshift = []
    contact_cwt_nonshift_slow = []
    contact_cwt_nonshift_fast = []

    print('starting contact ' + str(contact_i + 1))

    for trial in range(X_full.shape[0]):
        dataCWT = signal.cwt(X_full[trial, contact_i, :], signal.morlet2, widths)
        dataCWT = np.asarray(dataCWT)
        if shifts[trial]:
            contact_cwt_shift.append(dataCWT)
            if respTimes[trial] < shift_speed:
                contact_cwt_shift_fast.append(dataCWT)
            else:
                contact_cwt_shift_slow.append(dataCWT)
        else:
            contact_cwt_nonshift.append(dataCWT)
            if respTimes[trial] < shift_speed:
                contact_cwt_nonshift_fast.append(dataCWT)
            else:
                contact_cwt_nonshift_slow.append(dataCWT)

    print('Graphing...')

    # Takes the average values across shift and non-shift trials for the given contact
    contact_dir = str('/Users/nebras/Desktop/figs/contact' + str(contact_i + 1) + '/cwt/')

    if not os.path.isdir(contact_dir):
        os.makedirs(contact_dir)

    # Shifts
    average_shift = np.mean(contact_cwt_shift, axis=0)
    average_shift = np.abs(average_shift)
    plt.pcolormesh(t, freqs, average_shift, shading='gouraud')
    figname = (contact_dir + 'all_shift.png')
    plt.title('Shifts for contact: ' + str(contact_info['contact'][contact_i]) + '\n AAL Region: ' + str(contact_info['aal_label'][contact_i]) + '\n HO Region: ' + str(contact_info['ho_label'][contact_i]))
    plt.savefig(figname, dpi=300)
    plt.close()

    # Nonshifts
    average_nonshift = np.mean(contact_cwt_nonshift, axis=0)
    average_nonshift = np.abs(average_nonshift)
    plt.pcolormesh(t, freqs, average_nonshift, shading='gouraud')
    figname = (contact_dir + 'all_nonshift.png')
    plt.title('Non-shifts for contact: ' + str(contact_info['contact'][contact_i]) + '\n AAL Region: ' + str(contact_info['aal_label'][contact_i]) + '\n HO Region: ' + str(contact_info['ho_label'][contact_i]))
    plt.savefig(figname, dpi=300)
    plt.close()

    # Fast shifts
    average_shift_fast = np.mean(contact_cwt_shift_fast, axis=0)
    average_shift_fast = np.abs(average_shift_fast)
    plt.pcolormesh(t, freqs, average_shift_fast, shading='gouraud')
    figname = (contact_dir + 'fast_shift.png')
    plt.title('Fast shifts for contact: ' + str(contact_info['contact'][contact_i]) + '\n AAL Region: ' + str(contact_info['aal_label'][contact_i]) + '\n HO Region: ' + str(contact_info['ho_label'][contact_i]))
    plt.savefig(figname, dpi=300)
    plt.close()

    # Slow Shifts
    average_shift_slow = np.mean(contact_cwt_shift_slow, axis=0)
    average_shift_slow = np.abs(average_shift_slow)
    plt.pcolormesh(t, freqs, average_shift_slow, shading='gouraud')
    figname = (contact_dir + 'slow_shift.png')
    plt.title('Slow shifts for contact: ' + str(contact_info['contact'][contact_i]) + '\n AAL Region: ' + str(contact_info['aal_label'][contact_i]) + '\n HO Region: ' + str(contact_info['ho_label'][contact_i]))
    plt.savefig(figname, dpi=300)
    plt.close()

    # Fast non-shifts
    average_nonshift_fast = np.mean(contact_cwt_nonshift_fast, axis=0)
    average_nonshift_fast = np.abs(average_nonshift_fast)
    plt.pcolormesh(t, freqs, average_nonshift_fast, shading='gouraud')
    figname = (contact_dir + 'fast_nonshift.png')
    plt.title('Fast non-shifts for contact: ' + str(contact_info['contact'][contact_i]) + '\n AAL Region: ' + str(contact_info['aal_label'][contact_i]) + '\n HO Region: ' + str(contact_info['ho_label'][contact_i]))
    plt.savefig(figname, dpi=300)
    plt.close()

    # Slow non-shifts
    average_nonshift_slow = np.mean(contact_cwt_nonshift_slow, axis=0)
    average_nonshift_slow = np.abs(average_nonshift_slow)
    plt.pcolormesh(t, freqs, average_nonshift_slow, shading='gouraud')
    figname = (contact_dir + 'slow_nonshift.png')
    plt.title('Slow non-shifts for contact: ' + str(contact_info['contact'][contact_i]) + '\n AAL Region: ' + str(contact_info['aal_label'][contact_i]) + '\n HO Region: ' + str(contact_info['ho_label'][contact_i]))
    plt.savefig(figname, dpi=300)
    plt.close()

# Now we will save our new model output to the H5PY file
h5py_file = h5py.File(os.path.join(path_to_save, "processed_data.h5"), 'w')
h5py_file.create_dataset('data_full', data=X_full)
h5py_file.create_dataset('respTimes', data=Y)
h5py_file.create_dataset('shifts', data=shifts)

# h5py_file.attrs['electrodes'] = contact_info['electrode'].to_list()
h5py_file.attrs['contacts'] = contact_info['contact'].to_list()
h5py_file.attrs['regions'] = contact_info['ho_label'].to_list()
h5py_file.attrs['regions'] = contact_info['aal_label'].to_list()  # Can use this to change between HO and AAL labels PRN
h5py_file.close()