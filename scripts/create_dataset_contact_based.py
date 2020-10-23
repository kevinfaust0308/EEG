import sys
import socket
import matplotlib

matplotlib.use("Agg")

from scripts.utils import *

machine_hostname = socket.gethostname()

if machine_hostname == 'mpc':
    path = "/hpf/largeprojects/ccm_home/masgouri/eeg/data/"
else:
    path = r"Z:\tempytempyeeg\data"

try:
    filename = sys.argv[2]
except:
    # local debugging parameters
    filename = 'SEEG-SK-04'

path_to_load = os.path.join(path, filename, 'raw')
path_to_save = os.path.join(path, filename, 'processed')

num_regions = 10
percentile = 75


# NOTE: called in load_data
def get_contact_mapping(path_to_load):
    # This is the order of the contacts for each contact
    labels = []
    ft_structure = h5py.File(os.path.join(path_to_load, "ft_structure.mat"), 'r')
    for row in range(ft_structure['label'].shape[0]):
        obj = ft_structure[ft_structure['label'][row, 0]]
        curr_label = ''.join(chr(val) for val in obj[:])
        labels.append(curr_label)
    labels = np.asarray(labels)

    # Want to make the corresponding info mapping file in the same order. The below will do the sorting
    contact_mappings = pd.read_csv(os.path.join(path_to_load, "contact_mapping.csv"))
    contacts = contact_mappings['contact'].values

    indices = [np.where(labels == elem)[0][0] for elem in contacts]

    contact_mappings['indices'] = indices
    contact_mappings = contact_mappings.sort_values(['indices'])
    contact_mappings = contact_mappings.reset_index(drop=True)

    contact_info = contact_mappings[['electrode', 'contact', 'aal_label', 'exclude']].astype(str)

    if len(contact_info) != len(labels):
        print("ERROR, missing contacts! Exiting now.")
        sys.exit(1)

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

    return respTimes, shifts, contact_info


respTimes, shifts, contact_info = load_data(path_to_load)
print(respTimes.shape)

X_full = []
Y = []

for i in range(respTimes.shape[0]):
    trialindex = "%03d" % (i + 1)
    f = h5py.File(os.path.join(path_to_load, "trialdata_" + trialindex + ".mat"), 'r')
    A = f['trialdata'][:]
    A = np.transpose(A)

    # Extract FFT Parameters
    trialData_full = []

    for n in range(A.shape[0]):
        A_n = A[n]
        R = A_n[1500:3000]

        R_avg = np.mean(R.reshape(-1, 15), axis=1)

        # Change range of data to be non-negative (>= 0)
        R_avg -= np.sign(np.amin(R_avg)) * np.abs(np.amin(R_avg))
        A_n -= np.sign(np.amin(A_n[1250:3250])) * np.abs(np.amin(A_n[1250:3250]))

        trialData_full.append(A_n)

    trialData_full = np.asarray(trialData_full)
    # trialData_full /= np.amax(trialData_full[:, 1250:3250])
    curr_diff = np.amax(trialData_full[:, 1000:3500]) - np.amin(trialData_full[:, 1000:3500])
    print(str(trialindex) + " -- " + str(round(np.amin(trialData_full[:, 1000:3500]), 2)) + " - " + str(round(np.amax(trialData_full[:, 1000:3500]), 2)) + " -- " + str(round(np.average(trialData_full[:, 1000:3500]), 2)) + " -- " + str(round(curr_diff, 2)) + " -- " + str(round(respTimes[i], 2)))

    X_full.append(trialData_full)
    Y.append(respTimes[i])

    # print len(trialData_full)
    # fig = plt.figure()
    # plt.imshow(trialData_full)
    # plt.show()
    # sys.exit(1)

# Change into numpy arrays
X_full = np.asarray(X_full)
Y = np.asarray(Y)

# X_full /= np.amax(X_full[:, :, 1250:3250])

print("")
print(X_full.shape)
print(Y.shape)
print(shifts.shape)
print("")

if not os.path.isdir(path_to_save):
    os.makedirs(path_to_save)

contact_info, [X_full] = remove_regions('nan', contact_info, X_values=[X_full])

range_min = 1000
range_max = 4500

for trial_index in range(X_full.shape[0]):
    curr_diff = np.amax(X_full[trial_index, :, range_min:range_max]) - np.amin(X_full[trial_index, :, range_min:range_max])
    print(str(trial_index) + " -- " + str(round(np.amin(X_full[trial_index, :, range_min:range_max]), 2)) + " - " + str(round(np.amax(X_full[trial_index, :, range_min:range_max]), 2)) + " -- " + str(round(np.average(X_full[trial_index, :, range_min:range_max]), 2)) + " -- " + str(
        round(curr_diff, 2)) + " -- " + str(round(respTimes[trial_index], 2)))

X_full /= np.amax(X_full[:, :, range_min:range_max])
Y_i, p = calculate_percentile(Y, percentile=percentile)
# Only doing the differencing sorting on a subset of the data.
# At this point, X_full is sorted by contacts with most change
contact_info, [X_full] = sort_by_abs_difference(X_full[:, :, range_min:range_max], contact_info, additional_X=[X_full])

# NOTE: ALL THE MAIN STUFF IS DONE AT THIS POINT. THE REST IS LIKE VISUALIZATION OR GETTING TOP REGIONS (DOING DIFFERENT APPROACH ATM)

if 1:
    h5py_file = h5py.File(os.path.join(path_to_save, "processed_data.h5"), 'w')
    h5py_file.create_dataset('data_full', data=X_full)
    h5py_file.create_dataset('respTimes', data=Y)
    h5py_file.create_dataset('shifts', data=shifts)

    # h5py_file.attrs['electrodes'] = contact_info['electrode'].to_list()
    h5py_file.attrs['contacts'] = contact_info['contact'].to_list()
    h5py_file.attrs['regions'] = contact_info['aal_label'].to_list()

    h5py_file.close()
