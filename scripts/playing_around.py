from scipy import signal

from scripts.configs import *
from scripts.eval_utils import *
from scripts.data_utils import *

[X_full1, _], respTimes, num_regions, top_regions, top_region_indices = load_data(path_to_load)

Y, y_min, y_max = normalize_output(respTimes)

# NOTE: see how data spectogram looks like at different response time groups

def get_spectrogram(data, fs, nperseg=100, noverlap=50, fs_cap=None):
    f, t, Sxx = signal.spectrogram(data, fs)  # , nperseg=nperseg, noverlap=noverlap)
    fs_ind = None
    if fs_cap is not None:
        fs_ind = next(i for i, res in enumerate(f > fs_cap) if res) - 1
        # print('max frequency:', f[fs_ind])
    # print(f.shape)
    # print(t.shape)
    # print(Sxx.shape)
    return f[:fs_ind], t, Sxx[:, :, :fs_ind, :]

X_full1 = X_full1[:,:,100:4600]

# f, t, dats = get_spectrogram(X_full1, 2048, fs_cap=100)

f, t, dats = signal.spectrogram(X_full1, 2048, nperseg=256, noverlap=128)

plt.close('all')
plt.plot(X_full1[0,0,:])
plt.savefig(r'Z:\tempytempyeeg\data\SEEG-SK-04\eeg_orig.jpg', dpi=300)


plt.close('all')
plt.pcolormesh(t, f, dats[0,0,:,:], cmap='Greys')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.savefig(r'Z:\tempytempyeeg\data\SEEG-SK-04\specto_raw.jpg', dpi=300)



plt.close('all')
plt.pcolormesh(t, f, (10 * np.log10(dats))[0,0,:,:], cmap='Greys')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.savefig(r'Z:\tempytempyeeg\data\SEEG-SK-04\specto_scaled.jpg', dpi=300)



# NOTE: making a spectogram plot for each of the contacts (trials are averaged)
ranges = np.arange(0, Y.max() + np.finfo(float).eps, Y.max() / 5)
for i in range(1, len(ranges)):

    temp = dats[(ranges[i-1] <= Y) & (Y <= ranges[i])].mean(axis=0)
    for contact_i in range(len(temp)):
        plt.pcolormesh(t, f, temp[contact_i], cmap='Greys')
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (s)')
        # plt.show()
        sp = r'Z:\tempytempyeeg\results\SEEG-SK-04\specto_on_raw_dat_c{}_{:.2f}_to_{:.2f}.jpg'
        plt.savefig(sp.format(contact_i, ranges[i-1], ranges[i]))

# NOTE: using log scaling so that colormap is actually visible
loggeddats = 10 * np.log10(dats)
for i in range(1, len(ranges)):
    temp = loggeddats[(ranges[i-1] <= Y) & (Y <= ranges[i])].mean(axis=0)
    contact_i = 0
    plt.close('all')
    mesh = plt.pcolormesh(t, f, temp[contact_i], cmap='Greys')
    plt.colorbar(mesh)
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    # plt.show()
    sp = r'Z:\tempytempyeeg\results\SEEG-SK-04\TEMPYspecto_on_raw_dat_c{}_{:.2f}_to_{:.2f}.jpg'
    plt.savefig(sp.format(contact_i, ranges[i-1], ranges[i]))

    # for making data 100-4600 and average over every 5
    avgd_dats = np.mean(X_full1[:, :, 100:4600].reshape((365, 95, 900, 5)), axis=-1)
    f, t, avgd_dats = get_spectrogram(avgd_dats, 2048, fs_cap=100)

    ranges = np.arange(0, Y.max() + np.finfo(float).eps, Y.max() / 5)
    for i in range(1, len(ranges)):

        temp = avgd_dats[(ranges[i-1] <= Y) & (Y <= ranges[i])].mean(axis=0)
        for contact_i in range(len(temp)):
            plt.pcolormesh(t, f, temp[contact_i], cmap='Greys')
            plt.ylabel('Frequency (Hz)')
            plt.xlabel('Time (s)')
            # plt.show()
            sp = r'Z:\tempytempyeeg\results\SEEG-SK-04\specto_5avg_dat_c{}_{:.2f}_to_{:.2f}.jpg'
            plt.savefig(sp.format(contact_i, ranges[i-1], ranges[i]))



