import warnings
import os

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import keras
from keras.callbacks import EarlyStopping
from keras import backend as K

K.clear_session()

import tensorflow as tf

# TODO: since i dont have GPU on my local computer
# config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': 15})
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)
# keras.backend.set_session(sess)

from sklearn.metrics import accuracy_score, recall_score, precision_score, mean_squared_error
from sklearn.metrics import roc_auc_score, confusion_matrix

from scipy.stats import spearmanr
import pickle
import socket
import csv
import sys
import pathlib
from scipy import signal

from scripts.utils import *

machine_hostname = socket.gethostname()

if machine_hostname == 'mpc':
    path = "/hpf/largeprojects/ccm_home/masgouri/eeg"
    batch_size_div = 10

elif machine_hostname == 'saman':
    path = '/home/saman/Desktop/Git/closed-loop-scripts'
    batch_size_div = 5

elif machine_hostname == 'pdiamand-om01.local':
    path = '/Users/labuser/PycharmProjects/EEG'
    batch_size_div = 10

else:
    path = r"Z:/tempytempyeeg"
    batch_size_div = 5

try:
    run_id = sys.argv[1]
    filename = sys.argv[2]
    is_final_run = sys.argv[3]
except:
    # local debugging parameters
    run_id = 1
    filename = 'SEEG-SK-04'
    is_final_run = 1

path_to_load = os.path.join(path, 'data', filename, 'processed')
path_to_save = os.path.join(path, 'results', filename)

num_batches = 5  # Number of batches
num_classes = 1  # Number of outcome classes


if MODE_FREQUENCY:
    data_avg_points = 1  # How many datapoints to average over
    data_range = (100, 10100)
else:
    data_avg_points = 5
    data_range = (100, 4600)  # Range of data to select from full set

max_regions_limit = 2  # Upper limit of how many regions to select for run
final_run = int(is_final_run)  # If 0, then validate against validation datset. If 1, search through logs to find best-performing model, load those hyperparameters for training, then test on test dataset

# whether to cap the frequency data. None means don't
fs_cap = 100
fs_cap = None

for dir in ['plots', 'losses', 'attn_maps']:
    pathlib.Path(os.path.join(path_to_save, dir)).mkdir(parents=True, exist_ok=True)

# region_indices: the indexes of the contacts belonging to a region
# num_regions/top_regions: the #/names of the top x regions we can choose from (selected in create_dataset.py)


[X_full1, _], respTimes, num_regions, top_regions, top_region_indices = load_data(path_to_load)

Y, y_min, y_max = normalize_output(respTimes)


region_nums_to_use = []

if final_run:
    run_id = get_best_model(os.path.join(path_to_save, 'regression_results_topn_svrs.csv'))
    print(run_id)
    print("Loading hyperparameters from best model...")
    percentile, model_num, learning_rate, dropout_rate, epochs, loss, _, L1_units, L2_units, L3_units, L4_units, regions_to_use = \
        load_hyperparameters(os.path.join(path_to_save, 'regression_results_topn.csv'), run_id)

    for region in regions_to_use:
        region_nums_to_use.append(np.where(top_regions == region)[0].tolist()[0])

    run_id *= 1000

else:
    percentile, model_num, learning_rate, dropout_rate, epochs, loss, _, L1_units, L2_units, L3_units, L4_units = \
        generate_hyperparameters()

    region_nums_to_use = None

    # TODO: default is 256 and 256//2
    NPERSEG_RANDOM = random.choice([64, 128, 256, 512, 1024])
    # NOVERLAP_RANDOM = random.choice([NPERSEG_RANDOM // 2, NPERSEG_RANDOM // 4, NPERSEG_RANDOM // 8])
    NOVERLAP_RANDOM = NPERSEG_RANDOM // 2



# TODO:
NPERSEG_RANDOM = 256
NOVERLAP_RANDOM = NPERSEG_RANDOM // 2



[X_full], regions_to_use, num_graphs, region_nums_to_use, indices_to_use = select_regions(num_regions, max_regions_limit, top_regions, top_region_indices, data=[X_full1], region_nums_to_use=region_nums_to_use)

if final_run:
    epochs[0] = epochs[1] = epochs[2] = epochs[3] = epochs[4] = int(np.mean(epochs))

print('hyperparameters used:')
print('percentile: ' + str(percentile))
print('learning rate: ' + str(learning_rate))
print('dropout rate: ' + str(dropout_rate))
print('epochs: ' + str(epochs))
print('loss: ' + str(loss))
print('L1, L2, L3 : ' + str(L1_units) + ', ' + str(L2_units) + ', ' + str(L3_units))

print(top_region_indices)
print(region_nums_to_use)
print(indices_to_use)
print(regions_to_use)
print(X_full.shape)
print(Y.shape)
print(respTimes.shape)

epochs_es = []
targets = []
preds = []
mses = []

preds_train = []
targets_train = []

B = int(round(X_full.shape[0] / num_batches))
for step in range(0, num_batches):

    trainData, trainTarget, trainTarget_p, \
    validData, validTarget, \
    testData, testTarget = \
        calculate_batch(X_full, Y, B, step, num_batches, data_range, data_avg_points, percentile, validation=not final_run, test=True)

    # Augment training dataset
    if not MODE_FREQUENCY:
        # Add random noise
        data, target = augment_dataset_fft(trainData, trainTarget)

    # After shifting, we will already have a subset of our desired data. So can't do any shifting on it.
    # Have to small+large shift on original data and then get rid of any duplicates.
    # (Since when we do shifting, we will have original data + augmented shifted data)

    # Small data shift
    trainDataSS, trainTargetSS = augment_dataset_shift(
        trainData, trainTarget, np.percentile(trainTarget, percentile), data_range, data_avg_points, shift_by=5, left_shifts=2, right_shifts=2)
    # Large data shift (Response time is adjusted)
    trainDataLS, trainTargetLS = augment_dataset_shift(
        trainData, trainTarget, np.percentile(trainTarget, percentile), data_range, data_avg_points, shift_by=100, response_shift=True, left_shifts=1, right_shifts=3, y_min=y_min, y_max=y_max)
    # Combine the datas and get the indices of the unique rows
    trainData, unique_indices = np.unique(np.vstack([trainDataSS, trainDataLS]), axis=0, return_index=True)
    # Get the corresponding unique responses
    trainTarget = np.concatenate([trainTargetSS, trainTargetLS])[unique_indices]

    ###

    # Upsample each bin in training dataset to get an even-distribution across bins
    trainData, trainTarget = upsample_bins(trainData, trainTarget, num_batches)

    ### TODO:
    if MODE_FREQUENCY:

        # # NOTE: a check for testing which combinations would work for inverting...result: all the /4 and /8 fail
        # NPERSEG_RANDOMS = [128, 256, 512, 1024, 2048]
        # # NOVERLAP_RANDOMS = [NPERSEG_RANDOM // 2, NPERSEG_RANDOM // 4, NPERSEG_RANDOM // 8]
        # for NPERSEG_RANDOM in NPERSEG_RANDOMS:
        #     for NOVERLAP_RANDOM in [NPERSEG_RANDOM // 2, NPERSEG_RANDOM // 4, NPERSEG_RANDOM // 8]:
        #         try:
        #             _, _, temp = signal.stft(trainData, 2048, nperseg=NPERSEG_RANDOM, noverlap=NOVERLAP_RANDOM)
        #             signal.istft(temp, 2048, nperseg=NPERSEG_RANDOM, noverlap=NOVERLAP_RANDOM)
        #         except:
        #             print('FAILED NOLA: ', NPERSEG_RANDOM, NOVERLAP_RANDOM, NPERSEG_RANDOM / NOVERLAP_RANDOM)





        f, t, trainDataSTFT = signal.stft(trainData, 2048, nperseg=NPERSEG_RANDOM, noverlap=NOVERLAP_RANDOM)
        fs_ind = next(i for i, res in enumerate(f > fs_cap) if res) if fs_cap else None
        trainDataSTFT = trainDataSTFT[:, :, :fs_ind, :]

        if 0:
            trainDataISTFT = signal.istft(trainDataSTFT, 2048)  # we only using up to 100 frequency so its not a pure reconstruct but very similar

            # example of reverted back to signal
            plt.plot(trainDataISTFT[1][0][0])
            plt.show()

            upsampled = signal.resample(trainDataISTFT[1][0][0], 4500)
            plt.plot(upsampled)
            plt.show()

            import cv2
            resized = cv2.resize(reversee[1][0][0], (4500,1))
            plt.plot(resized[0])
            plt.show()

        trainData = trainDataSTFT
        # trainData = 10 * np.log10(signal.spectrogram(trainData, 2048)[2])


    ###

    # Standardize the dataset
    train_X_mean = np.mean(trainData)
    train_X_std = np.std(trainData)
    trainData -= train_X_mean
    trainData /= train_X_std
    trainData = np.expand_dims(trainData, axis=-1)

    print('train data shape:', trainData.shape)

    input_shape = trainData.shape[1:]
    model, model_name = create_model(input_shape, dropout_rate, num_classes, L1_units, L2_units, L3_units, activation_function='sigmoid')

    # NOTE: Initial weight file is in root directory for now
    # model.load_weights(os.path.join(path, 'init_weights.h5'))

    opt = keras.optimizers.RMSprop(lr=learning_rate)
    model.compile(loss=loss, optimizer=opt)

    if not final_run:
        # Find the hyperparameters which perform best on validation data

        # TODO:
        if MODE_FREQUENCY:

            f, t, validDataFreq = signal.stft(validData, 2048, nperseg=NPERSEG_RANDOM, noverlap=NOVERLAP_RANDOM)
            fs_ind = next(i for i, res in enumerate(f > fs_cap) if res) if fs_cap else None
            validDataFreq = validDataFreq[:, :, :fs_ind, :]

            validData = validDataFreq
            # validData = 10 * np.log10(signal.spectrogram(validData, 2048))

        ###

        validData -= train_X_mean
        validData /= train_X_std
        validData = np.expand_dims(validData, axis=-1)
        print('Validation data shape:', validData.shape)

        early_stopping_monitor = EarlyStopping(monitor='val_loss', mode='min', patience=200, restore_best_weights=True)

        history = model.fit(x=trainData, y=trainTarget, validation_data=(validData, validTarget),
                            # batch_size=trainData.shape[0] // batch_size_div, epochs=epochs,
                            batch_size=64, epochs=epochs,
                            callbacks=[early_stopping_monitor], verbose=1)

        # Plot loss outputs
        plot_loss_graphs(history.history, run_id, step, path_to_save)

        epochs_es.append(str(len(history.history['loss'])))  # Save the epoch the model stopped at

        data_to_predict = validData
        target_to_predict = validTarget

    else:

        # history = model.fit(x=trainData, y=trainTarget, batch_size=trainData.shape[0] // batch_size_div, epochs=epochs[step], verbose=1)
        # history = model.fit(x=trainData, y=trainTarget, batch_size=64, epochs=epochs[step], verbose=1)
        history = model.fit(x=trainData, y=trainTarget, batch_size=64, epochs=10, verbose=1)

        # model = keras.models.load_model(r'Z:\tempytempyeeg\results\SEEG-SK-04\STFT_notrim.h5')


        # TODO:
        if MODE_FREQUENCY:

            f, t, testDataFreq = signal.stft(testData, 2048, nperseg=NPERSEG_RANDOM, noverlap=NOVERLAP_RANDOM)
            fs_ind = next(i for i, res in enumerate(f > fs_cap) if res) if fs_cap else None
            testDataFreq = testDataFreq[:, :, :fs_ind, :]

            testData = testDataFreq
            # testData = 10 * np.log10(signal.spectrogram(testData, 2048)[2])

        ###

        testData -= train_X_mean
        testData /= train_X_std
        testData = np.expand_dims(testData, axis=-1)
        print('Test data shape:', testData.shape)

        data_to_predict = testData
        target_to_predict = testTarget


    # Log outputs and losses
    pred = model.predict(data_to_predict).reshape(-1)
    print(str(step + 1) + ": MSE:     " + str(round(mean_squared_error(target_to_predict, pred), 5)))

    mses.append(round(mean_squared_error(target_to_predict, pred), 5))
    targets = targets + target_to_predict.tolist()
    preds = preds + pred.tolist()

    # Plot GRAD-CAMS if looking at test dataset
    PLOT_GRAD_CAM = False
    # if final_run:
    if PLOT_GRAD_CAM:
        plot_gradcams(model, testData, pred, target_to_predict, step, path_to_save, train_X_mean, train_X_std)

    if not final_run:
        del validData, validTarget

    del pred, trainTarget, trainData, testData, model, history

    keras.backend.clear_session()

# Re-set the output range to the original range
targets = (np.asarray(targets) * y_max) + y_min
preds = (np.asarray(preds) * y_max) + y_min

# true vs pred, true vs pred with 20% error forgiveness, true vs pred with 1 STDDEV error forgiveness
mse_loss_0, r2_loss_0, accuracy_0 = plot_svr_plot(targets, preds, 0.0, 'Raw', run_id, path_to_save)
mse_loss_02, r2_loss_02, accuracy_02 = plot_svr_plot(targets, preds, 0.2, '0.2', run_id, path_to_save)
mse_loss_stddev, r2_loss_stddev, accuracy_stddev = plot_svr_plot(targets, preds, np.std(targets), 'STDDEV', run_id, path_to_save)

p = np.percentile(targets, 75)
print(p)
classification_t = []
classification_p = []
for t in targets:
    if t < p:
        classification_t.append(0)
    else:
        classification_t.append(1)

for t in preds:
    if t < p:
        classification_p.append(0)
    else:
        classification_p.append(1)

confusion_matrix_test = confusion_matrix(classification_t, classification_p)
print("Data Distribution for Test Set: ")
tr_unique, tr_counts = np.unique(classification_t, return_counts=True)
print(dict(zip(tr_unique, tr_counts)))
print("Confusion Matrix for Test Set: ")
print(confusion_matrix_test)
print("")

# TODO: save confusion matrix?

print("")
print("AUROC:      " + str(round(roc_auc_score(classification_t, classification_p), 2)))
print("Precision:  " + str(round(precision_score(classification_t, classification_p), 2)))
print("Recall:     " + str(round(recall_score(classification_t, classification_p), 2)))
print("Accuracy:   " + str(round(accuracy_score(classification_t, classification_p), 2)))
print("Spearman C: " + str(round(spearmanr(targets, preds)[0], 2)))
print("Spearman p: " + str(round(spearmanr(targets, preds)[1], 2)))

plot_box_plot(targets, preds, p, '75th Percentile', run_id, path_to_save)

if True or not final_run:

    # NOTE: so that we can have final results in a separate file. and we can do some experimentation and stuff and yeah
    FILE_PREFIX = 'FINAL_RUN_' if final_run else ''

    # First save the model parameters of the current run
    full_save_path = os.path.join(path_to_save, FILE_PREFIX + "regression_results_topn.csv")
    with open(full_save_path, 'a', newline='') as f:
        writer = csv.writer(f)

        if f.tell() == 0:
            # First time writing to file. Write header row.
            writer.writerow(
                ['Run ID', 'Model #', 'Model', 'Epochs', 'ES Epochs', 'Loss', 'Dropout Rate', 'Learning Rate',
                 '# Graphs', '# Batches', 'Percentile', 'Regions To Use', 'NPERSEG', 'NOVERLAP'])

        data = [
            run_id, model_num, model_name, epochs, ":".join(epochs_es), loss, dropout_rate, learning_rate, num_graphs,
            num_batches, percentile, regions_to_use, NPERSEG_RANDOM, NOVERLAP_RANDOM
        ]
        writer.writerow(data)

    # Save the model metrics of the current run
    full_save_path = os.path.join(path_to_save, FILE_PREFIX + "regression_results_topn_svrs.csv")
    with open(full_save_path, 'a', newline='') as f:
        writer = csv.writer(f)

        if f.tell() == 0:
            # First time writing to file. Write header row.
            writer.writerow(
                ['Run ID', 'MSE Loss Raw', 'R2 Loss Raw', 'Accuracy Raw', 'MSE Loss 0.2', 'R2 Loss 0.2', 'Accuracy 0.2',
                 'MSE Loss 1 STDDEV', 'R2 Loss 1 STDDEV', 'Accuracy 1 STDDEV', 'AUROC', 'Precision', 'Recall', 'Accuracy',
                 'Spearman Correlation'])

        data = [
            run_id, mse_loss_0, r2_loss_0, accuracy_0, mse_loss_02, r2_loss_02, accuracy_02, mse_loss_stddev, r2_loss_stddev,
            accuracy_stddev,
            round(roc_auc_score(classification_t, classification_p), 2),
            round(precision_score(classification_t, classification_p), 2),
            round(recall_score(classification_t, classification_p), 2),
            round(accuracy_score(classification_t, classification_p), 2),
            round(spearmanr(targets, preds)[0], 2)
        ]
        writer.writerow(data)

# Using the best run parameters, train using all the data to create a final model
if final_run:

    trainData, trainTarget, trainTarget_p, \
    _, _, \
    _, _ = \
        calculate_batch(X_full, Y, 0, 0, None, None, None, percentile, validation=False, test=False)

    # Augment training dataset
    trainData, trainTarget = augment_dataset(trainData, trainTarget, trainTarget_p, data_range, data_avg_points)

    # Upsample each bin in training dataset to get an even-distribution across bins
    trainData, trainTarget = upsample_bins(trainData, trainTarget, num_batches)

    # Standardize the dataset
    train_X_mean = np.mean(trainData)
    train_X_std = np.std(trainData)
    trainData -= train_X_mean
    trainData /= train_X_std
    trainData = np.expand_dims(trainData, axis=-1)

    print('train data shape:', trainData.shape)

    input_shape = trainData.shape[1:]
    model, model_name = create_model(input_shape, dropout_rate, num_classes, L1_units, L2_units, L3_units, activation_function='sigmoid')

    opt = keras.optimizers.RMSprop(lr=learning_rate)
    model.compile(loss=loss, optimizer=opt)

    history = model.fit(x=trainData, y=trainTarget,
                        batch_size=trainData.shape[0] // batch_size_div, epochs=np.mean(epochs, dtype=int), verbose=1)

    full_save_path = os.path.join(path_to_save, 'model.h5')  # TODO: or just weights?
    model.save(full_save_path)
