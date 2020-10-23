import tensorflow as tf
import numpy as np
import random
import pandas as pd

# from scripts import configs
from . import configs


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

    return percentile, model_num, learning_rate, dropout_rate, epochs, loss, num_graphs, L1_units, L2_units, L3_units, L4_units


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

    return percentile, model_num, learning_rate, dropout_rate, epochs, loss, num_graphs, L1_units, L2_units, L3_units, L4_units, regions_to_use  # , row['NPERSEG'][0], row['NOVERLAP'][0]


def get_best_model(results_svr_path):
    df = pd.read_csv(results_svr_path)
    df.sort_values(by=['Accuracy 1 STDDEV'], ascending=False, inplace=True)
    run_id = int(df.iloc[0]['Run ID'])
    return run_id


# validation, test -> set aside data for these.
# When not on final run, we set aside these but only use train and cross validate on val.
# When on final run, we set aside only test and train on everything else.
# When creating our final model, we dont set any aside and just train on all data.
def calculate_batch(X_full, Y, shift, B, step, num_batches, data_range, data_avg_points, validation=True, test=True):
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

    validShift, testShift = None, None

    # Indices of all the data not part of our validation/testing batch (so we can use for training)
    train_indices = np.ones(X_full.shape[0], dtype=bool)

    if test:
        # On the final step, we will have to loop around the data
        if step == num_batches - 1:
            testData, testTarget = X_full[i_batch:], Y[i_batch:]
            train_indices[i_batch:] = 0

            testShift = shift[i_batch:]
        else:
            testData, testTarget = X_full[i_batch:i_batch + B], Y[i_batch:i_batch + B]
            train_indices[i_batch:i_batch + B] = 0

            testShift = shift[i_batch:i_batch + B]

        testData = set_data_range(testData, data_range, data_avg_points)

    if validation:
        if step == num_batches - 1:
            validData, validTarget = X_full[0:B], Y[0:B]
            train_indices[0:B] = 0

            validShift = shift[0:B]

        else:
            validData, validTarget = X_full[i_batch + B:i_batch + (2 * B)], Y[i_batch + B:i_batch + (2 * B)]
            train_indices[i_batch + B:i_batch + (2 * B)] = 0

            validShift = shift[i_batch + B:i_batch + (2 * B)]

        validData = set_data_range(validData, data_range, data_avg_points)

    trainData, trainTarget, trainShift = X_full[train_indices], Y[train_indices], shift[train_indices]

    return trainData, trainTarget, validData, validTarget, testData, testTarget, trainShift, validShift, testShift


def create_model(input_shape, dropout_rate, num_classes, L1_units, L2_units, L3_units, activation_function):
    input_stft = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv3D(L1_units, padding='valid', kernel_size=(1, 3, 3), activation='relu')(input_stft)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.Model(inputs=input_stft, outputs=x)

    input_shift = tf.keras.Input(shape=(1,))

    x = tf.keras.layers.concatenate([x.output, input_shift])

    x = tf.keras.layers.Dense(L2_units, activation='relu')(x)
    x = tf.keras.layers.Dense(L3_units, activation='relu')(x)

    model_name = 'C2NN' + str(L1_units) + '_NN' + str(L2_units) + '_NN' + str(L3_units)

    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(num_classes, activation=activation_function)(x)

    x = tf.keras.Model(inputs=[input_stft, input_shift], outputs=x)

    return x, model_name

    inputs = tf.keras.Input(shape=input_shape)

    if not configs.mode_frequency:
        x = tf.keras.layers.Conv2D(L1_units, kernel_size=(1, input_shape[1] // 5), dilation_rate=2, activation='relu')(inputs)
        x = tf.keras.layers.Conv2D(L2_units, kernel_size=(1, input_shape[0] // 10), dilation_rate=1, activation='relu')(x)
    else:
        x = tf.keras.layers.Conv3D(L1_units, kernel_size=(1, 3, 3), activation='relu')(inputs)
        x = tf.keras.layers.Conv3D(L2_units, kernel_size=(input_shape[0], 3, 3), activation='relu')(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(L2_units, activation='relu')(x)
    x = tf.keras.layers.Dense(L3_units, activation='relu')(x)

    model_name = 'C2NN' + str(L1_units) + '_NN' + str(L2_units) + '_NN' + str(L3_units)

    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation=activation_function)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model, model_name
