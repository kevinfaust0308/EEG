import matplotlib

matplotlib.use('Agg')  # only comment this out when you are in pycharm and want to do visualiziation debugging
import matplotlib.pyplot as plt

import numpy as np
import os
from sklearn.metrics import mean_squared_error, r2_score

# from scripts import configs
from . import configs


def plot_gradcams(model, testData, preds, targets, step, path_to_save, xmean, xstd, attn_map_cutoff=0.7):
    from vis.visualization import visualize_cam, get_num_filters, visualize_activation, visualize_saliency
    from vis.utils import utils

    if configs.mode_frequency:
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
                raise Exception('On best run, type in the nperseg and noverlap params below and then comment out this exception.')

                img = signal.istft(img * xstd + xmean, 2048)[1][:, :10000]
                # NOTE: scaling or not scaling results in same shape. and then when we scale 0-1, they will be indentical
                attn_map_inverted = signal.istft(attn_map * xstd + xmean, 2048)[1][:, :10000]

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


def plot_svr_plot(target, classification, accuracy_window, title, run_id, path_to_save, batch_mode=False, shifts=None):
    # Mask of which indices are within our accuracy window
    mask = (classification + accuracy_window >= target) & (target >= classification - accuracy_window)

    plt.close('all')
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    if accuracy_window == 0.0:
        ax.scatter(x=target, y=classification, c='b', alpha=0.5)

    elif shifts is not None:
        mask2 = np.array(shifts, dtype=bool)

        ax.scatter(x=target[mask2], y=classification[mask2], c='g', alpha=0.5)
        ax.scatter(x=target[~mask2], y=classification[~mask2], c='r', alpha=0.5)

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

    if batch_mode:
        name = 'BATCH_'
    elif shifts is not None:
        name = 'SHIFTS_'
    else:
        name = ''

    fig.savefig(os.path.join(path_to_save, 'plots', '{}plots_scatter_{}_{}.png'.format(name, accuracy_window, run_id)))

    mse_loss = mean_squared_error(classification[~mask], target[~mask])
    r2_score_val = r2_score(classification[~mask], target[~mask])

    print("MSE Loss for AW-" + str(accuracy_window) + " :" + str(round(mse_loss, 5)))
    print("R2 score for AW-" + str(accuracy_window) + " :" + str(round(r2_score_val, 5)))
    print("Accuracy for AW-" + str(accuracy_window) + " :" + str(round(accuracy, 5)))
    print("")

    return mse_loss, r2_score_val, accuracy
