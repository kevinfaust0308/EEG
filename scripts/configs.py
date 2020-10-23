import os
import sys

path = r"Z:/tempytempyeeg"

try:
    run_id = int(sys.argv[1])
    filename = sys.argv[2]
    is_final_run = sys.argv[3]
except:
    # local debugging parameters
    run_id = 5000
    filename = 'SEEG-SK-04'
    is_final_run = 0

path_to_load = os.path.join(path, 'data', filename, 'processed')
path_to_save = os.path.join(path, 'results', filename)

num_batches = 5  # Number of batches
num_classes = 1  # Number of outcome classes

mode_frequency = True

if mode_frequency:
    data_avg_points = 1  # How many datapoints to average over
    data_range = (25, 4050)
else:
    data_avg_points = 5
    data_range = (25, 4050)  # Range of data to select from full set

final_run = int(is_final_run)  # If 0, then validate against validation datset. If 1, search through logs to find best-performing model, load those hyperparameters for training, then test on test dataset

# whether to cap the frequency data. None means don't
fs_cap = 100
fs_cap = None



CONTACT_MODE = True
max_regions_limit = 2  # Upper limit of how many regions to select for run

# Out of `TOP_X_CONTACTS`, select up to `MAX_CONTACT_LIMIT` contacts each train run
TOP_X_CONTACTS = 20
MAX_CONTACT_LIMIT = 3

USE_DATASET_API = True

USE_MULTI_GPU = True