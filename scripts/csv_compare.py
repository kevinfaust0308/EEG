import pandas as pd
import os

folders = [
    r'Z:\tempytempyeeg\results\SEEG-SK-04\conv3d (no fft augment)',
    r'Z:\tempytempyeeg\results\SEEG-SK-04\conv3d (no fft augment, specto hyperparams, no trim)'
]

dfs = {}
for i, folder in enumerate(folders):


    df_score = pd.read_csv(os.path.join(folder, 'regression_results_topn.csv'))
    df_hyperparam = pd.read_csv(os.path.join(folder, 'regression_results_topn_svrs.csv'))

    df = df_score.merge(df_hyperparam, on='Run ID')
    df.sort_values('Accuracy 1 STDDEV', inplace=True, ascending=False)

    dfs[i] = df
