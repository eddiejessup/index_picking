from __future__ import print_function, division

import glob
from os import path
import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt

matplotlib.style.use('ggplot')
xls_fnames = glob.glob('xls_binary/*.xls')

fig = plt.figure()
ax = fig.gca()


def weighted_mean(values, weights):
    return np.sum(weights * values) / np.sum(weights)


def weighted_std(values, weights):
    mean = weighted_mean(values, weights)
    return weighted_mean((values - mean) ** 2, weights)

for xls_fname in xls_fnames:
    name = path.splitext(path.basename(xls_fname))[0]

    info_df = pd.read_excel(xls_fname, sheetname=0)
    name_column = info_df.columns[0]
    val_column = info_df.columns[1]

    fixed_yearly_value_change_row = info_df[info_df[name_column] == 'Ongoing Charges Figures']
    fixed_yearly_value_change = fixed_yearly_value_change_row[val_column].iloc[0] / 100.0

    initial_value_change_row = info_df[info_df[name_column] == 'Initial Charge']
    initial_value_change = initial_value_change_row[val_column].iloc[0] / 100.0

    time_df_raw = pd.read_excel(xls_fname, sheetname=1)
    time_df_raw['As Of'] = pd.to_datetime(time_df_raw['As Of'])
    time_df = time_df_raw.set_index('As Of')
    time_df = time_df[time_df.index > datetime.datetime(2013, 1, 1)]
    buy_value = time_df['Bid']

    ratio_raw = (buy_value.values[1:] / buy_value.values[:-1])
    ratio_array = np.empty(ratio_raw.shape[0] + 1)
    ratio_array[1:] = ratio_raw
    ratio_array[0] = np.nan
    ratio_array_yearlyised = (ratio_array - 1.0) * 365 + 1.0
    time_df['ratio'] = ratio_array_yearlyised
    ratio = time_df['ratio']

    ratio_net = ratio - fixed_yearly_value_change

    delta = datetime.datetime.now() - time_df.index
    delta_days = delta.days
    weights = 1.0 / delta_days ** 0.2
    weights /= weights.sum()
    predicted_ratio_net = weighted_mean(ratio_net, weights)
    predicted_ratio_net_std = weighted_std(ratio_net, weights)
    predicted_ratio_net_uncertainty = predicted_ratio_net_std / predicted_ratio_net

    num_significant_data_points = (weights > 0.1 * weights.max()).sum()
    prediction_uncertainty = 1.0 / np.sqrt(num_significant_data_points)

    if predicted_ratio_net > 1.1 and predicted_ratio_net_uncertainty < 10.5 and prediction_uncertainty < 0.045:
        print('{} {:.2g} {:.2g} {}'.format(name,
                                           predicted_ratio_net - 1.0,
                                           predicted_ratio_net_uncertainty,
                                           prediction_uncertainty))
        (ratio - 1.0).cumsum().plot(ax=ax, legend=True, label=name, fontsize=6)

plt.savefig('hi.pdf')
