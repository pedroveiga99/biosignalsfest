import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tsfel


def calculate_r(signal):
    avg_ch9 = np.mean(signal[:, 2])
    avg_ch10 = np.mean(signal[:, 3])
    ac_ch9 = signal[:, 2].max() - signal[:, 2].min()
    ac_ch10 = signal[:, 3].max() - signal[:, 3].min()
    r = (ac_ch9 / avg_ch9) / (ac_ch10 / avg_ch10)

    return r


def is_rest(x):
    return ((31 * fs < x) & (x < 59 * fs) |
            (91 * fs < x) & (x < 119 * fs) |
            (151 * fs < x) & (x < 179 * fs) |
            (211 * fs < x) & (x < 239 * fs) |
            (271 * fs < x) & (x < 299 * fs) |
            (331 * fs < x) & (x < 359 * fs))


def is_medium(x):
    return ((61 * fs < x) & (x < 89 * fs) |
            (121 * fs < x) & (x < 149 * fs) |
            (181 * fs < x) & (x < 209 * fs))


def is_high(x):
    return ((241 * fs < x) & (x < 269 * fs) |
            (301 * fs < x) & (x < 329 * fs) |
            (361 * fs < x) & (x < 389 * fs))


def get_features(df, person):
    # Separate the diferent situations (1s buffer in the end and beginning)
    df['state'] = 'default'
    df['state'] = np.where(is_rest(df['nSeq']), 'rest_pressure', df['state'])
    df['state'] = np.where(is_medium(df['nSeq']), 'medium_pressure', df['state'])
    df['state'] = np.where(is_high(df['nSeq']), 'high_pressure', df['state'])

    df = df.drop(df[df['state'] == 'default'].index)

    df_rest = df[df['state'] == 'rest_pressure']
    df_medium = df[df['state'] == 'medium_pressure']
    df_high = df[df['state'] == 'high_pressure']

    for df_state in (df_rest, df_medium, df_high):
        data_separated = np.array_split(df_state, round(len(df_state) / (window_size * fs)))

        # Write features
        for block in data_separated:
            # normalize data?
            # normalizing using min-max
            block['CH9A'] = (block['CH9A'] - block['CH9A'].min()) / (block['CH9A'].max() - block['CH9A'].min())
            block['CH9B'] = (block['CH9B'] - block['CH9B'].min()) / (block['CH9B'].max() - block['CH9B'].min())

            features = tsfel.time_series_features_extractor(cfg, block[['CH9A', 'CH9B']], fs=fs)
            features.to_csv(f, header=False, index=False, line_terminator=',')
            f.write(block.iloc[0]['state'] + ',' + person + '\n')


path_pedro = 'data pedro/opensignals_spo2_pedro_2021-12-03_15-16-47.txt'
path_rodrigo = 'data pedro/opensignals_spo2_rodrigo_2021-12-03_14-57-47.txt'
path_diogo = 'data pedro/opensignals_spo2_diogo_2021-12-03_15-08-49.txt'

# "sampling rate": 1000,
# "resolution": [16, 16],
# "channels": [9, 10, 11],
# "sensor": ["SpO2", "SpO2", "%SpO2"],
# "label": ["CH9A", "CH9B", "%SpO2"],
# "column": ["nSeq", "DI", "CH9A", "CH9B", "%SpO2"]
# CH9A : Red light channel
# CH9B : IR light channel

fs = 1000
window_size = 3
cfg = tsfel.get_features_by_domain('spectral')
# f = open('features/features_spectral_3s_normalized.csv', 'w')
f = open('test.txt','w')

data_pedro = pd.read_csv(path_pedro, delim_whitespace=True,
                         header=0, names=["nSeq", "DI", "CH9A", "CH9B", "SpO2"],
                         skiprows=3, usecols=("nSeq", "CH9A", "CH9B", "SpO2"))
data_rodrigo = pd.read_csv(path_rodrigo, delim_whitespace=True,
                           header=0, names=["nSeq", "DI", "CH9A", "CH9B", "SpO2"],
                           skiprows=3, usecols=("nSeq", "CH9A", "CH9B", "SpO2"))
data_diogo = pd.read_csv(path_diogo, delim_whitespace=True,
                         header=0, names=["nSeq", "DI", "CH9A", "CH9B", "SpO2"],
                         skiprows=3, usecols=("nSeq", "CH9A", "CH9B", "SpO2"))

# Write header
temp = np.array_split(data_pedro, round(len(data_pedro) / (window_size * fs)))
header_string = ''
header = tsfel.time_series_features_extractor(cfg, temp[0][['CH9A', 'CH9B']], fs=fs)
for col in header.columns:
    header_string += col + ','
f.write(header_string + 'state,person' + '\n')

get_features(data_pedro, 'pedro')
get_features(data_rodrigo, 'rodrigo')
get_features(data_diogo, 'diogo')

# ver package tsfresh

f.close()
