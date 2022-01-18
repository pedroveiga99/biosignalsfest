import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import novainstrumentation as ni
import tsfel


def calculate_r(signal):
    avg_ch9 = np.mean(signal[:, 2])
    avg_ch10 = np.mean(signal[:, 3])
    ac_ch9 = signal[:, 2].max() - signal[:, 2].min()
    ac_ch10 = signal[:, 3].max() - signal[:, 3].min()
    r = (ac_ch9 / avg_ch9) / (ac_ch10 / avg_ch10)

    return r


def is_rest(x):
    return ((30 * fs < x) & (x < 60 * fs) |
            (90 * fs < x) & (x < 120 * fs) |
            (150 * fs < x) & (x < 180 * fs) |
            (210 * fs < x) & (x < 240 * fs) |
            (270 * fs < x) & (x < 300 * fs) |
            (330 * fs < x) & (x < 360 * fs))


def is_medium(x):
    return ((60 * fs < x) & (x < 90 * fs) |
            (120 * fs < x) & (x < 150 * fs) |
            (180 * fs < x) & (x < 210 * fs))


def is_high(x):
    return ((240 * fs < x) & (x < 270 * fs) |
            (300 * fs < x) & (x < 330 * fs) |
            (360 * fs < x) & (x < 390 * fs))


def get_features(df):
    # Separate the diferent situations
    df['state'] = 'default'
    df['state'] = np.where(is_rest(df['nSeq']), 'rest_pressure', df['state'])
    df['state'] = np.where(is_medium(df['nSeq']), 'medium_pressure', df['state'])
    df['state'] = np.where(is_high(df['nSeq']), 'high_pressure', df['state'])

    df = df.drop(df[df['state'] == 'default'].index)

    data_separated = np.array_split(df, round(len(df) / (window_size * fs)))

    # Write features
    for block in data_separated:
        features = tsfel.time_series_features_extractor(cfg, block[['CH9A', 'CH9B']], fs=fs)
        features.to_csv(f, header=False, index=False, line_terminator=',')
        f.write(block.iloc[0]['state'] + '\n')


path_pedro = 'data pedro/opensignals_spo2_pedro_2021-12-03_15-16-47.txt'
path_rodrigo = 'data pedro/opensignals_spo2_rodrigo_2021-12-03_14-57-47.txt'
path_diogo = 'data pedro/opensignals_spo2_diogo_2021-12-03_15-08-49.txt'

# "sampling rate": 1000,
# "resolution": [16, 16],
# "channels": [9, 10, 11],
# "sensor": ["SpO2", "SpO2", "%SpO2"],
# "label": ["CH9A", "CH9B", "%SpO2"],
# "column": ["nSeq", "DI", "CH9A", "CH9B", "%SpO2"]

fs = 1000
window_size = 1
cfg = tsfel.get_features_by_domain('temporal')
f = open('features_temporal.csv', 'w')

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
f.write(header_string + 'state' + '\n')

get_features(data_pedro)
get_features(data_rodrigo)
get_features(data_diogo)

f.close()
