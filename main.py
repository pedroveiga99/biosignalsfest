import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import novainstrumentation as ni


def calculate_r(signal):
    avg_ch9 = np.mean(signal[:, 2])
    avg_ch10 = np.mean(signal[:, 3])
    ac_ch9 = signal[:, 2].max() - signal[:, 2].min()
    ac_ch10 = signal[:, 3].max() - signal[:, 3].min()
    r = (ac_ch9/avg_ch9)/(ac_ch10/avg_ch10)

    return r

path = 'data pedro/opensignals_spo2_pedro_2021-12-03_15-16-47.txt'
raw_data = np.loadtxt(path)
raw_data = pd.read_csv(path, delim_whitespace=True,\
                       header=0, names=["nSeq", "DI", "CH9A", "CH9B", "%SpO2"],\
                       skiprows=2, usecols=("nSeq", "CH9A", "CH9B", "%SpO2"))

# "sampling rate": 1000,
# "resolution": [16, 16],
# "channels": [9, 10, 11],
# "sensor": ["SpO2", "SpO2", "%SpO2"],
# "label": ["CH9A", "CH9B", "%SpO2"],
# "column": ["nSeq", "DI", "CH9A", "CH9B", "%SpO2"]


# fs = 1000
#
# rest1 = raw_data[30*fs:60*fs, :]
# rest2 = raw_data[90*fs:120*fs, :]
# rest3 = raw_data[150*fs:180*fs, :]
# rest4 = raw_data[150*fs:180*fs, :]
# rest5 = raw_data[210*fs:240*fs, :]
# rest6 = raw_data[270*fs:300*fs, :]
# rest7 = raw_data[330*fs:360*fs, :]
# rest = np.concatenate((rest1, rest2, rest3, rest4, rest5, rest6, rest7))
#
# medium_pressure1 = raw_data[60*fs:90*fs, :]
# medium_pressure2 = raw_data[120*fs:150*fs, :]
# medium_pressure3 = raw_data[180*fs:210*fs, :]
# medium_pressure = np.concatenate((medium_pressure1, medium_pressure2, medium_pressure3))
#
# high_pressure1 = raw_data[240*fs:270*fs, :]
# high_pressure2 = raw_data[300*fs:330*fs, :]
# high_pressure3 = raw_data[360*fs:390*fs, :]
# high_pressure = np.concatenate((high_pressure1, high_pressure2, high_pressure3))
#
# block_size = 2 # ter cuidado para ser divisivel pelo tamanho do vetor
#
# data_rest = []
#
# for i in range(0, 30*7, block_size):
#     data_rest.append(rest[i*fs: (i+block_size)*fs, :])
#
# data_medium = []
# data_high = []
# for i in range(0, 30*3, block_size):
#     data_medium.medium_pressure(rest[i*fs: (i+block_size)*fs, :])
#     data_high.high_pressure(rest[i*fs: (i+block_size)*fs, :])
#
# r_rest = []
# r_medium = []
# r_high = []
# for block in data_rest: # em cada bloco de 3segundos
#     r_rest.append(calculate_r(block))
# for block in data_medium:
#     r_medium.append(calculate_r(block))
# for block in data_high:
#     r_high.append(calculate_r(block))
#
# plt.plot(r_rest)
# plt.figure()
# plt.plot(r_medium)
# plt.figure()
# plt.plot(r_high)
# plt.show()