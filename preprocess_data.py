import pandas as pd
import numpy as np
import os
from scipy.io import loadmat
from matplotlib import pyplot as plt
from scipy.fft import fft, fftfreq, fftshift

def load_raw_mat(path):
    sampling_freq = 48000
    mat = loadmat(path)
    num = path.split('/')[-1].strip('.mat').split('_')[-1]

    df = pd.DataFrame({
        'DE': mat[f"X{num}_DE_time"].ravel(),
        'FE': mat[f"X{num}_FE_time"].ravel()
        })
    
    df['time'] = df.index * sampling_freq

    return df

def plot_raw(df):

    plt.figure()
    plt.plot(df['time'], df['DE'], '.', label='Drive End')
    plt.plot(df['time'], df['FE'], '.', label='Fan End')
    plt.legend()
    plt.show()

def my_fft(array, sampling_freq):
    N = len(array)

    yf = fft(array)
    xf = fftfreq(N, 1/sampling_freq)

    xf_shifted = fftshift(xf)
    yf_shifted = fftshift(yf)

    half_point = N // 2
    xf_pos = xf_shifted[half_point:]
    yf_pos = np.abs(yf_shifted[half_point:])
    yf_dB = 20 * np.log10(yf_pos)

    return xf_pos, yf_dB
    # plt.figure()
    # plt.plot(xf_pos, yf_dB, '.')
    # plt.show()

def win_do_fft(array, win_width, slide):
    sampling_freq = 48000
    array = np.asarray(array)

    n_loops = round((len(array) - win_width) / slide + 0.5)
    xf = np.zeros((n_loops, win_width // 2))
    yf = np.zeros((n_loops, win_width // 2))
    t = np.zeros(n_loops)

    for i, i1 in enumerate(range(win_width, len(array), slide)):
        i0 = i1 - win_width
        win = array[i0:i1]

        win_xf, win_yf = my_fft(win, sampling_freq)

        t[i] = (i1+1) / sampling_freq
        xf[i,:] = win_xf
        yf[i,:] = win_yf

    # plt.pcolormesh(t, xf[0,:], yf.T, shading='auto')
    # plt.xlabel("X-axis (t)")
    # plt.ylabel("Y-axis (xf)")
    # plt.title("Heat Plot")
    # plt.colorbar(label="Corresponding yf value")
    # plt.show()

    return t, xf, yf

def plot_spectrum(t, xf, DE_yf, FE_yf, path, win_width, slide):

    plt.figure()
    plt.subplot(2,1,1)
    plt.suptitle(f"Heat Plot: {path}")

    plt.pcolormesh(t, xf[0,:], DE_yf.T, shading='auto')
    plt.xlabel("t (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title('Drive End')
    plt.colorbar(label="Corresponding yf value")

    plt.subplot(2,1,2)
    plt.pcolormesh(t, xf[0,:], FE_yf.T, shading='auto')
    plt.xlabel("t (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(f"Fan End")
    plt.colorbar(label="Corresponding yf value")

    path = path.replace('raw/', f'processed/{win_width}_{slide}/images/').replace('.mat', f"_freq_{win_width}_{slide}.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)

def save_processed_bulk(t, xf, DE_yf, FE_yf, path, win_width, slide):
    path = path.replace('raw/', f'processed/{win_width}_{slide}/').replace('.mat', f"_freq_{win_width}_{slide}.npy")
    freq = np.hstack([DE_yf, FE_yf])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, freq)


all_paths = ['raw/OR007_6_1_136.mat', 'raw/OR021_6_1_239.mat', 'raw/Time_Normal_1_098.mat', 'raw/IR021_1_214.mat', 'raw/IR014_1_175.mat', 'raw/IR007_1_110.mat', 'raw/B021_1_227.mat', 'raw/B014_1_190.mat', 'raw/B007_1_123.mat', 'raw/OR014_6_1_202.mat']
win_width=128
slide=16
for path in all_paths:

    df = load_raw_mat(path)
    t, DE_xf, DE_yf = win_do_fft(df['DE'], win_width=win_width, slide=slide)
    t, FE_xf, FE_yf = win_do_fft(df['FE'], win_width=win_width, slide=slide)

    save_processed_bulk(t, DE_xf, DE_yf, FE_yf, path, win_width, slide)
    plot_spectrum(t, DE_xf, DE_yf, FE_yf, path, win_width, slide)