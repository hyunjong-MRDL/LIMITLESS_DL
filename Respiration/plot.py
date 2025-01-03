import os
from scipy.fft import fft, fftshift
import numpy as np
import matplotlib.pyplot as plt

def plot_by_index(data_type, time_dict, beam_dict, index):
    Times, Amps = time_dict[index], beam_dict[index]
    filename = f"./Plots/{data_type}/{index}.jpg"
    if not os.path.exists(filename):
        fft_amp = fftshift(fft(np.array(Amps)))
        N = len(Amps)
        fs = 1/0.015
        freqs = np.linspace(0, N-1, N) * (fs/N)
        fig, ax = plt.subplots(2, 1, figsize=(18, 12))
        ax[0].plot(Times, Amps)
        ax[0].set_title("Time domain data")
        ax[0].set_xlabel("Time (s)"), ax[0].set_ylabel("Amplitude (cm)")
        ax[1].plot(freqs, np.abs(fft_amp))
        ax[1].set_title("Frequency domain data")
        ax[1].set_xlabel("Frequency (Hz)"), ax[0].set_ylabel("Amplitude (cm)")
        fig.savefig(filename)
        plt.close()
    
    else:
        print(f"Figure already exists: {filename}")

def plot_whole_data(data_type, Times, Amps):
    filename = f"./Plots/{data_type}/Whole_data.jpg"
    if not os.path.exists(filename):
        fft_amp = fftshift(fft(np.array(Amps)))
        N = len(Amps)
        fs = 1/0.015
        freqs = np.linspace(0, N-1, N) * (fs/N)
        fig, ax = plt.subplots(2, 1, figsize=(18, 12))
        ax[0].plot(Times, Amps)
        ax[0].set_title("Time domain data")
        ax[0].set_xlabel("Time (s)"), ax[0].set_ylabel("Amplitude (cm)")
        ax[1].plot(freqs, np.abs(fft_amp))
        ax[1].set_title("Frequency domain data")
        ax[1].set_xlabel("Frequency (Hz)"), ax[0].set_ylabel("Amplitude (cm)")
        fig.savefig(filename)
        plt.close()
    
    else:
        print(f"Figure already exists: {filename}")