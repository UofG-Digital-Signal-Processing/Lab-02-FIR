import numpy as np


def high_pass_design(sampling_rate, cutoff_frequency, frequency_resolution=1):
    fs = sampling_rate
    M = int(fs / frequency_resolution)
    w_c = int(cutoff_frequency / fs * M)
    # Define the ideal frequency response
    X = np.ones(M)
    X[0:w_c] = 0
    X[M - w_c:M - 1] = 0
    # Compute h(n) using the inverse FFT
    x = np.fft.ifft(X)
    x = np.real(x)
    # Mirror h(n)
    h = np.zeros(M)
    h[0:int(M / 2)] = x[int(M / 2):M]
    h[int(M / 2):M] = x[0:int(M / 2)]
    # Window h(n) using a Hamming window
    h = h * np.hamming(M)

    return h


def band_stop_design(sampling_rate, cutoff_frequency, frequency_resolution=1):
    fs = sampling_rate
    M = int(fs / frequency_resolution)
    w_1 = int(cutoff_frequency[0] / fs * M)
    w_2 = int(cutoff_frequency[1] / fs * M)
    # Define the ideal frequency response
    X = np.ones(M)
    X[w_1:w_2 + 1] = 0
    X[M - w_2:M - w_1 + 1] = 0
    # Compute h(n) using the inverse FFT
    x = np.fft.ifft(X)
    x = np.real(x)
    # Mirror h(n)
    h = np.zeros(M)
    h[0:int(M / 2)] = x[int(M / 2):M]
    h[int(M / 2):M] = x[0:int(M / 2)]
    # Window h(n) using a Hamming window
    h = h * np.hamming(M)

    return h
