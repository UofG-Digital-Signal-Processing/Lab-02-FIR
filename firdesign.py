import numpy as np


def high_pass_design(sampling_rate, cutoff_frequency):
    M = sampling_rate
    w_c = cutoff_frequency
    # Define the ideal frequency response
    x = np.ones(M)
    x[0:w_c] = 0
    x[M - w_c:M - 1] = 0
    # Compute h(n) using the inverse FFT
    h = np.fft.ifft(x)
    # Mirror h(n)
    tmp = np.copy(h[0:int(M / 2)])
    h[0:int(M / 2)] = x[int(M / 2):M]
    x[int(M / 2):M] = tmp
    h = np.real(h)
    # Window h(n) using a Hamming window
    h = h * np.hamming(M)

    return h


def band_stop_design(sampling_rate, cutoff_frequency):
    M = sampling_rate
    w_1 = cutoff_frequency[0]
    w_2 = cutoff_frequency[1]
    # Define the ideal frequency response
    x = np.ones(M)
    x[w_1:w_2] = 0
    x[M - w_2:M - w_1] = 0
    # Compute h(n) using the inverse FFT
    h = np.fft.ifft(x)
    # Mirror h(n)
    tmp = np.copy(h[0:int(M / 2)])
    h[0:int(M / 2)] = x[int(M / 2):M]
    x[int(M / 2):M] = tmp
    h = np.real(h)
    # Window h(n) using a Hamming window
    h = h * np.hamming(M)

    return h
