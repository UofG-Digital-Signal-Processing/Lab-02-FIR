import numpy as np


class FirFilter:
    def __init__(self, _h):
        self.h = _h
        self.M = len(_h)
        # Ring buffer
        self.buffer = np.zeros(self.M)
        self.offset = self.M - 1

    def do_filter(self, input):
        self.buffer[self.offset] = input
        # Move the offset
        self.offset -= 1
        if self.offset < 0:
            self.offset = self.M - 1
        # Calculate the output
        output = 0
        for i in range(self.M):
            output += self.h[i] * self.buffer[(i + self.offset) % self.M]
        return output

    def do_total_filter(self, input):
        n = len(input)
        output = np.zeros(n)
        for i in range(n):
            output[i] = self.do_filter(input[i])
        return output

    def do_filter_adaptive(self, signal, noise, learning_rate):
        # Calculate thr error
        canceller = self.do_filter(noise)
        error = signal - canceller
        # Update the h(n)
        for i in range(self.M):
            self.h[i] += error * learning_rate * self.buffer[i]
        return error


class BandStopFilter(FirFilter):
    def __init__(self, sample_rate, w_1, w_2, freq_resolution=1):
        h = _band_stop_design(sample_rate, w_1, w_2, freq_resolution)
        super().__init__(h)


class HighPassFilter(FirFilter):
    def __init__(self, sample_rate, w_1, freq_resolution=1):
        h = _high_pass_design(sample_rate, w_1, freq_resolution)
        super().__init__(h)


class LmsFilter(FirFilter):
    def __init__(self, sample_rate, freq_resolution):
        h = np.zeros(int(sample_rate / freq_resolution))
        super().__init__(h)
        self.sample_rate = sample_rate

    def _do_filter_adaptive(self, signal, noise, learning_rate):
        # Calculate thr error
        canceller = super.do_filter(noise)
        error = signal - canceller
        # Update the h(n)
        for i in range(self.M):
            self.h[i] += error * learning_rate * self.buffer[i]
        return error

    def do_total_filter_adaptive(self, input, noise_freq, learning_rate):
        n = len(input)
        output = np.zeros(n)
        for i in range(n):
            noise = np.sin(2.0 * np.pi * noise_freq / self.sample_rate * i)
            output[i] = self.do_filter_adaptive(input[i], noise, learning_rate)
        return output


def _band_stop_design(sampling_rate, w_1, w_2, freq_resolution=1):
    fs = sampling_rate
    M = int(fs / freq_resolution)
    w_1 = int(w_1 / fs * M)
    w_2 = int(w_2 / fs * M)
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


def _high_pass_design(sampling_rate, w_1, freq_resolution=1):
    fs = sampling_rate
    M = int(fs / freq_resolution)
    w_c = int(w_1 / fs * M)
    # Define the ideal frequency response
    X = np.ones(M)
    X[0:w_c + 1] = 0
    X[M - w_c:M + 1] = 0
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
