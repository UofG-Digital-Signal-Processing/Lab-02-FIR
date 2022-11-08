import abc

import numpy as np


class FirFilter:
    def __init__(self, _h):
        self.h = _h
        self.M = len(_h)
        # Ring buffer
        self.buffer = np.zeros(self.M)
        self.offset = self.M - 1

    # Define the abstract design method
    @abc.abstractmethod
    def _design_filter(self, w):
        pass

    def _do_filter(self, input):
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
            output[i] = self._do_filter(input[i])
        return output


class BandStopFilter(FirFilter):
    def __init__(self, _sample_rate, _w_1, _w_2, _freq_resolution=1):
        self.sample_rate = _sample_rate
        self.freq_resolution = _freq_resolution
        h = self._design_filter([_w_1, _w_2])
        super().__init__(h)

    def _design_filter(self, w):
        fs = self.sample_rate
        M = int(fs / self.freq_resolution)
        w_1 = int(w[0] / fs * M)
        w_2 = int(w[1] / fs * M)
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


class HighPassFilter(FirFilter):
    def __init__(self, _sample_rate, _w_c, _freq_resolution=1):
        self.sample_rate = _sample_rate
        self.freq_resolution = _freq_resolution
        h = self._design_filter(_w_c)
        super().__init__(h)

    def _design_filter(self, w):
        fs = self.sample_rate
        M = int(fs / self.freq_resolution)
        w_c = int(w / fs * M)
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


class LmsFilter(FirFilter):
    def __init__(self, _sample_rate, _freq_resolution=1):
        self.sample_rate = _sample_rate
        self.freq_resolution = _freq_resolution
        h = self._design_filter(None)
        super().__init__(h)

    def _design_filter(self, w):
        h = np.zeros(int(self.sample_rate / self.freq_resolution))
        return h

    def __do_filter_adaptive(self, signal, noise, learning_rate):
        # Calculate thr error
        canceller = super()._do_filter(noise)
        output = signal - canceller
        # Update the h(n)
        for i in range(self.M):
            self.h[i] += output * learning_rate * self.buffer[(i + self.offset) % self.M]
        return output

    def do_total_filter_adaptive(self, input, noise_freq, learning_rate):
        n = len(input)
        output = np.zeros(n)
        for i in range(n):
            noise = np.sin(2.0 * np.pi * noise_freq / self.sample_rate * i)
            output[i] = self.__do_filter_adaptive(input[i], noise, learning_rate)
        return output
