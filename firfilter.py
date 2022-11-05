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

    def do_filter_adaptive(self, signal, noise, learning_rate):
        # Calculate thr error
        canceller = self.do_filter(noise)
        error = signal - canceller
        # Update the h(n)
        for i in range(self.M):
            self.h[i] += error * learning_rate * self.buffer[i]
        return error
