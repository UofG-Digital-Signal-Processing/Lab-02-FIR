import numpy as np


def cal_time_domain(data, sample_rate):
    amplitude = data
    time = np.linspace(0, len(amplitude) / sample_rate, num=len(amplitude))

    return time, amplitude


def cal_frequency_domain(data, sample_rate):
    ft = np.fft.fft(data)
    frequency = np.fft.fftfreq(data.size, d=1.0 / sample_rate)
    frequency = frequency[:int(len(frequency) / 2)]
    amplitude = np.abs(ft)
    amplitude = amplitude[:int(len(amplitude) / 2)]

    return frequency, amplitude


def cal_frequency_domain_db(data, sample_rate):
    frequency, amplitude = cal_frequency_domain(data, sample_rate)
    amplitude = 20 * np.log10(amplitude)

    return frequency, amplitude
