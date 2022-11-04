import numpy as np

import firdesign
from firfilter import FirFilter

if __name__ == '__main__':
    file_path = 'ECG_1000Hz.dat'
    sample_rate = 1000
    band_stop_w_1 = 49
    band_stop_w_2 = 51
    high_pass_w_c = 2
    data = np.loadtxt(file_path)
    n = len(data)
    # Create bands-top filter
    band_stop_h = firdesign.band_stop_design(sample_rate, [band_stop_w_1, band_stop_w_2])
    band_stop_filter = FirFilter(band_stop_h)
    # Create high-pass filter
    high_pass_h = firdesign.high_pass_design(sample_rate, high_pass_w_c)
    high_pass_filter = FirFilter(high_pass_h)
    # Remove the 50Hz interference by the band-stop filter
    band_stop_output = np.zeros(n)
    for i in range(n):
        band_stop_output[i] = band_stop_filter.do_filter(data[i])
    # Process the baseline wander by the high-pass filter
    high_pass_output = np.zeros(n)
    for i in range(n):
        high_pass_output[i] = high_pass_filter.do_filter(band_stop_output[i])

    # TODO Plot & Compare the results
