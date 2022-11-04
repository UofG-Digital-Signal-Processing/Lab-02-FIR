import numpy as np
import firdesign
import matplotlib.pyplot as plt


h = firdesign.band_stop_design(500, [45, 55])
plt.plot(h)
plt.show()