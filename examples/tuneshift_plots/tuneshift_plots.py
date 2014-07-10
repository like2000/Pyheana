from Pyheana import *
import numpy as np

# Set larger fonts globally.
import MyStyle as ms
ms.set_mystyle()


# Directory containing bunch (or slice) monitor input files of the (intensity) sweep.
input_path = '/data/TECH/Experiments/LHC/4TeV/PyHT/data/run_LHC_4TeV_iSwp_O0_newTracker_noSigmaz_NSL250/'

# SETTINGS
analyzer        = 'sussix'        # Use 'sussix' or 'fft'.
scan_parameter  = 'intensity'  # Use 'intensity' (intensity sweep), or ...
output_path     = './'
output_filename = 'tuneshift_' + analyzer + '_iSwp_O0_noSigmaz_NSL250_PyHT.png'

# READ DATA AND ATTRIBUTES FROM HDF5.
file_list          = get_file_list(input_path, '*.h5')
rawdata, file_list = read_bunch_data(file_list[:], format='h5')
L = get_attributes(file_list, ['Q_x', 'Q_y', 'Q_s', 'beta_x', 'beta_y', scan_parameter], format='h5')

# CREATE TUNESHIFT PLOTS.
# Returns a dictionary with keys 'horizontal' and 'vertical' denoting the horizontal and the vertical planes
# respectively. The values of the dictionary are 2-tuples, consisting of (spectral_lines, spectral_intensity)
# (data for each value of the scan parameter). 
spectra = calculate_tuneshifts(rawdata, L['beta_x'], L['beta_y'], L['Q_x'], L['Q_y'], L['Q_s'], analyzer)

# Plot and save horizontal spectrogram.
plane = 'horizontal'
plot_tuneshifts(spectra[plane], scan_values=L[scan_parameter],
                xlabel='intensity [particles]', ylabel=plane+' mode number',
                xlimits=((0., 7.1e11)), ylimits=((-4,2)))

plt.savefig(output_path + plane + '_' + output_filename, dpi=300)
plt.show()


# Plot and save vertical spectrogram.
plane = 'vertical'
plot_tuneshifts(spectra[plane], scan_values=L[scan_parameter],
                xlabel='intensity [particles]', ylabel=plane+' mode number',
                xlimits=((0., 7.1e11)), ylimits=((-4,2)))

plt.savefig(output_path + plane + '_' + output_filename, dpi=300)
plt.show()
