from Pyheana import *
import numpy as np

# Set larger fonts globally.
import MyStyle as ms
ms.set_mystyle()


# Directory containing bunch (or slice) monitor input files of the (intensity) sweep.
input_path = '/data/TECH/Experiments/LHC/4TeV/PyHT/data/run_LHC_4TeV_iSwp_O0_BBResonator_newTracker/'


# SETTINGS
analyzer        = 'sussix'     # Use 'sussix' or 'fft'.
scan_parameter  = 'intensity'  # Use 'intensity' (intensity sweep), or ...
output_path     = './'
output_filename = 'tuneshift_' + analyzer + '_iSwp_O0_BBResonatorDipX_PyHT.png'


# READ DATA AND ATTRIBUTES FROM HDF5.
file_list          = get_file_list(input_path, '*.h5')
rawdata, file_list = read_bunch_data(file_list[:], format='h5')

L = get_attributes(file_list, ['Q_x', 'Q_y', 'Q_s', 'beta_x', 'beta_y', scan_parameter], format='h5')


# CREATE TUNESHIFT PLOTS.
# Note that calling TuneshiftPlots.calculate_spectrogram(analyzer) calculates and saves data for both planes.
# Using TuneshiftPlots.prepare_plot(...), the data can be plotted for the plane and analyzer indicated (see below).
tuneshift_plots = TuneshiftPlots(rawdata, L[scan_parameter], L['beta_x'], L['beta_y'], L['Q_x'], L['Q_y'], L['Q_s'])
tuneshift_plots.calculate_spectrogram(analyzer=analyzer)

# Save and show horizontal plot.
plane = 'horizontal'
tuneshift_plots.prepare_plot(plane=plane, analyzer=analyzer,
                             xlabel='intensity [particles]', ylabel=plane+' mode number',
                             xlimits=((0., 9.1e11)), ylimits=((-5,3)))
plt.savefig(output_path + plane + '_' + output_filename, dpi=300)
plt.show()

# Save and show vertical plot.
plane = 'vertical'
tuneshift_plots.prepare_plot(plane=plane, analyzer=analyzer,
                             xlabel='intensity [particles]', ylabel=plane+' mode number',
                             xlimits=((0., 9.1e11)), ylimits=((-5,3)))
plt.savefig(output_path + plane + '_' + output_filename, dpi=300)
plt.show()
