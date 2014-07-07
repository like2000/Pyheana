from Pyheana import *
import numpy as np

# Set larger fonts.
import MyStyle as ms
ms.set_mystyle()


# INPUT SETTINGS
scan_parameter  = 'intensity'  # Use 'intensity', or 'i_oct_d' (octupole current sweep), or ...

input_path      = '/data/TECH/Experiments/LHC/4TeV/PyHT/data/run_LHC_4TeV_iSwp_O0_BBResonator_newTracker/'
file_list       = get_file_list(input_path, '*.h5')
data, file_list = read_bunch_data(file_list[:], format='h5')
L               = get_attributes(file_list, [scan_parameter], format='h5')

# OUTPUT SETTINGS
output_path     = './'
output_filename = 'thresholds_iSwp_O0_BBResonatorDipX_PyHT.png'

# Plot parameters.
plane   = 'horizontal'
xlabel  = 'turns'
ylabel  = 'intensity [particles]'
zlabel  = plane + ' normalized emittance [$\mu m$]' 
xlimits = ((0, 16384))
ylimits = ((0., 8.6e11)) 
zlimits = ((0., 10.))    

# CREATE PLOT, SAVE AND SHOW.
threshold_plots = ThresholdPlots(rawdata=data, scan_values=L[scan_parameter])

threshold_plots.prepare_plot(plane, xlabel, ylabel, zlabel, xlimits, ylimits, zlimits)
plt.savefig(output_path + plane + '_' + output_filename, dpi=300)
plt.show()
