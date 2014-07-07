from Pyheana import *
import numpy as np

# Set larger fonts.
import MyStyle as ms
ms.set_mystyle()

# Input path containing particle and bunch data.
path = '/data/TECH/Experiments/LHC/4TeV/PyHT/data/run_LHC_4TeV_I50e10_O100_newTracker_wPart/'
particle_file = path + 'particles_001.h5part'
bunch_file    = path + 'slices_001.h5'

# Instantiate TuneFootprintPlots object. Pass bunch and particle data paths and define what range of data should
# be read in (turns, how many particles).
# - It must be ensured by the user to read the correct range of turns here when later using .create_single_plot
#   or .create_plot_series().
footprints = TuneFootprintPlots(bunch_file, particle_file,
                                init_turn_to_read=4096, n_turns_to_read=4096, macroparticles_stride=2)

# Set axis ranges of footprint plot and emittance plots. If these methods are not called by the user, some default
# values are assumed.
footprints.set_emittance_plot_range(horizontal_ylimits=((0, 160)), vertical_ylimits=((0, 16)), xlimits=((0, 8192)))
footprints.set_footprint_plot_range(xlimits=((0.290, 0.322)), ylimits=((0.315, 0.333)), zlimits=((None, None)))

# (I)  GENERATE A SINGLE TUNE FOOTPRINT FRAME.
# - zaxis defines what variable to use for the color axis (possible so far are: slice_index, j_x, j_y, j_xy)
# (j denotes action).
# - Note: zaxis values are always evaluated ONLY at the right boundary of the window, which can be problematic 
# (e.g. for oscillating values with a period smaller or similar to the window width).
# - to use j_z (longitudinal action), one must adapt the ._prepare_input() method to also load z, dp, and eta.
footprints.create_single_plot(turn=7500, window_width=128, zaxis='j_x', zlabel='$J_x [m]$')
plt.savefig('./tune_footprint_example_jx.png', dpi=300)
plt.show()

# (II) GENERATE A SERIES OF SUBSEQUENT TUNE FOOTPRINTS.
# - .create_plot_series() actually calls .create_single_plot() several times and saves its output to png files at the
# given path.
# - The method returns a list of paths pointing to the images that have been created.
# output_files = footprints.create_plot_series(init_turn=4500, final_turn=7500, window_width=128, step=128,
#                                              zaxis='j_x', zlabel='$J_x [m]$', output_path='./')
# plt.show()

# OPTIONAL
# The list of images returned by .create_plot_series() can be added together and stored as a movie.
# - Note: mencoder must be installed on your system.
# footprints.make_movie(output_files, movie_name='example_animation_jx.mpg', fps=3)
# plt.show()
