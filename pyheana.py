from Pyheana import *

import MyStyle as ms
ms.set_mystyle()

# path = '/home/kli/backup/CERN/HL-LHC'
# path = '/home/kli/backup/CERN/HL-LHC/LHC-7TeV-doubleharmonic-base-Intensity-Chromaticity'
# path = '/home/kli/backup/CERN/HL-LHC'
# path = '/home/kli/backup/CERN/SPS/HeadTail_SPS_26GeV_Linear-Density'
path = '/home/kli/backup/CERN/SPS/HeadTail_SPS_Ramp-ElrDen'


file_list = get_file_list(path)


A, file_list = read_bunch_data(file_list[::8], format='hdf5')
t, x, xp, y, yp, z, dp = A[:7,:,:]
sigma_x, sigma_y, sigma_z, sigma_dp, epsn_x, epsn_y, epsn_z = A[7:14,:,:]
T = plt.ones(t.shape).T * plt.arange(len(t)); T = T.T

# plot_risetimes(T, x, file_list=file_list)
# plt.savefig("risetimes-" +path.split('/')[-1] + ".png", dpi=200)

# plot_coherent((T, y), (T, epsn_y), file_list=file_list, scan_line='Average_electron_cloud_density_along_the_ring_(1/m^3):')
# plot_coherent((T, sigma_dz), (T, epsn_z), file_list=file_list, scan_line='Number_of_particles_per_bunch:')
# plt.savefig("coherent-" +path.split('/')[-1] + ".png", dpi=200)

# plot_tuneshift(A, file_list, 'Number_of_particles_per_bunch:', scan_parameter='Intensity')
# plot_tuneshift(A, file_list, 'Average_electron_cloud_density_along_the_ring_(1/m^3):', scan_parameter='Density')
# plt.savefig("tuneshift-" +path.split('/')[-1] + ".png", dpi=200)
# plt.show()

# A = read_particle_data(file_list[3], format='ascii')
# print A.shape
# x, xp, y, yp, dz, dp = A[:,::1,::1]
# plot_footprint(x, xp, 20.13, y, yp, 20.18)
# plot_spectrum(x, xp, 20.13)
# plt.show()

# beta = plt.amax(dz[:,0]) / plt.amax(dp[:,0])
# r = dz[:,0] ** 2 + (beta * dp[:,0]) ** 2
# plot_particles(dz, dp, r)
# plt.savefig("bucket-" +path.split('/')[-1] + ".png", dpi=200)
# plt.show()


# A = read_slice_data(file_list[80], format='hdf5')
# plot_slices(A)

