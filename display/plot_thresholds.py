import matplotlib
import pylab as plt
import scipy.io as sio
from Pyheana.load.load_data import *


# file_list = get_file_list(path)


# A, file_list = read_bunch_data(file_list[::1], format='ascii'); A = A[:,10000:,:]
# t, x, xp, y, yp, dz, dp = A[:7,:,:]
# sigma_x, sigma_y, sigma_dz, sigma_dp, epsn_x, epsn_y, epsn_z = A[7:14,:,:]

# T = plt.arange(len(t)) + 1 + 10000
# varray = plt.array(get_value_from_cfg(file_list, "Number_of_particles_per_bunch:"))


def plot_thresholds(x, y, z):

    vmax = 10

    # fig, (ax) = plt.subplots(1, figsize=(11, 8))
    # n_turns, n_files = x.shape
    # cmap = plt.get_cmap('jet')
    # c = [cmap(i) for i in plt.linspace(0, 1, n_files)]
    # ax.set_color_cycle(c)

    # k = plt.amax(y, axis=0).argsort()[::-1]
    # x, y = x[:,k], y[:,k]
    # ax.plot(y)
    # ax.set_ylim(-10, 10)

    # # Mask array
    # a = np.ma.masked_less_equal(A, 0)
    # a = np.ma.masked_greater_equal(a, 10000)

    # Contour plot
    fig, (ax) = plt.subplots(1, figsize=(11, 8))
    cmap = plt.cm.get_cmap('jet', 2)
    ax.patch.set_facecolor(cmap(range(2))[-1])
    cmap = plt.cm.get_cmap('jet')

    xx, yy = plt.meshgrid(x[:,0], z)
    ct = ax.contourf(xx, yy, y.T, levels=plt.linspace(0, vmax, 201), vmin=0, vmax=vmax, cmap=cmap)
    ax.set_xlabel('Turns')
    ax.set_ylabel('Intensity [1e11 ppb]')
    cb = plt.colorbar(ct)
    cb.set_label('Horizontal norm. emittance [$\mu$m]')
    plt.tight_layout()

