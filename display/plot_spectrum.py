from __future__ import division


import PySussix
import matplotlib
import pylab as plt
import scipy.io as sio
#~ from mpi4py import MPI
from Pyheana.load.fit_data import *
from Pyheana.load.load_data import *
from scipy.stats import gaussian_kde, norm


def fft_spectrogram(signals=None, axis=None, tunes=None, scan_values=None):

    palette = cropped_cmap(cmap='hot')

    sx, sy = signals[0], signals[1]
    ax1, ax2 = axis[0], axis[1]
    Qx, Qy, Qs = tunes[0], tunes[1], tunes[2]

    n_lines = 1000
    n_turns, n_files = sx.shape
    oxx, axx = plt.zeros((n_lines, n_files)), plt.zeros((n_lines, n_files))
    oyy, ayy = plt.zeros((n_lines, n_files)), plt.zeros((n_lines, n_files))

    for i in xrange(n_files):
        t = plt.linspace(0, 1, n_turns)
        ax = plt.absolute(plt.fft(sx[:,i]))
        ay = plt.absolute(plt.fft(sy[:,i]))
        # Amplitude normalisation
        ax /= plt.amax(ax, axis=0)
        ay /= plt.amax(ay, axis=0)
        # Tunes
        if i==0:
            tunexfft = t[plt.argmax(ax[:n_turns/2], axis=0)]
            tuneyfft = t[plt.argmax(ay[:n_turns/2], axis=0)]
            print "\n*** Tunes from FFT"
            print "    tunex:", tunexfft, ", tuney:", tuneyfft, "\n"
        # Tune normalisation
        ox = (t - (Qx[i] % 1)) / Qs[i]
        oy = (t - (Qy[i] % 1)) / Qs[i]
        #~oxfft = (tfft-tunexfft)/self.Qs[j]
        #~oyfft = (tfft-tuneyfft)/self.Qs[j]
        #oyfft = tfft

# def get_spectrum(x, xp, Qx, comm):

#     # Initialise Sussix object
#     SX = PySussix.Sussix()

#     x, xp = x.T, xp.T # (turns, particles)
#     n_turns, n_particles = x.shape

#     # Floquet transformation
#     beta = plt.amax(x) / plt.amax(xp)
#     xp *= beta

#     if comm.rank == 0:
#         SX.sussix_inp(nt1=1, nt2=n_turns, idam=1, ir=0, tunex=Qx % 1, tuney=Qx % 1)
#     comm.Barrier()

#     n_subparticles = n_particles / comm.size
#     n_remain = n_particles % n_subparticles
#     tmptunes = plt.zeros(n_subparticles)

#     for i in xrange(n_particles):
#         if (n_subparticles * comm.rank < i < n_subparticles * (comm.rank + 1)):
#             SX.sussix(x[:,i], xp[:,i], x[:,i], xp[:,i], x[:,i], xp[:,i])
#             tmptunes[i - n_subparticles * comm.rank] = plt.absolute(SX.ox[plt.argmax(SX.ax)])

#     # if comm.rank == 0:
#     #     tunes = plt.zeros(n_particles)

#     tunes = np.array(comm.gather(tmptunes, root=0)).flatten()

#     if comm.rank == 0:
#         print tunes.shape
#         #sys.exit()

#     return tunes

def get_spectrum(x, xp, Qx):

    # Initialise Sussix object
    SX = PySussix.Sussix()

    x, xp = x.T, xp.T # (turns, particles)
    n_turns, n_particles = x.shape
    print x.shape

    # Floquet transformation
    # beta = plt.amax(x) / plt.amax(xp)
    # xp *= beta

    SX.sussix_inp(nt1=1, nt2=n_turns, idam=1, ir=0, tunex=Qx % 1, tuney=Qx % 1)

    # n_subparticles = n_particles / comm.size
    # n_remain = n_particles % n_subparticles
    # tmptunes = plt.zeros(n_subparticles)

    tunes = plt.zeros(n_particles)
    for i in xrange(n_particles):
        print 'Run sussix... particle',i 
        SX.sussix(x[:,i], xp[:,i], x[:,i], xp[:,i], x[:,i], xp[:,i])
        tunes[i] = plt.absolute(SX.ox[plt.argmax(SX.ax)])

    print tunes.shape
    #sys.exit()

    return tunes

def plot_footprint(x, xp, Qx, y, yp, Qy):

    tunes_x = get_spectrum(x, xp, Qx)
    tunes_y = get_spectrum(y, yp, Qy)

    plt.scatter(tunes_x, tunes_y)

def plot_spectrum(x, xp, Qx):
    
    #~ if comm.rank == 0:
    fig, (ax1) = plt.subplots(1, figsize=(11,8), sharex=True)

    tunes = get_spectrum(x, xp, Qx)
    #~ tunes = get_spectrum(x, xp, Qx, comm)
    
    #~ if comm.rank == 0:
    tunes *= 1 / Qx

    # Floquet transformation
    beta = plt.amax(x) / plt.amax(xp)
    xp *= beta

    # Action
    x0, xp0 = x[:,0], xp[:,0]

    r2 = x0 ** 2 + xp0 ** 2
    r2 *= 1 / plt.amax(r2)
    sigma_x = plt.std(x0)

    # Plot
    sc = ax1.scatter(x0, tunes, c=r2, marker='x')
    ax1.set_xlim(-0.3, 0.3)
    ax1.set_ylim(0, 1.5)
    ax1.set_xlabel('dz position along bunch [m]')
    ax1.set_ylabel(r'$Q_s / Q_{s0}$', fontsize=24)

    cb = plt.colorbar(sc)
    cb.set_label(r'$J / J_{\rm{max}}$', fontsize=24)

    plt.tight_layout()

    # ax1.hist(tunes, 100)
    # ax1.set_xlim(plt.amin(tunes), plt.amax(tunes))
    # # Histograms
    # smoothed_histogram(ax2, tunes, xaxis='x')
    # ax2.ticklabel_format(useOffset=False)

def hamiltonian(dz, dp):

    pass

def smoothed_histogram(ax, x, xaxis='x'):

    if xaxis == 'x':
        t = plt.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 1000)
        f = kde_scipy(x, t, 0.2)
        f = normalise(f)
        ax.plot(t, f, c='purple', lw=2)
        ax.fill_between(t, 0, f, facecolor='purple', alpha=0.1)

        sigma, mu, xi, yia, yib = fitgauss(t, f)
        ax.plot(xi, yia, c='firebrick', lw=2, ls='--')
        ax.text(0.99, 0.99, '$\mu$: {:1.2e}\n$\sigma$: {:1.2e}'.format(mu, sigma), fontsize=20, color='firebrick', horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)
    elif xaxis == 'y':
        t = plt.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 1000)
        f = kde_scipy(x, t, 0.2)
        f = normalise(f)
        ax.plot(f, t, c='purple', lw=2)
        ax.fill_betweenx(t, 0, f, facecolor='purple', alpha=0.1)

        sigma, mu, xi, yia, yib = fitgauss(t, f)
        ax.plot(yia, xi, c='firebrick', lw=2, ls='--')
        ax.text(0.99, 0.99, '$\mu$: {:1.2e}\n$\sigma$: {:1.2e}'.format(mu, sigma), fontsize=20, color='firebrick', horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)
    else:
        raise ValueError
    
def kde_scipy(x, x_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scipy"""
    kde = gaussian_kde(x, bw_method=bandwidth, **kwargs)

    return kde.evaluate(x_grid)
