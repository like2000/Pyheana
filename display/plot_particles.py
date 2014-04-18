import matplotlib
import matplotlib.gridspec as gridspec
import pylab as plt
from scipy.stats import gaussian_kde, norm
from Pyheana.load.fit_data import *
from Pyheana.load.load_data import *


def plot_particles(x, y, z):

    plt.figure(figsize=(12,10))

    # pdf2, bins2, patches = plt.hist(x[:,0], 100, histtype='stepfilled')
    # pdf3, bins3, patches = plt.hist(y[:,0], 100, orientation='horizontal', histtype='stepfilled')
    # plt.clf()

    # plt.ion()
    gs = gridspec.GridSpec(2, 2, width_ratios=[3,1], height_ratios=[1, 3])
    ax1 = plt.subplot(gs[2])
    ax2 = plt.subplot(gs[0], sharex=ax1)
    ax3 = plt.subplot(gs[3], sharey=ax1)
    # ax3 = plt.subplot(gs[3], sharex=ax1)

    [n_particles, n_turns] = x.shape
    xlimits = plt.amax(plt.absolute(x))
    ylimits = plt.amax(plt.absolute(y))
    zlimits = plt.amax(plt.absolute(z))
    
    # for i in range(n_turns):
    #     plt.clf()
    #     ax = plt.gca()
    #     ax.set_xlim(-xlimits, xlimits)
    #     ax.set_ylim(-ylimits, ylimits)
    #     plt.scatter(x[:,i], y[:,i], c=z, marker='o', lw=0)
    #     plt.draw()

    # Scatterplot
    z = plt.ones((n_particles, n_turns)).T * z; z = z.T
    ax1.set_xlim(-8e-1, 8e-1)
    ax1.set_ylim(-4e-3, 4e-3)
    # ax1.set_xlim(-4e-1, 4e-1)
    # ax1.set_ylim(-5e-4, 5e-4)
    ax1.set_xlabel('$\Delta$ z [m]', fontsize=22)
    ax1.set_ylabel('$\delta$', fontsize=26)
    ax1.scatter(x, y, c=z, marker='o', lw=0)

    # Histograms
    smoothed_histogram(ax2, x[:,:].flatten(), xaxis='x')
    smoothed_histogram(ax3, y[:,:].flatten(), xaxis='y')

    for l in ax2.get_xticklabels() + ax3.get_yticklabels():
        l.set_visible(False)
    for l in ax1.get_xticklabels() + ax3.get_xticklabels():
        l.set_rotation(45)

    plt.tight_layout()

    # # from mayavi.modules.grid_plane import GridPlane
    # # from mayavi.modules.outline import Outline
    # # from mayavi.modules.volume import Volume
    # # from mayavi.scripts import mayavi2
    # from mayavi import mlab
    # mlab.options.backend = 'envisage'
    # # # graphics card driver problem
    # # # workaround by casting int in line 246 of enthought/mayavi/tools/figure.py
    # # #mlab.options.offscreen = True
    # # #enthought.mayavi.engine.current_scene.scene.off_screen_rendering = True

    # mlab.figure(bgcolor=(0.9,0.9,0.9), fgcolor=(0.2,0.2,0.2))
    # aspect = (0, 10, 0, 16, -6, 6)
    # s = mlab.points3d(x[:,0], y[:,0], z[:,0], r, colormap='jet', resolution=6,
    #                   scale_factor=1e-3, scale_mode='none')
    # mlab.outline(line_width=1)
    # for i in range(n_turns):
    #     print i
    #     s.mlab_source.set(x=x[:,i], y=y[:,i], z=z[:,i])

    return ax1

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
    # Note that scipy weights its bandwidth by the covariance of the
    # input data.  To make the results comparable to the other methods,
    # we divide the bandwidth by the sample standard deviation here.
    # kde = gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1), **kwargs)
    kde = gaussian_kde(x, bw_method=bandwidth, **kwargs)

    return kde.evaluate(x_grid)

def fit_gauss(bins, pdf):

    mean, sigma = norm.fit(pdf)
    print mean, sigma
    
    x = plt.linspace(bins[0], bins[-1], 1000)
    y = norm.pdf(x, loc=mean, scale=sigma)

    return x, y



    
def smoothed_histogram_window(ax, pdf, bins):

    wsize = 20
    bins = 0.5 * (bins[:-1] + bins[1:])
    
    # for wsize in range(5, 30, 5):
    # extending the data at beginning and at the end
    # to apply the window at the borders
    ps = plt.r_[pdf[wsize-1:0:-1], pdf, pdf[-1:-wsize:-1]]
    w = plt.hanning(wsize)
    pc = plt.convolve(w/w.sum(), ps, mode='valid')
    pc = pc[wsize/2:len(ps)-wsize/2]
    pc = pc[0:len(bins)]
    # plt.plot(bins, pc)
    # plt.fill_between(bins, 0, pc, alpha=0.1)

    return pc, bins

def circular_moments(pdf):

    c_sin = []
    c_cos = []
    for k in xrange(5):
        c_sin.append(integrate.quad(pdf, low, upp, weight='cos', wvar=k))
        c_cos.append(integrate.quad(pdf, low, upp, weight='sin', wvar=k))

    print c_sin
    print c_cos
