'''
@author Kevin Li, Michael Schenk
@date 10.07.14 (last update)
@Collection of methods for the creation of threshold plots from bunch data.
@copyright CERN
'''
import matplotlib
import pylab as plt
import scipy.io as sio
from Pyheana.load.load_data import *


def plot_thresholds_2(ax, x, y, z, zlimits=(0, 100), zlabel=''):

    # Prepare input data.
    # x axis
    # t = x
    # turns = plt.ones(t.shape).T * plt.arange(len(t))
    # turns = turns.T

    cmap = plt.cm.get_cmap('jet', 2)
    ax[0].patch.set_facecolor(cmap(range(2))[-1])
    cmap = plt.cm.get_cmap('jet')

    t = range(len(x))
    xx, yy = plt.meshgrid(t, y)

    threshold_plot = ax[0].contourf(x, y, z.T, levels=plt.linspace(zlimits[0], zlimits[1], 201),
                                   vmin=zlimits[0], vmax=zlimits[1], cmap=cmap)
    cb = plt.colorbar(threshold_plot, ax[1], orientation='vertical')
    cb.set_label(zlabel)

    # plt.tight_layout()

    return threshold_plot


def plot_thresholds(rawdata, scan_values, plane='horizontal',
                    xlabel='turns', ylabel='intensity [particles]', zlabel='normalized emittance',
                    xlimits=((0.,8192)), ylimits=((0.,7.1e11)), zlimits=((0., 10.))):

    # Prepare input data.
    # x axis
    t = rawdata[0,:,:]
    turns = plt.ones(t.shape).T * plt.arange(len(t))
    turns = turns.T

    # z axis
    epsn_abs = {}
    epsn_abs['horizontal'] = plt.absolute(rawdata[11,:,:])
    epsn_abs['vertical']   = plt.absolute(rawdata[12,:,:])

    # Prepare plot environment.
    ax11, ax13 = _create_axes(xlabel, ylabel, zlabel, xlimits, ylimits, zlimits)
    cmap = plt.cm.get_cmap('jet', 2)
    ax11.patch.set_facecolor(cmap(range(2))[-1])
    cmap = plt.cm.get_cmap('jet')

    x, y = plt.meshgrid(turns[:,0], scan_values)
    z = epsn_abs[plane]

    threshold_plot = ax11.contourf(x, y, z.T, levels=plt.linspace(zlimits[0], zlimits[1], 201),
                                   vmin=zlimits[0], vmax=zlimits[1], cmap=cmap)
    cb = plt.colorbar(threshold_plot, ax13, orientation='vertical')
    cb.set_label(zlabel)

    plt.tight_layout()


def _create_axes(xlabel, ylabel, zlabel, xlimits, ylimits, zlimits):

    # Plot environment
    ratio = 20
    fig = plt.figure(figsize=(22, 12))

    gs = matplotlib.gridspec.GridSpec(1, ratio)
    ax11 = plt.subplot(gs[0,:ratio-1])
    ax13 = plt.subplot(gs[:,ratio-1])

    # Prepare plot axes
    ax11.grid(color='w')
    ax11.set_axis_bgcolor('0.35')
    ax11.set_xlabel(xlabel);
    ax11.set_ylabel(ylabel)
    if xlimits:
        ax11.set_xlim(xlimits)
    if ylimits:
        ax11.set_ylim(ylimits)

    return ax11, ax13
