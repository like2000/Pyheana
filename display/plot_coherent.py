import PySussix
import matplotlib
import pylab as plt
from Pyheana.load.fit_data import *
from Pyheana.load.load_data import *
from operator import methodcaller


def plot_coherent(*args, **kwargs):

    majorFormatter = plt.FormatStrFormatter('%3.2e')
    sctfcFormatter = plt.ScalarFormatter(useOffset=False, useMathText=True)
    sctfcFormatter.set_scientific(True)

    data_pairs = args
    n_data_pairs = len(data_pairs)
    slices = plt.s_[:]
    
    if kwargs is not None:
        for key, value in kwargs.iteritems():
            if key == 'file_list':
                file_list = value
            if key == 'scan_line':
                scan_line = value
    varray = plt.array(get_value_from_cfg(file_list, scan_line))

    x = data_pairs[0][0]
    n_files = x.shape[-1]
    cmap = plt.get_cmap('jet')
    c = [cmap(i) for i in plt.linspace(0, 1, n_files)]

    if n_data_pairs == 1:
        ax1 = ptl.subplot(111)
        ax1.set_color_cycle(c)

        x, y = data_pairs[0][0].T, data_pairs[0][1].T
        k = plt.amax(y, axis=0).argsort()[::-1]
        x, y = x[:,k], y[:,k]
        ax1.plot(x, y)

    elif n_data_pairs == 2:
        fig1, (ax1, ax2) = plt.subplots(2, sharex=True)
        fig1.set_size_inches((12, 8), forward=True)
        ax1.set_color_cycle(c)
        ax2.set_color_cycle(c)

        # First data set for sorting
        x, y = data_pairs[0][0], data_pairs[0][1]
        # k = plt.amax(y, axis=0).argsort()[::-1]; varray = varray[k]
        # plot_data_on_axis(ax1, x, convert_to_mm(y), c, k)
        ax1.plot(x, y)

        x, y = data_pairs[1][0], data_pairs[1][1]
        # plot_data_on_axis(ax2, x, y, c, k)
        ax2.plot(x, y)

        # [l.set_linewidth(2) for l in ax2.lines]
        ax2.set_ylim(2, 6)

    elif n_data_pairs == 4:
        fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True)
        fig1.set_size_inches((16, 9), forward=True)

        # First data set for sorting
        x, y = data_pairs[0][0], data_pairs[0][1]
        k = plt.amax(y, axis=0).argsort()[::-1]; varray = varray[k]
        plot_data_on_axis(ax1, x, y, c, k)
        
        x, y = data_pairs[1][0], data_pairs[1][1]
        plot_data_on_axis(ax2, x, y, c, k)

        x, y = data_pairs[2][0], data_pairs[2][1]
        plot_data_on_axis(ax3, x, y, c, k)

        x, y = data_pairs[3][0], data_pairs[3][1]
        plot_data_on_axis(ax4, x, y, c, k)

        [l.set_linewidth(2) for l in ax2.lines + ax4.lines]
        ax4.set_ylim(2, 6)
        
    else:
        raise ValueError(n_data_pairs, "data pairs still needs to be implemented!")

    fig1.subplots_adjust(left=0.2, right=0.8, bottom=0.15, top=0.95)
    fontsize = 22
    ax1.set_ylabel("Vertical\nposition [mm]", fontsize=fontsize, ha='center', labelpad=40)
    ax2.set_ylabel("Vertical\nnorm. emittance [$\mu$m]", fontsize=fontsize, ha='center', labelpad=40)
    for ax in (ax1.get_xticklabels() + ax1.get_yticklabels() + 
               ax2.get_xticklabels() + ax2.get_yticklabels()):
        ax.set_fontsize(fontsize)
    # ax2.set_ylabel("Horizontal\nnormalized emittance [$\mu$m]", ha='center', labelpad=40)
    ax2.set_xlabel("Turn")
    ax2.set_xlim((0, plt.amax(x)))
    lines = ax1.get_lines()
    labels = [l.get_label() for l in lines]
    varray = ['{:1.2e}'.format(v) for v in varray]
    plt.figlegend(lines, varray, loc='right', ncol=1, fontsize='x-large')
    # plt.tight_layout()

def convert_to_mm(x):

    x *= 1e3
    
    return x

def plot_data_on_axis(ax, x, y, c, k):
    ax.set_color_cycle(c)

    x, y = x[:,k], y[:,k]
    ax.plot(x, y)
