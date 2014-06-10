import PySussix
import matplotlib
import pylab as plt
import scipy.io as sio
from Pyheana.load.load_data import *


def axes_setup(scan_parameter, plane='y'):

    # xlimits = (0, 7e11)
    # ylimits = (-8, 8)
    # xlimits = (0, 3e12)
    # ylimits = (-2, 8)
    #xlimits = (0, 6.2e11)
    xlimits = (0, 1000)
    ylimits = (-3, 2)
    xlabel = scan_parameter

    # fig1, (ax11, ax12) = plt.subplots(2, sharex=True, sharey=True)
    # fig2, (ax21, ax22) = plt.subplots(2, sharex=True, sharey=True)

    # majorFormatter = plt.FormatStrFormatter('%3.2e')
    # sctfcFormatter = plt.ScalarFormatter(useOffset=False, useMathText=True)
    # sctfcFormatter.set_scientific(True)

    # Plot environment
    ratio = 20
    if len(plane) == 1:
        fig = plt.figure(figsize=(12, 8))
            
        gs = matplotlib.gridspec.GridSpec(1, ratio)
        ax11 = plt.subplot(gs[0,:ratio-1])
        ax13 = plt.subplot(gs[:,ratio-1])

        # Prepare plot axes
        if plane == 'x':
            ax11.set_ylabel('Horizontal mode number')
        elif plane == 'y':
            ax11.set_ylabel('Vertical mode number')
        ax11.grid(color='w')
        ax11.set_axis_bgcolor('0.35')
        ax11.set_xlabel(xlabel);
        ax11.set_xlim(xlimits)
        ax11.set_ylim(ylimits)

        return [ax11, ax13]

    elif len(plane) == 2:
        fig = plt.figure(figsize=(12, 10))

        gs = matplotlib.gridspec.GridSpec(2, ratio)
        ax11 = plt.subplot(gs[0,:ratio-1])
        ax12 = plt.subplot(gs[1,:ratio-1])
        ax13 = plt.subplot(gs[:,ratio-1])

        # Prepare plot axes
        ax11.set_ylabel('Horizontal mode number')
        ax12.set_ylabel('Vertical mode number')
        for ax in ax11, ax12:
            ax.grid(color='w')
            ax.set_axis_bgcolor('0.35')
            ax.set_xlabel(xlabel);
            ax.set_xlim(xlimits)
            ax.set_ylim(ylimits)

        return [ax11, ax12, ax13]

    else:
        raise ValueError

    # fig2 = plt.figure(figsize=(12, 10))
    # gs = matplotlib.gridspec.GridSpec(2, 4)
    # ax21 = plt.subplot(gs[0,:3])
    # ax22 = plt.subplot(gs[1,:3])
    # ax23 = plt.subplot(gs[:,3])
        
    # ax21.set_ylabel('Horizontal mode number')
    # ax22.set_ylabel('Vertical mode number')
    # for ax in ax21, ax22:
    #     ax.grid(color='w')
    #     ax.set_axis_bgcolor('0.35')
    #     ax.set_xlabel(xlabel);
    #     ax.set_xlim(xlimits)
    #     ax.set_ylim(ylimits)

def plot_tuneshift(data, file_list, beta_x=None, beta_y=None, Qx=None, Qy=None, Qs=None,
                   varray=None, scan_parameter=None):
    
    # Prepare data
    A = data
    [n_cols, n_turns, n_files]= A.shape

    # Prepare signals
    T = range(n_turns)
    window_width = n_turns
    sx = plt.zeros((window_width, n_files), dtype='complex')
    sy = plt.zeros((window_width, n_files), dtype='complex')
    for i in range(n_files):
        t, x, xp, y, yp, z, dp = A[:7,:,i]

        # Floquet transformation
        xp *= beta_x[i]
        yp *= beta_y[i]

        # Signal
        sx[:,i] = x[:window_width] - 1j * xp[:window_width]
        sy[:,i] = y[:window_width] - 1j * yp[:window_width]

    # FFT - Spectrogram
    ox, ax, oy, ay  = fft_spectrum((sx, sy), tunes=(Qx, Qy, Qs))
    plot_spectrogram(ox, ax, oy, ay, scan_parameter, varray)

    # Sussix - Spectrogram
    # ox, ax, oy, ay = sussix_spectrum((sx, sy), tunes=(Qx, Qy, Qs))
    # plot_spectrogram(ox, ax, oy, ay, scan_parameter, varray)

    #liney = ax2.axvline(6.5, c='orange', ls='--', lw=2)
    #liney = ax2.axvline(13.5, c='orange', ls='--', lw=2)
    #liney = ax2.axhline(1, c='g', ls='--', lw=2)

    #if Input['MakeMovie']==True:
    #for i in range(len(x)):
        #linex = ax1.axvline(x[i], c='r', lw=2)
        #liney = ax2.axvline(x[i], c='r', lw=2)
        #draw()
        #tmpname = 'Tuneshift--'+str('%03d' % (i+1))+'.'+Input['Format']
        #savefig(tmpname)
        #ax1.lines.remove(linex)
        #ax2.lines.remove(liney)
    
    ## x-tuneshift fit
    ##~ ix0 = abs(y[-21:]).argmin()
    ##~ tunexmax.append(y[-21+ix0])
    ##~ ax1.scatter(xx[:20,0], tunexmax[:20], s=100, c='r', marker='>', alpha=1)

def plot_spectrogram(ox, ax, oy, ay, scan_parameter=None, scan_values=None):

    palette = cropped_cmap(cmap='hot')
    ax1, ax3 = axes_setup(scan_parameter)
    # ax1, ax2, ax3 = axes_setup(scan_parameter, plane=('x', 'y'))

    ax *= 1 / plt.amax(ax)
    ay *= 1 / plt.amax(ay)
    xx = plt.ones(ox.shape) * plt.array(scan_values, dtype='float64')
    n_lines, n_files = xx.shape
    for i in xrange(n_files):
        # x, y, z = xx[:,i], ox[:,i], ax[:,i]
        # sc1 = ax1.scatter(x, y, s=192*plt.log(1+z), c=z, cmap=palette, edgecolors='None')
        x, y, z = xx[:,i], oy[:,i], ay[:,i]
        sc1 = ax1.scatter(x, y, s=192*plt.log(1+z), c=z, cmap=palette, edgecolors='None')

    # Colorbar
    cb = plt.colorbar(sc1, ax3, orientation='vertical')
    cb.set_label('Power [normalised]')

    plt.tight_layout()

def cropped_cmap(n_segments = 12, cmap='jet'):

    new_cmap = plt.cm.get_cmap(cmap, n_segments)(range(n_segments))
    new_cmap = plt.delete(new_cmap, (0, -1), 0)
    palette = matplotlib.colors.LinearSegmentedColormap.from_list("palette", new_cmap)

    return palette

def fft_spectrum(signals, tunes=None, scan_values=None):

    sx, sy = signals[0], signals[1]
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

        # Sort
        CX = plt.rec.fromarrays([ox, ax], names='ox, ax')
        CX.sort(order='ax')
        CY = plt.rec.fromarrays([oy, ay], names='oy, ay')
        CY.sort(order='ay')
        ox, ax, oy, ay = CX.ox[-n_lines:], CX.ax[-n_lines:], CY.oy[-n_lines:], CY.ay[-n_lines:]
        oxx[:,i], axx[:,i], oyy[:,i], ayy[:,i] = ox, ax, oy, ay

    return [oxx, axx, oyy, ayy]

def sussix_spectrum(signals=None, tunes=None, scan_values=None):

    # Initialise Sussix object
    SX = PySussix.Sussix()
    
    sx, sy = signals[0], signals[1]
    Qx, Qy, Qs = tunes[0], tunes[1], tunes[2]
    x, xp, y, yp = sx.real, sx.imag, sy.real, sy.imag
    
    n_lines = 600
    n_turns, n_files = sx.shape
    oxx, axx = plt.zeros((n_lines, n_files)), plt.zeros((n_lines, n_files))
    oyy, ayy = plt.zeros((n_lines, n_files)), plt.zeros((n_lines, n_files))

    for i in xrange(n_files):
        SX.sussix_inp(nt1=1, nt2=n_turns, idam=2, ir=0, tunex=Qx[i] % 1, tuney=Qy[i] % 1)
        SX.sussix(x[:,i], xp[:,i], y[:,i], yp[:,i], sx[:,i], sx[:,i])
        # Amplitude normalisation
        SX.ax /= plt.amax(SX.ax)
        SX.ay /= plt.amax(SX.ay)
        # Tunes
        SX.ox = plt.absolute(SX.ox)
        SX.oy = plt.absolute(SX.oy)
        if i==0:
            tunexsx = SX.ox[plt.argmax(SX.ax)]
            tuneysx = SX.oy[plt.argmax(SX.ay)]
            print "\n*** Tunes from Sussix"
            print "    tunex", tunexsx, ", tuney", tuneysx, "\n"
        # Tune normalisation
        SX.ox = (SX.ox - (Qx[i] % 1)) / Qs[i]
        SX.oy = (SX.oy - (Qy[i] % 1)) / Qs[i]
        #~SX.ox = (SX.ox-tunexsx)/self.Qs[j]
        #~SX.oy = (SX.oy-tuneysx)/self.Qs[j]

        # Sort
        CX = plt.rec.fromarrays([SX.ox, SX.ax], names='ox, ax')
        CX.sort(order='ax')
        CY = plt.rec.fromarrays([SX.oy, SX.ay], names='oy, ay')
        CY.sort(order='ay')
        ox, ax, oy, ay = CX.ox, CX.ax, CY.oy, CY.ay
        oxx[:,i], axx[:,i], oyy[:,i], ayy[:,i] = ox, ax, oy, ay

    return [oxx, axx, oyy, ayy]
