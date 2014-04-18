import PySussix
import matplotlib
import pylab as plt
import scipy.io as sio
from Pyheana.load.load_data import *


def axes_setup(scan_parameter):

    # xlimits = (0, 7e11)
    # ylimits = (-8, 8)
    # xlimits = (0, 3e12)
    # ylimits = (-2, 8)
    xlimits = (0, 5e11)
    ylimits = (-3, 3)
    xlabel = scan_parameter

    # fig1, (ax11, ax12) = plt.subplots(2, sharex=True, sharey=True)
    # fig2, (ax21, ax22) = plt.subplots(2, sharex=True, sharey=True)

    # majorFormatter = plt.FormatStrFormatter('%3.2e')
    # sctfcFormatter = plt.ScalarFormatter(useOffset=False, useMathText=True)
    # sctfcFormatter.set_scientific(True)

    # Plot environment
    fig = plt.figure(figsize=(12, 10))
    ratio = 20
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
        
    return [ax11, ax12, ax13]

def plot_tuneshift(data, file_list, beta_x=None, beta_y=None, Qx=None, Qy=None, Qs=None,
                   varray=None, scan_parameter=None):

    # Set up axes
    ax11, ax12, ax13 = axes_setup(scan_parameter)
    ax21, ax22, ax23 = axes_setup(scan_parameter)
    
    # Prepare data
    A = data
    [n_cols, n_turns, n_files]= A.shape
    # # Get parameters from cfg files
    # if file_list:
    #     try:
    #         beta_x = get_value_from_cfg(file_list, 'Horizontal_beta_function_at_the_kick_sections_[m]:')
    #         beta_y = get_value_from_cfg(file_list, 'Vertical_beta_function_at_the_kick_sections_[m]:')
    #         Qx = get_value_from_cfg(file_list, 'Horizontal_tune:')
    #         Qy = get_value_from_cfg(file_list, 'Vertical_tune:')
    #         Qs = get_value_from_cfg(file_list, 'Synchrotron_tune:')
    #     except IndexError:
    #         beta_x = get_value_from_new_cfg(file_list, 'betax')
    #         beta_y = get_value_from_new_cfg(file_list, 'betay')
    #         Qx = get_value_from_new_cfg(file_list, 'Qx')
    #         Qy = get_value_from_new_cfg(file_list, 'Qy')
    #         Qs = get_value_from_new_cfg(file_list, 'Synchrotron_tune:')
    # else:
    #     beta_x = plt.ones(n_files) * 100
    #     beta_y = plt.ones(n_files) * 100
    #     Qx = plt.ones(n_files) * 0.1
    #     Qy = plt.ones(n_files) * 0.1
    #     Qs = plt.ones(n_files) * 0.1
    # if scan_line:
    #     try:
    #         varray = get_value_from_cfg(file_list, scan_line)
    #     except IndexError:
    #         varray = get_value_from_new_cfg(file_list, scan_line)
    # else:
    #     varray = plt.arange(n_files)

    # Prepare signals
    T = range(n_turns)
    window_width = n_turns
    sx = plt.zeros((window_width, n_files), dtype='complex')
    sy = plt.zeros((window_width, n_files), dtype='complex')
    for i in range(n_files):
        t, x, xp, y, yp, dz, dp = A[:7,:,i]

        # Floquet transformation
        xp *= beta_x[i]
        yp *= beta_y[i]

        # Signal
        sx[:,i] = x[:window_width] - 1j * xp[:window_width]
        sy[:,i] = y[:window_width] - 1j * yp[:window_width]

    # FFT - Spectrogram
    sc11, sc12 = fft_spectrogram(signals=(sx, sy), axis=(ax11, ax12), tunes=(Qx, Qy, Qs),
                                 scan_values = varray)

    # Sussix - Spectrogram
    sc21, sc22 = sussix_spectrogram(signals=(sx, sy), axis=(ax21, ax22), tunes=(Qx, Qy, Qs),
                                    scan_values = varray)
 
    # Colorbar
    cb = plt.colorbar(sc11, ax13, orientation='vertical')
    cb = plt.colorbar(sc21, ax23, orientation='vertical')
    cb.set_label('Power [normalised]')
    # cb2 = fig2.colorbar(sc21, ax=ax21, orientation='vertical')
    # cb2.set_label('Power [normalised]', fontsize=fontsize)
    # for ticklabel in cb1.ax.get_yticklabels() + cb2.ax.get_yticklabels():
    #     ticklabel.set_fontsize(lfontsize)

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

    plt.tight_layout()

def cropped_cmap(n_segments = 12, cmap='jet'):

    new_cmap = plt.cm.get_cmap(cmap, n_segments)(range(n_segments))
    new_cmap = plt.delete(new_cmap, (0, -1), 0)
    palette = matplotlib.colors.LinearSegmentedColormap.from_list("palette", new_cmap)

    return palette

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

        # Sort
        CX = plt.rec.fromarrays([ox, ax], names='ox, ax')
        CX.sort(order='ax')
        CY = plt.rec.fromarrays([oy, ay], names='oy, ay')
        CY.sort(order='ay')
        ox, ax, oy, ay = CX.ox[-n_lines:], CX.ax[-n_lines:], CY.oy[-n_lines:], CY.ay[-n_lines:]
        oxx[:,i], axx[:,i], oyy[:,i], ayy[:,i] = ox, ax, oy, ay

        # xx, yy = plt.meshgrid(plt.array(scan_values, dtype='float64'), ox)
        # yy, zz = ox, ax
        # sc1 = ax1.scatter(xx[:,i], yy, s=160*plt.log(1+zz), c=zz, cmap=palette, edgecolors='None')
        # xx, yy = plt.meshgrid(plt.array(scan_values, dtype='float64'), oy)
        # yy, zz = oy, ay
        # sc2 = ax2.scatter(xx[:,i], yy, s=160*plt.log(1+zz), c=zz, cmap=palette, edgecolors='None')

    axx *= 1 / plt.amax(axx)
    ayy *= 1 / plt.amax(ayy)
    xx = plt.ones((n_lines, n_files)) * plt.array(scan_values, dtype='float64')
    for i in xrange(n_files):
        x, y, z = xx[:,i], oxx[:,i], axx[:,i]
        sc1 = ax1.scatter(x, y, s=192*plt.log(1+z), c=z, cmap=palette, edgecolors='None')
        x, y, z = xx[:,i], oyy[:,i], ayy[:,i]
        sc2 = ax2.scatter(x, y, s=192*plt.log(1+z), c=z, cmap=palette, edgecolors='None')

    return [sc1, sc2]

def sussix_spectrogram(signals=None, axis=None, tunes=None, scan_values=None):

    # Initialise Sussix object
    SX = PySussix.Sussix()

    palette = cropped_cmap(cmap='hot')
    
    sx, sy = signals[0], signals[1]
    ax1, ax2 = axis[0], axis[1]
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

        # xx, yy = plt.meshgrid(plt.array(scan_values, dtype='float64'), ox)
        # yy, zz = ox, ax
        # # sc1 = ax1.scatter(xx[:,i], yy, s=192*plt.log(1+zz), c=zz, cmap=palette, edgecolors='None')
        # sc1 = ax1.scatter(xx[:,i], yy, s=104*zz, c=zz, cmap=palette, edgecolors='None')
        # xx, yy = plt.meshgrid(plt.array(scan_values, dtype='float64'), oy)
        # yy, zz = oy, ay
        # # sc2 = ax2.scatter(xx[:,i], yy, s=192*plt.log(1+zz), c=zz, cmap=palette, edgecolors='None')
        # sc2 = ax2.scatter(xx[:,i], yy, s=104*zz, c=zz, cmap=palette, edgecolors='None')

    axx *= 1 / plt.amax(axx)
    ayy *= 1 / plt.amax(ayy)
    xx = plt.ones((600, n_files)) * plt.array(scan_values, dtype='float64')
    for i in xrange(n_files):
        x, y, z = xx[:,i], oxx[:,i], axx[:,i]
        sc1 = ax1.scatter(x, y, s=192*plt.log(1+z), c=z, cmap=palette, edgecolors='None')
        x, y, z = xx[:,i], oyy[:,i], ayy[:,i]
        sc2 = ax2.scatter(x, y, s=192*plt.log(1+z), c=z, cmap=palette, edgecolors='None')

    # sio.savemat("Tunespectrum.mat", {'ox':oxx, 'ax':axx, 'oy':oyy, 'ay':ayy, 'xx':xx})
    
    return [sc1, sc2]
