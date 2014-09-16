'''
@author Kevin Li, Michael Schenk
@date 10.07.14 (last update)
@brief Collection of methods to generate tuneshift plots.
@copyright CERN
'''
import PySussix
import matplotlib
import pylab as plt
import scipy.io as sio


def calculate_tuneshifts(rawdata, beta_x, beta_y, Q_x, Q_y, Q_s, analyzer='fft', n_lines=None):

    # Prepare signals and perform spectral analysis.
    sx, sy = _prepare_input_signals(rawdata, beta_x, beta_y)
    if analyzer == 'fft':
        spectra = _calculate_spectra_fft(sx, sy, Q_x, Q_y, Q_s, n_lines=1000)
    elif analyzer == 'sussix':
        spectra = _calculate_spectra_sussix(sx, sy, Q_x, Q_y, Q_s, n_lines=600)
    else:
        print 'Invalid analyzer argument.'

    return spectra

def calculate_1d_sussix_spectrum(turn, window_width, x, xp, qx):
    macroparticlenumber = len(x)
    tunes = plt.zeros(macroparticlenumber)

    # Initialise Sussix object
    SX = PySussix.Sussix()
    SX.sussix_inp(nt1=1, nt2=window_width, idam=1, ir=0, tunex=qx)

    n_particles_to_analyse_10th = int(macroparticlenumber/10)
    print 'Running SUSSIX analysis ...'
    for i in xrange(macroparticlenumber):
        if not i%n_particles_to_analyse_10th: print '  Particle', i
        SX.sussix(x[i,turn:turn+window_width], xp[i,turn:turn+window_width],
                  x[i,turn:turn+window_width], xp[i,turn:turn+window_width],
                  x[i,turn:turn+window_width], xp[i,turn:turn+window_width]) # this line is not used by sussix!
        tunes[i] = plt.absolute(SX.ox[plt.argmax(SX.ax)])

    return tunes

def plot_tuneshifts_2(ax, spectrum, scan_values):

    (spectral_lines, spectral_intensity) = spectrum

    # Normalize power.
    normalized_intensity = spectral_intensity / plt.amax(spectral_intensity)

    # Prepare plot environment.
    palette    = _create_cropped_cmap()

    x_grid = plt.ones(spectral_lines.shape) * plt.array(scan_values, dtype='float64')
    for file_i in xrange(len(scan_values)):
        x, y, z = x_grid[:,file_i], spectral_lines[:,file_i], normalized_intensity[:,file_i]
        tuneshift_plot = ax[0].scatter(x, y, s=192*plt.log(1+z), c=z, cmap=palette, edgecolors='None')

    # Colorbar
    cb = plt.colorbar(tuneshift_plot, ax[1], orientation='vertical')
    cb.set_label('Power [normalised]')

    # plt.tight_layout()

def plot_tuneshifts(spectrum, scan_values, xlabel='intensity [particles]', ylabel='mode number',
                    xlimits=((0.,7.1e11)), ylimits=((-4,2))):

    (spectral_lines, spectral_intensity) = spectrum

    # Normalize power.
    normalized_intensity = spectral_intensity / plt.amax(spectral_intensity)

    # Prepare plot environment.
    ax11, ax13 = _create_axes(xlabel, ylabel, xlimits, ylimits)
    palette    = _create_cropped_cmap()

    x_grid = plt.ones(spectral_lines.shape) * plt.array(scan_values, dtype='float64')
    for file_i in xrange(len(scan_values)):
        x, y, z = x_grid[:,file_i], spectral_lines[:,file_i], normalized_intensity[:,file_i]
        tuneshift_plot = ax11.scatter(x, y, s=192*plt.log(1+z), c=z, cmap=palette, edgecolors='None')

    # Colorbar
    cb = plt.colorbar(tuneshift_plot, ax13, orientation='vertical')
    cb.set_label('Power [normalised]')

    plt.tight_layout()

def _prepare_input_signals(rawdata, beta_x, beta_y):

    n_variables, n_turns, n_files = rawdata.shape

    sx = plt.zeros((n_turns, n_files), dtype='complex')
    sy = plt.zeros((n_turns, n_files), dtype='complex')

    for file_i in range(n_files):
        x, xp, y, yp = rawdata[1:5,:,file_i]

        # Floquet transformation
        xp *= beta_x[file_i]
        yp *= beta_y[file_i]

        # Signal
        sx[:,file_i] = x - 1j * xp
        sy[:,file_i] = y - 1j * yp

    return sx, sy


def _create_axes(xlabel, ylabel, xlimits, ylimits):

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
    ax11.set_xlim(xlimits)
    ax11.set_ylabel(ylabel)
    ax11.set_ylim(ylimits)

    return ax11, ax13
        
                
def _create_cropped_cmap(n_segments = 12, cmap='hot'):

    new_cmap = plt.cm.get_cmap(cmap, n_segments)(range(n_segments))
    new_cmap = plt.delete(new_cmap, (0, -1), 0)
    palette = matplotlib.colors.LinearSegmentedColormap.from_list("palette", new_cmap)
        
    return palette


def _calculate_spectra_fft(sx, sy, Q_x, Q_y, Q_s, n_lines):

    n_turns, n_files = sx.shape
        
    # Allocate memory for output.
    oxx, axx = plt.zeros((n_lines, n_files)), plt.zeros((n_lines, n_files))
    oyy, ayy = plt.zeros((n_lines, n_files)), plt.zeros((n_lines, n_files))

    for file_i in xrange(n_files):
        t = plt.linspace(0, 1, n_turns)
        ax = plt.absolute(plt.fft(sx[:, file_i]))
        ay = plt.absolute(plt.fft(sy[:, file_i]))

        # Amplitude normalisation
        ax /= plt.amax(ax, axis=0)
        ay /= plt.amax(ay, axis=0)
    
        # Tunes
        if file_i==0:
            tunexfft = t[plt.argmax(ax[:n_turns/2], axis=0)]
            tuneyfft = t[plt.argmax(ay[:n_turns/2], axis=0)]
            print "\n*** Tunes from FFT"
            print "    tunex:", tunexfft, ", tuney:", tuneyfft, "\n"

        # Tune normalisation
        ox = (t - (Q_x[file_i] % 1)) / Q_s[file_i]
        oy = (t - (Q_y[file_i] % 1)) / Q_s[file_i]
    
        # Sort
        CX = plt.rec.fromarrays([ox, ax], names='ox, ax')
        CX.sort(order='ax')
        CY = plt.rec.fromarrays([oy, ay], names='oy, ay')
        CY.sort(order='ay')
        ox, ax, oy, ay = CX.ox[-n_lines:], CX.ax[-n_lines:], CY.oy[-n_lines:], CY.ay[-n_lines:]
        oxx[:,file_i], axx[:,file_i], oyy[:,file_i], ayy[:,file_i] = ox, ax, oy, ay

    spectra = {}
    spectra['horizontal'] = (oxx, axx)
    spectra['vertical']   = (oyy, ayy)
        
    return spectra
    

def _calculate_spectra_sussix(sx, sy, Q_x, Q_y, Q_s, n_lines):

    n_turns, n_files = sx.shape

    # Allocate memory for output.        
    oxx, axx = plt.zeros((n_lines, n_files)), plt.zeros((n_lines, n_files))
    oyy, ayy = plt.zeros((n_lines, n_files)), plt.zeros((n_lines, n_files))

    # Initialise Sussix object.
    SX = PySussix.Sussix()
    
    x, xp, y, yp = sx.real, sx.imag, sy.real, sy.imag
    for file_i in xrange(n_files):
        SX.sussix_inp(nt1=1, nt2=n_turns, idam=2, ir=0, tunex=Q_x[file_i] % 1, tuney=Q_y[file_i] % 1)
        SX.sussix(x[:,file_i], xp[:,file_i], y[:,file_i], yp[:,file_i], sx[:,file_i], sx[:,file_i])

        # Amplitude normalisation
        SX.ax /= plt.amax(SX.ax)
        SX.ay /= plt.amax(SX.ay)

        # Tunes
        SX.ox = plt.absolute(SX.ox)
        SX.oy = plt.absolute(SX.oy)
        if file_i==0:
            tunexsx = SX.ox[plt.argmax(SX.ax)]
            tuneysx = SX.oy[plt.argmax(SX.ay)]
            print "\n*** Tunes from Sussix"
            print "    tunex", tunexsx, ", tuney", tuneysx, "\n"

        # Tune normalisation
        SX.ox = (SX.ox - (Q_x[file_i] % 1)) / Q_s[file_i]
        SX.oy = (SX.oy - (Q_y[file_i] % 1)) / Q_s[file_i]
    
        # Sort
        CX = plt.rec.fromarrays([SX.ox, SX.ax], names='ox, ax')
        CX.sort(order='ax')
        CY = plt.rec.fromarrays([SX.oy, SX.ay], names='oy, ay')
        CY.sort(order='ay')
        ox, ax, oy, ay = CX.ox, CX.ax, CY.oy, CY.ay
        oxx[:,file_i], axx[:,file_i], oyy[:,file_i], ayy[:,file_i] = ox, ax, oy, ay

    spectra = {}
    spectra['horizontal'] = (oxx, axx)
    spectra['vertical']   = (oyy, ayy)
        
    return spectra
