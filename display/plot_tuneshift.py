'''
@class TuneshiftPlots
@author Kevin Li, Michael Schenk
@date 07.07.14 (last update)
@Class for creation of tuneshift plots from bunch data.
@copyright CERN
'''
import PySussix
import matplotlib
import pylab as plt
import scipy.io as sio


class TuneshiftPlots(object):

    def __init__(self, rawdata, scan_values, beta_x, beta_y, Q_x, Q_y, Q_s):

        self.scan_values = scan_values
        self.Q_x = Q_x
        self.Q_y = Q_y
        self.Q_s = Q_s

        self._prepare_input_signals(rawdata, beta_x, beta_y)

        # Dictionaries to store results of different cases (horizontal, vertical) x (fft, sussix).
        self.spectral_lines = {}
        self.spectral_intensity = {}

        
    def calculate_spectrogram(self, analyzer):

        if analyzer == 'fft':
            self._calculate_fft_spectrogram()
        elif analyzer == 'sussix':
            self._calculate_sussix_spectrogram()
        else:
            print 'Invalid analyzer argument.'
            
        
    def prepare_plot(self, plane='horizontal', analyzer='fft', xlabel='intensity [particles]', ylabel='mode number',
                     xlimits=((0.,7.1e11)), ylimits=((-4,2))):

        if not plane in ('horizontal', 'vertical'):
            raise ValueError, 'Invalid plane argument.'

        if not analyzer in ('fft', 'sussix'):
            raise ValueError, 'Invalid analyzer argument.'

        if not self.spectral_lines.has_key((analyzer,plane)):
            self.calculate_spectrogram(analyzer)

        # Set up plot environment.
        self._create_axes(xlabel, ylabel, xlimits, ylimits)
        self._create_cropped_cmap()
        
        # Normalize power.
        norm_intensity = self.spectral_intensity[(analyzer,plane)] / plt.amax(self.spectral_intensity[(analyzer,plane)])
        spec_lines = self.spectral_lines[(analyzer,plane)]
        
        x_grid = plt.ones(spec_lines.shape) * plt.array(self.scan_values, dtype='float64')
        for file_i in xrange(self.n_files):
            x, y, z = x_grid[:,file_i], spec_lines[:,file_i], norm_intensity[:,file_i]
            tuneshift_plot = self.ax11.scatter(x, y, s=192*plt.log(1+z), c=z, cmap=self.palette, edgecolors='None')

        # Colorbar
        cb = plt.colorbar(tuneshift_plot, self.ax13, orientation='vertical')
        cb.set_label('Power [normalised]')
                
        plt.tight_layout()

                
    def _prepare_input_signals(self, rawdata, beta_x, beta_y):

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

        self.sx = sx
        self.sy = sy
        self.n_turns = n_turns
        self.n_files = n_files


    def _calculate_fft_spectrogram(self, n_lines=1000):

        # Allocate memory for output.
        oxx, axx = plt.zeros((n_lines, self.n_files)), plt.zeros((n_lines, self.n_files))
        oyy, ayy = plt.zeros((n_lines, self.n_files)), plt.zeros((n_lines, self.n_files))

        for file_i in xrange(self.n_files):
            t = plt.linspace(0, 1, self.n_turns)
            ax = plt.absolute(plt.fft(self.sx[:, file_i]))
            ay = plt.absolute(plt.fft(self.sy[:, file_i]))

            # Amplitude normalisation
            ax /= plt.amax(ax, axis=0)
            ay /= plt.amax(ay, axis=0)

            # Tunes
            if file_i==0:
                tunexfft = t[plt.argmax(ax[:self.n_turns/2], axis=0)]
                tuneyfft = t[plt.argmax(ay[:self.n_turns/2], axis=0)]
                print "\n*** Tunes from FFT"
                print "    tunex:", tunexfft, ", tuney:", tuneyfft, "\n"

            # Tune normalisation
            ox = (t - (self.Q_x[file_i] % 1)) / self.Q_s[file_i]
            oy = (t - (self.Q_y[file_i] % 1)) / self.Q_s[file_i]
    
            # Sort
            CX = plt.rec.fromarrays([ox, ax], names='ox, ax')
            CX.sort(order='ax')
            CY = plt.rec.fromarrays([oy, ay], names='oy, ay')
            CY.sort(order='ay')
            ox, ax, oy, ay = CX.ox[-n_lines:], CX.ax[-n_lines:], CY.oy[-n_lines:], CY.ay[-n_lines:]
            oxx[:,file_i], axx[:,file_i], oyy[:,file_i], ayy[:,file_i] = ox, ax, oy, ay

        self.spectral_lines.update({ ("fft", "horizontal"): oxx, ("fft", "vertical"): oyy })
        self.spectral_intensity.update({ ("fft", "horizontal"): axx, ("fft", "vertical"): ayy })


    def _calculate_sussix_spectrogram(self, n_lines=600):

        # Initialise Sussix object.
        SX = PySussix.Sussix()

        # Allocate memory for output.        
        oxx, axx = plt.zeros((n_lines, self.n_files)), plt.zeros((n_lines, self.n_files))
        oyy, ayy = plt.zeros((n_lines, self.n_files)), plt.zeros((n_lines, self.n_files))

        x, xp, y, yp = self.sx.real, self.sx.imag, self.sy.real, self.sy.imag
        for file_i in xrange(self.n_files):
            SX.sussix_inp(nt1=1, nt2=self.n_turns, idam=2, ir=0, tunex=self.Q_x[file_i] % 1, tuney=self.Q_y[file_i] % 1)
            SX.sussix(x[:,file_i], xp[:,file_i], y[:,file_i], yp[:,file_i], self.sx[:,file_i], self.sx[:,file_i])

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
            SX.ox = (SX.ox - (self.Q_x[file_i] % 1)) / self.Q_s[file_i]
            SX.oy = (SX.oy - (self.Q_y[file_i] % 1)) / self.Q_s[file_i]
    
            # Sort
            CX = plt.rec.fromarrays([SX.ox, SX.ax], names='ox, ax')
            CX.sort(order='ax')
            CY = plt.rec.fromarrays([SX.oy, SX.ay], names='oy, ay')
            CY.sort(order='ay')
            ox, ax, oy, ay = CX.ox, CX.ax, CY.oy, CY.ay
            oxx[:,file_i], axx[:,file_i], oyy[:,file_i], ayy[:,file_i] = ox, ax, oy, ay

        self.spectral_lines.update({ ("sussix", "horizontal"): oxx, ("sussix", "vertical"): oyy })
        self.spectral_intensity.update({ ("sussix", "horizontal"): axx, ("sussix", "vertical"): ayy })

        
    def _create_axes(self, xlabel, ylabel, xlimits, ylimits):

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

        self.ax11 = ax11
        self.ax13 = ax13

                
    def _create_cropped_cmap(self, n_segments = 12, cmap='hot'):

        new_cmap = plt.cm.get_cmap(cmap, n_segments)(range(n_segments))
        new_cmap = plt.delete(new_cmap, (0, -1), 0)
        palette = matplotlib.colors.LinearSegmentedColormap.from_list("palette", new_cmap)

        self.palette = palette
