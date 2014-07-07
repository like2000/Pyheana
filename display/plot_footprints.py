'''
@class TuneFootprintPlots
@author Kevin Li, Michael Schenk
@date 07.07.14 (last update)
@Class for creation of footprint plots from particle and bunch data.
@copyright CERN
'''
import matplotlib
import matplotlib.colors as colors
import matplotlib.cm as cm
from matplotlib.ticker import MaxNLocator

import PySussix
import Pyheana.load as ld
import h5py
import numpy as np
import pylab as plt
import scipy.io as sio
import os, sys


class TuneFootprintPlots(object):

    def __init__(self, bunch_data_file, particle_data_file, init_turn_to_read=0, n_turns_to_read=1024,
                 macroparticles_stride=1):

        self.init_turn_to_read     = init_turn_to_read
        self.n_turns_to_read       = n_turns_to_read
        self.macroparticles_stride = macroparticles_stride
        
        self._prepare_input(particle_data_file, bunch_data_file)
        self.set_emittance_plot_range()
        self.set_footprint_plot_range()

                
    def set_emittance_plot_range(self, horizontal_ylimits=((0, 10)), vertical_ylimits=((0, 10)),
                                 xlimits=((0,16384))):

        self.horizontal_emittance_ylimits = horizontal_ylimits
        self.vertical_emittance_ylimits   = vertical_ylimits
        self.emittance_xlimits            = xlimits

                
    def set_footprint_plot_range(self, xlimits=((0.290, 0.322)), ylimits=((0.315,0.333)), zlimits=((0,250))):

        self.footprint_xlimits = xlimits
        self.footprint_ylimits = ylimits
        self.footprint_zlimits = zlimits
        
        
    def create_single_plot(self, turn=0, window_width=64, zaxis='slice_index', zlabel='slice index'):

        print ''
        print 'Calculating footprint for turns [%d, %d] ...' % (turn, turn+window_width)
        
        self._create_axes()
        
        # Calculate tune footprint using sussix. Choose z axis.
        # ! WARNING: Z AXIS VALUES ARE ONLY EVALUATED AT THE RIGHT BOUNDARY OF THE WINDOW !
        if zaxis == 'slice_index':
            zcol = self.slice_index[:, turn-self.init_turn_to_read+window_width]
        elif zaxis == 'j_x':
            zcol = self.x[:, turn-self.init_turn_to_read+window_width] ** 2 + \
                   self.xp[:, turn-self.init_turn_to_read+window_width] ** 2
            zcol = zcol / self.beta_x
        elif zaxis == 'j_y':
            zcol = self.y[:, turn-self.init_turn_to_read+window_width] ** 2 + \
                   self.yp[:, turn-self.init_turn_to_read+window_width] ** 2
            zcol = zcol / self.beta_y
        elif zaxis == 'j_xy':
            j_x  = self.x[:, turn-self.init_turn_to_read+window_width] ** 2 + \
                   self.xp[:, turn-self.init_turn_to_read+window_width] ** 2
            j_x  = j_x / self.beta_x
            j_y  = self.y[:, turn-self.init_turn_to_read+window_width] ** 2 + \
                   self.yp[:, turn-self.init_turn_to_read+window_width] ** 2
            j_y  = j_y / self.beta_y
            zcol = j_x + j_y
        elif zaxis == 'j_z':
            print 'Not supported for the moment.'
            zcol = 'black'
        else:
            print 'No z axis chosen or unknown type (known are slice_index, j_x, j_y, j_xy, j_z).'
            zcol = 'black'
            
        tunes_x, tunes_y = self._calculate_sussix_spectrum(turn-self.init_turn_to_read, window_width)
        tune_fp = self.ax1.scatter(tunes_x, tunes_y, c=zcol, marker='o', lw=0,
                                   vmin=self.footprint_zlimits[0], vmax=self.footprint_zlimits[1])

        # Add colorbar if needed.
        if not zcol == 'black':
            cbaxes = self.fig.add_axes([0.17, 0.764, 0.5, 0.01]) 
            cb = plt.colorbar(tune_fp, ax=self.ax1, cax = cbaxes, orientation='horizontal')
            cb.set_clim(vmin=self.footprint_zlimits[0], vmax=self.footprint_zlimits[1])
            cb.formatter.set_powerlimits((2, 2))
            cb.set_label(zlabel)
            cb.outline.set_linewidth(1)
            cb.locator = MaxNLocator(nbins=6)
            cb.update_ticks()

        # Add main tune.
        self.ax1.plot([self.q_x, self.footprint_xlimits[1]], [self.q_y, self.q_y], color='black', linestyle='dashed', lw=1.)
        self.ax1.plot([self.q_x, self.q_x], [self.q_y, self.footprint_ylimits[1]], color='black', linestyle='dashed', lw=1.)
        self.ax1.plot(self.q_x, self.q_y, marker='x', mec='black', mfc='None', mew=1., ms=10)

        # Add emittance plots and rectangle indicating where turn window is.
        self.ax2.plot(self.epsn_x, c='blue', lw=1)
        self.ax3.plot(self.epsn_y, c='green', lw=1)
        self.ax2.axvspan(turn, turn+window_width, facecolor='red', alpha=0.6)
        
        # Add projection histograms.
        weights = np.ones(len(tunes_x))/self.n_particles_to_analyse
        bins_x  = plt.linspace(self.footprint_xlimits[0], self.footprint_xlimits[1], 100)
        x_y, x_x, x_o = self.ax4.hist(tunes_x, bins_x, weights=weights, color='grey', alpha=0.7)
        bins_y = plt.linspace(self.footprint_ylimits[0], self.footprint_ylimits[1], 100)
        y_y, y_x, y_o = self.ax5.hist(tunes_y, bins_y, weights=weights, orientation='horizontal', color='grey', alpha=0.7)

        self.ax4.axvline(self.q_x, lw=1., color='black', linestyle='dashed')
        self.ax5.axhline(self.q_y, lw=1., color='black', linestyle='dashed')


    def create_plot_series(self, init_turn=0, final_turn=1024, window_width=64, step=64,
                           zaxis='slice_index', zlabel='slice index', output_path='./'):

        files = []
        for i in range(init_turn, final_turn+window_width, step):
            self.create_single_plot(turn=i, window_width=window_width, zaxis=zaxis, zlabel=zlabel)
            file_name = output_path + 'turn_%06d.png'%(i)
            plt.savefig(file_name)
            files.append(file_name)

        return files
    

    def make_movie(self, files, movie_name='animation.mpg', fps=3):
        ''' Needs mencoder '''
        # Mencoder accepts list of files in a .txt.
        file_list_txt = open('list.txt', 'w')
        for filename in files:
            file_list_txt.write("%s\n"%(filename))
        file_list_txt.close()
        os.system("mencoder 'mf://@list.txt' -mf type=png:fps=%d -ovc lavc -lavcopts vcodec=wmv2 -oac copy -o %s"%(fps, movie_name))
        

    def _calculate_sussix_spectrum(self, turn, window_width):
    
        # Initialise Sussix object
        SX = PySussix.Sussix()
        SX.sussix_inp(nt1=1, nt2=window_width, idam=2, ir=0, tunex=self.q_x, tuney=self.q_y)

        tunes_x = plt.zeros(self.n_particles_to_analyse)
        tunes_y = plt.zeros(self.n_particles_to_analyse)

        n_particles_to_analyse_10th = int(self.n_particles_to_analyse/10)
        print 'Running SUSSIX analysis ...'
        for i in xrange(self.n_particles_to_analyse):
            if not i%n_particles_to_analyse_10th: print '  Particle', i
            SX.sussix(self.x[i,turn:turn+window_width], self.xp[i,turn:turn+window_width],
                      self.y[i,turn:turn+window_width], self.yp[i,turn:turn+window_width],
                      self.x[i,turn:turn+window_width], self.xp[i,turn:turn+window_width]) # this line is not used by sussix!
            tunes_x[i] = plt.absolute(SX.ox[plt.argmax(SX.ax)])
            tunes_y[i] = plt.absolute(SX.oy[plt.argmax(SX.ay)])
                
        return tunes_x, tunes_y

        
    def _prepare_input(self, particle_data_file, bunch_data_file):

        # Load particle data.
        keys_to_read  = ['x', 'y', 'xp', 'yp', 'slice_index']
        particle_data = ld.read_particle_data(filename=particle_data_file, format='h5', keys_to_read=keys_to_read,
                                              init_turn_to_read=self.init_turn_to_read,
                                              n_turns_to_read=self.n_turns_to_read,
                                              macroparticles_stride=self.macroparticles_stride)
        
        x, xp = particle_data['x'], particle_data['xp']
        y, yp = particle_data['y'], particle_data['yp']
        # z, dp = particle_data['z'], particle_data['dp']
        slice_index = particle_data['slice_index']

        # Load bunch data and attributes.
        bunch_file_handle = h5py.File(bunch_data_file, 'r')
        bunch_data        = bunch_file_handle['Bunch']

        Q_x    = bunch_file_handle.attrs['Q_x']
        Q_y    = bunch_file_handle.attrs['Q_y']
        beta_x = bunch_file_handle.attrs['beta_x']
        beta_y = bunch_file_handle.attrs['beta_y']
        # Q_s    = bunch_file_handle.attrs['Q_s']
        # C      = bunch_file_handle.attrs['C']
        # eta    = bunch_file_handle.attrs['eta']
        # beta_z = C * eta / (2. * np.pi * Q_s)

        # Preprocess particle and bunch data and make them class attributes.
        # Floquet transformation
        xp *= beta_x
        yp *= beta_y

        self.x  = x
        self.y  = y
        self.xp = xp
        self.yp = yp
        self.slice_index = slice_index
        self.n_particles_to_analyse = len(x[:,0])

        self.beta_x = beta_x
        self.beta_y = beta_y                
        self.q_x    = Q_x%1
        self.q_y    = Q_y%1
        self.epsn_x = bunch_data['epsn_x']
        self.epsn_y = bunch_data['epsn_y']

        self.n_total_turns = len(self.epsn_x)
                        
        
    def _create_axes(self):

        if hasattr(self, 'fig'):
            self.fig.clf()
            self.ax1.cla()
            self.ax2.cla()
            self.ax3.cla()
            self.ax4.cla()
            self.ax5.cla()
        else:
            self.fig = plt.figure(figsize=(15, 16))

        plt.subplots_adjust(left=0.1, right=0.92, top=0.96, bottom=0.06)
        
        gs = matplotlib.gridspec.GridSpec(35, 5)
        ax1 = plt.subplot(gs[6:25, 0:4])                # footprint
        ax2 = plt.subplot(gs[28:,   :])                 # emittance plots
        ax3 = ax2.twinx()
        ax4 = plt.subplot(gs[0:5,  0:4], sharex=ax1)    # projection histogram qx
        ax5 = plt.subplot(gs[6:25, 4],   sharey=ax1)    # projection histogram qy

        # Prepare plot axes
        ax1.set_xlabel('horiz. fract. tune $q_x$')
        ax1.set_xlim(self.footprint_xlimits)
        ax1.set_ylabel('vert. fract. tune $q_y$')
        ax1.set_ylim(self.footprint_ylimits)

        # Emittance plots.
        ax2.set_xlabel('turns')
        ax2.set_xlim(self.emittance_xlimits)
        ax2.set_ylabel('$\epsilon_x [\mu m]$')
        ax2.set_ylim(self.horizontal_emittance_ylimits)
        
        ax3.set_xlim(self.emittance_xlimits)
        ax3.set_ylabel('$\epsilon_y [\mu m]$')
        ax3.set_ylim(self.vertical_emittance_ylimits)

        plt.setp(ax4.get_xticklabels(), visible=False)
        plt.setp(ax5.get_yticklabels(), visible=False)
        ax4.yaxis.set_major_locator(MaxNLocator(3))
        ax5.xaxis.set_major_locator(MaxNLocator(3))

        self.ax1 = ax1
        self.ax2 = ax2
        self.ax3 = ax3
        self.ax4 = ax4
        self.ax5 = ax5        
