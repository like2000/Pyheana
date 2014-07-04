import matplotlib
import pylab as plt
import scipy.io as sio
from Pyheana.load.load_data import *



class ThresholdPlots:

    def __init__(self, rawdata, scan_values):

        self.scan_values = scan_values
        self._prepare_input_signals(rawdata)
        
        
    def create_plot(self, plane='horizontal', xlabel='turns', ylabel='intensity [particles]', zlabel='normalized emittance',
                    xlimits=((0.,8192)), ylimits=((0.,7.1e11)), zlimits=((0., 10.))):
    
        self._create_axes(xlabel, ylabel, zlabel, xlimits, ylimits, zlimits)
        
        cmap = plt.cm.get_cmap('jet', 2)
        self.ax11.patch.set_facecolor(cmap(range(2))[-1])
        cmap = plt.cm.get_cmap('jet')

        x, y = plt.meshgrid(self.turns[:,0], self.scan_values)
        z = self.epsn_abs[plane]
        threshold_plot = self.ax11.contourf(x, y, z.T, levels=plt.linspace(zlimits[0], zlimits[1], 201),
                                            vmin=zlimits[0], vmax=zlimits[1], cmap=cmap)
        cb = plt.colorbar(threshold_plot, self.ax13, orientation='vertical')
        cb.set_label(zlabel)

        plt.tight_layout()

        
    def _prepare_input_signals(self, rawdata):

        # x axis
        t = rawdata[0,:,:]
        turns = plt.ones(t.shape).T * plt.arange(len(t))
        turns = turns.T
        self.turns = turns
        
        # z axis
        self.epsn_abs = {}
        self.epsn_abs['horizontal'] = plt.absolute(rawdata[11,:,:])
        self.epsn_abs['vertical']   = plt.absolute(rawdata[12,:,:])
            

    def _create_axes(self, xlabel, ylabel, zlabel, xlimits, ylimits, zlimits):

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
