from pylab import *


def set_mystyle():
    params = {
        'figure.subplot.left': 0.15,
        'figure.subplot.right': 0.95,
        'figure.subplot.bottom': 0.15,
        'figure.subplot.top': 0.9,
        'figure.subplot.wspace': 0.2,
        'figure.subplot.hspace': 0.2,
        'axes.labelsize': 22,
        'axes.linewidth': 2,
        'legend.fancybox': True,
        'legend.fontsize': 'x-large',
        'xtick.labelsize': 22,
        'ytick.labelsize': 22,
        'xtick.major.pad': 14,
        'ytick.major.pad': 14,
        'text.fontsize': 'xx-large'
    }
    rcParams.update(params)


# # Format axes
# majorFormatter = FormatStrFormatter('%3.2e')
# sctfcFormatter = ScalarFormatter(useOffset=False, useMathText=True)
# sctfcFormatter.set_scientific(True)

# gcf().subplots_adjust(left=None, bottom=None, right=0.80, top=None, wspace=None, hspace=None)
