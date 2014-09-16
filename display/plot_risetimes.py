from Pyheana.load.fit_data import *
import pylab as plt


def plot_risetimes(a, b, **kwargs):

    # plt.ion()
    # if kwargs is not None:
    #     for key, value in kwargs.iteritems():
    #         if key == 'file_list':
    #             file_list = value
    #         if key == 'scan_line':
    #             scan_line = value
    # varray = plt.array(get_value_from_cfg(file_list, scan_line))

    n_files = a.shape[-1]
    cmap = plt.get_cmap('jet')
    c = [cmap(i) for i in plt.linspace(0, 1, n_files)]

    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    [ax.set_color_cycle(c) for ax in (ax1, ax2)]

    r = []
    for i in xrange(n_files):
        x, y = a[:,i], b[:,i]
        # xo, yo = x, y #, get_envelope(x, y)
        xo, yo = get_envelope(x, y)
        p = plt.polyfit(xo, np.log(yo), 1)

        # Right way to fit... a la Nicolas - the fit expert!
        l = ax1.plot(x, plt.log(plt.absolute(y)))
        lcolor = l[-1].get_color()
        ax1.plot(xo, plt.log(yo), color=lcolor, marker='o', mec=None)
        ax1.plot(x, p[1] + x * p[0], color=lcolor, ls='--', lw=3)

        l = ax2.plot(x, y)
        lcolor = l[-1].get_color()
        ax2.plot(xo, yo, 'o', color=lcolor)
        xi = plt.linspace(plt.amin(x), plt.amax(x))
        yi = plt.exp(p[1] + p[0] * xi)
        ax2.plot(xi, yi, color=lcolor, ls='--', lw=3)

        print p[1], p[0], 1 / p[0]
        # plt.draw()
        # ax1.cla()
        # ax2.cla()

        r.append(1/p[0])

    ax2.set_ylim(0, 1000)
    plt.figure(2)
    plt.plot(r, lw=3, c='purple')
    # plt.gca().set_ylim(0, 10000)

    # ax3 = plt.subplot(111)
    # ax3.semilogy(x, y)
    # ax3.semilogy(xo, yo)

    return r
