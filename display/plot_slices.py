import matplotlib
import pylab as plt
from scipy import interpolate
from Pyheana.load.load_data import *


def plot_slices(data):

    [n_cols, n_slices, n_turns] = data.shape
    A = data[2,:,:] * data[13,:,:]

    # Color cycle
    n_colors = n_turns
    colmap = plt.get_cmap('jet')
    c = [colmap(i) for i in plt.linspace(0., 1., n_colors)]

    fig1 = plt.figure(figsize=(12, 8))
    # ax1 = plt.gca()
    # [ax1.plot(A[:,i], c=c[i]) for i in range(1, n_turns, 1)]
    # plt.show()

    # Smoothing
    X = plt.arange(0, n_slices, 1)
    Y = plt.arange(0, n_turns, 1)
    A = A[X,:][:,Y]
    
    Xi = plt.linspace(X[0], X[-1], 1000)
    Yi = plt.linspace(Y[0], Y[-1], 1000)
    sp = interpolate.RectBivariateSpline(X, Y, A)
    Ai = sp(Xi, Yi)

    X, Y = plt.meshgrid(X, Y)
    X, Y = X.T, Y.T
    Xi, Yi = plt.meshgrid(Xi, Yi)
    Xi, Yi = Xi.T, Yi.T

    #fig = figure(1)
    #ax3d = fig.gca(projection='3d')
    #pl = ax3d.plot_wireframe(Xi, Yi, Ai, \
        #rstride=ns, cstride=ns, cmap=cm.jet, linewidth=0.1, alpha=0.3)
    #cset = ax3d.contourf(Xi, Yi, Ai, zdir='z')#, offset=-100)
    #cset = ax3d.contourf(Xi, Yi, Ai, zdir='x')#, offset=-40)
    #cset = ax3d.contourf(Xi, Yi, Ai, zdir='y')#, offset=40)
    #ax3d.zaxis.set_major_locator(LinearLocator(10))
    #ax3d.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    #ax3d.view_init(elev=25, azim=-150)
    #ax3d.set_xlabel('Slices')
    #ax3d.set_ylabel('Turns')

    #fig = figure(2)
    #ax3d = fig.gca(projection='3d')
    #pl = ax3d.plot_wireframe(X, Y, A, \
        #rstride=ns, cstride=ns, cmap=cm.jet, linewidth=0.1, alpha=0.3)
    #cset = ax3d.contourf(X, Y, A, zdir='z')#, offset=-100)
    #cset = ax3d.contourf(X, Y, A, zdir='x')#, offset=-40)
    #cset = ax3d.contourf(X, Y, A, zdir='y')#, offset=40)
    #ax3d.zaxis.set_major_locator(LinearLocator(10))
    #ax3d.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    #ax3d.view_init(elev=25, azim=-150)

    #show()

    from mayavi.modules.grid_plane import GridPlane
    from mayavi.modules.outline import Outline
    from mayavi.modules.volume import Volume
    from mayavi.scripts import mayavi2
    from mayavi import mlab
    mlab.options.backend = 'envisage'
    # graphics card driver problem
    # workaround by casting int in line 246 of enthought/mayavi/tools/figure.py
    #mlab.options.offscreen = True
    #enthought.mayavi.engine.current_scene.scene.off_screen_rendering = True

    mlab.figure(bgcolor=(1,1,1), fgcolor=(0.2,0.2,0.2))
    aspect = (0, 10, 0, 20, -6, 6)
    ranges = (plt.amin(X), plt.amax(X), plt.amin(Y), plt.amax(Y), plt.amin(A), plt.amax(A))
    # s = mlab.surf(Xi, Yi, Ai, colormap='jet', representation='surface',
    #               warp_scale=1e-3)
    s = mlab.surf(Xi, Yi, Ai, colormap='jet', representation='surface', extent=aspect,
                  warp_scale='auto')
    mlab.outline(line_width=1)
    mlab.axes(x_axis_visibility=True,
              xlabel='Slice No.', ylabel='Turn No.', zlabel='BPM signal', ranges=ranges)

    #mlab.title(('Electron cloud dynamics - slice passage: %03d/%03d' % (i, n_blocks)), size=0.25)
    mlab.view(azimuth=230, elevation=60)   
    mlab.show()
