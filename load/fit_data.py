import numpy as np
import pylab as plt
from scipy.optimize import leastsq


def get_envelope(x, y):

    # get peaks
    ixmm = np.diff(np.sign(np.diff(y))).nonzero()[0] + 1 # local min+max
    ixmin = (np.diff(np.sign(np.diff(y))) > 0).nonzero()[0] + 1 # local min
    ixmax = (np.diff(np.sign(np.diff(y))) < 0).nonzero()[0] + 1 # local max
    xm = x[ixmax]
    ym = y[ixmax]

    return xm, ym

def fit_risetimes(p, x, y):
    # get peaks
    ixmm = np.diff(np.sign(np.diff(y))).nonzero()[0] + 1 # local min+max
    ixmin = (np.diff(np.sign(np.diff(y))) > 0).nonzero()[0] + 1 # local min
    ixmax = (np.diff(np.sign(np.diff(y))) < 0).nonzero()[0] + 1 # local max
    xp = x[ixmax]
    yp = y[ixmax]

    # define functions
    #fitfunc = lambda p, x: p[0]*exp(x/p[1])
    #fitfunc = lambda p, x: p[0] + exp(p[1]*x)
    fitfunc = lambda p, x: p[0] + p[1] * np.exp(p[2]*x)
    errfunc = lambda p, xi, yi: fitfunc(p, xi) - yi

    # fit peaks
    p1, success = leastsq(errfunc, p, args=(xp, yp))

    xi = np.linspace(x.min(), x.max(), 1000)
    yi = fitfunc(p1, xi)

    return [p1, xp, yp, xi, yi]

def fitgeneral(fitfunc, data, p0, fitrange):
    '''
        general fitting function.
        input: data [x,y], fitfunction (lambda/inline form), starting value
        output: [fitparameters, fitdata[xi, yi]]
    '''
    # error function
    errfunc = lambda p, xi, yi: fitfunc(p, xi) - yi
    p1, success = leastsq(errfunc, p0[:], args=(data[0][fitrange], data[1][fitrange]))

    #print 'quality of least squares fit: '+str(errfunc)

    xi = np.linspace(data[0].min(), data[0].max(), 1000)
    yi = fitfunc(p1, xi)
    return [p1, xi, yi]

def fitlinear(x, y):
    # fit the first set
    fitfunc = lambda p, xi: p[0]*xi + p[1] # target function
    errfunc = lambda p, xi, yi: fitfunc(p, xi) - yi # distance to the target function
    p0 = [0,0] # initial guess for the parameters
    p1, success = leastsq(errfunc, p0[:], args=(x, y))

    xi = np.linspace(x.min(), x.max(), 100)
    yi = fitfunc(p1,xi)
    return [p1,xi,yi]

def fitgauss(x, y):
    # calculate moments
    mu = sum(x * y) / sum(y)
    sigma = np.sqrt(sum((x - mu) ** 2 * y) / sum(y))

    xi  = np.linspace(x.min(), x.max(), 100)
    yib = np.exp(-1 / 2. * (xi - mu) ** 2 / sigma ** 2) * y.max() #* 1/sqrt(2*pi)*1/sigma

    # fit the first set
    fitfunc = lambda p, xi: np.exp(-1/2. * (xi - p[1]) ** 2 / p[0] ** 2) * y.max() # target function
    errfunc = lambda p, xi, yi: fitfunc(p, xi) - yi # distance to the target function
    p0 = [sigma, mu] # initial guess for the parameters
    p1, success = leastsq(errfunc, p0[:], args=(x, y))

    # re-assign moments
    [sigma, mu] = p1
    yia = fitfunc(p1, xi)
    return [sigma, mu, xi, yia, yib]

def normalise(x):

    x = x/x.max()

    return x
