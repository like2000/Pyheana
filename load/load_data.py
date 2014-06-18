import glob, h5py, natsort, sys
import scipy.io as sio
import pylab as plt
import numpy as np


def get_file_list(path, substring=None):

    file_list = glob.glob(path + '/' + substring)
    file_list = sorted(list(file_list))

    print '--> Found files'
    for i, f in enumerate(file_list):
        print i+1, f
    print 'in:\n', path

    return file_list

def read_bunch_data(file_list, format='ascii'):

    A, R = [], []
    
    if format == 'ascii':
        data_list = [f.replace('.cfg', '_prt.dat') for f in file_list]

        for i, d in enumerate(data_list):
            try:
                A.append(plt.loadtxt(d).T)
            except IOError:
                print '*** WARNING: no file ', d
                # R.append((i, file_list[i]))
                A.append(A[-1])
            except ValueError:
                print i, d
                raise ValueError
    elif format == 'hdf5':
        data_list = [f.replace('.cfg', '.prt.h5') for f in file_list]

        for i, d in enumerate(data_list):
            try:
                hf = h5py.File(d, 'r')
                A.append(
                    plt.insert(hf['Fields']['Data'], 0, hf['Time']['time'], axis=1).T)
            except IOError:
                print '*** WARNING: no file ', d
                R.append((i, file_list[i]))
    elif format == 'h5':
        data_list = file_list
        A = None

        for i, d in enumerate(data_list):
            try:
                hf = h5py.File(d, 'r')['Bunch']
                if A == None:
                    keys = hf.keys()
                    a = hf[keys[0]]
                    A = plt.zeros((len(data_list), len(keys) + 1, len(a)))
                A[i, 0, :] = plt.arange(len(hf['mean_x']))
                A[i, 1, :] = hf['mean_x']
                A[i, 2, :] = hf['mean_xp']
                A[i, 3, :] = hf['mean_y']
                A[i, 4, :] = hf['mean_yp']
                A[i, 5, :] = hf['mean_z']
                A[i, 6, :] = hf['mean_dp']
                A[i, 7, :] = hf['sigma_x']
                A[i, 8, :] = hf['sigma_y']
                A[i, 9, :] = hf['sigma_z']
                A[i, 10, :] = hf['sigma_dp']
                A[i, 11, :] = hf['epsn_x']
                A[i, 12, :] = hf['epsn_y']
                A[i, 13, :] = hf['epsn_z']
                A[i, 14, :] = hf['n_macroparticles']
                # A.append(plt.array(
                #     [,
                #      hf['mean_x'], hf['mean_xp'], hf['mean_y'], hf['mean_yp'], hf['mean_dz'], hf['mean_dp'],
                #      hf['sigma_x'], hf['sigma_y'], hf['sigma_dz'], hf['sigma_dp'],
                #      hf['epsn_x'], hf['epsn_y'], hf['epsn_z'], hf['n_macroparticles']]).T)
            except IOError:
                print '*** WARNING: no file ', d
                R.append((i, file_list[i]))
    else:
        raise(ValueError('*** Unknown format: ', format))

    for r in R:
        file_list.remove(r[1])
        A = plt.delete(A, r[0], axis=0)

    A = plt.array(A)
    A = plt.rollaxis(A, 0, 3)
    [n_cols, n_turns, n_files] = A.shape

    return A, file_list

def read_slice_data(filename, format='ascii'):

    print "--> Reading data from file", filename
    A, filename = read_big_data(filename, type='hdtl', format=format)
    # sio.savemat(filename.replace(".dat", ".mat"), {'A':A})
    sio.savemat(filename.replace(".h5", ".mat"), {'A':A})
    print filename.replace(".h5", ".mat")

    return A


def read_particle_data(filename, format='ascii', n_steps_to_read=None, n_macroparticles_to_read=None):

    A = {}

    print "--> Reading data from file", filename
    if format == ('ascii' or 'hdf5'):
        A, filename = read_big_data(filename, type='prb', format=format)
    elif format == 'h5':
        hf = h5py.File(filename, 'r')

        # Check whether h5 file has structure with 'Step#..' keys. 
        if 'Step#0' in hf.keys():
            # Preallocate memory.
            if not n_steps_to_read:
                n_steps_to_read  = len(hf.keys())
            if not n_macroparticles_to_read:
                n_macroparticles_to_read = len((hf[hf.keys()[0]])['x'])

            x   = np.zeros((n_macroparticles_to_read, n_steps_to_read))
            xp  = np.zeros((n_macroparticles_to_read, n_steps_to_read))
            y   = np.zeros((n_macroparticles_to_read, n_steps_to_read))
            yp  = np.zeros((n_macroparticles_to_read, n_steps_to_read))
            z   = np.zeros((n_macroparticles_to_read, n_steps_to_read))
            dp  = np.zeros((n_macroparticles_to_read, n_steps_to_read))
            c   = np.zeros((n_macroparticles_to_read, n_steps_to_read))
            idd = np.zeros((n_macroparticles_to_read, n_steps_to_read))

            # Read data from h5 file.
            for i in range(0, n_steps_to_read):
                step = hf['Step#' + str(i)]

                x[:,i]   = step['x'][:n_macroparticles_to_read]
                xp[:,i]  = step['xp'][:n_macroparticles_to_read]
                y[:,i]   = step['y'][:n_macroparticles_to_read]
                yp[:,i]  = step['yp'][:n_macroparticles_to_read]
                z[:,i]   = step['z'][:n_macroparticles_to_read]
                dp[:,i]  = step['dp'][:n_macroparticles_to_read]
                c[:,i]   = step['c'][:n_macroparticles_to_read]
                idd[:,i] = step['id'][:n_macroparticles_to_read]

            # Build dictionary
            A['x']  = x
            A['xp'] = xp
            A['y']  = y
            A['yp'] = yp
            A['z']  = z
            A['dp'] = dp
            A['c']  = c
            A['id'] = idd

        # No 'Step#..' structure.
        else:
            if n_macroparticles_to_read or n_steps_to_read:
                print 'Reading in extract of particle data not yet implemented ...'

            keys = hf.keys()
            a = hf[keys[0]]
            A['x']  = hf['x'][:]
            A['xp'] = hf['xp'][:]
            A['y']  = hf['y'][:]
            A['yp'] = hf['yp'][:]
            A['z']  = hf['z'][:]
            A['dp'] = hf['dp'][:]
            A['id'] = hf['id'][:]
            A['c']  = hf['c'][:]

    else:
        raise(ValueError('*** Unknown format: ', format))

    return A


def read_big_data(filename, type=None, format='ascii'):

    if format == 'ascii':
        data = filename.replace('.cfg', '_' + type + '.dat')
    elif format == 'hdf5':
        data = filename.replace('.cfg', '.' + type + '.h5')

    try:
        A = plt.np.load(data.replace('.dat', '.npy'))
    except IOError:
        if format == 'ascii':
            n_blocks = get_first_newline(data)
            A = plt.loadtxt(data).T
            [n_cols, n_turns, n_blocks] = [A.shape[0], A.shape[1] / n_blocks, n_blocks]
            A = plt.reshape(A, (n_cols, n_turns, n_blocks))
            A = plt.rollaxis(A, 1, 3)

            plt.np.save(data.replace('.dat', '.npy'), A)
        elif format == 'hdf5':
            A = h5py.File(data, 'r')
            turns = natsort.natsorted(A.keys(), signed=False)
            cols = ['z0', 'x', 'xp', 'y', 'yp', 'z', 'zp']
            try:
                n_cols, n_particles, n_turns = len(cols), len(A[turns[0]][cols[0]]), len(turns)

                B = plt.zeros((n_cols, n_particles, n_turns))
                for i, d in enumerate(cols):
                    for j, g in enumerate(turns):
                        B[i, :, j] = A[g][d]
            except KeyError:
                B = A['Fields']['Data'][:].T

            A = B

    return A, data

def get_first_newline(data):

    newline = 0
    with open(data, 'r') as file:
        for line in file:
            if line != '\n':
                newline += 1
            else:
                break

    return newline

def get_value_from_cfg(file_list, linestring=None):

    A = []
    found = 0

    for f in file_list:
        with open(f, 'r') as file:
            for line in file:
                if line == '\n':
                    continue
                if line.split()[0] == linestring:
                    A.append(float(line.split()[1]))
                    found = 1
                    break
    if not found:
        raise IndexError('Entry "' + linestring + '" not found!')

    return A

def get_value_from_new_cfg(file_list, linestring=None):

    A = []
    found = 0

    for f in file_list:
        with open(f, 'r') as file:
            for line in file:
                if line == '\n':
                    continue
                if line.split('=')[0].strip() == linestring:
                    A.append(float(line.split()[2].strip(',')))
                    found = 1
                    break
    if not found:
        A = get_value_from_cfg(file_list, linestring)

    return A

def get_value_from_h5(file_list, linestring):

    A = []
    found = 0

    for f in file_list:
        with h5py.File(f, 'r') as file:
            A.append(file.attrs[linestring])
    #         for line in file:
    #             if line == '\n':
    #                 continue
    #             if line.split('=')[0].strip() == linestring:
    #                 A.append(float(line.split()[2].strip(',')))
    #                 found = 1
    #                 break
    # if not found:
    #     A = get_value_from_cfg(file_list, linestring)

    return A

def get_attributes(file_list, searchstring, keys=None, format='cfg'):

    if not keys:
        keys = searchstring
    L = plt.zeros((len(keys), len(file_list)))

    if format == 'cfg':
        for i, k in enumerate(searchstring):
            L[i,:] = get_value_from_cfg(file_list, k)
    elif format == 'new_cfg':
        for i, k in enumerate(searchstring):
            L[i,:] = get_value_from_new_cfg(file_list, k)
    elif format == 'h5':
        for i, k in enumerate(searchstring):
            L[i,:] = get_value_from_h5(file_list, k)
    else:
        raise ValueError

    return dict(zip(keys, L))
