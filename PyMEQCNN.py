# /bin/env python
# Modified from Ross et al (2018)
# PyMEQCNN (Christopher Salim, 2019)
# Additions from previous version: disabled the -V input to simplify the usage, added the csv save
# feature, also enabled the user to save the plot obtained from the running result immediately. 

import string
import time
import argparse as ap
import sys
import os

import numpy as np
import obspy.core as oc
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras import losses
from keras.models import model_from_json
import tensorflow as tf
import matplotlib as mpl
import pylab as plt
import pandas as pd
mpl.rcParams['pdf.fonttype'] = 42

#####################
# Hyperparameters
min_proba = 0.95 # Minimum softmax probability for phase detection
freq_min = 6.0
freq_max = 12.0
filter_data = False
decimate_data = True # If false, assumes data is already 100 Hz samprate
n_shift = 400 # Number of samples to shift the sliding window at a time
n_gpu = 0 # Number of GPUs to use (if any)
#####################
batch_size = 300

half_dur = 2.0
only_dt = 0.01
n_win = int(half_dur/only_dt)
n_feat = 2*n_win

#-------------------------------------------------------------

def sliding_window(data, size, stepsize=1, padded=False, axis=-1, copy=True):
    """
    Calculate a sliding window over a signal
    Parameters
    ----------
    data : numpy array
        The array to be slided over.
    size : int
        The sliding window size
    stepsize : int
        The sliding window stepsize. Defaults to 1.
    axis : int
        The axis to slide over. Defaults to the last axis.
    copy : bool
        Return strided array as copy to avoid sideffects when manipulating the
        output array.
    Returns
    -------
    data : numpy array
        A matrix where row in last dimension consists of one instance
        of the sliding window.
    Notes
    -----
    - Be wary of setting `copy` to `False` as undesired sideffects with the
      output values may occurr.
    Examples
    --------
    >>> a = numpy.array([1, 2, 3, 4, 5])
    >>> sliding_window(a, size=3)
    array([[1, 2, 3],
           [2, 3, 4],
           [3, 4, 5]])
    >>> sliding_window(a, size=3, stepsize=2)
    array([[1, 2, 3],
           [3, 4, 5]])
    See Also
    --------
    pieces : Calculate number of pieces available by sliding
    """
    if axis >= data.ndim:
        raise ValueError(
            "Axis value out of range"
        )

    if stepsize < 1:
        raise ValueError(
            "Stepsize may not be zero or negative"
        )

    if size > data.shape[axis]:
        raise ValueError(
            "Sliding window size may not exceed size of selected axis"
        )

    shape = list(data.shape)
    shape[axis] = np.floor(data.shape[axis] / stepsize - size / stepsize + 1).astype(int)
    shape.append(size)

    strides = list(data.strides)
    strides[axis] *= stepsize
    strides.append(data.strides[axis])

    strided = np.lib.stride_tricks.as_strided(
        data, shape=shape, strides=strides
    )

    if copy:
        return strided.copy()
    else:
        return strided

if __name__ == "__main__":
    parser = ap.ArgumentParser(
        prog='gpd_predict.py',
        description='Automatic picking of seismic waves using'
                    'Generalized Phase Detection')
    parser.add_argument(
        '-I',
        type=str,
        default=None,
        help='Input file')
    parser.add_argument(
        '-O',
        type=str,
        default=None,
        help='Output file')
    parser.add_argument(
        '-P',
        default=True,
        action='store_false',
        help='Suppress plotting output')
    parser.add_argument(
        '-V',
        default=False,
        action='store_true',
        help='verbose')
    args = parser.parse_args()

    plot = args.P

    # Reading in input file
    fdir = []
    evid = []
    staid = []
    with open(args.I) as f:
        for line in f:
            tmp = line.split()
            fdir.append([tmp[0], tmp[1], tmp[2]])
    nsta = len(fdir)
    print(nsta)
    # load json and create model
    json_file = open('model_architecture.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json, custom_objects={'tf':tf})

    # load weights into new model
    model.load_weights("model_weights.h5")
    print("Loaded model from disk")

    if n_gpu > 1:
        from keras.utils import multi_gpu_model
        model = multi_gpu_model(model, gpus=n_gpu)

    ofile = open(args.O, 'w')
    station=[]
    pick_p_fix = []
    pick_s_fix = []
    for i in range(nsta):
        fname = fdir[i][0].split("/")
        if not os.path.isfile(fdir[i][0]):
            print("%s doesn't exist, skipping" % fdir[i][0])
            continue
        if not os.path.isfile(fdir[i][1]):
            print("%s doesn't exist, skipping" % fdir[i][1])
            continue
        if not os.path.isfile(fdir[i][2]):
            print("%s doesn't exist, skipping" % fdir[i][2])
            continue
        st = oc.Stream()
        st += oc.read(fdir[i][0])
        st += oc.read(fdir[i][1])
        st += oc.read(fdir[i][2])
        st = st.normalize()
        latest_start = np.max([x.stats.starttime for x in st])
        earliest_stop = np.min([x.stats.endtime for x in st])
        st.trim(latest_start, earliest_stop)

        st.detrend(type='linear')
        if filter_data:
            st.filter(type='bandpass', freqmin=freq_min, freqmax=freq_max, corners=2, zerophase=True)
        if decimate_data:
            st.interpolate(100.0)
        chan = st[0].stats.channel
        sr = st[0].stats.sampling_rate

        dt = st[0].stats.delta
        net = st[0].stats.network
        sta = st[0].stats.station
        if sta == 'TES':
            sta = str(fname[0][16:20])

        print("Reshaping data matrix for sliding window")
        tt = (np.arange(0, st[0].data.size, n_shift) + n_win) * dt
        tt_i = np.arange(0, st[0].data.size, n_shift) + n_feat
        #tr_win = np.zeros((tt.size, n_feat, 3))
        sliding_N = sliding_window(st[0].data, n_feat, stepsize=n_shift)
        sliding_E = sliding_window(st[1].data, n_feat, stepsize=n_shift)
        sliding_Z = sliding_window(st[2].data, n_feat, stepsize=n_shift)
        tr_win = np.zeros((sliding_N.shape[0], n_feat, 3))
        tr_win[:,:,0] = sliding_N
        tr_win[:,:,1] = sliding_E
        tr_win[:,:,2] = sliding_Z
        tr_win = tr_win / np.max(np.abs(tr_win), axis=(1,2))[:,None,None]
        tt = tt[:tr_win.shape[0]]
        tt_i = tt_i[:tr_win.shape[0]]

        ts = model.predict(tr_win, verbose=True, batch_size=batch_size)

        prob_S = ts[:,1]
        prob_P = ts[:,0]
        prob_N = ts[:,2]
        
        station.append(sta)
        pick_p = np.where(prob_P>=0.95)
        pick_p_new = np.array(pick_p)
        np.savetxt('pick_P ' + str(sta), pick_p_new*n_shift/100)
        pick_p_new = pick_p_new[0]
        pick_p_fix.append(pick_p_new*n_shift/100)
        
        pick_s = np.where(prob_S>=0.95)
        pick_s_new = np.array(pick_s)
        np.savetxt('pick_S ' + str(sta), pick_s_new*n_shift/100)
        pick_s_new = pick_s_new[0]
        pick_s_fix.append(pick_s_new*n_shift/100)

        p_picks = []
        s_picks = []
        for picks in pick_p_new:
            stamp_pick = st[0].stats.starttime + tt[picks]
            p_picks.append(stamp_pick)
            ofile.write("%s %s P %s\n" % (net, sta, stamp_pick.isoformat()))
            np.savetxt('p_picks' + str(i), p_picks)
        for picks in pick_s_new:
            stamp_pick = st[0].stats.starttime + tt[picks]
            s_picks.append(stamp_pick)
            ofile.write("%s %s S %s\n" % (net, sta, stamp_pick.isoformat()))
        fig = plt.figure(figsize=(8, 12))
        ax = []
        ax.append(fig.add_subplot(4,1,1))
        ax.append(fig.add_subplot(4,1,2,sharex=ax[0],sharey=ax[0]))
        ax.append(fig.add_subplot(4,1,3,sharex=ax[0],sharey=ax[0]))
        ax.append(fig.add_subplot(4,1,4,sharex=ax[0]))
        for j in range(3):
            ax[j].plot(np.arange(st[j].data.size)*dt, st[j].data, c='k', \
                       lw=0.5)
            ax[j].set_xlabel('Time (s)')
            ax[j].set_ylabel('Normalized value')
        ax[0].set_title('Picking in N-S component')
        ax[1].set_title('Picking in E-W component')
        ax[2].set_title('Picking in U-D component')
        ax[3].plot(tt, ts[:,0], c='r', lw=0.5)
        ax[3].plot(tt, ts[:,1], c='b', lw=0.5)
        ax[3].axhline(min_proba,linestyle='--')
        ax[3].set_title('Probability plot')
        for p_pick in p_picks:
            for k in range(3):
                for xc in pick_p_new:
                    ax[k].axvline(x=xc*4, c='r', lw=0.5)
        for s_pick in s_picks:
            for k in range(3):
                for xc in pick_s_new:
                    ax[k].axvline(x=xc*4, c='b', lw=0.5)
        plt.tight_layout()
        fig.savefig('Probability_plot_station' + str(i+1))
    df = pd.DataFrame({'Station': station, 'tp': pick_p_fix, 'ts': pick_s_fix})
    df.to_csv(r'./result_dataframe.csv', index = None, header=True)
    ofile.close()
