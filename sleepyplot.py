""" plotting functions for Dataset objects """

import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import itertools


def plotEEG(d, raw=True, filtered=False, spindles=False, spindle_rejects=False):
    """ plot multichannel EEG 
    
    Parameters
    ----------
    d: instance of ioeeg Dataset class
    raw: bool, optional, default: True
        Option to plot raw EEG
    filtered: bool, optional, default: False
        Option to plot filtered EEG
    spindles: bool, optional, default: False
        Option to plot spindle detections
    spindle_rejects: bool, optional, default: False
        Option to plot rejected spindle detections
        
    Returns
    -------
    matplotlib.pyplot figure instance
    """
    data = []
    title = []
    
    # import data
    if raw == True:
        raw = d.data
        data.append(raw)
        title.append('Raw')
    if filtered == True:    
        filtd = d.spindle_calcs.loc(axis=1)[:, 'Filtered']
        data.append(filtd)
        title.append('Filtered')

    # flatten events list by channel for plotting
    if spindles == True:
        sp_eventsflat = [list(itertools.chain.from_iterable(d.spindle_events[i])) for i in d.spindle_events.keys()]
    if spindle_rejects == True:
        sp_rej_eventsflat = [list(itertools.chain.from_iterable(d.spindle_rejects[i])) for i in d.spindle_rejects.keys()]   

    # plot data    
    fig, axs = plt.subplots(len(data), 1, sharex=True, figsize=(10,10), squeeze=False)
    fig.subplots_adjust(hspace=.1, top=.9, bottom=.1, left=.05, right=.95)
    
    for dat, ax, t in zip(data, axs.flatten(), title):
        for i, c in enumerate(d.eeg_channels):
            # normalize each channel to [0, 1]
            dat_ser = pd.Series(dat[(c, t)], index=dat.index)
            norm_dat = (dat_ser - min(dat_ser))/(max(dat_ser)-min(dat_ser)) - i # subtract i for plotting offset
            ax.plot(norm_dat, linewidth=.5, color='C0')
            
            # plot spindles
            if spindles == True:
                sp_events_TS = [pd.Timestamp(x) for x in sp_eventsflat[i]]
                spins = pd.Series(index=norm_dat.index)
                spins[sp_events_TS] = norm_dat[sp_events_TS]
                ax.plot(spins, color='orange', alpha=0.5)
            if spindle_rejects == True:
                sp_rejs_TS = [pd.Timestamp(x) for x in sp_rej_eventsflat[i]]
                spin_rejects = pd.Series(index=norm_dat.index)
                spin_rejects[sp_rejs_TS] = norm_dat[sp_rejs_TS]
                ax.plot(spin_rejects, color='red', alpha=0.5)
        
        ax.set_title(t)
        ax.set_yticks(list(np.arange(0.5, -(len(d.eeg_channels)-1), -1)))
        ax.set_yticklabels(d.eeg_channels)
        ax.margins(x=0) # remove white space margins between data and y axis
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # set overall parameters
    fig.suptitle(d.in_num)
    plt.xlabel('Time')

    return fig