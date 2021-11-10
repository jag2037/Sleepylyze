""" plotting functions for Dataset objects 
    
    To Do:
        Edit hyp_stats plots to take transitions.HypStats object instead of ioeeg.Dataset object
        Remove redundant plotting fns added into EKG classs
        Add subsetEEG function to break up concatenated NREM segments for plotting. Will require adjustments
        to specified detections added to plot.
"""

import itertools
import igraph as ig
import math
import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np 
import os
import pandas as pd
import shapely.geometry as SG

from matplotlib.widgets import Slider
from pandas.plotting import register_matplotlib_converters
from scipy.signal import find_peaks, butter, sosfiltfilt
from scipy import interpolate
register_matplotlib_converters()



def plotEEG(d, raw=True, filtered=False, spindles=False, spindle_rejects=False):
    """ plot multichannel EEG w/ option for double panel raw & filtered. For short, pub-ready
        figures. Use vizeeg for data inspection 

        red = spindle rejects by time domain criteria; dark red = spindle rejects by frequency domain criteria
    
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
        sp_rej_t_eventsflat = [list(itertools.chain.from_iterable(d.spindle_rejects_t[i])) for i in d.spindle_rejects_t.keys()]
        sp_rej_f_eventsflat = [list(itertools.chain.from_iterable(d.spindle_rejects_f[i])) for i in d.spindle_rejects_f.keys()]   

    # set channels for plotting
    channels = [x[0] for x in d.data.columns]

    # plot data    
    fig, axs = plt.subplots(len(data), 1, sharex=True, figsize=(10,10), squeeze=False)
    fig.subplots_adjust(hspace=.1, top=.9, bottom=.1, left=.05, right=.95)
    
    for dat, ax, t in zip(data, axs.flatten(), title):
        for i, c in enumerate(channels):
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
                # plot time-domain rejects
                sp_rejs_t_TS = [pd.Timestamp(x) for x in sp_rej_t_eventsflat[i]]
                spin_rejects_t = pd.Series(index=norm_dat.index)
                spin_rejects_t[sp_rejs_t_TS] = norm_dat[sp_rejs_t_TS]
                ax.plot(spin_rejects_t, color='red', alpha=0.5)
                # plot frequency-domain rejects
                sp_rejs_f_TS = [pd.Timestamp(x) for x in sp_rej_f_eventsflat[i]]
                spin_rejects_f = pd.Series(index=norm_dat.index)
                spin_rejects_f[sp_rejs_f_TS] = norm_dat[sp_rejs_f_TS]
                ax.plot(spin_rejects_f, color='darkred', alpha=0.5)
        
        ax.set_title(t)
        ax.set_yticks(list(np.arange(0.5, -(len(channels)-1), -1)))
        ax.set_yticklabels(channels)
        ax.margins(x=0) # remove white space margins between data and y axis
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # set overall parameters
    fig.suptitle(d.metadata['file_info']['in_num'])
    plt.xlabel('Time')

    return fig, axs



def plotEEG_singlechan(d, chan, raw=True, filtered=False, rms=False, thresholds=False, spindles=False, spindle_rejects=False):
    """ plot single channel EEG. Options for multipaneled calculations. Not for concatenated datasets
    
    Parameters
    ----------
    d: instance of ioeeg Dataset class
    chan: str
        channel to plot
    raw: bool, optional, default: True
        Option to plot raw EEG panel
    filtered: bool, optional, default: False
        Option to plot filtered EEG panel
    rms: bool, optional, default: False
        Option to plot filtered EEG panel with RMS and RMS moving average
    thresholds: bool, optional, default: False
        Option to plot spindle threshold lines on rms panel
    spindles: bool, optional, default: False
        Option to plot filtered EEG with spindle detection panel
    spindle_rejects: bool, optional, default: False
        Option to plot filtered EEG with spindle rejection panel.
        Note: Spindles and spindle_rejects plot on same panel if 
        both True

    Returns
    -------
    matplotlib.pyplot figure instance
    """
    data = []
    dtype = []
    labels = []
    c = chan
    
    # import data
    if raw == True:
        raw_data = d.data[c, 'Raw'] 
    if filtered == True or rms == True or spindles == True or spindle_rejects == True:
        filtd_data = d.spindle_calcs.loc(axis=1)[c, 'Filtered']

    # set data to plot
    if raw == True:
        #raw = d.data[c, 'Raw']
        data.append(raw_data)
        dtype.append('raw')
        labels.append('Raw Signal')
    if filtered == True:    
        #filtd = d.spindle_calcs.loc(axis=1)[c, 'Filtered']
        data.append(filtd_data)
        dtype.append('filtd')
        labels.append('Filtered Signal')
    if rms == True:
        data.append(filtd_data)
        dtype.append('filtd+rms')
        labels.append('Filtered Signal')
    if spindles == True or spindle_rejects == True:
        data.append(filtd_data)
        labels.append('Filtered Signal')
        if spindles == True and spindle_rejects == False:
            dtype.append('filtd+spin')
        elif spindles == False and spindle_rejects == True:
            dtype.append('filtd+rej')
        elif spindles == True and spindle_rejects == True:
            dtype.append('filtd+spin+rej')


    # pull out thresholds for labels
    loSD = d.metadata['spindle_analysis']['sp_loSD']
    hiSD = d.metadata['spindle_analysis']['sp_hiSD']

    # plot data    
    fig, axs = plt.subplots(len(data), 1, sharex=True, figsize=(18,6), squeeze=False)
    fig.subplots_adjust(hspace=.1, top=.9, bottom=.1, left=.05, right=.95)
    
    for dat, ax, dt, label in zip(data, axs.flatten(), dtype, labels):
        # plot EEG
        ax.plot(dat, linewidth=.5, color='C0', label=label)
            
        # plot filtered EEG w/ rms & thresholds
        if dt == 'filtd+rms':
            ax.plot(d.spRMS[c], label='RMS', color='green')
            ax.plot(d.spRMSmavg[c], label='RMS moving average', color='orange')
        if dt == 'filtd+rms' and thresholds == True:
            ax.axhline(d.spThresholds[c].loc['Low Threshold'], linestyle='solid', color='grey', label = f'Mean RMS + {loSD} SD')
            ax.axhline(d.spThresholds[c].loc['High Threshold'], linestyle='dashed', color='grey', label = f'Mean RMS + {hiSD} SD')
        
        # plot spindles
        if dt =='filtd+spin' or dt =='filtd+spin+rej':
            sp_valuesflat = []
            sp_eventsflat = []
            for n in range(len(d.spindle_events[c])):
                for m in range(len(d.spindle_events[c][n])):
                    sp_valuesflat.append(dat[d.spindle_events[c][n][m]])
                    sp_eventsflat.append(d.spindle_events[c][n][m])
            sp_events_TS = [pd.Timestamp(x) for x in sp_eventsflat]
            spins = pd.Series(index=dat.index)
            spins[sp_events_TS] = dat[sp_events_TS]
            ax.plot(spins, color='orange', alpha=0.5, label='Spindle Detection')
        
        # plot spindle rejections
        if dt == 'filtd+rej' or dt == 'filtd+spin+rej':
            # plot time-domain rejects
            sp_rej_t_valuesflat = []
            sp_rej_t_eventsflat = []
            for n in range(len(d.spindle_rejects_t[c])):
                for m in range(len(d.spindle_rejects_t[c][n])):
                    sp_rej_t_valuesflat.append(dat[d.spindle_rejects_t[c][n][m]])
                    sp_rej_t_eventsflat.append(d.spindle_rejects_t[c][n][m])
            sp_rej_t_events_TS = [pd.Timestamp(x) for x in sp_rej_t_eventsflat]
            spin_rejects_t = pd.Series(index=dat.index)
            spin_rejects_t[sp_rej_t_events_TS] = dat[sp_rej_t_events_TS]
            ax.plot(spin_rejects_t, color='red', alpha=0.5, label='Rejected Detection (T)')

            # plot frequency-domain rejects
            sp_rej_f_valuesflat = []
            sp_rej_f_eventsflat = []
            for n in range(len(d.spindle_rejects_f[c])):
                for m in range(len(d.spindle_rejects_f[c][n])):
                    sp_rej_f_valuesflat.append(dat[d.spindle_rejects_f[c][n][m]])
                    sp_rej_f_eventsflat.append(d.spindle_rejects_f[c][n][m])
            sp_rej_f_events_TS = [pd.Timestamp(x) for x in sp_rej_f_eventsflat]
            spin_rejects_f = pd.Series(index=dat.index)
            spin_rejects_f[sp_rej_f_events_TS] = dat[sp_rej_f_events_TS]
            ax.plot(spin_rejects_f, color='darkred', alpha=0.5, label='Rejected Detection (F)')
            
        ax.legend(loc='lower left')
        #ax.set_title(t)
        #ax.set_yticks(list(np.arange(0.5, -(len(chan)-1), -1)))
        #ax.set_yticklabels(chan)
        ax.margins(x=0) # remove white space margins between data and y axis
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # plot minor axes
        seconds = mdates.SecondLocator()
        ax.xaxis.set_minor_locator(seconds)
        ax.grid(axis='x', which='minor', linestyle=':')
        ax.grid(axis='x', which='major')
       
    # set overall parameters
    fig.suptitle(d.metadata['file_info']['in_num'])
    plt.xlabel('Time')
    
    return fig

def vizeeg(d, raw=True, filtered=False, spindles=False, spindle_rejects=False, slider=True, win_width=15, raw_lowpass=True, 
        lowpass_freq=25, lowpass_order=4):
    """ vizualize multichannel EEG w/ option for double panel raw and/or filtered. Optimized for
        inspecting spindle detections (title/axis labels removed for space)
        Spindles rejected based on time-domain criteria are plotted in red; rejections based on 
        frequency-domain criteria are plotted in darkred.
    
    Parameters
    ----------
    d: instance of ioeeg Dataset class
    raw: bool, optional, default: True
        Option to plot raw EEG
    filtered: bool, optional, default: False
        Option to plot spindle filtered EEG
    spindles: bool, optional, default: False
        Option to plot spindle detections
    spindle_rejects: bool, optional, default: False
        Option to plot rejected spindle detections
    slider: bool (default: False)
        Option to implement an X-axis slider instead of built-in matplotlib zoom. Useful
        for inspecting long segments of EEG with a set window
    win_width: int (default: 15)
        If using slider option, number of seconds to set window width
    raw_lowpass: bool (default: True)
        Whether to plot the lowpass filtered raw data [in place of the unchanged raw data]
    lowpass_freq: int (default: 25)
        Frequency to lowpass the raw data for visualization (if not already applied)
    lowpass_order: int (default: 4)
        Butterworth lowpass filter order to be used if lowpass_raw is not None (doubles for filtfilt)
        
    Returns
    -------
    matplotlib.pyplot figure instance
    """
    
    # Set figure size (double height if plotting both raw & filtered)
    if raw == True & filtered == True:
        figsize = (14, 14)
    else:
        figsize = (14, 7)
        
        data = []
    title = []
    
    # import data
    if raw == True:
        if not raw_lowpass:
            # use the unchanged raw data
            raw_data = d.data
        elif raw_lowpass:
            # use the lowpass filtered raw data
            try:
                # check if filtered data exists
                raw_lowpass_data = d.data_lowpass
            except AttributeError:
                # apply lowpass filter
                d.lowpass_raw(lowpass_freq, lowpass_order)
                raw_lowpass_data = d.data_lowpass

    if filtered == True:
        filtd = d.spindle_calcs.loc(axis=1)[:, 'Filtered']
    
    # set data to plot (title corresponds to multiindex level 2 in data df)
    if raw == True:
        if not raw_lowpass:
            # plot the unchanged data
            data.append(raw_data)
            title.append('Raw')
        elif raw_lowpass:
            # plot the lowpass data
            data.append(raw_lowpass_data)
            title.append('raw_lowpass')
    if filtered == True:    
        data.append(filtd)
        title.append('Filtered')
 

    # flatten events list by channel for plotting
    if spindles == True:
        sp_eventsflat = [list(itertools.chain.from_iterable(d.spindle_events[i])) for i in d.spindle_events.keys()]
    if spindle_rejects == True:
        # time-domain rejects
        sp_rej_t_eventsflat = [list(itertools.chain.from_iterable(d.spindle_rejects_t[i])) for i in d.spindle_rejects_t.keys()]
        # frequency domain rejects
        sp_rej_f_eventsflat = [list(itertools.chain.from_iterable(d.spindle_rejects_f[i])) for i in d.spindle_rejects_f.keys()]  

    # set channels for plotting
    channels = [x[0] for x in d.data.columns if x[0] not in ['EKG', 'EOG_L', 'EOG_R', 'REF']]
    
    # set offset multiplier (distance between channels in plot)
    mx = 0.1
    
    # plot data    
    fig, axs = plt.subplots(len(data), 1, sharex=True, figsize=figsize, squeeze=False)
    fig.subplots_adjust(hspace=.1, top=.9, bottom=.1, left=.05, right=.95)
    
    yticks = []
    
    for dat, ax, t in zip(data, axs.flatten(), title):
        for i, c in enumerate(channels):
            # normalize each channel to [0, 1] -> can also simply subtract the mean (cleaner looking), but
            # normalization preserves relative differences between channels while putting them on a common scale
            dat_ser = pd.Series(dat[(c, t)], index=dat.index)
            norm_dat = (dat_ser - min(dat_ser))/(max(dat_ser)-min(dat_ser)) - i*mx # subtract i for plotting offset
            yticks.append(np.nanmedian(norm_dat))
            ax.plot(norm_dat, linewidth=.5, color='C0')
            
            # plot spindles
            if spindles == True:
                sp_events_TS = [pd.Timestamp(x) for x in sp_eventsflat[i]]
                spins = pd.Series(index=norm_dat.index)
                spins[sp_events_TS] = norm_dat[sp_events_TS]
                ax.plot(spins, color='orange', alpha=0.5)
            if spindle_rejects == True:
                # plot time-domain rejects
                sp_rejs_t_TS = [pd.Timestamp(x) for x in sp_rej_t_eventsflat[i]]
                spin_t_rejects = pd.Series(index=norm_dat.index)
                spin_t_rejects[sp_rejs_t_TS] = norm_dat[sp_rejs_t_TS]
                ax.plot(spin_t_rejects, color='red', alpha=0.5)
                # plot frequency-domain rejects
                sp_rejs_f_TS = [pd.Timestamp(x) for x in sp_rej_f_eventsflat[i]]
                spin_f_rejects = pd.Series(index=norm_dat.index)
                spin_f_rejects[sp_rejs_f_TS] = norm_dat[sp_rejs_f_TS]
                ax.plot(spin_f_rejects, color='darkred', alpha=0.5)
        
        # remove title to maximize on-screen plot area
        #ax.set_title(t)
        
        # set y axis params
        ax.set_yticks(yticks)
        ax.set_yticklabels(channels)
        ax.set_ylim(bottom = yticks[-1]-3*mx, top=yticks[0]+3*mx)

        ax.margins(x=0) # remove white space margins between data and y axis
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # if data roughly 5 mins or less, set minor x-axes
    if (d.data.index[-1] - d.data.index[0]).total_seconds() < 400:
        seconds = mdates.SecondLocator()
        ax.xaxis.set_minor_locator(seconds)
        ax.grid(axis='x', which='minor', linestyle=':')
        ax.grid(axis='x', which='major')
    
    # set overall parameters
    #fig.tight_layout(pad=0)  # remove figure padding --> this pushes slider onto fig
    
    # remove labels to maximize on-screen plot area
    #plt.xlabel('Time')
    #fig.suptitle(d.metadata['file_info']['in_num'])

    # option to use x-axis slider insted of matplotlib zoom
    if slider:
        # plot minor axes --> requires slider for segments longer than 5mins
        seconds = mdates.SecondLocator()
        ax.xaxis.set_minor_locator(seconds)
        ax.grid(axis='x', which='minor', linestyle=':')
        ax.grid(axis='x', which='major')

        # set initial window
        x_min_index = 0
        x_max_index = win_width*int(d.s_freq)

        x_min = d.data.index[x_min_index]
        x_max = d.data.index[x_max_index]
        x_dt = x_max - x_min
        
        y_min, y_max = plt.axis()[2], plt.axis()[3]

        plt.axis([x_min, x_max, y_min, y_max])

        axcolor = 'lightgoldenrodyellow'
        axpos = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor=axcolor)
        
        slider_max = len(d.data) - x_max_index - 1

        # set slider position object
        spos = Slider(axpos, 'Pos', matplotlib.dates.date2num(x_min), matplotlib.dates.date2num(d.data.index[slider_max]))

        # format date names
        #plt.gcf().autofmt_xdate()
        
        # create slider update function
        def update(val):
            pos = spos.val
            xmin_time = matplotlib.dates.num2date(pos)
            xmax_time = matplotlib.dates.num2date(pos) + x_dt
            ax.axis([xmin_time, xmax_time, y_min, y_max])
            fig.canvas.draw_idle()

        # update slider position on click
        spos.on_changed(update)

    #return fig, axs
    return fig


def plotLFP(d, raw=True, filtered=True, thresholds=True, spindles=True, spindle_rejects=True, raw_lowpass=True, lowpass_freq=25, 
            lowpass_order=4, win_frac=None, xlim=None):
    """ plot dual-channel LFP w/ option for double panel raw & filtered.

        red = spindle rejects by time domain criteria; dark red = spindle rejects by frequency domain criteria
    
    Parameters
    ----------
    d: instance of ioeeg Dataset class
    raw: bool, optional, default: True
        Option to plot raw EEG
    filtered: bool, optional, default: True
        Option to plot filtered EEG
    thresholds: bool, optional, default: True
        Option to plot spindle detection thresholds
    spindles: bool, optional, default: True
        Option to plot spindle detections
    spindle_rejects: bool, optional, default: True
        Option to plot rejected spindle detections
    raw_lowpass: bool (default: True)
        Whether to plot the lowpass filtered raw data [in place of the unchanged raw data]
    lowpass_freq: int (default: 25)
        Frequency to lowpass the raw data for visualization (if not already applied)
    lowpass_order: int (default: 4)
        Butterworth lowpass filter order to be used if lowpass_raw is not None (doubles for filtfilt)
    win_frac: str or None (default: None)
        window count, if plotting x-axis in windows (ex. '3/4' for window 3 of 4)
    xlim: tuple of DateTimeIndex
        x-axis values to be used for x-limits

    Returns
    -------
    matplotlib.pyplot figure instance
    """
    data = []
    title = []
    
    # import data
    if raw == True:
        if not raw_lowpass:
            # use the unchanged raw data
            raw_data = d.data
        elif raw_lowpass:
            # use the lowpass filtered raw data
            try:
                # check if filtered data exists
                raw_lowpass_data = d.data_lowpass
            except AttributeError:
                # apply lowpass filter
                d.lowpass_raw(lowpass_freq, lowpass_order)
                raw_lowpass_data = d.data_lowpass

    if filtered == True or thresholds == True:
        filtd = d.spindle_calcs.loc(axis=1)[:, 'Filtered']
    
    # set data to plot (title corresponds to multiindex level 2 in data df)
    if raw == True:
        if not raw_lowpass:
            # plot the unchanged data
            data.append(raw_data)
            title.append('Raw')
        elif raw_lowpass:
            # plot the lowpass data
            data.append(raw_lowpass_data)
            title.append('raw_lowpass')
    if filtered == True:    
        data.append(filtd)
        title.append('Filtered')
    if thresholds == True:
        data.append(filtd)
        title.append('Filtered')

    # flatten events list by channel for plotting
    if spindles == True:
        sp_eventsflat = [list(itertools.chain.from_iterable(d.spindle_events[i])) for i in d.spindle_events.keys()]
    if spindle_rejects == True:
        sp_rej_t_eventsflat = [list(itertools.chain.from_iterable(d.spindle_rejects_t[i])) for i in d.spindle_rejects_t.keys()]
        sp_rej_f_eventsflat = [list(itertools.chain.from_iterable(d.spindle_rejects_f[i])) for i in d.spindle_rejects_f.keys()]   

    # set channels for plotting
    channels = [x[0] for x in d.data.columns]

    # plot data    
    fig, axs = plt.subplots(len(data), 1, sharex=True, figsize=(18,6), squeeze=False)
    fig.subplots_adjust(hspace=.2, top=.9, bottom=.1, left=.05, right=.95)
    
    for (e, dat), ax, t in zip(enumerate(data), axs.flatten(), title):
        for i, c in enumerate(channels):
            # set labels for only the first filtered channel (prevent duplicate legends)
            if i == 0:
                loSD = d.metadata['spindle_analysis']['sp_loSD']
                hiSD = d.metadata['spindle_analysis']['sp_hiSD']
                labels = {'RMS': 'RMS', 'RMS mavg': 'RMS mavg', 'lo_thres':f'RMS + {loSD} SD','hi_thres':f'RMS + {hiSD} SD', 'spindles':'Spindle Detection', 
                          'spindle_rejects_t': 'Rejected Detection (time-domain)', 'spindle_rejects_f':'Rejected Detection (frequency-domain)'}
            else:
                label_keys = ['RMS', 'RMS mavg', 'lo_thres', 'hi_thres', 'spindles', 'spindle_rejects_t', 'spindle_rejects_f']
                labels = {k:'_nolegend_' for k in label_keys}
            
            # normalize each channel to [0, 1]; plot signal on 1st & 2nd panels
            dat_ser = pd.Series(dat[(c, t)], index=dat.index)
            norm_dat = (dat_ser - min(dat_ser))/(max(dat_ser)-min(dat_ser)) - i # subtract i for plotting offset
            ax.plot(norm_dat, linewidth=.5, color='C0')
            
            # plot thresholds on the second panel
            if (thresholds == True) & (e == 1):
                # RMS
                rms_ser = d.spRMS[c].RMS
                norm_rms = (rms_ser - min(dat_ser))/(max(dat_ser)-min(dat_ser)) - i
                ax.plot(norm_rms, linewidth=.8, color='green', label = labels['RMS'])
                # RMS moving average
                rmsmavg_ser = d.spRMSmavg[c].RMSmavg
                norm_rmsmavg = (rmsmavg_ser - min(dat_ser))/(max(dat_ser)-min(dat_ser)) - i
                ax.plot(norm_rmsmavg, linewidth=.8, color='orange', label = labels['RMS mavg'])
                # threshold values
                norm_lo = (d.spThresholds[c].loc['Low Threshold'] - min(dat_ser))/(max(dat_ser)-min(dat_ser)) - i
                norm_hi = (d.spThresholds[c].loc['High Threshold'] - min(dat_ser))/(max(dat_ser)-min(dat_ser)) - i
                ax.axhline(norm_lo, linestyle='solid', color='grey', label = labels['lo_thres'])
                ax.axhline(norm_hi, linestyle='dashed', color='grey', label = labels['hi_thres'])
            
            # plot spindles on the 3rd panel
            if (spindles == True) & (e == 2):
                sp_events_TS = [pd.Timestamp(x) for x in sp_eventsflat[i]]
                spins = pd.Series(index=norm_dat.index)
                spins[sp_events_TS] = norm_dat[sp_events_TS]
                ax.plot(spins, color='orange', alpha=0.5, label=labels['spindles'])
            if (spindle_rejects == True) & (e == 2):
                # plot time-domain rejects
                sp_rejs_t_TS = [pd.Timestamp(x) for x in sp_rej_t_eventsflat[i]]
                spin_rejects_t = pd.Series(index=norm_dat.index)
                spin_rejects_t[sp_rejs_t_TS] = norm_dat[sp_rejs_t_TS]
                ax.plot(spin_rejects_t, color='red', alpha=0.5, label=labels['spindle_rejects_t'])
                # plot frequency-domain rejects
                sp_rejs_f_TS = [pd.Timestamp(x) for x in sp_rej_f_eventsflat[i]]
                spin_rejects_f = pd.Series(index=norm_dat.index)
                spin_rejects_f[sp_rejs_f_TS] = norm_dat[sp_rejs_f_TS]
                ax.plot(spin_rejects_f, color='darkred', alpha=0.5, label=labels['spindle_rejects_f'])
        
        # set subplot title
        if t == 'Raw':
            subtitle = 'Original Signal'
        elif t == 'raw_lowpass':
            lp_filtfreq = d.metadata['visualizations']['lowpass_freq']
            subtitle = f'{lp_filtfreq} Hz Lowpass Filtered Signal'
        elif t == 'Filtered':
            sp_filtfreqs = d.metadata['spindle_analysis']['sp_filtwindow']
            subtitle = f'{sp_filtfreqs[0]}-{sp_filtfreqs[1]} Hz Bandpass Filtered Signal'

        # set xlimit for windowing
        if xlim is not None:
            ax.set_xlim(xlim)

        # set subplot params
        ax.set_title(subtitle, pad=5, fontsize='medium')
        ax.set_yticks(list(np.arange(0.5, -(len(channels)-1), -1)))
        ax.set_yticklabels(channels)
        ax.margins(x=0) # remove white space margins between data and y axis
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # plot minor axes
        seconds = mdates.SecondLocator()
        ax.xaxis.set_minor_locator(seconds)
        ax.grid(axis='x', which='minor', linestyle=':')
        ax.grid(axis='x', which='major')
    
    # set overall parameters
    fig_title = d.metadata['file_info']['in_num'] + ' ' + d.metadata['file_info']['path'].split('\\')[1] + ' ' + d.metadata['file_info']['path'].split('.')[0].split('_')[-1]
    if win_frac is not None:
            frac = win_frac.split('/')
            fig_title = fig_title + f' (Figure {frac[0]} of {frac[1]})'
    fig.suptitle(fig_title)
    fig.legend(ncol=2, loc='upper right', fancybox=True, framealpha=0.5)
    plt.xlabel('Time')


    return fig



### Spindle Methods ###

def plot_spindlepower_chan_i(n, chan, show_peaks='spins', dB=False, spin_type='true_spins'):
    """ Plot individual spindle spectra for a given channel 

        Parameters
        ----------
        n: nrem.NREM object
        chan: str
            channel to plot
        show_peaks: bool or str (default: 'spins')
            which peaks to plot. 'spins' plots only peaks in the spindle range (options: None, 'spins', 'all')
        spin_type: str (default: 'true_spins')
            type of spindle to plot (options: 'true_spins', 'rejects')
            note: 'rejects' option plots spindles rejected in the frequency domain, not in the time domain
    """
    
    # set data to plot
    if spin_type == 'true_spins':
        psd_dict = n.spindle_psd_i
    elif spin_type == 'rejects':
        psd_dict = n.spindle_psd_i_rejects

    # end if no spindles found matching the criteria
    if len(psd_dict[chan]) < 1:
        print(f'No {spin_type} found for channel {chan}')
        return

    # set figure & subplot params
    ncols = int(np.sqrt(len(psd_dict[chan])))
    nrows = len(psd_dict[chan])//ncols + (len(psd_dict[chan]) % ncols > 0) 
    fig, axs = plt.subplots(nrows = nrows, ncols = ncols, figsize=(ncols*3, ncols*2))
    fig.subplots_adjust(hspace=0.8, wspace=0.5)

    # move axes into a list for plotting if only one subplot
    try:
        axs_flat = axs.flatten()
    except AttributeError:
        axs_flat = [axs]

    # plot spindles
    for spin, ax in zip(psd_dict[chan], axs_flat):    
        # transform units
        if dB == True:
            pwr = 10 * np.log10(psd_dict[chan][spin].values)
            ylabel = 'Power (dB)'
        else:
            pwr = psd_dict[chan][spin].values
            ylabel = 'Power (mV^2/Hz)'
            # set y-axis to scientific notation
            ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

        # highlight spindle range. aquamarine or lavender works here too
        spin_range = n.metadata['spindle_analysis']['spin_range']
        ax.axvspan(spin_range[0], spin_range[1], color='lavender', alpha=0.8, zorder=1)
        # plot spectrum
        ax.plot(psd_dict[chan][spin].index, pwr, color='black', alpha=0.9, linewidth=0.8, zorder=2)

        # grab the peaks on the power spectrum
        p_idx, props = find_peaks(psd_dict[chan][spin])
        peaks = psd_dict[chan][spin].iloc[p_idx]
        # plot all peaks
        if show_peaks == 'all':
            ax.scatter(x=peaks.index, y=peaks.values, alpha=0.5, zorder=3)
        # plot only peaks in the spindle range
        elif show_peaks == 'spins':
            peaks_spins = peaks[(peaks.index > spin_range[0]) & (peaks.index < spin_range[1])]
            ax.scatter(x=peaks_spins.index, y=peaks_spins.values, alpha=0.5, zorder=3)


        # set subplot params
        ax.set_xlim(0, 25)
        ax.margins(y=0)
        ax.set_xticks([5, 10, 15, 20])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title(spin, size='x-small')
        ax.tick_params(axis='both', labelsize='x-small', labelleft=True) # turn labelleft=False to remove y-tick labels
    
    # delete empty subplots --> this can probably be combined with previous loop
    for i, ax in enumerate(axs_flat):
        if i >= len(psd_dict[chan]):
            fig.delaxes(ax)
    
    # set figure params   
    fig.tight_layout(pad=1, rect=[0, 0, 1, 0.93])
    fig.text(0.5, 0, 'Frequency (Hz)', ha='center')
    fig.text(0, 0.5, ylabel, va='center', rotation='vertical')
    fig.suptitle(n.metadata['file_info']['fname'].split('.')[0] + f'\nSpindle Power {chan}: {spin_type}')

    return fig


def spec_spins(n, chan, x, labels=True, raw_lowpass = True):
    """ Vizualize individual peak detections, looking at % of > 4Hz power w/in the spindle range
        
        Parameters
        ----------
        n: nrem.NREM object
            compatible with psd_type = 'i' under analyze_spindles method
        chan: str
            Channel to plot
        x: int
            Spindle # to plot
        labels: bool (default: True)
            Whether to print axis labels
        raw_lowpass: bool (default: True)
            Whether to plot the lowpass filtered raw data [in place of the unchanged raw data]

        Returns
        -------
        matplotlib.Figure
    """
    spin_range = n.metadata['spindle_analysis']['spin_range']
    prune_range = n.metadata['spindle_analysis']['prune_range']
    lowpass_freq = n.metadata['visualizations']['lowpass_freq']

    # set data to plot
    try:
        psd = n.spindle_psd_i[chan][x]
        if raw_lowpass:
            # set spindle to lowpass data
            zpad_spin = n.spindles_zpad_lowpass[chan][x]
        else:
            # use original data
            zpad_spin = n.spindles_zpad[chan][x]
        spin_perc = n.spindle_multitaper_calcs[chan][f'perc_{prune_range[0]}-{prune_range[1]}Hzpwr_in_spin_range'].loc[x]
        status = 'accepted'
    except KeyError:
        psd = n.spindle_psd_i_rejects[chan][x]
        if raw_lowpass:
            # set spindle to lowpass data
            zpad_spin = n.spindles_zpad_rejects_lowpass[chan][x]
        else:
            # use original data
            zpad_spin = n.spindles_zpad_rejects[chan][x]
        spin_perc = n.spindle_multitaper_calcs_rejects[chan][f'perc_{prune_range[0]}-{prune_range[1]}Hzpwr_in_spin_range'].loc[x]
        status = 'rejected'
    

    # subset of power w/in prune range
    psd_subset = psd[(psd.index >= prune_range[0]) & (psd.index <= prune_range[1])]
    # power in spindle range
    psd_spins = psd[(psd.index >= spin_range[0]) & (psd.index <= spin_range[1])]
    
    # plot the peak detections
    fig, axs = plt.subplots(3, 1, figsize=(5,5))
    plt.subplots_adjust(top=0.88, bottom=0.125, hspace=0.5)
    
    # plot the raw spindle + zpad
    ## set zpad label
    if raw_lowpass:
        zpad_label = f'{lowpass_freq}Hz Lowpass Filtered Signal'
    else:
        zpad_label = 'Original Signal'
    axs[0].plot(zpad_spin, alpha=1, lw=0.8, label=zpad_label)
    # convert x-tick labels from samples to ms
    xticks = axs[0].get_xticks().tolist()
    ms_xticks = [int(sample/n.s_freq*1000) for sample in xticks]
    axs[0].set_xticklabels(ms_xticks)
    if labels:
        axs[0].set_ylabel('Amplitude (mV)', size = 'small')
        axs[0].set_xlabel('Time (ms)', size = 'small')
        
    # plot the whole spectrum
    axs[1].plot(psd, c='black', lw=0.8, label='Power Spectrum')
    axs[1].axvspan(spin_range[0], spin_range[1], color='grey', alpha=0.2, zorder=0)
    if labels:
        axs[1].set_ylabel('Power (mv$^2$/Hz)', size = 'small')
        axs[1].set_xlabel('Frequency (Hz)', size = 'small')      
    
    # plot just the subset of the spectrum used for pruning
    axs[2].plot(psd_subset, c='black', lw=0.8, zorder=3) 
    axs[2].axvspan(spin_range[0], spin_range[1], color='grey', alpha=0.2, label = 'Spindle Range', zorder=0)
    axs[2].fill_between(psd_subset.index, psd_subset.values, zorder=0, alpha=0.3, color='pink')
    axs[2].fill_between(psd_subset.index, psd_subset.values, where=[sub in psd_spins.index for sub in psd_subset.index], zorder=1, alpha=1, color='white')
    axs[2].fill_between(psd_subset.index, psd_subset.values, where=[sub in psd_spins.index for sub in psd_subset.index], zorder=2, alpha=0.8, color='pink')
    if labels:
        axs[2].set_ylabel('Power (mv$^2$/Hz)', size = 'small')
        axs[2].set_xlabel('Frequency (Hz)', size = 'small')
    
    for ax in axs.flatten():
        ax.tick_params(labelsize=9)
        # set y-axis to scientific notation
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    
    fig.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0), fontsize='x-small')
    fig.suptitle(f'Channel {chan} Spindle #{x}\nSpindle range comprises {spin_perc}% of {prune_range[0]}-{prune_range[1]}Hz power ({status})', size = 'medium')
    plt.xticks(fontsize=8)
    
    return fig


def spec_peaks_SD(n, chan, x, labels=True):
    """ Vizualize spectral peaks from individual spindles
        This looks at mean power >4Hz + 1 SD as a potential frequency domain threshold
        Plots three panels: Upper = spindle tracing (w/ zero-pad) , 
            Center = spectrum w/ peaks, Lower = > 4Hz spectrum w/ peaks

        *NOTE this is not used in final detection criteria

        Parameters
        ----------
        n: nrem.NREM object
            compatible with psd_type = 'i' under analyze_spindles method
        chan: str
            Channel to plot
        x: int
            Spindle # to plot
        labels: bool (default: True)
            Whether to print axis labels

        Returns
        -------
        matplotlib.Figure
    """
    
    spin_range = n.metadata['spindle_analysis']['spin_range']
    prune_frange = [4, 25] # change this to pull the value from metadata
    s = 1 # pull from metadata --> number of standard deviations above mean

    psd = n.spindle_psd_i[chan][x]
    zpad_spin = n.spindles_zpad[chan][x]
    psd_subset = psd[(psd.index >= prune_frange[0]) & (psd.index <= prune_frange[1])]

    # grab the peaks on the power spectrum
    p_idx, props = find_peaks(psd)
    peaks = psd.iloc[p_idx]
    peaks_spins = peaks[(peaks.index > spin_range[0]) & (peaks.index < spin_range[1])]

    
    # plot the peak detections
    fig, axs = plt.subplots(3, 1, figsize=(5,5))
    plt.subplots_adjust(top=0.9, bottom=0.1, hspace=0.5)
    #fig.set_tight_layout(True)
    
    # plot the raw spindle + zpad
    axs[0].plot(zpad_spin, alpha=1, lw=0.8, label='raw signal')
    # convert x-tick labels from samples to ms
    xticks = axs[0].get_xticks().tolist()
    ms_xticks = [int(sample/n.s_freq*1000) for sample in xticks]
    axs[0].set_xticklabels(ms_xticks)
    if labels:
        axs[0].set_ylabel('Amplitude (mV)', size = 'small')
        axs[0].set_xlabel('Time (ms)', size = 'small')
        
    # plot the whole spectrum + peaks
    axs[1].plot(psd, c='black', lw=0.8, label='power spectrum')
    axs[1].axvspan(spin_range[0], spin_range[1], color='grey', alpha=0.2, label = 'Spindle Range', zorder=0)
    axs[1].scatter(x=peaks.index, y=peaks.values, c='grey', alpha=0.8, label='spectral peaks')
    if labels:
        axs[1].set_ylabel('Power (mv$^2$/Hz)', size = 'small')
        axs[1].set_xlabel('Frequency (Hz)', size = 'small')      
    
    # plot just the subset of the spectrum used for pruning + mean & SD
    axs[2].plot(psd_subset, c='black', lw=0.8) 
    axs[2].axvspan(spin_range[0], spin_range[1], color='grey', alpha=0.2, label = 'Spindle Range', zorder=0)
    axs[2].axhline(psd_subset.mean(), c='orange', linestyle = '-', label = 'mean power')
    axs[2].axhline(psd_subset.mean() + s*psd_subset.std(), c='orange', linestyle=':', label = f'mean+{s}SD')
    axs[2].scatter(x=peaks_spins.index, y=peaks_spins.values, c='grey', alpha=0.8, label='spectral peaks')
    if labels:
        axs[2].set_ylabel('Power (mv$^2$/Hz)', size = 'small')
        axs[2].set_xlabel('Frequency (Hz)', size = 'small')
    
    for ax in axs.flatten():
        ax.tick_params(labelsize=9)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    
    #fig.legend()
    fig.suptitle(f'Channel {chan} Spindle #{x}', size = 'medium')
    plt.xticks(fontsize=8)

    return fig



def plot_spins(n, datatype='Raw'):
    """ plot all spindle detections by channel 
        
        Params
        ------
        datatype: str (default: 'Raw')
            Data to plot [Options: 'Raw', 'spfilt']
    """
    
    exclude = ['EKG', 'EOG_L', 'EOG_R']
    eeg_chans = [x for x in n.spindles.keys() if x not in exclude]
    ncols = 6
    nrows = len(eeg_chans)//ncols + (len(eeg_chans) % ncols > 0) 
    fig, axs = plt.subplots(nrows = nrows, ncols = ncols, sharex=True, figsize=(15, 7))
    fig.subplots_adjust(hspace=0.5)
    
    for chan, ax in zip(n.spindles.keys(), axs.flatten()):
        if chan not in exclude:
            # set color iterator -- for other colors look at ocean, gnuplot, prism
            color=iter(plt.cm.nipy_spectral(np.linspace(0, 1, len(n.spindles[chan]))))
            for i in n.spindles[chan]:
                c = next(color)
                ax.plot(n.spindles[chan][i][datatype], c=c, alpha=1, lw=0.8)
            # set subplot params
            ax.set_xlim([-1800, 1800])
            ax.set_title(chan, fontsize='medium')
            ax.tick_params(axis='both', which='both', labelsize=8)

    # delete empty subplots --> this can probably be combined with previous loop
    for i, ax in enumerate(axs.flatten()):
        if i >= len(eeg_chans):
            fig.delaxes(ax)
                 
    # set figure params   
    fig.tight_layout(pad=1, rect=[0, 0, 1, 0.95])
    fig.text(0.5, 0, 'Time (ms)', ha='center')
    fig.text(0, 0.5, 'Amplitude (mV)', va='center', rotation='vertical')
    fig.suptitle(n.metadata['file_info']['fname'].split('.')[0])

    return fig

def plot_spin_means(n, datatype='Raw', spins=True, count=True, buffer=False, err='sem', spin_color='black', count_color='dodgerblue', buff_color='lightblue'):
    """ plot all spindle detections by channel 
    
        Note: Removed buffer option bc buffer calculations not maintained in nrem module (11-26-19)

        Parameters
        ----------
        datatype: str (default: 'Raw')
            data to plot [options: 'Raw', 'spfilt']
        spins: bool (default: True)
            plot spindle averages
        count: bool (default: True)
            plot overlay of spindle count at each timedelta
        # buffer: bool (default:False)
        #     plot average data +/- 3s from zero-neg spindle peaks. 
        #     Note: this has an effect of washing out spindles features due to asymmetry in spindle distribution 
        #     around the negative peak and corresponding averaging of spindle with non-spindle data
        err: str (default:'sem')
            type of error bars to use [options: 'std', 'sem']
        spin_color: str (default: 'black')
            color for plotting spindles
        buff_color: str (default:'lightblue')
            color for plotting buffer data
    """
    
    exclude = ['EKG', 'EOG_L', 'EOG_R']
    eeg_chans = [x for x in n.spindles.keys() if x not in exclude]
    
    # set channel locations
    locs = {'FPz': [4, 0],'Fp1': [3, 0],'Fp2': [5, 0],'AF7': [1, 1],'AF8': [7, 1],'F7': [0, 2],'F8': [8, 2],'F3': [2, 2],'F4': [6, 2],'F1': [3, 2],
            'F2': [5, 2],'Fz': [4, 2],'FC5': [1, 3],'FC6': [7, 3],'FC1': [3, 3],'FC2': [5, 3],'T3': [0, 4],'T4': [8, 4],'C3': [2, 4],'C4': [6, 4],
            'Cz': [4, 4],'CP5': [1, 5],'CP6': [7, 5],'CP1': [3, 5],'CP2': [5, 5],'CPz': [4, 5],'P3': [2, 6],'P4': [6, 6],'Pz': [4, 6],'T5': [0, 6],
            'T6': [8, 6],'POz': [4, 7],'PO7': [1, 7],'PO8': [7, 7],'O1': [2, 8],'O2': [6, 8],'Oz': [4, 8]}
    
    fig, ax = plt.subplots(9,9, figsize=(15, 13))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    for chan in n.spindles.keys():
        if chan not in exclude:
            # if buffer:
            #     data = n.spindle_buffer_means
            #     ax.plot(data[(chan, 'mean')], alpha=1, color=buff_color, label='Overall Average', lw=1)
            #     ax.fill_between(data.index, data[(chan, 'mean')] - data[(chan, err)], data[(chan, 'mean')] + data[(chan, err)], 
            #                     color=buff_color, alpha=0.2)
            if spins:
                data = n.spindle_means[datatype]
                ax[locs[chan][1], locs[chan][0]].plot(data[(chan, 'mean')], alpha=1, color=spin_color, label='Spindle Average', lw=1)
                ax[locs[chan][1], locs[chan][0]].fill_between(data.index, data[(chan, 'mean')] - data[(chan, err)], data[(chan, 'mean')] + data[(chan, err)], 
                                color=spin_color, alpha=0.2)
                if count:
                    ax1 = ax[locs[chan][1], locs[chan][0]].twinx()
                    ax1.plot(data[chan, 'count'], color=count_color, alpha=0.3)
                    ax1.fill_between(data.index, 0, data[(chan, 'count')], color=count_color, alpha=0.3)
                    max_count = len(n.spindles[chan])
                    ticks = np.linspace(0, max_count, num=5, dtype=int)
                    ax1.set_yticks(ticks=ticks)
                    ax1.set_yticklabels(labels=ticks, color=count_color)
                    ax1.tick_params(axis='y', labelsize=8) #color=count_color)

            # set subplot params
            ax[locs[chan][1], locs[chan][0]].set_xlim([-1800, 1800])
            ax[locs[chan][1], locs[chan][0]].set_title(chan, fontsize='medium')
            ax[locs[chan][1], locs[chan][0]].tick_params(axis='both', which='both', labelsize=8)

    # remove unused plots
    coords = [[x, y] for x in range(0, 9) for y in range(0,9)]
    unused = [c for c in coords if  c not in locs.values()]
    for u in unused:
        fig.delaxes(ax[u[1], u[0]])

    # set figure params
    #fig.legend()   
    fig.tight_layout(pad=1, rect=[0, 0, 1, 0.95])
    fig.text(0.5, 0, 'Time (ms)', ha='center', size='large')
    fig.text(0, 0.5, 'Amplitude (mV)', va='center', rotation='vertical', color=spin_color, size='large')
    fig.text(1, 0.5, 'Spindle Count', va='center', rotation=270, color=count_color, size='large')
    fig.suptitle(n.metadata['file_info']['fname'].split('.')[0] + f'\nSpindle Averages ({datatype})')

    return fig


def plot_spin_clust_means(n, datatype='Raw', spins=True, count=True, err='sem', spin_color='black', count_color='dodgerblue'):
    """ plot mean spindles by cluster 

        Parameters
        ----------
        cluster: int
            which cluster to plot [options: 0, 1]
        datatype: str (default: 'Raw')
            data to plot [options: 'Raw', 'spfilt']
        spins: bool (default: True)
            plot spindle averages
        count: bool (default: True)
            plot overlay of spindle count at each timedelta
        err: str (default:'sem')
            type of error bars to use [options: 'std', 'sem']
        spin_color: str (default: 'black')
            color for plotting spindles
    """
    
    fig, axs = plt.subplots(1, 2, figsize=(7,3), sharey=True)
    
    for ax, clust in zip(axs.flatten(), n.spindle_aggregates_clust):
        if spins:
            data = n.spindle_clust_means[datatype]
            ax.plot(data[(clust, 'mean')], alpha=1, color=spin_color, label='Spindle Average', lw=1)
            ax.fill_between(data.index, data[(clust, 'mean')] - data[(clust, err)], data[(clust, 'mean')] + data[(clust, err)], 
                            color=spin_color, alpha=0.2)
            if count:
                ax1 = ax.twinx()
                ax1.plot(data[clust, 'count'], color=count_color, alpha=0.3)
                ax1.fill_between(data.index, 0, data[(clust, 'count')], color=count_color, alpha=0.3)
                max_count = len(n.spindle_aggregates_clust[clust]['Raw'].columns)
                ticks = np.linspace(0, max_count, num=5, dtype=int)
                ax1.set_yticks(ticks=ticks)
                ax1.set_yticklabels(labels=ticks, color=count_color)
                ax1.tick_params(axis='y', labelsize=8) #color=count_color)

        # set subplot params
        ax.set_xlim([-1000, 1000])
        ax.set_title('Cluster ' + str(clust), fontsize='medium')
        ax.set_xlabel('Time (ms)', size='large')
        ax.set_ylabel('Amplitude (mV)', size='large')
        ax.tick_params(axis='both', which='both', labelsize=8)

    # set figure params
    #fig.legend()   
    fig.tight_layout(pad=1, rect=[0, 0, 1, 0.92])
    if count:
        fig.text(1, 0.5, 'Spindle Count', va='center', rotation=270, color=count_color, size='large')
    fig.suptitle(n.metadata['file_info']['fname'].split('.')[0] + f'\nSpindle Averages ({datatype})')

    return fig

def plot_avg_spindle(n, datatype='Raw', spins=True, count=True, err='sem', spin_color='black', count_color='dodgerblue'):
    """ plot the average spindle tracing across the head 
        For use in comparison with cluster averages
    """
    fig, ax = plt.subplots()
    
    if spins:
        data = n.spindle_means_all[datatype]
        ax.plot(data['mean'], alpha=1, color=spin_color, label='Spindle Average', lw=1)
        ax.fill_between(data.index, data['mean'] - data[err], data['mean'] + data[err], 
                        color=spin_color, alpha=0.2)
        if count:
            ax1 = ax.twinx()
            ax1.plot(data['count'], color=count_color, alpha=0.3)
            ax1.fill_between(data.index, 0, data['count'], color=count_color, alpha=0.3)
            max_count = len(n.spindle_aggregates_all['Raw'].columns)
            ticks = np.linspace(0, max_count, num=5, dtype=int)
            ax1.set_yticks(ticks=ticks)
            ax1.set_yticklabels(labels=ticks, color=count_color)
            ax1.tick_params(axis='y', labelsize=8) #color=count_color)

    # set subplot params
    ax.set_xlim([-1000, 1000])
    #ax.set_title('Cluster ' + str(clust), fontsize='medium')
    ax.set_xlabel('Time (ms)', size='large')
    ax.set_ylabel('Amplitude (mV)', size='large')
    ax.tick_params(axis='both', which='both', labelsize=8)

    # set figure params
    #fig.legend()   
    fig.tight_layout(pad=1, rect=[0, 0, 1, 0.92])
    if count:
        fig.text(1, 0.5, 'Spindle Count', va='center', rotation=270, color=count_color, size='large')
    fig.suptitle(n.metadata['file_info']['fname'].split('.')[0] + f'\nSpindle Averages ({datatype})')

    return fig


def plot_spindlepower_chan(n, chan, dB=True):
    """ Plot spindle power spectrum for a single channel """

    # transform units
    if dB == True:
        pwr = 10 * np.log10(n.spindle_psd_concat[chan].values)
        ylabel = 'Power (dB)'
    else:
        pwr = n.spindle_psd_concat[chan].values
        ylabel = 'Power (mV^2/Hz)'
    
    fig, ax = plt.subplots()
    
    # plot just spectrum
    ax.plot(n.spindle_psd_concat[chan].index, pwr, color='black', alpha=0.9, linewidth=0.8)
    ax.axvspan(9, 16, color='lavender', alpha=0.8)
        
    ax.set_xlim(0, 25)
    ax.margins(y=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.xlabel('Frequency (Hz)')
    plt.ylabel(ylabel)
    plt.title((n.metadata['file_info']['fname'].split('.')[0] + '\n\n' + chan + ' Concatenated Spindle Power'), size='medium', weight='semibold')

    return fig


def plot_spindlepower(n, dB=True):
    """ Plot spindle power spectrum (from concatenated spindles) for all channels """
    
    exclude = ['EKG', 'EOG_L', 'EOG_R']
    eeg_chans = [x for x in n.spindle_psd_concat.keys() if x not in exclude]
    
    # set subplot parameters
    if len(eeg_chans) < 1:
        print('No concatened spindles detected to plot.')
        return
    elif len(eeg_chans) < 6:
        ncols = len(eeg_chans)
    else:
        ncols = int(len(eeg_chans)/6)
    nrows = len(eeg_chans)//ncols + (len(eeg_chans) % ncols > 0) 
    fig, axs = plt.subplots(nrows = nrows, ncols = ncols, figsize=(ncols*3, ncols*2))
    fig.subplots_adjust(hspace=0.8, wspace=0.5)

    # move axes into a list for plotting if only one subplot
    try:
        axs_flat = axs.flatten()
    except AttributeError:
        axs_flat = [axs]

    for chan, ax in zip(eeg_chans, axs_flat):    
        # transform units
        if dB == True:
            pwr = 10 * np.log10(n.spindle_psd_concat[chan].values)
            ylabel = 'Power (dB)'
        else:
            pwr = n.spindle_psd_concat[chan].values
            ylabel = 'Power (mV^2/Hz)'
            # set y-axis to scientific notation
            ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

        # plot spectrum
        ax.plot(n.spindle_psd_concat[chan].index, pwr, color='black', alpha=0.9, linewidth=0.8)
        # highlight spindle range. aquamarine or lavender works here too
        spin_range = n.metadata['spindle_analysis']['spin_range']
        ax.axvspan(spin_range[0], spin_range[1], color='grey', alpha=0.2)

        # set subplot params
        ax.set_xlim(0, 25)
        ax.margins(y=0)
        ax.set_xticks([5, 10, 15, 20])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title(chan, size='medium', weight='bold')
    
    # delete empty subplots --> this can probably be combined with previous loop
    for i, ax in enumerate(axs_flat):
        if i >= len(eeg_chans):
            fig.delaxes(ax)
    
    # set figure params   
    fig.tight_layout(pad=1, rect=[0, 0, 1, 0.93])
    fig.text(0.5, 0, 'Frequency (Hz)', ha='center')
    fig.text(0, 0.5, ylabel, va='center', rotation='vertical')
    fig.suptitle(n.metadata['file_info']['fname'].split('.')[0] + '\nSpindle Power')

    return fig

def plot_spindlepower_headplot(n, dB=True):
    """ 
        Headplot of spindle power spectrum for all channels 
        NOTE: only for FS128 (12-11-19)
    """
    
# set channel locations
    locs = {'FPz': [4, 0],'Fp1': [3, 0],'Fp2': [5, 0],'AF7': [1, 1],'AF8': [7, 1],'F7': [0, 2],'F8': [8, 2],'F3': [2, 2],'F4': [6, 2],'F1': [3, 2],
            'F2': [5, 2],'Fz': [4, 2],'FC5': [1, 3],'FC6': [7, 3],'FC1': [3, 3],'FC2': [5, 3],'T3': [0, 4],'T4': [8, 4],'C3': [2, 4],'C4': [6, 4],
            'Cz': [4, 4],'CP5': [1, 5],'CP6': [7, 5],'CP1': [3, 5],'CP2': [5, 5],'CPz': [4, 5],'P3': [2, 6],'P4': [6, 6],'Pz': [4, 6],'T5': [0, 6],
            'T6': [8, 6],'POz': [4, 7],'PO7': [1, 7],'PO8': [7, 7],'O1': [2, 8],'O2': [6, 8],'Oz': [4, 8]}
    
    fig, ax = plt.subplots(9,9, figsize=(12, 12))
    plt.subplots_adjust(hspace=0.3, wspace=0.3) # use this or tight_layout

    for chan in locs.keys():    
        # transform units
        if dB == True:
            pwr = 10 * np.log10(n.spindle_psd_concat[chan].values)
            ylabel = 'Power (dB)'
        else:
            pwr = n.spindle_psd_concat[chan].values
            ylabel = 'Power (mV^2/Hz)'

        # plot spectrum
        #ax = plt.subplot()
        ax[locs[chan][1], locs[chan][0]].plot(n.spindle_psd_concat[chan].index, pwr, color='black', alpha=0.9, linewidth=0.8)
        # highlight spindle range. aquamarine or lavender works here too
        ax[locs[chan][1], locs[chan][0]].axvspan(9, 16, color='grey', alpha=0.2)

        # set subplot params
        ax[locs[chan][1], locs[chan][0]].set_xlim(0, 25)
        ax[locs[chan][1], locs[chan][0]].margins(y=0)
        ax[locs[chan][1], locs[chan][0]].set_xticks([5, 10, 15, 20])
        ax[locs[chan][1], locs[chan][0]].tick_params(axis='both', labelsize=7)
        ax[locs[chan][1], locs[chan][0]].spines['top'].set_visible(False)
        ax[locs[chan][1], locs[chan][0]].spines['right'].set_visible(False)
        ax[locs[chan][1], locs[chan][0]].set_title(chan, size='small', weight='semibold')
        ax[locs[chan][1], locs[chan][0]].title.set_position([.5, 0.75])
        #ax[locs[chan][1], locs[chan][0]].text(0.0, 0.0, chan)
        
    # remove unused plots
    coords = [[x, y] for x in range(0, 9) for y in range(0,9)]
    unused = [c for c in coords if  c not in locs.values()]
    for u in unused:
        fig.delaxes(ax[u[1], u[0]])

    # set labels
    fig.text(0.5, 0.08, 'Frequency (Hz)', ha='center', size='large', weight='semibold')
    fig.text(0.08, 0.5, ylabel, va='center', rotation='vertical', size='large', weight='semibold')

    return fig


def plot_gottselig(n, datatype='calcs', plot_peaks=True, smoothed=True):
    """ plot gottselig normalization for all channels 
    
        Parameters
        ----------
        datatype: str (default: 'calcs')
            which data to plot [options: 'calcs', 'normed_pwr']
        plot_peaks: bool (default: True)
            whether to plot peak detections [only if datatype='normed_pwr']
        smoothed: bool (default: False)
            whether to plot rms smoothed signal used for peak calculations [only if datatype='normed_pwr']
    """
    
    exclude = ['EKG', 'EOG_L', 'EOG_R']
    eeg_chans = [x for x in n.spindle_psd_concat.keys() if x not in exclude]
     # set subplot parameters
    if len(eeg_chans)/6 < 1:
        ncols = 1
    else:
        ncols = int(len(eeg_chans)/6)
    nrows = len(eeg_chans)//ncols + (len(eeg_chans) % ncols > 0) 
    fig, axs = plt.subplots(nrows = nrows, ncols = ncols, figsize=(ncols*3, ncols*2))
    fig.subplots_adjust(hspace=0.8, wspace=0.5)
    
    for chan, ax in zip(eeg_chans, axs.flatten()):
        data = n.spindle_psd_concat[chan]
        data_normed = n.spindle_psd_concat_norm[chan]
        
        if datatype == 'calcs':
            # first plot
            ax.scatter(data_normed['values_to_fit'].index, data_normed['values_to_fit'].values, alpha=0.8, color='mediumslateblue', linewidths=0, marker='s', label='Normalization Range')
            ax.plot(data.index, 10*np.log10(data.values), color='black', label = 'Power Spectrum')
            ax.plot(data_normed['exp_fit_line'], color='mediumblue', label = 'Exponential fit')
            ax.set_title(chan)
        
        elif datatype == 'normed_pwr':
            # second plot
            ax.plot(data_normed['normed_pwr'], color='black', lw=0.8, label='Normalized power', zorder=2)
            ax.axvspan(9, 16, color='lightgrey', alpha=0.8, label = 'Spindle Range', zorder=1)
            ax.set_title(chan)

            if smoothed:
                # plot smoothed psd
                ax.plot(n.psd_concat_norm_peaks[chan]['smoothed_data'], lw=1.2, alpha=0.8, color='lime', zorder=3)
           
            if plot_peaks:
                # plot peak detections
                peaks = n.psd_concat_norm_peaks[chan]['peaks']
                ax.scatter(x=peaks.index, y=peaks.values, color='magenta', alpha=0.8, marker=7, zorder=4)
        
        # set subplot params
        ax.margins(y=0)
        #ax.set_xticks([5, 10, 15, 20])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title(chan, size='medium')
    
    # delete empty subplots
    for i, ax in enumerate(axs.flatten()):
        if i >= len(eeg_chans):
            fig.delaxes(ax)
    
    # set figure params   
    fig.tight_layout(pad=1, rect=[0, 0, 1, 0.93])
    fig.text(0.5, 0, 'Frequency (Hz)', ha='center', size='large')
    fig.text(0, 0.5, 'Power (dB)', va='center', rotation='vertical', size='large')
    fig.suptitle(n.metadata['file_info']['fname'].split('.')[0] + '\n\nGottselig Normalization', size='large')


    return fig

def plot_gottselig_headplot(n, datatype='calcs', plot_peaks=True, smoothed=True):
    """ plot gottselig normalization headplot for all channels 
        NOTE: only for FS128 (12-11-19)

        To Do: Change plots for consistent y-axis
    
        Parameters
        ----------
        datatype: str (default: 'calcs')
            which data to plot [options: 'calcs', 'normed_pwr']
        plot_peaks: bool (default: True)
            whether to plot peak detections [only if datatype='normed_pwr']
        smoothed: bool (default: False)
            whether to plot rms smoothed signal used for peak calculations [only if datatype='normed_pwr']
    """
    
    # set channel locations
    locs = {'FPz': [4, 0],'Fp1': [3, 0],'Fp2': [5, 0],'AF7': [1, 1],'AF8': [7, 1],'F7': [0, 2],'F8': [8, 2],'F3': [2, 2],'F4': [6, 2],'F1': [3, 2],
            'F2': [5, 2],'Fz': [4, 2],'FC5': [1, 3],'FC6': [7, 3],'FC1': [3, 3],'FC2': [5, 3],'T3': [0, 4],'T4': [8, 4],'C3': [2, 4],'C4': [6, 4],
            'Cz': [4, 4],'CP5': [1, 5],'CP6': [7, 5],'CP1': [3, 5],'CP2': [5, 5],'CPz': [4, 5],'P3': [2, 6],'P4': [6, 6],'Pz': [4, 6],'T5': [0, 6],
            'T6': [8, 6],'POz': [4, 7],'PO7': [1, 7],'PO8': [7, 7],'O1': [2, 8],'O2': [6, 8],'Oz': [4, 8]}
    
    fig, ax = plt.subplots(9,9, figsize=(12, 12))
    plt.subplots_adjust(hspace=0.3, wspace=0.3) # use this or tight_layout
    
    for chan in locs.keys():    
        data = n.spindle_psd_concat[chan]
        data_normed = n.spindle_psd_concat_norm[chan]
        
        if datatype == 'calcs':
            # first plot
            ax[locs[chan][1], locs[chan][0]].scatter(data_normed['values_to_fit'].index, data_normed['values_to_fit'].values, alpha=0.8, color='mediumslateblue', linewidths=0, marker='s', label='Normalization Range')
            ax[locs[chan][1], locs[chan][0]].plot(data.index, 10*np.log10(data.values), color='black', label = 'Power Spectrum')
            ax[locs[chan][1], locs[chan][0]].plot(data_normed['exp_fit_line'], color='mediumblue', label = 'Exponential fit')
            ax[locs[chan][1], locs[chan][0]].set_title(chan)
        
        elif datatype == 'normed_pwr':
            # second plot
            ax[locs[chan][1], locs[chan][0]].plot(data_normed['normed_pwr'], color='black', lw=0.8, label='Normalized power', zorder=2)
            ax[locs[chan][1], locs[chan][0]].axvspan(9, 16, color='lavender', alpha=0.8, label = 'Spindle Range', zorder=1)
            ax[locs[chan][1], locs[chan][0]].set_title(chan)

            if smoothed:
                # plot smoothed psd
                ax[locs[chan][1], locs[chan][0]].plot(n.psd_concat_norm_peaks[chan]['smoothed_data'], lw=1.2, color='lime', alpha=0.8, label='Smoothed Spectrum', zorder=3)
           
            if plot_peaks:
                # plot peak detections
                peaks = n.psd_concat_norm_peaks[chan]['peaks']
                ax[locs[chan][1], locs[chan][0]].scatter(x=peaks.index, y=peaks.values, color='magenta', marker=7, alpha=0.8, label ='Spindle Peak', zorder=4)
        
        
        # set subplot params
        ax[locs[chan][1], locs[chan][0]].margins(y=0)
        ax[locs[chan][1], locs[chan][0]].set_xlim(0, 25)
        ax[locs[chan][1], locs[chan][0]].margins(y=0)
        ax[locs[chan][1], locs[chan][0]].set_xticks([5, 10, 15, 20])
        ax[locs[chan][1], locs[chan][0]].tick_params(axis='both', labelsize=7)
        ax[locs[chan][1], locs[chan][0]].spines['top'].set_visible(False)
        ax[locs[chan][1], locs[chan][0]].spines['right'].set_visible(False)
        ax[locs[chan][1], locs[chan][0]].set_title(chan, size='small', weight='semibold')
        ax[locs[chan][1], locs[chan][0]].title.set_position([.5, 0.75])
    
    # remove unused plots
    coords = [[x, y] for x in range(0, 9) for y in range(0,9)]
    unused = [c for c in coords if  c not in locs.values()]
    for u in unused:
        fig.delaxes(ax[u[1], u[0]])
    
    # set labels
    fig.text(0.5, 0.08, 'Frequency (Hz)', ha='center', size='large', weight='semibold')
    fig.text(0.08, 0.5, 'Power (dB)', va='center', rotation='vertical', size='large', weight='semibold')

    return fig


### Slow Oscillation Methods ###
def plot_so(n, datatype='Raw'):
    """ plot all slow oscillation detections by channel 
        
        Params
        ------
        datatype: str (default: 'Raw')
            Data to plot [Options: 'Raw', 'sofilt']
    """
    
    # set channel locations
    locs = {'FPz': [4, 0],'Fp1': [3, 0],'Fp2': [5, 0],'AF7': [1, 1],'AF8': [7, 1],'F7': [0, 2],'F8': [8, 2],'F3': [2, 2],'F4': [6, 2],'F1': [3, 2],
            'F2': [5, 2],'Fz': [4, 2],'FC5': [1, 3],'FC6': [7, 3],'FC1': [3, 3],'FC2': [5, 3],'T3': [0, 4],'T4': [8, 4],'C3': [2, 4],'C4': [6, 4],
            'Cz': [4, 4],'CP5': [1, 5],'CP6': [7, 5],'CP1': [3, 5],'CP2': [5, 5],'CPz': [4, 5],'P3': [2, 6],'P4': [6, 6],'Pz': [4, 6],'T5': [0, 6],
            'T6': [8, 6],'POz': [4, 7],'PO7': [1, 7],'PO8': [7, 7],'O1': [2, 8],'O2': [6, 8],'Oz': [4, 8]}

    exclude = ['EKG', 'EOG_L', 'EOG_R']
    eeg_chans = [x for x in n.spindles.keys() if x not in exclude]

    fig, ax = plt.subplots(9,9, figsize=(15, 13))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    for chan in n.so.keys():
        if chan not in exclude:
            # set color iterator -- for other colors look at ocean, gnuplot, prism
            color=iter(plt.cm.nipy_spectral(np.linspace(0, 1, len(n.so[chan]))))
            for i in n.so[chan]:
                c = next(color)
                ax[locs[chan][1], locs[chan][0]].plot(n.so[chan][i][datatype], c=c, alpha=1, lw=0.8)
            # set subplot params
            ax[locs[chan][1], locs[chan][0]].set_xlim([-2500, 2500])
            ax[locs[chan][1], locs[chan][0]].set_title(chan, fontsize='medium')
            ax[locs[chan][1], locs[chan][0]].tick_params(axis='both', which='both', labelsize=8)

    # remove unused plots
    coords = [[x, y] for x in range(0, 9) for y in range(0,9)]
    unused = [c for c in coords if  c not in locs.values()]
    for u in unused:
        fig.delaxes(ax[u[1], u[0]])
                 
    # set figure params   
    fig.tight_layout(pad=1, rect=[0, 0, 1, 0.95])
    fig.text(0.5, 0, 'Time (ms)', ha='center')
    fig.text(0, 0.5, 'Amplitude (mV)', va='center', rotation='vertical')
    fig.suptitle(n.metadata['file_info']['fname'].split('.')[0])

    fig.tight_layout()

    return fig



### SpSO Methods ###

def plot_spsomap(n):
    """ Plot histogram mapping of spso """
    fig, ax = plt.subplots(figsize=(14, 3))

    ax.fill_between(x = n.so_bool.index, y1=0, y2=n.so_bool.T.sum(), alpha=0.5, color='blue', label='Slow Oscillations')
    ax.fill_between(x = n.spin_bool.index, y1=0, y2=n.spin_bool.T.sum(), alpha=0.5, color='green', label='Spindles')
    ax.set_xlabel('Time')
    ax.set_ylabel('Channel Count')
    ax.margins(x=0)
    ax.legend()


def plot_spso_chan_subplots(n, chan, so_dtype='sofilt', sp_dtype='spfilt'):
    """ Subplot individual slow oscillations with overriding spindle detections (one subplot per SO) """
    
    if sp_dtype == 'spfilt':
        spin_data = n.spfiltEEG
    elif sp_dtype == 'spsofilt':
        spin_data = n.spsofiltEEG
    elif sp_dtype == 'sofilt':
        spin_data = n.sofiltEEG
    
    height = 2/3 * int(len(n.so_spin_map[chan]))
    fig, axs = plt.subplots(nrows=int(len(n.so_spin_map[chan])/3)+1, ncols=3, figsize=(10, height))
    fig.subplots_adjust(hspace=0.4)
    
    for ax, (so, spins) in zip(axs.flatten(), n.so_spin_map[chan].items()):
        ax.plot(n.so[chan][so].time, n.so[chan][so][so_dtype])
        for spin in spins:
            ax.plot(spin_data[(chan, 'Filtered')].loc[n.spindle_events[chan][spin]], lw=1)
        ax.tick_params(axis='x', labelsize='small', rotation=15., pad=.1)
    
    # delete empty subplots --> this can probably be combined with previous loop
    for i, ax in enumerate(axs.flatten()):
        if i >= len(n.so_spin_map[chan]):
            fig.delaxes(ax)

    fig.text(0.5, 0, 'Time (ms)', ha='center')
    fig.text(0, 0.5, 'Amplitude (mV)', va='center', rotation='vertical')
    fig.suptitle(n.metadata['file_info']['fname'].split('.')[0])

def plot_spso_chan(n, chan, so_dtype='sofilt', sp_dtype='spsofilt', spin_tracings=False, plot_dist=True, so_tracings=True):
    """ Plot individual slow oscillations with overriding spindle detections 
    
        Parameters
        ----------
        chan: str
            channel to plot
        so_dtype: str (default: 'sofilt')
            slow oscillation data to plot [Options: 'sofilt', 'spsofilt']
        sp_dtype: str (default: 'spsofilt')
            spindle data to plot [Options: 'spfilt', 'spsofilt']
            *Note: spfilt is broken ATM
        spin_tracings: bool (default: False)
            whether to plot spindle tracings
        plot_dist: bool (default: True)
            whether to plot spindle distribution
        so_tracings: bool (default: True)
            whether to plot so tracings (if False, will plot SO mean)
    """
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    so_dict = {}
    for so_id, df in n.spso_aggregates[chan].items():
        if so_tracings:
            # plot slow oscillation
            ax.plot(df[so_dtype], color='black', alpha=0.2)
        else:
            # grab the slow oscillations to calculate mean
            so_dict[chan+'_'+str(so_id)] = df[df.index.notna()][so_dtype]

        # grab spindle columns
        spin_cols = [x for x in df.columns if x.split('_')[0] == 'spin']
        for spin in spin_cols:
            # get index & cluster of spindle
            spin_idx = int(spin_cols[0].split('_')[1])
            clust = int(n.spindle_stats_i[(n.spindle_stats_i.chan == chan) & (n.spindle_stats_i.spin == spin_idx)].cluster.values)

            if spin_tracings:
                # plot spindle
                c = plt.get_cmap('RdYlBu', 2)(clust)
                hx = matplotlib.colors.rgb2hex(c[:-1])
                ax.plot(df[sp_dtype][df[spin].notna()], lw=3, color=hx, alpha=0.5)

    # plot SO mean
    if so_tracings == False:
        so_df = pd.DataFrame(so_dict)
        mean = so_df.mean(axis=1)
        sd = so_df.std(axis=1)
        if len(mean) > 0:
            ax.plot(mean, color='black')
            ax.fill_between(mean.index, mean-sd, mean+sd, color='black', alpha=0.3)

    if plot_dist:
        # plot normalized distribution of each cluster for each timepoint
        ax1 = ax.twinx()
        for clust, dct in n.spin_dist['by_chan'][chan].items():
            # set color
            c = plt.get_cmap('RdYlBu', 2)(int(clust))
            hx = matplotlib.colors.rgb2hex(c[:-1])
            # plot normed distribution
            ax1.plot(dct['dist_norm'], color=c, label='Cluster ' + clust)
            ax1.fill_between(dct['dist_norm'].index, 0, dct['dist_norm'].values, color=c, alpha=0.3)
        ax1.set_ylabel('Proportion of spindles present')
        ax1.legend()

    ax.tick_params(axis='x', rotation=15., pad=.1)
    ax.tick_params(axis='y', rotation=0, pad=.1)
    ax.set_ylabel('Ampltiude (mV)')
    ax.set_xlabel('Time (ms)')

    fig.suptitle(n.metadata['file_info']['fname'].split('.')[0])
    fig.tight_layout()
    
    return fig


def plot_spso(n, so_dtype='sofilt', sp_dtype='spsofilt', spin_tracings=False, plot_dist=True, so_tracings=True, cmap='winter', ylims=None, legend=False):
    """ Plot individual slow oscillations with overriding spindle detections 
    
        Parameters
        ----------
        so_dtype: str (default: 'sofilt')
            slow oscillation data to plot [Options: 'sofilt', 'spsofilt']
        sp_dtype: str (default: 'spsofilt')
            spindle data to plot [Options: 'spfilt', 'spsofilt']
            *Note: spfilt is broken ATM
        spin_tracings: bool (default: False)
            whether to plot spindle tracings
        plot_dist: bool (default: True)
            whether to plot spindle distribution
        so_tracings: bool (default: True)
            whether to plot SO tracings (if set to False, mean SO will be plotted)
        cmap: str (default:'winter')
            matplotlib colormap. usually 'winter' or 'RdYlBu'
        ylims: tuple or None (default: None)
            y limits for SO tracings axis. for use if an outlier is skewing the axis
        legend: bool (default: False)
            whether to plot the legend
        
    """
    
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.rcParams["font.family"] = "Arial"
    
    so_dict = {}
    for chan in n.spso_aggregates.keys():
        for so_id, df in n.spso_aggregates[chan].items():
            if so_tracings:
                # plot slow oscillation
                ax.plot(df[so_dtype], color='black', alpha=0.2)
                if ylims is not None:
                    ax.set_ylim(ylims)
            else:
                # grab the slow oscillations to calculate mean
                so_dict[chan+'_'+str(so_id)] = df[df.index.notna()][so_dtype]

            # grab spindle columns
            spin_cols = [x for x in df.columns if x.split('_')[0] == 'spin']
            for spin in spin_cols:
                # get index & cluster of spindle
                spin_idx = int(spin_cols[0].split('_')[1])
                clust = int(n.spindle_stats_i[(n.spindle_stats_i.chan == chan) & (n.spindle_stats_i.spin == spin_idx)].cluster.values)
                
                if spin_tracings:
                    # plot spindle
                    c = plt.get_cmap(cmap, 2)(clust)
                    hx = matplotlib.colors.rgb2hex(c[:-1])
                    ax.plot(df[sp_dtype][df[spin].notna()], lw=3, color=hx, alpha=0.8)

    # plot SO mean
    if so_tracings == False:
        so_df = pd.DataFrame(so_dict)
        mean = so_df.mean(axis=1)
        sd = so_df.std(axis=1)
        if len(mean) > 0:
            ax.plot(mean, color='black', lw=2)
            ax.fill_between(mean.index, mean-sd, mean+sd, color='grey', alpha=0.3)

    if plot_dist:
        # plot normalized distribution of each cluster for each timepoint
        ax1 = ax.twinx()
        for clust, dct in n.spin_dist['all'].items():
            # set color
            c = plt.get_cmap(cmap, 2)(int(clust))
            hx = matplotlib.colors.rgb2hex(c[:-1])
            # plot normed distribution
            ax1.plot(dct['dist_norm'], color=c, label='Cluster ' + clust, lw=2)
            ax1.fill_between(dct['dist_norm'].index, 0, dct['dist_norm'].values, color=c, alpha=0.4)
            ax1.set_ylim(0, 1.0)
        ax1.set_ylabel('Spindles present (%)', size=30, rotation=270, va='bottom')
        ax1.set_yticklabels(['0', '20', '40', '60', '80', '100'])
        ax1.tick_params(axis='y', rotation=0, pad=.1, labelsize=20)
        if legend:
            ax1.legend()

    ax.tick_params(axis='x', rotation=15., pad=.1, labelsize=20)
    ax.tick_params(axis='y', rotation=0, pad=.1, labelsize=20)
    ax.set_ylabel('Ampltiude (mV)', size=30)
    ax.set_xlabel('Time (ms)', size=30)

    #fig.suptitle(n.metadata['file_info']['fname'].split('.')[0])
    #fig.tight_layout()
    
    return fig



def plot_spso_headplot(n, so_dtype='sofilt', sp_dtype='spsofilt', spin_tracings=False, plot_dist=True, so_tracings=True):
    """ Plot headplot of slow oscillations with spindle distribution by cluster

        Parameters
        ----------
        chan: str
            channel to plot
        so_dtype: str (default: 'sofilt')
            slow oscillation data to plot [Options: 'sofilt', 'spsofilt']
        sp_dtype: str (default: 'spsofilt')
            spindle data to plot [Options: 'spfilt', 'spsofilt']
            *Note: spfilt is broken ATM
        spin_tracings: bool (default: False)
            whether to plot spindle tracings
        plot_dist: bool (default: True)
            whether to plot spindle distribution
    """

    # set channel locations
    locs = {'FPz': [4, 0],'Fp1': [3, 0],'Fp2': [5, 0],'AF7': [1, 1],'AF8': [7, 1],'F7': [0, 2],'F8': [8, 2],'F3': [2, 2],'F4': [6, 2],'F1': [3, 2],
            'F2': [5, 2],'Fz': [4, 2],'FC5': [1, 3],'FC6': [7, 3],'FC1': [3, 3],'FC2': [5, 3],'T3': [0, 4],'T4': [8, 4],'C3': [2, 4],'C4': [6, 4],
            'Cz': [4, 4],'CP5': [1, 5],'CP6': [7, 5],'CP1': [3, 5],'CP2': [5, 5],'CPz': [4, 5],'P3': [2, 6],'P4': [6, 6],'Pz': [4, 6],'T5': [0, 6],
            'T6': [8, 6],'POz': [4, 7],'PO7': [1, 7],'PO8': [7, 7],'O1': [2, 8],'O2': [6, 8],'Oz': [4, 8]}
    
    fig, ax = plt.subplots(9,9, figsize=(15, 13))
    plt.subplots_adjust(hspace=0.3, wspace=0.3) # use this or tight_layout
    
    for chan in locs.keys():
        so_dict = {}
        for so_id, df in n.spso_aggregates[chan].items():
            if so_tracings:
                # plot slow oscillation
                ax[locs[chan][1], locs[chan][0]].plot(df[so_dtype], color='black', alpha=0.2)
            else:
                # grab the slow oscillations to calculate mean
                so_dict[chan+'_'+str(so_id)] = df[df.index.notna()][so_dtype]

            # grab spindle columns
            spin_cols = [x for x in df.columns if x.split('_')[0] == 'spin']
            for spin in spin_cols:
                # get index & cluster of spindle
                spin_idx = int(spin_cols[0].split('_')[1])
                clust = int(n.spindle_stats_i[(n.spindle_stats_i.chan == chan) & (n.spindle_stats_i.spin == spin_idx)].cluster.values)

                if spin_tracings:
                    # plot spindle
                    c = plt.get_cmap('RdYlBu', 2)(clust)
                    hx = matplotlib.colors.rgb2hex(c[:-1])
                    ax[locs[chan][1], locs[chan][0]].plot(df[sp_dtype][df[spin].notna()], lw=3, color=hx, alpha=0.5)

        # plot SO mean
        if so_tracings == False:
            so_df = pd.DataFrame(so_dict)
            mean = so_df.mean(axis=1)
            sd = so_df.std(axis=1)
            if len(mean) > 0:
                ax[locs[chan][1], locs[chan][0]].plot(mean, color='black')
                ax[locs[chan][1], locs[chan][0]].fill_between(mean.index, mean-sd, mean+sd, color='black', alpha=0.3)

        ax[locs[chan][1], locs[chan][0]].set_title(chan, size='small')
        ax[locs[chan][1], locs[chan][0]].tick_params(axis='x', rotation=0, pad=.1, labelsize='x-small')
        ax[locs[chan][1], locs[chan][0]].tick_params(axis='y', rotation=0, pad=.1, labelsize='x-small')
        ax[locs[chan][1], locs[chan][0]].spines['top'].set_visible(False)

        if plot_dist:
            # plot normalized distribution of each cluster for each timepoint
            ax1 = ax[locs[chan][1], locs[chan][0]].twinx()
            for clust, dct in n.spin_dist['by_chan'][chan].items():
                # set color
                c = plt.get_cmap('RdYlBu', 2)(int(clust))
                hx = matplotlib.colors.rgb2hex(c[:-1])
                # plot normed distribution
                ax1.plot(dct['dist_norm'], color=c, label='Cluster ' + clust)
                ax1.fill_between(dct['dist_norm'].index, 0, dct['dist_norm'].values, color=c, alpha=0.3)
                ax1.set_ylim(0, 1.0)
                ax1.tick_params(axis='both', labelsize='x-small')
                ax1.spines['top'].set_visible(False)
    
    # remove unused plots
    coords = [[x, y] for x in range(0, 9) for y in range(0,9)]
    unused = [c for c in coords if  c not in locs.values()]
    for u in unused:
        fig.delaxes(ax[u[1], u[0]])

    #fig.suptitle(n.metadata['file_info']['fname'].split('.')[0])
    
    # set labels
    fig.text(0.5, 0.9, n.metadata['file_info']['fname'].split('.')[0], ha='center', size='large', weight='semibold')
    fig.text(0.5, 0.08, 'Time (ms)', ha='center', size='large', weight='semibold')
    fig.text(0.08, 0.5, 'Amplitude (mV)', va='center', rotation='vertical', size='large', weight='semibold')
    fig.text(0.93, 0.5, 'Proportion of Spindles Present', va='center', rotation=270, size='large', weight='semibold')
    
    return fig


#### Topoplot methods ####

def plot_tstats_topo(n, col):
    """ Plot topoplots for nrem.spindle_tstats column [col = column name] """
    
    # some parameters
    N = 300             # number of points for interpolation
    xy_center = [4,4]   # center of the plot
    radius = 4          # radius

    # set channel locations
    locs = {'FPz': [4, 8],
             'Fp1': [3, 8],
             'Fp2': [5, 8],
             'AF7': [1, 7],
             'AF8': [7, 7],
             'F7': [0, 6],
             'F8': [8, 6],
             'F3': [2, 6],
             'F4': [6, 6],
             'F1': [3, 6],
             'F2': [5, 6],
             'Fz': [4, 6],
             'FC5': [1, 5],
             'FC6': [7, 5],
             'FC1': [3, 5],
             'FC2': [5, 5],
             'T3': [0, 4],
             'T4': [8, 4],
             'C3': [2, 4],
             'C4': [6, 4],
             'Cz': [4, 4],
             'CP5': [1, 3],
             'CP6': [7, 3],
             'CP1': [3, 3],
             'CP2': [5, 3],
             'CPz': [4, 3],
             'P3': [2, 2],
             'P4': [6, 2],
             'Pz': [4, 2],
             'T5': [0, 2],
             'T6': [8, 2],
             'POz': [4, 1],
             'PO7': [1, 1],
             'PO8': [7, 1],
             'O1': [2, 0],
             'O2': [6, 0],
             'Oz': [4, 0]}

    # make figure
    fig, ax = plt.subplots()
    fig.subplots_adjust(hspace=0.2, wspace=0.2)

    # set data
    data = n.spindle_tstats[col]

    z = []
    x,y = [],[]
    for chan, loc in locs.items():
        x.append(loc[0])
        y.append(loc[1])
        z.append(data[chan])


    xi = np.linspace(-2, 8, N)
    yi = np.linspace(-2, 8, N)
    zi = interpolate.griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')

    # set points > radius to not-a-number. They will not be plotted.
    # the dr/2 makes the edges a bit smoother
    dr = xi[1] - xi[0]
    for i in range(N):
        for j in range(N):
            r = np.sqrt((xi[i] - xy_center[0])**2 + (yi[j] - xy_center[1])**2)
            if (r - dr/2) > radius:
                zi[j,i] = "nan"

    # set aspect = 1 to make it a circle
    ax.set_aspect(1)

    # use different number of levels for the fill and the lines
    # vmin and vmax set the color ranges
#         vmin, vmax = -120, 850
#         v = np.linspace(vmin, vmax, 60, endpoint=True)
#         CS = ax.contourf(xi, yi, zi, v, cmap = plt.cm.jet, zorder = 1, vmin=vmin, vmax=vmax)
    CS = ax.contourf(xi, yi, zi, 60, cmap = plt.cm.jet, zorder = 1)
    # add contour lines
    #ax.contour(xi, yi, zi, 15, colors = "grey", zorder = 2)



    # add the data points
    ax.scatter(x, y, marker = 'o', c = 'black', s = 15, zorder = 3)

    # draw a circle
    # change the linewidth to hide the 
    circle = matplotlib.patches.Circle(xy = xy_center, radius = radius, edgecolor = "k", facecolor = "none")
    ax.add_patch(circle)

    # make the axis invisible 
    for loc, spine in ax.spines.items():
        # use ax.spines.items() in Python 3
        spine.set_linewidth(0)

    #remove the ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Add some body parts. Hide unwanted parts by setting the zorder low
    # add two ears
    circle = matplotlib.patches.Ellipse(xy = [0,radius], width = 0.5, height = 1.0, angle = 0, edgecolor = "k", facecolor = "w", zorder = 0)
    ax.add_patch(circle)
    circle = matplotlib.patches.Ellipse(xy = [radius*2, radius], width = 0.5, height = 1.0, angle = 0, edgecolor = "k", facecolor = "w", zorder = 0)
    ax.add_patch(circle)
    # add a nose
    xy = [[3.25,7.7], [4,8.75],[4.75,7.7]]
    polygon = matplotlib.patches.Polygon(xy = xy, facecolor = "w", edgecolor='black', zorder = 0)
    ax.add_patch(polygon) 

    # set axes limits
    ax.set_xlim(-0.5, radius*2+0.5)
    ax.set_ylim(-0.5, radius*2+1)


    # make a color bar
    cbar = fig.colorbar(CS)
    # set title
    fig.suptitle(col)

    return fig


#### Export Methods ####

def export_spindle_figs(n, export_dir, ext='png', dpi=300, transparent=False, spindle_spectra=False, spins_i=False, spindle_means=True, psd_concat=True):
    """ Produce and export all spindle figures 
    
        Parameters
        ----------
        export_dir: str
            directory to export figures
        ext: str (default: 'png')
            figure ext (e.g. png, eps, jpg)
        dpi: int (default: 300)
            dots per inch
        transparent: bool (default: False)
            render background as transparent
        spindle_spectra: bool (default: True)
            whether to plot spindle spectra subplot figure for each channel
        spins_i: bool (default: True)
            whether to plot indvidiual spindle spectra (spec_spins) figures for each channel
        spindle_means: bool (default: True)
            whether to plot spindle means and tracing overlays
        psd_concat: bool (default: True)
            whether to plot concated spindle spectra headplots and gottselig plots

        
        Returns
        -------
        *To be completed
    
    """
    
    print(f'Spindle figure export directory: {export_dir}\n')  
    
    # make export directory if doesn't exit
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)
    # set base for savename
    fname = n.metadata['file_info']['fname'].split('.')[0]
    
    # make subdirectory for channel spectra
    psd_dir = os.path.join(export_dir, 'channel_spectra')
    if not os.path.exists(psd_dir):
        os.makedirs(psd_dir)
    
    if spindle_spectra or spins_i:                       
        # by channels
        exclude = ['EKG', 'EOG_L', 'EOG_R']
        for chan in n.spindles.keys():
            if chan not in exclude:
                print(f'\nExporting {chan} spectra figures...')
                # make subdirectory for channel
                chan_dir = os.path.join(psd_dir, chan)
                if not os.path.exists(chan_dir):
                        os.makedirs(chan_dir)

                if spindle_spectra:
                    print('\tExporting spindle power subplots...')
                    # individual spindle spectra (accepted)
                    filename = f'{fname}_SpindleSpectra_Accepted.{ext}'
                    savename = os.path.join(chan_dir, filename)
                    fig = plot_spindlepower_chan_i(n, chan)
                    try:
                        fig.savefig(savename, dpi=dpi, transparent=transparent)
                    except AttributeError:
                        pass
                    else:
                        plt.close(fig)                  
                    # individual spindle spectra (rejected)
                    filename = f'{fname}_SpindleSpectra_Rejected.{ext}'
                    savename = os.path.join(chan_dir, filename)
                    fig = plot_spindlepower_chan_i(n, chan, spin_type='rejects')
                    try:
                        fig.savefig(savename, dpi=dpi, transparent=transparent)
                    except AttributeError:
                        pass
                    else:
                        plt.close(fig)

                if spins_i:
                    print('\tExporting spec_spins figures...')
                    # make subdirectory for individual spins
                    spin_dir = os.path.join(chan_dir, 'individual_spectra')
                    if not os.path.exists(spin_dir):
                            os.makedirs(spin_dir)              
                    # individual [accepted] spindles
                    for spin in n.spindle_psd_i[chan].keys():
                        filename = f'{fname}_SpinPSD_{chan}_Accepted_{spin}.{ext}'
                        savename = os.path.join(spin_dir, filename)
                        fig = spec_spins(n, chan, spin)
                        fig.savefig(savename, dpi=dpi, transparent=transparent)
                        plt.close(fig)
                    # individual [rejected] spindles
                    for spin in n.spindle_psd_i_rejects[chan].keys():
                        filename = f'{fname}_SpinPSD_{chan}_Rejected_{spin}.{ext}'
                        savename = os.path.join(spin_dir, filename)
                        fig = spec_spins(n, chan, spin)
                        fig.savefig(savename, dpi=dpi, transparent=transparent)
                        plt.close(fig)
    
    
    ## Overall
    if spindle_means:
        print('Exporting spindle overlays & means...')
        # overlaid spindle tracings (raw & filtered)
        filename = f'{fname}_spindles_raw.{ext}'
        savename = os.path.join(export_dir, filename)
        fig = plot_spins(n)
        fig.savefig(savename, dpi=dpi, transparent=transparent)
        plt.close(fig)

        filename = f'{fname}_spindles_filt.{ext}'
        savename = os.path.join(export_dir, filename)
        fig = plot_spins(n, datatype='spfilt')
        fig.savefig(savename, dpi=dpi, transparent=transparent)
        plt.close(fig)

        # spindle means (raw & filtered)
        filename = f'{fname}_SpindleMeans_raw.{ext}'
        savename = os.path.join(export_dir, filename)
        fig = plot_spin_means(n)
        fig.savefig(savename, dpi=dpi, transparent=transparent)
        plt.close(fig)

        filename = f'{fname}_SpindleMeans_filt.{ext}'
        savename = os.path.join(export_dir, filename)
        fig = plot_spin_means(n, datatype ='spfilt')
        fig.savefig(savename, dpi=dpi, transparent=transparent)
        plt.close(fig)
    
    if psd_concat:
        print('Exporting concatenated spindle power figures...')
        # concatenated spectra headplot
        filename = f'{fname}_SpectraConcat.{ext}'
        savename = os.path.join(export_dir, filename)
        fig = plot_spindlepower_headplot(n)
        fig.savefig(savename, dpi=dpi, transparent=transparent)
        plt.close(fig)

        # gottselig norm headplots (calcs & subtracted)
        filename = f'{fname}_SpectraConcat_NormFit.{ext}'
        savename = os.path.join(export_dir, filename)
        fig = plot_gottselig_headplot(n)
        fig.savefig(savename, dpi=dpi, transparent=transparent)
        plt.close(fig)

        filename = f'{fname}_SpectraConcat_NormPwr.{ext}'
        savename = os.path.join(export_dir, filename)
        fig = plot_gottselig_headplot(n, datatype='normed_pwr')
        fig.savefig(savename, dpi=dpi, transparent=transparent)
        plt.close(fig)
    
    print('\nDone.')


### Macroarchitecture methods ###

def plot_sleepcycles(d, plt_stages='all', logscale=True, normx=True):
    """ 
        Plot cycle length for each sleep stage vs cycle # 
        -> This should show if there are trends in cycle length
        based on number of times that stage has been cycled through
        
        Params
        ------
        d: transitions.HypStats or ioeeg.Dataset object
        plt_stages: str or list of string (default: 'all')
            stages to plot
        logscale: bool (default:True)
            plot the y-axis on a log scale
        normx: bool (default: True)
            normalize x-axis according to total number of cycles for that stage
    """

    if plt_stages == 'all':
        stages = ['awake', 'rem', 's1', 's2', 'ads', 'sws']
    elif type(plt_stages) == str:
        stages = [plt_stages]
    elif type(plt_stages) == list:
        stages = plt_stages

    fig, ax = plt.subplots()

    for key in d.hyp_stats.keys():
        if key in stages:
            if d.hyp_stats[key]['n_cycles'] > 0:
                if normx == True:
                    x = [int(x)/d.hyp_stats[key]['n_cycles'] for x in d.hyp_stats[key]['cycle_lengths'].keys()]
                    xlabel = 'Normalized Cycle'
                else:
                    x = d.hyp_stats[key]['cycle_lengths'].keys()
                    xlabel = 'Cycle'
                if logscale == True:
                    y = [math.log10(y) for y in d.hyp_stats[key]['cycle_lengths'].values()]
                    ylabel = 'Cycle length [log(seconds)]'
                else:
                    y = d.hyp_stats[key]['cycle_lengths'].values()
                    ylabel = 'Cycle length (seconds)'
                ax.plot(x, y, label = key)
    ax.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    return fig

def cycles_boxplot(d, yscale='min'):
    """ boxplot of cycle lengths 
    
        Params
        ------
        yscale: str (default: 'min')
            y scaling (options: 'min', 'sec')
        
        Note: see style example here http://blog.bharatbhole.com/creating-boxplots-with-matplotlib/
    """
    ylist = []
    xticklabels = []
    for key in d.hyp_stats.keys():
        if type(d.hyp_stats[key]) == dict and 'n_cycles' in d.hyp_stats[key].keys():
            if d.hyp_stats[key]['n_cycles'] > 0:
                xticklabels.append(key + '\n(n='+ str(d.hyp_stats[key]['n_cycles']) + ')')
                if yscale == 'sec':
                    ylist.append(list(d.hyp_stats[key]['cycle_lengths'].values()))
                    ylabel = 'Cycle Length (sec)'
                elif yscale == 'min':
                    ylist.append([y/60. for y in d.hyp_stats[key]['cycle_lengths'].values()])
                    ylabel = 'Cycle Length (min)'
                
    fig, ax = plt.subplots()
    
    bp = ax.boxplot(ylist, notch=False, patch_artist=True)
    
    # change box outline & fill
    for box in bp['boxes']:
        box.set(color='lightgray')
        box.set(facecolor='lightgray')
    # change median color
    for median in bp['medians']:
        median.set(color='grey', lw=2)
    # change whisker color
    for whisker in bp['whiskers']:
        whisker.set(color='grey', lw=2)
    # change cap color
    for cap in bp['caps']:
        cap.set(color='grey', lw = 2)
    # change the style of fliers and their fill
    for flier in bp['fliers']:
        flier.set(markerfacecolor='grey', marker='o', markeredgecolor='white', markersize=8, alpha=0.5, lw=.01)
    
    ax.set_xticklabels(xticklabels)
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)
    
    plt.xlabel('\nSleep Stage')
    plt.ylabel(ylabel)
    plt.suptitle(d.in_num + ' (' + d.metadata['start_date'] + ')')
    
    return fig

def cycles_scatterbox(d, yscale='min'):
    """ boxplot of cycle lengths w/ colored scatterplot of datapoints
    
        Params
        ------
        yscale: str (default: 'min')
            y scaling (options: 'min', 'sec')
        
        Note: see style example here http://blog.bharatbhole.com/creating-boxplots-with-matplotlib/
    """

    ylist = []
    xticklabels = []
    colors = []
    for key in d.hyp_stats.keys():
        if type(d.hyp_stats[key]) == dict and 'n_cycles' in d.hyp_stats[key].keys():
            if d.hyp_stats[key]['n_cycles'] > 0:
                # set xtick labels
                xticklabels.append(key + '\n(n='+ str(d.hyp_stats[key]['n_cycles']) + ')')
                # set y values
                if yscale == 'sec':
                    ylist.append(list(d.hyp_stats[key]['cycle_lengths'].values()))
                    ylabel = 'Cycle Length (sec)'
                elif yscale == 'min':
                    ylist.append([y/60. for y in d.hyp_stats[key]['cycle_lengths'].values()])
                    ylabel = 'Cycle Length (min)'
                # set colors
                if key == 'awake':
                    colors.append('darkgrey')
                elif key == 's1':
                    colors.append('orange')
                elif key == 's2':
                    colors.append('lime')
                elif key == 'ads':
                    colors.append('blue')
                elif key == 'sws':
                    colors.append('purple')

    fig, ax = plt.subplots()
    bp = ax.boxplot(ylist, notch=False, patch_artist=True)
    
    # change box outline & fill
    for box in bp['boxes']:
        box.set(color='lightgray', alpha=1)
    # change median color
    for median in bp['medians']:
        median.set(color='grey', lw=2)
    # change whisker color
    for whisker in bp['whiskers']:
        whisker.set(color='grey', lw=2)
    # change cap color
    for cap in bp['caps']:
        cap.set(color='grey', lw = 2)
    # change the style of fliers and their fill
    for flier in bp['fliers']:
        flier.set(markerfacecolor='grey', marker=None, markeredgecolor='white', markersize=8, alpha=0.5, lw=.01)
        
    # add scatterplot for datapoints
    for i in range(len(ylist)):
        y = ylist[i]
        # create jitter by randomly distributing x vals
        x = np.random.normal(1+i, 0.1, size=len(y))
        ax.plot(x, y, 'r.', markerfacecolor=colors[i], markeredgecolor=colors[i], markersize=12, 
                markeredgewidth=0.8, alpha=0.5, zorder=3) #zorder sets scatter on top

    ax.set_xticklabels(xticklabels)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.grid(axis='both', linestyle=':', linewidth=1)
    plt.xlabel('\nSleep Stage')
    plt.ylabel(ylabel)
    plt.suptitle(d.in_num + ' (' + d.start_date + ')')
    
    return fig
    

def plot_hyp(d):
    """ plot hypnogram for ioeeg.Dataset instance """
    fig, ax = plt.subplots(figsize = (30,5))
    
    ax.plot(d.hyp, color='lightseagreen', lw=2)
    ax = plt.gca()
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.set_yticklabels(['Awake', 'REM', 'Stage 1', 'Stage 2', 'Alpha-Delta\nSleep     ', 'SWS'])

    fmt = mdates.DateFormatter('%I:%M%p')
    ax.xaxis.set_major_formatter(fmt)
    ax.margins(x=.01)
    
    plt.xlabel('Time')
    plt.ylabel('Sleep Stage')
    plt.suptitle(d.metadata['in_num'] + ' (' + d.metadata['start_date'] + ')')
    
    return fig


def plot_transitions(h, savefig=False, savedir=None):
    """ plot sleep stage transition diagram. 
        
        NOTE: This is for preliminary visualization. For high quality
        figures use plot_transitions.R
            *for 3d plots use R code, or see https://stackoverflow.com/questions/16907564/how-can-i-plot-a-3d-graph-in-python-with-igraph
            or this http://bommaritollc.com/2011/02/21/plotting-3d-graphs-with-python-igraph-and-cairo-cn220-example/
    """

    # set data
    m = np.matrix(h.transition_perc)/2
    net=ig.Graph.Weighted_Adjacency(m.tolist(), attr='width', loops=False)

    # set node params [see https://www.cs.rhul.ac.uk/home/tamas/development/igraph/tutorial/tutorial.html]
    node_size = np.array((h.epoch_counts['Awake'], h.epoch_counts['REM'], h.epoch_counts['Stage 1'], h.epoch_counts['Stage 2'], h.epoch_counts['Alpha-Delta'], h.epoch_counts['SWS']))
    net.vs['size'] = node_size/4
    net.vs['shape'] = 'circle'
    net.vs['color'] = ['lightgrey', 'red', 'orange', 'lime', 'blue', 'purple']

    label_dist = [0 if x>200 else 1.3 for x in node_size]
    net.vs['label_dist'] = label_dist
    net.vs['label_angle'] = [np.pi/2, np.pi, np.pi, 0, np.pi*3/2, 0]
    net.vs['label_size'] = 12
    net.vs['label'] = ['Awake', 'REM', 'Stage 1', 'Stage 2', 'Alpha-Delta', 'SWS']

    # set edge params
    net.es['curved'] = True # this can be set to a float also (ex. 0.3)
    net.es['arrow_size'] = 0.8
    net.es['arrow_width'] = 1.5
    net.es['color'] = 'darkgrey'

    # set layout
    l = [(0, 0), (-2, 2), (-1, 1), (1, 1), (0, 2), (2, 2)]
    layout = ig.Layout(l)

    # set visual style
    visual_style = {}
    visual_style['bbox'] = (500, 500)
    visual_style['margin'] = 110

    fig = ig.plot(net, layout=layout, **visual_style)

    if savefig == True:
        savename = h.in_num + '_TransitionPlot_' + h.start_date + '.png'
        if savedir is None:
            savedir = os.getcwd()
        save = os.path.join(savedir, savename)
        fig.save(save)

    return fig


### EKG Methods ###

def plotEKG(ekg, rpeaks=False):
    """ plot EKG class instance """
    fig = plt.figure(figsize = [18, 6])
    plt.plot(ekg.data)

    if rpeaks == True:
        plt.scatter(ekg.rpeaks.index, ekg.rpeaks.values, color='red')

    return fig

def plot_ekgibi(ekg, rpeaks=True, ibi=True):
    """ plot EKG class instance """
    # set number of panels
    if ibi == True:
        plots = ['ekg', 'ibi']
        data = [ekg.data, ekg.rpeaks_df['ibi_ms']]
        
    else:
        plots = ['ekg']
        data = [ekg.data]

    fig, axs = plt.subplots(len(plots), 1, sharex=True, figsize = [9.5, 6])
    
    for dat, ax, plot in zip(data, axs, plots):
        if plot == 'ekg' and rpeaks == True:
            ax.plot(dat)
            ax.scatter(ekg.rpeaks.index, ekg.rpeaks.values, color='red')
        elif plot == 'ibi':
            ax.plot(dat, color='grey', marker='.', markersize=8, markerfacecolor=(0, 0, 0, 0.8), markeredgecolor='None')
        ax.margins(x=0)
        # show microseconds for mouse-over
        ax.format_xdata = lambda d: mdates.num2date(d).strftime('%H:%M:%S.%f')[:-3]


def plotHTI(ekg):
    """ plot histogram of HRV Triangular Index (bin size 1/128sec) 
        Note: 1/128 bin size is used for consistency with literature """
    fig = plt.figure()
    # may have to remove histtype or change to 'step' if plotting multiple overlaid datasets
    plt.hist(ekg.rr, bins=np.arange(min(ekg.rr), max(ekg.rr) + 7.8125, 7.8125), histtype='stepfilled')
    return fig


def plotPS(ekg, method, dB=False, bands=False):
    """ Plot power spectrum """
    
     # set title
    title = ekg.metadata['file_info']['in_num'] + ' ' + ekg.metadata['file_info']['start_date'] + '\n' + ekg.metadata['file_info']['sleep_stage'] + ' ' + ekg.metadata['file_info']['cycle'] + ' ' + ekg.metadata['file_info']['epoch']

    # set data to plot
    if method == 'mt':
        #title = 'Multitaper'
        psd = ekg.psd_mt
    elif method == 'welch':
        #title = 'Welch'
        psd = ekg.psd_welch
    
    # transform units
    if dB == True:
        pwr = 10 * np.log10(psd['pwr'])
        ylabel = 'Power spectral density (dB)'
    else:
        pwr = psd['pwr']/1e6 # convert to seconds
        ylabel = 'Power spectral density (s^2/Hz)'
    
    fig, ax = plt.subplots()
    
    # plot just spectrum
    if bands == False:
        ax.plot(psd['freqs'], pwr)
    
    # or plot spectrum colored by frequency band
    elif bands == True:
        # use matplotlib.patches.Patch to make objects for legend w/ data
        ax.plot(psd['freqs'], pwr, color='black')
        
        yline = SG.LineString(list(zip(psd['freqs'],pwr)))
        #ax.plot(yline, color='black')
        
        colors = [None, 'yellow', 'orange', 'tomato']
        for (key, value), color in zip(ekg.psd_fband_vals.items(), colors):
            if value['idx'] is not None:
                # get intercepts & plot vertical lines for bands
                xrange = [float(x) for x in ekg.freq_stats[key]['freq_range'][1:-1].split(",")] 
                xline = SG.LineString([(xrange[1], min(pwr)), (xrange[1], max(pwr))])
                coords = np.array(xline.intersection(yline))            
                ax.vlines(coords[0], 0, coords[1], colors='black', linestyles='dotted')
                
                # fill spectra by band
                ax.fill_between(psd['freqs'], pwr, where = [xrange[0] <= x <=xrange[1] for x in psd['freqs']], 
                                facecolor=color, alpha=.6)    
        
    ax.set_xlim(0, 0.4)
    ax.margins(y=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.xlabel('Frequency (Hz)')
    plt.ylabel(ylabel)
    plt.suptitle(title)

    return fig



