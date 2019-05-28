""" plotting functions for Dataset objects 
    
    To Do:
        Edit hyp_stats plots to take transitions.HypStats object instead of ioeeg.Dataset object
"""

import itertools
import math
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np 
import os
import pandas as pd
import shapely.geometry as SG


def plotEEG(d, raw=True, filtered=False, spindles=False, spindle_rejects=False):
    """ plot multichannel EEG w/ option for double panel raw & filtered 
    
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

    # set channels based on dataset
    channels = [col[0] for col in data.columns]

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
                sp_rejs_TS = [pd.Timestamp(x) for x in sp_rej_eventsflat[i]]
                spin_rejects = pd.Series(index=norm_dat.index)
                spin_rejects[sp_rejs_TS] = norm_dat[sp_rejs_TS]
                ax.plot(spin_rejects, color='red', alpha=0.5)
        
        ax.set_title(t)
        ax.set_yticks(list(np.arange(0.5, -(len(channels)-1), -1)))
        ax.set_yticklabels(channels)
        ax.margins(x=0) # remove white space margins between data and y axis
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # set overall parameters
    fig.suptitle(d.in_num)
    plt.xlabel('Time')

    return fig, axs



def plotEEG_singlechan(d, chan, raw=True, filtered=False, rms=False, thresholds=False, spindles=False, spindle_rejects=False):
    """ plot single channel EEG. Options for multipaneled calculations
    
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
    if filtered == True:    
        #filtd = d.spindle_calcs.loc(axis=1)[c, 'Filtered']
        data.append(filtd_data)
        dtype.append('filtd')
    if rms == True:
        data.append(filtd_data)
        dtype.append('filtd+rms')
    if spindles == True or spindle_rejects == True:
        data.append(filtd_data)
        if spindles == True and spindle_rejects == False:
            dtype.append('filtd+spin')
        elif spindles == False and spindle_rejects == True:
            dtype.append('filtd+rej')
        elif spindles == True and spindle_rejects == True:
            dtype.append('filtd+spin+rej')

    
    # plot data    
    fig, axs = plt.subplots(len(data), 1, sharex=True, figsize=(18,6), squeeze=False)
    fig.subplots_adjust(hspace=.1, top=.9, bottom=.1, left=.05, right=.95)
    
    for dat, ax, dt in zip(data, axs.flatten(), dtype):
        # plot EEG
        ax.plot(dat, linewidth=.5, color='C0')
            
        # plot filtered EEG w/ rms & thresholds
        if dt == 'filtd+rms':
            ax.plot(d.spRMS[c], label='RMS')
            ax.plot(d.spRMSmavg[c], label='RMS moving average')
        if dt == 'filtd+rms' and thresholds == True:
            ax.axhline(d.spThresholds[c].loc['Low Threshold'], linestyle='dashed', color='grey', label = 'Mean RMS + 1 SD')
            ax.axhline(d.spThresholds[c].loc['High Threshold'], linestyle='dashed', color='grey', label = 'Mean RMS + 1.5 SD')
        
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
            sp_rej_valuesflat = []
            sp_rej_eventsflat = []
            for n in range(len(d.spindle_rejects[c])):
                for m in range(len(d.spindle_rejects[c][n])):
                    sp_rej_valuesflat.append(dat[d.spindle_rejects[c][n][m]])
                    sp_rej_eventsflat.append(d.spindle_rejects[c][n][m])
            sp_rej_events_TS = [pd.Timestamp(x) for x in sp_rej_eventsflat]
            spin_rejects = pd.Series(index=dat.index)
            spin_rejects[sp_rej_events_TS] = dat[sp_rej_events_TS]
            ax.plot(spin_rejects, color='red', alpha=0.5, label='Rejected Detection')
            
        ax.legend(loc='lower left')
        #ax.set_title(t)
        #ax.set_yticks(list(np.arange(0.5, -(len(chan)-1), -1)))
        #ax.set_yticklabels(chan)
        ax.margins(x=0) # remove white space margins between data and y axis
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
       
    # set overall parameters
    fig.suptitle(d.in_num)
    plt.xlabel('Time')
    
    return fig

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
    plt.suptitle(d.in_num + ' (' + d.start_date + ')')
    
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
    plt.suptitle(d.in_num + ' (' + d.start_date + ')')
    
    return fig



### EKG Methods ###

def plotEKG(ekg, rpeaks=False):
    """ plot EKG class instance """
    fig = plt.figure(figsize = [18, 6])
    plt.plot(ekg.data)

    if rpeaks == True:
        plt.scatter(ekg.rpeaks.index, ekg.rpeaks.values, color='red')

    return fig

def plotHTI(ekg):
    """ plot histogram of HRV Triangular Index (bin size 1/128sec) 
        Note: 1/128 bin size is used for consistency with literature """
    fig = plt.figure()
    # may have to remove histtype or change to 'step' if plotting multiple overlaid datasets
    plt.hist(ekg.rr_int, bins=np.arange(min(ekg.rr_int), max(ekg.rr_int) + 7.8125, 7.8125), histtype='stepfilled')
    return fig


def plotPS(ekg, method, dB=False, bands=False):
    """ Plot power spectrum """
    
    # set data to plot
    if method == 'mt':
        title = 'Multitaper'
        psd = ekg.psd_mt
    elif method == 'welch':
        title = 'Welch'
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
        ax.plot(psd['freqs'], pwr, color='grey')
        
        yline = SG.LineString(list(zip(psd['freqs'],pwr)))
        #ax.plot(yline, color='black')
        
        colors = [None, 'blue', 'purple', 'green']
        for (key, value), color in zip(ekg.psd_fband_vals.items(), colors):
            if value['idx'] is not None:
                # get intercepts & plot vertical lines for bands
                xrange = [float(x) for x in ekg.freq_stats[key]['freq_range'][1:-1].split(",")] 
                xline = SG.LineString([(xrange[1], min(pwr)), (xrange[1], max(pwr))])
                coords = np.array(xline.intersection(yline))            
                ax.vlines(coords[0], 0, coords[1], colors='grey', linestyles='dotted')
                
                # fill spectra by band
                ax.fill_between(psd['freqs'], pwr, where = [xrange[0] <= x <=xrange[1] for x in psd['freqs']], 
                                facecolor=color, alpha=.09)    
        
    ax.set_xlim(0, 0.4)
    ax.margins(y=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.xlabel('Frequency (Hz)')
    plt.ylabel(ylabel)
    plt.suptitle(title)

    return fig