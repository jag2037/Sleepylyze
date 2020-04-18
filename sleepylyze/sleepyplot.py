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
from scipy.signal import find_peaks
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
            ax.plot(d.spRMS[c], label='RMS', color='green')
            ax.plot(d.spRMSmavg[c], label='RMS moving average', color='orange')
        if dt == 'filtd+rms' and thresholds == True:
            ax.axhline(d.spThresholds[c].loc['Low Threshold'], linestyle='solid', color='grey', label = 'Mean RMS + 1 SD')
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

def vizeeg(d, raw=True, filtered=False, spindles=False, spindle_rejects=False, slider=True, win_width=15):
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
        # time-domain rejects
        sp_rej_t_eventsflat = [list(itertools.chain.from_iterable(d.spindle_rejects_t[i])) for i in d.spindle_rejects_t.keys()]
        # frequency domain rejects
        sp_rej_f_eventsflat = [list(itertools.chain.from_iterable(d.spindle_rejects_f[i])) for i in d.spindle_rejects_f.keys()]  

    # set channels for plotting
    channels = [x[0] for x in d.data.columns if x[0] not in ['EKG', 'EOG_L', 'EOG_R']]
    
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

    # set figure & subplot params
    ncols = int(np.sqrt(len(psd_dict[chan])))
    nrows = len(psd_dict[chan])//ncols + (len(psd_dict[chan]) % ncols > 0) 
    fig, axs = plt.subplots(nrows = nrows, ncols = ncols, figsize=(16, 12))
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
        ax.tick_params(axis='both', labelsize='x-small', labelleft=False)
    
    # delete empty subplots --> this can probably be combined with previous loop
    for i, ax in enumerate(axs_flat):
        if i >= len(psd_dict[chan]):
            fig.delaxes(ax)
    
    # set figure params   
    fig.tight_layout(pad=1, rect=[0, 0, 1, 0.93])
    fig.text(0.5, 0, 'Frequency (Hz)', ha='center', size='large', weight='semibold')
    fig.text(0, 0.5, ylabel, va='center', rotation='vertical', size='large', weight='semibold')
    fig.suptitle(n.metadata['file_info']['fname'].split('.')[0] + f'\n\nSpindle Power {chan}', size='large', weight='semibold')

    return fig


def spec_peaks(n, chan, x, labels=True):
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

        Returns
        -------
        matplotlib.Figure
    """
    spin_range = n.metadata['spindle_analysis']['spin_range']
    prune_range = n.metadata['spindle_analysis']['prune_range']

    # set data to plot
    try:
        psd = n.spindle_psd_i[chan][x]
        zpad_spin = n.spindles_zpad[chan][x]
        spin_perc = n.spindle_multitaper_calcs[chan][f'perc_{prune_range[0]}-{prune_range[1]}Hzpwr_in_spin_range'].loc[x]
        status = 'accepted'
    except KeyError:
        psd = n.spindle_psd_i_rejects[chan][x]
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
    axs[0].plot(zpad_spin, alpha=1, lw=0.8, label='Raw Signal')
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

### Spindle Methods ###

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
    ncols = 6
    nrows = len(eeg_chans)//ncols + (len(eeg_chans) % ncols > 0) 
    fig, axs = plt.subplots(nrows = nrows, ncols = ncols, sharex=True, figsize=(18, 10))
    fig.subplots_adjust(hspace=0.5)
    
    for chan, ax in zip(n.spindles.keys(), axs.flatten()):
        if chan not in exclude:
            # if buffer:
            #     data = n.spindle_buffer_means
            #     ax.plot(data[(chan, 'mean')], alpha=1, color=buff_color, label='Overall Average', lw=1)
            #     ax.fill_between(data.index, data[(chan, 'mean')] - data[(chan, err)], data[(chan, 'mean')] + data[(chan, err)], 
            #                     color=buff_color, alpha=0.2)
            if spins:
                data = n.spindle_means[datatype]
                ax.plot(data[(chan, 'mean')], alpha=1, color=spin_color, label='Spindle Average', lw=1)
                ax.fill_between(data.index, data[(chan, 'mean')] - data[(chan, err)], data[(chan, 'mean')] + data[(chan, err)], 
                                color=spin_color, alpha=0.2)
                if count:
                    ax1 = ax.twinx()
                    ax1.plot(data[chan, 'count'], color=count_color, alpha=0.3)
                    ax1.fill_between(data.index, 0, data[(chan, 'count')], color=count_color, alpha=0.3)
                    max_count = len(n.spindles[chan])
                    ticks = np.linspace(0, max_count, num=5, dtype=int)
                    ax1.set_yticks(ticks=ticks)
                    ax1.set_yticklabels(labels=ticks, color=count_color)
                    ax1.tick_params(axis='y', labelsize=8) #color=count_color)

            # set subplot params
            ax.set_xlim([-1800, 1800])
            ax.set_title(chan, fontsize='medium')
            ax.tick_params(axis='both', which='both', labelsize=8)

    # delete empty subplots --> this can probably be combined with previous loop
    for i, ax in enumerate(axs.flatten()):
        if i >= len(eeg_chans):
            fig.delaxes(ax)

    # set figure params
    #fig.legend()   
    fig.tight_layout(pad=1, rect=[0, 0, 1, 0.95])
    fig.text(0.5, 0, 'Time (ms)', ha='center', size='large')
    fig.text(0, 0.5, 'Amplitude (mV)', va='center', rotation='vertical', color=spin_color, size='large')
    fig.text(1, 0.5, 'Spindle Count', va='center', rotation=270, color=count_color, size='large')
    fig.suptitle(n.metadata['file_info']['fname'].split('.')[0] + f'\nSpindle Averages ({datatype})')

    return fig


def plot_spindlepower_chan(n, chan, dB=True):
    """ Plot spindle power spectrum for a single channel """

    # transform units
    if dB == True:
        pwr = 10 * np.log10(n.spindle_psd[chan].values)
        ylabel = 'Power (dB)'
    else:
        pwr = n.spindle_psd[chan].values
        ylabel = 'Power (mV^2/Hz)'
    
    fig, ax = plt.subplots()
    
    # plot just spectrum
    ax.plot(n.spindle_psd[chan].index, pwr, color='black', alpha=0.9, linewidth=0.8)
    ax.axvspan(9, 16, color='lavender', alpha=0.8)
        
    ax.set_xlim(0, 25)
    ax.margins(y=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.xlabel('Frequency (Hz)')
    plt.ylabel(ylabel)
    plt.title((n.metadata['file_info']['fname'].split('.')[0] + '\n\n' + chan + ' Spindle Power'), size='medium', weight='semibold')

    return fig


def plot_spindlepower(n, dB=True):
    """ Plot spindle power spectrum for all channels """
    
    exclude = ['EKG', 'EOG_L', 'EOG_R']
    eeg_chans = [x for x in n.spindle_psd.keys() if x not in exclude]
    ncols = 6
    nrows = len(eeg_chans)//ncols + (len(eeg_chans) % ncols > 0) 
    fig, axs = plt.subplots(nrows = nrows, ncols = ncols, figsize=(16, 12))
    fig.subplots_adjust(hspace=0.8, wspace=0.5)
    
    for chan, ax in zip(eeg_chans, axs.flatten()):    
        # transform units
        if dB == True:
            pwr = 10 * np.log10(n.spindle_psd[chan].values)
            ylabel = 'Power (dB)'
        else:
            pwr = n.spindle_psd[chan].values
            ylabel = 'Power (mV^2/Hz)'

        # plot spectrum
        ax.plot(n.spindle_psd[chan].index, pwr, color='black', alpha=0.9, linewidth=0.8)
        # highlight spindle range. aquamarine or lavender works here too
        ax.axvspan(9, 16, color='lavender', alpha=0.8)

        # set subplot params
        ax.set_xlim(0, 25)
        ax.margins(y=0)
        ax.set_xticks([5, 10, 15, 20])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title(chan, size='medium', weight='bold')
    
    # delete empty subplots --> this can probably be combined with previous loop
    for i, ax in enumerate(axs.flatten()):
        if i >= len(eeg_chans):
            fig.delaxes(ax)
    
    # set figure params   
    fig.tight_layout(pad=1, rect=[0, 0, 1, 0.93])
    fig.text(0.5, 0, 'Frequency (Hz)', ha='center', size='large', weight='semibold')
    fig.text(0, 0.5, ylabel, va='center', rotation='vertical', size='large', weight='semibold')
    fig.suptitle(n.metadata['file_info']['fname'].split('.')[0] + '\n\nSpindle Power', size='large', weight='semibold')

    return fig

def plot_spindlepower_headplot(n, dB=True):
    """ 
        Headplot of spindle power spectrum for all channels 
        NOTE: only for FS128 (12-11-19)
    """
    
    # set channel locations
    locs = {'FPz': [4, 0],
             'Fp1': [3, 0],
             'FP2': [5, 0],
             'AF7': [1, 1],
             'AF8': [7, 1],
             'F7': [0, 2],
             'F8': [8, 2],
             'F3': [2, 2],
             'F4': [6, 2],
             'F1': [3, 2],
             'F2': [5, 2],
             'Fz': [4, 2],
             'FC5': [1, 3],
             'FC6': [7, 3],
             'FC1': [3, 3],
             'FC2': [5, 3],
             'T3': [0, 4],
             'T4': [8, 4],
             'C3': [2, 4],
             'C4': [6, 4],
             'Cz': [4, 4],
             'CP5': [1, 5],
             'CP6': [7, 5],
             'CP1': [3, 5],
             'CP2': [5, 5],
             'CPz': [4, 5],
             'P3': [2, 6],
             'P4': [6, 6],
             'Pz': [4, 6],
             'T5': [0, 6],
             'T6': [8, 6],
             'POz': [4, 7],
             'PO7': [1, 7],
             'PO8': [7, 7],
             'O1': [2, 8],
             'O2': [6, 8],
             'Oz': [4, 8]}
    
    fig, ax = plt.subplots(9,9, figsize=(12, 12))
    plt.subplots_adjust(hspace=0.3, wspace=0.3) # use this or tight_layout

    for chan in locs.keys():    
        # transform units
        if dB == True:
            pwr = 10 * np.log10(n.spindle_psd[chan].values)
            ylabel = 'Power (dB)'
        else:
            pwr = n.spindle_psd[chan].values
            ylabel = 'Power (mV^2/Hz)'

        # plot spectrum
        #ax = plt.subplot()
        ax[locs[chan][1], locs[chan][0]].plot(n.spindle_psd[chan].index, pwr, color='black', alpha=0.9, linewidth=0.8)
        # highlight spindle range. aquamarine or lavender works here too
        ax[locs[chan][1], locs[chan][0]].axvspan(9, 16, color='lavender', alpha=0.8)

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


def plot_gottselig(n, datatype='calcs'):
    """ plot gottselig normalization for all channels 
    
        Parameters
        ----------
        datatype: str (default: 'calcs')
            which data to plot [options: 'calcs', 'normed_pwr']
    """
    
    exclude = ['EKG', 'EOG_L', 'EOG_R']
    eeg_chans = [x for x in n.spindle_psd.keys() if x not in exclude]
    ncols = 6
    nrows = len(eeg_chans)//ncols + (len(eeg_chans) % ncols > 0) 
    fig, axs = plt.subplots(nrows = nrows, ncols = ncols, figsize=(20, 15))
    fig.subplots_adjust(hspace=0.8, wspace=0.5)
    
    for chan, ax in zip(eeg_chans, axs.flatten()):
        data = n.spindle_psd[chan]
        data_normed = n.spindle_psd_norm[chan]
        
        if datatype == 'calcs':
            # first plot
            ax.scatter(data_normed['values_to_fit'].index, data_normed['values_to_fit'].values, alpha=0.8, color='mediumslateblue', linewidths=0, marker='s', label='Normalization Range')
            ax.plot(data.index, 10*np.log10(data.values), color='black', label = 'Power Spectrum')
            ax.plot(data_normed['exp_fit_line'], color='mediumblue', label = 'Exponential fit')
            ax.set_title(chan)
        
        elif datatype == 'normed_pwr':
            # second plot
            ax.plot(data_normed['normed_pwr'], color='black', label='Normalized power')
            ax.axvspan(9, 16, color='lavender', alpha=0.8, label = 'Spindle Range')
            ax.set_title(chan)
        
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

def plot_gottselig_headplot(n, datatype='calcs'):
    """ plot gottselig normalization headplot for all channels 
        NOTE: only for FS128 (12-11-19)

        To Do: Change plots for consistent y-axis
    
        Parameters
        ----------
        datatype: str (default: 'calcs')
            which data to plot [options: 'calcs', 'normed_pwr']
    """
    
    # set channel locations
    locs = {'FPz': [4, 0],
             'Fp1': [3, 0],
             'FP2': [5, 0],
             'AF7': [1, 1],
             'AF8': [7, 1],
             'F7': [0, 2],
             'F8': [8, 2],
             'F3': [2, 2],
             'F4': [6, 2],
             'F1': [3, 2],
             'F2': [5, 2],
             'Fz': [4, 2],
             'FC5': [1, 3],
             'FC6': [7, 3],
             'FC1': [3, 3],
             'FC2': [5, 3],
             'T3': [0, 4],
             'T4': [8, 4],
             'C3': [2, 4],
             'C4': [6, 4],
             'Cz': [4, 4],
             'CP5': [1, 5],
             'CP6': [7, 5],
             'CP1': [3, 5],
             'CP2': [5, 5],
             'CPz': [4, 5],
             'P3': [2, 6],
             'P4': [6, 6],
             'Pz': [4, 6],
             'T5': [0, 6],
             'T6': [8, 6],
             'POz': [4, 7],
             'PO7': [1, 7],
             'PO8': [7, 7],
             'O1': [2, 8],
             'O2': [6, 8],
             'Oz': [4, 8]}
    
    fig, ax = plt.subplots(9,9, figsize=(12, 12))
    plt.subplots_adjust(hspace=0.3, wspace=0.3) # use this or tight_layout
    
    for chan in locs.keys():    
        data = n.spindle_psd[chan]
        data_normed = n.spindle_psd_norm[chan]
        
        if datatype == 'calcs':
            # first plot
            ax[locs[chan][1], locs[chan][0]].scatter(data_normed['values_to_fit'].index, data_normed['values_to_fit'].values, alpha=0.8, color='mediumslateblue', linewidths=0, marker='s', label='Normalization Range')
            ax[locs[chan][1], locs[chan][0]].plot(data.index, 10*np.log10(data.values), color='black', label = 'Power Spectrum')
            ax[locs[chan][1], locs[chan][0]].plot(data_normed['exp_fit_line'], color='mediumblue', label = 'Exponential fit')
            ax[locs[chan][1], locs[chan][0]].set_title(chan)
        
        elif datatype == 'normed_pwr':
            # second plot
            ax[locs[chan][1], locs[chan][0]].plot(data_normed['normed_pwr'], color='black', label='Normalized power')
            ax[locs[chan][1], locs[chan][0]].axvspan(9, 16, color='lavender', alpha=0.8, label = 'Spindle Range')
            ax[locs[chan][1], locs[chan][0]].set_title(chan)
        
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
    
    exclude = ['EKG', 'EOG_L', 'EOG_R']
    eeg_chans = [x for x in n.spindles.keys() if x not in exclude]
    ncols = 6
    nrows = len(eeg_chans)//ncols + (len(eeg_chans) % ncols > 0) 
    fig, axs = plt.subplots(nrows = nrows, ncols = ncols, sharex=True, figsize=(15, 7))
    fig.subplots_adjust(hspace=0.5)
    
    for chan, ax in zip(n.so.keys(), axs.flatten()):
        if chan not in exclude:
            # set color iterator -- for other colors look at ocean, gnuplot, prism
            color=iter(plt.cm.nipy_spectral(np.linspace(0, 1, len(n.so[chan]))))
            for i in n.so[chan]:
                c = next(color)
                ax.plot(n.so[chan][i][datatype], c=c, alpha=1, lw=0.8)
            # set subplot params
            ax.set_xlim([-2500, 2500])
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


def plot_spso_chan(n, chan, so_dtype='sofilt', sp_dtype='spfilt'):
    """ Plot individual slow oscillations with overriding spindle detections """
    
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


def plot_spso(n):
    """ Plot individual slow oscillations with overriding spindle detections """
    
    ncols = 6
    nrows = len(n.spso_aggregates)//ncols + (len(n.spso_aggregates) % ncols > 0) 
    fig, axs = plt.subplots(nrows = nrows, ncols = ncols, sharex=True, figsize=(15, 7))
    
    for ax, (chan, so_keys) in zip(axs.flatten(), n.spso_aggregates.items()):
        
        color_so=iter(plt.cm.winter(np.linspace(0, 1, len(so_keys))))
        for so in so_keys:
            ax.plot(n.spso_aggregates[chan][so]['sofilt'], c='black', alpha=0.3, lw=1)
            for spin in n.spso_aggregates[chan][so].columns[4:]:
                ax.plot(n.spso_aggregates[chan][so][spin], c=np.random.rand(3,), alpha=0.8, lw=1)
        ax.set_title(chan)
    
    # delete extra subplots
    for e, ax in enumerate(axs.flatten()):
        if e >= len(n.spso_aggregates):
            fig.delaxes(ax)
        
    fig.tight_layout(pad=1, rect=[0, 0, 1, 0.95])
    fig.text(0.5, 0, 'Time (ms)', ha='center')
    fig.text(0, 0.5, 'Amplitude (mV)', va='center', rotation='vertical')
    fig.suptitle(n.metadata['file_info']['fname'].split('.')[0])
    
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