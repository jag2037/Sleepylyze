""" plotting functions for Dataset objects 
    
    To Do:
        Make plot_hyp() function
"""

import matplotlib
#matplotlib.use('Qt5Agg') # uncomment for scrollable plot
import itertools
import matplotlib.pyplot as plt
import numpy as np 
import os
import pandas as pd
import shapely.geometry as SG
from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

class ScrollableWindow(QtWidgets.QMainWindow):
    """ Potentially for use to see spindle detections """
    def __init__(self, fig, ax, step=0.1):
        plt.close("all")
        if not QtWidgets.QApplication.instance():
            self.app = QtWidgets.QApplication(sys.argv)
        else:
            self.app = QtWidgets.QApplication.instance() 

        QtWidgets.QMainWindow.__init__(self)
        self.widget = QtWidgets.QWidget()
        self.setCentralWidget(self.widget)
        self.widget.setLayout(QtWidgets.QVBoxLayout())
        self.widget.layout().setContentsMargins(0,0,0,0)
        self.widget.layout().setSpacing(0)

        self.fig = fig
        self.ax = ax
        self.canvas = FigureCanvas(self.fig)
        self.canvas.draw()
        self.scroll = QtWidgets.QScrollBar(QtCore.Qt.Horizontal)
        self.step = step
        self.setupSlider()
        self.nav = NavigationToolbar(self.canvas, self.widget)
        self.widget.layout().addWidget(self.nav)
        self.widget.layout().addWidget(self.canvas)
        self.widget.layout().addWidget(self.scroll)

        self.canvas.draw()
        self.show()
        self.qapp.exec_()

    def setupSlider(self):
        self.lims = np.array(self.ax.get_xlim())
        self.scroll.setPageStep(self.step*100)
        self.scroll.actionTriggered.connect(self.update)
        self.update()

    def update(self, evt=None):
        r = self.scroll.value()/((1+self.step)*100)
        l1 = self.lims[0]+r*np.diff(self.lims)
        l2 = l1 +  np.diff(self.lims)*self.step
        self.ax.set_xlim(l1,l2)
        print(self.scroll.value(), l1,l2)
        self.fig.canvas.draw_idle()


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

def plotEKG(ekg, rpeaks=False):
    """ plot EKG class instance """
    fig = plt.figure(figsize = [18, 6])
    plt.plot(ekg.data)

    if rpeaks == True:
        plt.scatter(ekg.rpeaks.index, ekg.rpeaks.values, color='red')

    return fig

def plotEKG_slider(ekg, rpeaks=False):
    """ plot EKG class instance """
    fig, ax = plt.subplots(figsize = [18, 6])
    ax.plot(ekg.data)

    if rpeaks == True:
        plt.scatter(ekg.rpeaks.index, ekg.rpeaks.values, color='red')
    
    # create x-axis slider
    x_min_index = 0
    x_max_index = 2500

    x_min = ekg.data.index[x_min_index]
    x_max = ekg.data.index[x_max_index]
    x_dt = x_max - x_min
    
    y_min, y_max = plt.axis()[2], plt.axis()[3]

    plt.axis([x_min, x_max, y_min, y_max])

    axcolor = 'lightgoldenrodyellow'
    axpos = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor=axcolor)
    
    slider_max = len(ekg.data) - x_max_index - 1

    spos = Slider(axpos, 'Pos', matplotlib.dates.date2num(x_min), matplotlib.dates.date2num(ekg.data.index[slider_max]))

    # format date names
    #plt.gcf().autofmt_xdate()
    
    def update(val):
        pos = spos.val
        xmin_time = matplotlib.dates.num2date(pos)
        xmax_time = matplotlib.dates.num2date(pos) + x_dt
        ax.axis([xmin_time, xmax_time, y_min, y_max])
        fig.canvas.draw_idle()

    spos.on_changed(update)
    
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