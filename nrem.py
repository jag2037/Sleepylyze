""" This file contains a class and methods for Non-REM EEG segments 
    
    Notes:
        - Analysis should output # of NaNs in the data

    TO DO: 
        For self.detect_spindles(), move attributes into metadata['analysis_info'] dict
"""

import datetime
import os
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, sosfreqz

class NREM:
    """ General class for nonREM EEG segments """
    def __init__(self, fname, fpath, epoched=False):
        filepath = os.path.join(fpath, fname)

        in_num, start_date, slpstage, cycle = fname.split('_')[:4]
        self.metadata = {'file_info':{'in_num': in_num, 'fname': fname, 'path': filepath,
                                    'sleep_stage': slpstage,'cycle': cycle} }
        if epoched is True:
            self.metadata['file_info']['epoch'] = fname.split('_')[4]

        self.load_segment()

    def load_segment(self):
        """ Load eeg segment and extract sampling frequency. """
        
        data = pd.read_csv(self.metadata['file_info']['path'], header = [0, 1], index_col = 0, parse_dates=True)
        
        # Check cycle length against 5 minute duration minimum
        cycle_len_secs = (data.index[-1] - data.index[0]).total_seconds()
        self.data = data
        
        diff = data.index.to_series().diff()[1:2]
        s_freq = 1000000/diff[0].microseconds

        self.metadata['file_info']['start_time'] = data.index[0]
        self.metadata['analysis_info'] = {'s_freq': s_freq, 'cycle_len_secs': cycle_len_secs}
        self.s_freq = s_freq

        print('EEG successfully imported.')


    ## Spindle Detection Methods ##

    # make attributes
    def spindle_attributes(self):
        """ create attributes for spindle detection """
        try:
            self.channels
        except AttributeError:
            # create if doesn't exist
            self.channels = [x[0] for x in self.data.columns]

        dfs =['spfiltEEG', 'spRMS', 'spRMSmavg'] # for > speed, don't store spRMS as an attribute
        [setattr(self, df, pd.DataFrame(index=self.data.index)) for df in dfs]
        self.spThresholds = pd.DataFrame(index=['Mean RMS', 'Low Threshold', 'High Threshold'])
        self.spindle_events = {}
        self.spindle_rejects = {}
    
    # step 1: make filter
    def make_butter_sp(self, wn, order):
        """ Make Butterworth bandpass filter [Parameters/Returns]"""
        nyquist = self.s_freq/2
        wn_arr=np.asarray(wn)
        if np.any(wn_arr <=0) or np.any(wn_arr >=1):
            wn_arr = wn_arr/nyquist # must remake filter for each pt bc of differences in s_freq
   
        self.sp_sos = butter(order, wn_arr, btype='bandpass', output='sos')
        print(f"Zero phase butterworth filter successfully created: order = {order}x{order} bandpass = {wn}")
    
    # step 2: filter channels
    def spfilt(self, i):
        """ Apply Butterworth bandpass to signal by channel """

        # separate NaN and non-NaN values to avoid NaN filter output on cleaned data
        data_nan = self.data[i][self.data[i]['Raw'].isna()]
        data_notnan = self.data[i][self.data[i]['Raw'].isna() == False]

        # filter notNaN data & add column to notNaN df
        data_notnan_filt = sosfiltfilt(self.sp_sos, data_notnan.to_numpy(), axis=0)
        data_notnan['Filt'] = data_notnan_filt

        # merge NaN & filtered notNaN values, sort on index
        filt_chan = data_nan['Raw'].append(data_notnan['Filt']).sort_index()

        # add channel to main dataframe
        self.spfiltEEG[i] = filt_chan
    
    # steps 3-4: calculate RMS & smooth   
    def rms_smooth(self, i, sp_mw):
        """ Calculate moving RMS (rectify) & smooth the EEG """
        mw = int(sp_mw*self.s_freq) # convert moving window size from seconds to samples
    
        # convolve for rolling RMS
        datsq = np.power(self.spfiltEEG[i], 2)
        window = np.ones(mw)/float(mw)
        # convolution mode 'valid' will remove edge effects, but also introduce a time shift
        # and downstream erors because it changes the length of the rms data
        rms = np.sqrt(np.convolve(datsq, window, 'same')) 
        #spinfilt_RMS = pd.DataFrame(rms, index=self.data.index) --> add this back for > speed
        self.spRMS[i] = rms # for > speed, don't store spinfilt_RMS[i] as an attribute
    
        # smooth with moving average
        rms_avg = self.spRMS[i].rolling(mw, center=True).mean()
        self.spRMSmavg[i] = rms_avg
    
    # step 5: set thresholds
    def set_thres(self, i):
        """ set spindle detection threshold levels, in terms of multiples of RMS SD """
        mean_rms = float(np.mean(self.spRMSmavg[i]))
        det_lo = float(mean_rms + self.metadata['spindle_analysis']['sp_loSD']*np.std(self.spRMSmavg[i]))
        det_hi = float(mean_rms + self.metadata['spindle_analysis']['sp_hiSD']*np.std(self.spRMSmavg[i]))
        self.spThresholds[i] = [mean_rms, det_lo, det_hi]
    
    # step 6: detect spindles
    def get_spindles(self, i):

        # vectorize data for detection looping
        lo, hi = self.spThresholds[i]['Low Threshold'], self.spThresholds[i]['High Threshold'] 
        mavg_varr, mavg_iarr = np.asarray(self.spRMSmavg[i]), np.asarray(self.spRMSmavg[i].index)
        
        # initialize spindle event list & set pointer to 0
        self.spindle_events[i] = []
        x=0

        while x < len(self.data):
            # if value crosses high threshold, start a fresh spindle
            if mavg_varr[x] >= hi:
                spindle = []

                # count backwards to find previous low threshold crossing
                for h in range(x, -1, -1): 
                    if mavg_varr[h] >= lo:
                        spindle.insert(0, mavg_iarr[h]) # add value to the beginning of the spindle
                    else:
                        break
       
                # count forwards to find next low threshold crossing
                for h in range(x, len(self.data), 1):
                    # if above low threshold, add to current spindle
                    if mavg_varr[h] >= lo and x < (len(self.data)-1):
                        spindle.append(mavg_iarr[h])
                    # if above low threshold and last value, add to current spindle and add spindle to events list
                    elif mavg_varr[h] >= lo and x == (len(self.data)-1): ## untested
                        spindle.append(mavg_iarr[h])
                        self.spindle_events[i].append(spindle)
                    # otherwise finish spindle & add to spindle events list
                    elif mavg_varr[h] < lo:
                        self.spindle_events[i].append(spindle)
                        break
        
                # advance the pointer to the end of the spindle
                x = h
            # if value doesn't cross high threshold, advance
            else:
                x += 1
                
    # step 7: check spindle length
    def duration_check(self):
        """ Move spindles outside of set duration to reject list"""
        print('Checking spindle duration...')
        sduration = [x*self.s_freq for x in self.metadata['spindle_analysis']['sp_duration']]
    
        for i in self.spindle_events:
            self.spindle_rejects[i] = [x for x in self.spindle_events[i] if not sduration[0] <= len(x) <= sduration[1]]
            self.spindle_events[i] = [x for x in self.spindle_events[i] if sduration[0] <= len(x) <= sduration[1]]            

        dur = self.metadata['spindle_analysis']['sp_duration']
        print(f'Spindles shorter than {dur[0]}s and longer than {dur[1]}s removed.')
                    
    # set multiIndex
    def spMultiIndex(self):
        """ combine dataframes into a multiIndex dataframe"""
        # reset column levels
        self.spfiltEEG.columns = pd.MultiIndex.from_arrays([self.channels, np.repeat(('Filtered'), len(self.channels))],names=['Channel','datatype'])
        self.spRMS.columns = pd.MultiIndex.from_arrays([self.channels, np.repeat(('RMS'), len(self.channels))],names=['Channel','datatype'])
        self.spRMSmavg.columns = pd.MultiIndex.from_arrays([self.channels, np.repeat(('RMSmavg'), len(self.channels))],names=['Channel','datatype'])

        # list df vars for index specs
        dfs =[self.spfiltEEG, self.spRMS, self.spRMSmavg] # for > speed, don't store spinfilt_RMS as an attribute
        calcs = ['Filtered', 'RMS', 'RMSmavg']
        lvl0 = np.repeat(self.channels, len(calcs))
        lvl1 = calcs*len(self.channels)    
    
        # combine & custom sort
        self.spindle_calcs = pd.concat(dfs, axis=1).reindex(columns=[lvl0, lvl1])
        
        
    def detect_spindles(self, wn=[8, 16], order=4, sp_mw=0.2, loSD=0, hiSD=1.5, duration=[0.5, 3.0]):  
        """ Detect spindles by channel [Params/Returns] """

        self.metadata['spindle_analysis'] = {'sp_filtwindow': wn, 'sp_filtorder': order, 
            'sp_RMSmw': sp_mw, 'sp_loSD': loSD, 'sp_hiSD': hiSD, 'sp_duration': duration}

        #self.s_freq = self.metadata['analysis_info']['s_freq']
    
        # set attributes
        self.spindle_attributes()
        # Make filter
        self.make_butter_sp(wn, order)

        # loop through channels (all channels for plotting ease)
        for i in self.channels:
           # if i not in ['EOG_L', 'EOG_R', 'EKG']:
                # Filter
                self.spfilt(i)
                # Calculate RMS & smooth
                self.rms_smooth(i, sp_mw)
                # Set detection thresholds
                self.set_thres(i)
                # Detect spindles
                self.get_spindles(i)
        
        # Check spindle duration
        self.duration_check()
        print('Spindle detection complete.')
        # combine dataframes
        print('Combining dataframes...')
        self.spMultiIndex()
        print('done.')

    def analyze_spindles(self, zmethod='trough'):
        """ starting code for spindle statistics/visualizations 

        Parameters
        ----------
        zmethod: str (default: 'trough')
            method used to assign 0-center to spindles [options: 'trough', 'middle']. Trough assigns zero-center to
            the deepest negative trough. Middle assigns zero center to the midpoint in time.
        self.spindle_events: dict
            dict of timestamps when spindles occur (created from self.detect_spindles())
        self.data: pd.DataFrame
            df containing raw EEG data

        Returns
        -------
        self.spindles: nested dict of dfs
            nested dict with spindle data by channel {channel: {spindle_num:spindle_data}}
        """
        ## create dict of dataframes for spindle analysis
        print('Creating individual dataframes...')
        
        self.metadata['spindle_analysis']['zmethod'] = zmethod

        spindles = {}
        for chan in self.spindle_events.keys():
            spindles[chan] = {}
            for i, spin in enumerate(self.spindle_events[chan]):
                # create individual df for each spindle
                spin_data = self.data[chan]['Raw'].loc[self.spindle_events[chan][i]]
                #spin_data_normed = (spin_data - min(spin_data))/(max(spin_data)-min(spin_data))
                
                # set new index so that each spindle is centered around zero
                if zmethod == 'middle':
                    # this method could use some work
                    half_length = len(spin)/2
                    t_id = np.linspace(-half_length, half_length, int(2*half_length//1))
                    # convert from samples to ms
                    id_ms = t_id * (1/self.metadata['analysis_info']['s_freq']*1000)
                elif zmethod == 'trough':
                    id_ms = (spin_data.index - spin_data.idxmin()).total_seconds()*1000
                    
                # create new dataframe
                spindles[chan][i] = pd.DataFrame(index=id_ms)
                spindles[chan][i].index.name='id_ms'
                spindles[chan][i]['time'] = spin_data.index
                spindles[chan][i]['Raw'] = spin_data.values
                #spindles[chan][i]['Raw_normed'] = spin_data_normed.values
        
        self.spindles = spindles
        print('Dataframes created. Spindle data stored in obj.spindles.')


    ## Slow Oscillation Detection Methods ##

    def so_attributes(self):
        """ make attributes for slow oscillation detection """
        try:
            self.channels
        except AttributeError:
            # create if doesn't exist
            self.channels = [x[0] for x in self.data.columns]
        
        dfs = ['sofiltEEG']
        [setattr(self, df, pd.DataFrame(index=self.data.index)) for df in dfs]
        self.so_events = {}
        self.so_rejects = {}

    def make_butter_so(self, wn, order):
        """ Make Butterworth bandpass filter [Parameters/Returns]"""
        
        nyquist = self.s_freq/2
        wn_arr = np.asarray(wn)
        
        if np.any(wn_arr <=0) or np.any(wn_arr >=1):
            wn_arr = wn_arr/nyquist # must remake filter for each pt bc of differences in s_freq
   
        self.so_sos = butter(order, wn_arr, btype='bandpass', output='sos')
        print(f"Zero phase butterworth filter successfully created: order = {order}x{order}, bandpass = {wn}")

    def sofilt(self, i):
        """ Apply Slow Oscillation Butterworth bandpass to signal by channel 
        
            Parameters
            ----------
            i : str
                channel to filter
            
            Returns
            -------
            self.sofiltEEG: pandas.DataFrame
                filtered EEG data
        """

        # separate NaN and non-NaN values to avoid NaN filter output on cleaned data
        data_nan = self.data[i][self.data[i]['Raw'].isna()]
        data_notnan = self.data[i][self.data[i]['Raw'].isna() == False]

        # filter notNaN data & add column to notNaN df
        data_notnan_filt = sosfiltfilt(self.so_sos, data_notnan.to_numpy(), axis=0)
        data_notnan['SOFilt'] = data_notnan_filt

        # merge NaN & filtered notNaN values, sort on index
        filt_chan = data_nan['Raw'].append(data_notnan['SOFilt']).sort_index()

        # add channel to main dataframe
        self.sofiltEEG[i] = filt_chan

    def get_so(self, i, posx_thres, npeak_thres, negpos_thres):
        """ Detect slow oscillations. Based on detection algorithm from Molle 2011 
            
            Parameters
            ----------
            posx_thres: list of float (default: [0.9, 2])
                threshold of consecutive positive-negative zero crossings in seconds
            npeak_thres: int (default: -80)
                negative peak threshold in microvolts
            negpos_thres: int (default: 140)
                minimum amplitude threshold for negative to positive peaks
        """

        self.so_events[i] = {}
        n = 0
        
        # convert thresholds (time to timedelta & uv to mv)
        posx_thres_td = [pd.Timedelta(s, 's') for s in posx_thres]
        npeak_mv = npeak_thres*(10**-3)
        negpos_mv = negpos_thres*(10**-3)

        # convert channel data to series
        chan_dat = self.sofiltEEG[i]

        # get zero-crossings
        mask = chan_dat > 0
        # shift pos/neg mask by 1 and compare
        mask_shift = np.insert(np.array(mask), 0, None)
        mask_shift = pd.Series(mask_shift[:-1], index=mask.index)
        # pos-neg are True; neg-pos are False
        so_zxings = mask[mask != mask_shift]
        self.so_zxings = so_zxings

        # get intervals between subsequent positive-negative crossings
        pn_xings = so_zxings[so_zxings==True]
        intvls = pn_xings.index.to_series().diff()

        # loop through intervals
        for e, v in enumerate(intvls):
            # if interval is between >=0.9sec and <=2sec
            if e != 0 and posx_thres_td[0] <= v <= posx_thres_td[1]:
                # find negative & positive peaks
                npeak_time = chan_dat.loc[intvls.index[e-1]:intvls.index[e]].idxmin()
                npeak_val = chan_dat.loc[npeak_time]
                ppeak_time = chan_dat.loc[intvls.index[e-1]:intvls.index[e]].idxmax()
                ppeak_val = chan_dat.loc[ppeak_time]
                # if negative peak is < than threshold
                if npeak_val < npeak_mv:
                    # if negative-positive peak amplitude is >= than threshold
                    if np.abs(npeak_val) + np.abs(ppeak_val) >= negpos_mv:
                        self.so_events[i][n] = {'zcross1': intvls.index[e-1], 'zcross2': intvls.index[e], 'npeak': npeak_time, 
                                           'ppeak': ppeak_time, 'npeak_minus2s': npeak_time - datetime.timedelta(seconds=2), 
                                           'npeak_plus2s': npeak_time + datetime.timedelta(seconds=2)}
                        n += 1

    def soMultiIndex(self):
        """ combine dataframes into a multiIndex dataframe"""
        # reset column levels
        self.sofiltEEG.columns = pd.MultiIndex.from_arrays([self.channels, np.repeat(('Filtered'), len(self.channels))],names=['Channel','datatype'])
        #self.spRMS.columns = pd.MultiIndex.from_arrays([self.channels, np.repeat(('RMS'), len(self.channels))],names=['Channel','datatype'])
        #self.spRMSmavg.columns = pd.MultiIndex.from_arrays([self.channels, np.repeat(('RMSmavg'), len(self.channels))],names=['Channel','datatype'])

        # list df vars for index specs
        dfs =[self.sofiltEEG] # for > speed, don't store spinfilt_RMS as an attribute
        calcs = ['Filtered']
        lvl0 = np.repeat(self.channels, len(calcs))
        lvl1 = calcs*len(self.channels)    
    
        # combine & custom sort
        self.so_calcs = pd.concat(dfs, axis=1).reindex(columns=[lvl0, lvl1])

    def detect_so(self, wn=[0.1, 4], order=2, posx_thres = [0.9, 2], npeak_thres = -80, negpos_thres = 140):
        """ Detect slow oscillations by channel
        
            Parameters
            ----------
            wn: list (default: [0.1, 4])
                Butterworth filter window
            order: int (default: 2)
                Butterworth filter order (default of 2x2 from Massimini et al., 2004)
            posx_thres: list of float (default: [0.9, 2])
                threshold of consecutive positive-negative zero crossings in seconds
            npeak_thres: int (default: -80)
                negative peak threshold in microvolts
            negpos_thres: int (default: 140)
                minimum amplitude threshold for negative to positive peaks
            
            Returns
            -------
            self.so_sos
            self.so_filtEEG
            self.so_calcs
            self.so_zxings
            self.so_events
            self.so_rejects
        """
        
        self.metadata['so_analysis'] = {'so_filtwindow': wn, 'so_filtorder': order, 'posx_thres': posx_thres,
                                        'npeak_thres': npeak_thres, 'negpos_thres': negpos_thres}
        
        # set attributes
        self.so_attributes()
        
        # make butterworth filter
        self.make_butter_so(wn, order)
        
        # loop through channels (all channels for plotting ease)
        for i in self.channels:
                # Filter
                self.sofilt(i)
                # Detect SO
                self.get_so(i, posx_thres, npeak_thres, negpos_thres)
                
        # combine dataframes
        print('Combining dataframes...')
        self.soMultiIndex()
        print('done.')

    def analyze_so(self, zmethod='trough'):
        """ starting code for slow oscillation statistics/visualizations """

        ## create dict of dataframes for slow oscillation analysis
        print('Creating individual dataframes...')

        so = {}
        for chan in self.so_events.keys():
            so[chan] = {}
            for i, s in self.so_events[chan].items():
                # create individual df for each spindle
                start = self.so_events[chan][i]['npeak_minus2s']
                end = self.so_events[chan][i]['npeak_plus2s']
                so_data = self.data[chan]['Raw'].loc[start:end]
                
                # set new index so that each SO is zero-centered around the negative peak
                ms1 = list(range(-2000, 0, int(1/self.metadata['analysis_info']['s_freq']*1000)))
                ms2 = [-x for x in ms1[::-1]]
                id_ms = ms1 + [0] + ms2
                
                # create new dataframe
                so[chan][i] = pd.DataFrame(index=id_ms)
                so[chan][i].index.name='id_ms'
                
                # if the SO is not a full 2s from the beginning
                if start < self.data.index[0]:
                    # extend the df index to the full 2s
                    time_freq = str(int(1/self.metadata['analysis_info']['s_freq']*1000000))+'us'
                    time = pd.date_range(start=start, end=end, freq=time_freq)
                    so[chan][i]['time'] = time
                    # append NaNs onto the end of the EEG data
                    nans = np.repeat(np.NaN, len(time)-len(so_data))
                    data_extended = list(nans) + list(so_data.values)
                    so[chan][i]['Raw'] = data_extended
                # if the SO is not a full 2s from the end
                elif end > self.data.index[-1]:
                    # extend the df index to the full 2s
                    time_freq = str(int(1/self.metadata['analysis_info']['s_freq']*1000000))+'us'
                    time = pd.date_range(start=start, end=end, freq=time_freq)
                    so[chan][i]['time'] = time
                    # append NaNs onto the end of the EEG data
                    nans = np.repeat(np.NaN, len(time)-len(so_data))
                    data_extended = list(so_data.values) + list(nans)
                    so[chan][i]['Raw'] = data_extended
                else:
                    so[chan][i]['time'] = so_data.index
                    so[chan][i]['Raw'] = so_data.values
        
        self.so = so
        print('Dataframes created. Slow oscillation data stored in obj.so.')