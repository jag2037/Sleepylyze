""" This file contains a class and methods for Non-REM EEG segments 
    
    Notes:
        - Analysis should output # of NaNs in the data

    TO DO: 
        For self.detect_spindles(), move attributes into metadata['analysis_info'] dict
"""

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

        print('EEG successfully imported.')


    ## Spindle Detection Methods ##

    # make attributes
    def spindle_attributes(self):    
        dfs =['spfiltEEG', 'spRMS', 'spRMSmavg'] # for > speed, don't store spRMS as an attribute
        [setattr(self, df, pd.DataFrame(index=self.data.index)) for df in dfs]
        self.spThresholds = pd.DataFrame(index=['Mean RMS', 'Low Threshold', 'High Threshold'])
        self.spindle_events = {}
        self.spindle_rejects = {}
    
    # step 1: make filter
    def make_butter(self):
        """ Make Butterworth bandpass filter [Parameters/Returns]"""
        nyquist = self.s_freq/2
        wn=np.asarray(self.metadata['spindle_analysis']['sp_filtwindow'])
        if np.any(wn <=0) or np.any(wn >=1):
            wn = wn/nyquist # must remake filter for each pt bc of differences in s_freq
   
        self.sos = butter(self.metadata['spindle_analysis']['sp_filtorder'], wn, btype='bandpass', output='sos')
        print("Zero phase butterworth filter successfully created: order =", self.metadata['spindle_analysis']['sp_filtorder'],
            "bandpass =", wn*nyquist)
    
    # step 2: filter channels
    def filt_EEG_singlechan(self, i):
        """ Apply Butterworth bandpass to signal by channel """

        # separate NaN and non-NaN values to avoid NaN filter output on cleaned data
        data_nan = self.data[i][self.data[i]['Raw'].isna()]
        data_notnan = self.data[i][self.data[i]['Raw'].isna() == False]

        # filter notNaN data & add column to notNaN df
        data_notnan_filt = sosfiltfilt(self.sos, data_notnan.to_numpy(), axis=0)
        data_notnan['Filt'] = data_notnan_filt

        # merge NaN & filtered notNaN values, sort on index
        filt_chan = data_nan['Raw'].append(data_notnan['Filt']).sort_index()

        # add channel to main dataframe
        self.spfiltEEG[i] = filt_chan
    
    # steps 3-4: calculate RMS & smooth   
    def rms_smooth(self, i):
        """ Calculate moving RMS (rectify) & smooth the EEG """
        mw = int(self.metadata['spindle_analysis']['sp_RMSmw']*self.s_freq) # convert moving window size from seconds to samples
    
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
    def get_spindles(self, i, lo, hi, mavg_varr, mavg_iarr):
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
    
        self.spindle_rejects = {}
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

        self.s_freq = self.metadata['analysis_info']['s_freq']
    
        # set attributes
        self.spindle_attributes()
        # Make filter
        self.make_butter()

        # For all channels (easier for plotting purposes)
        self.channels = [x[0] for x in self.data.columns]
        for i in self.channels:
           # if i not in ['EOG_L', 'EOG_R', 'EKG']:
                # Filter
                self.filt_EEG_singlechan(i)
                # Calculate RMS & smooth
                self.rms_smooth(i)
                # Set detection thresholds
                self.set_thres(i)
                # Vectorize data for detection looping
                lo, hi = self.spThresholds[i]['Low Threshold'], self.spThresholds[i]['High Threshold'] 
                mavg_varr, mavg_iarr = np.asarray(self.spRMSmavg[i]), np.asarray(self.spRMSmavg[i].index)
                # Detect spindles
                self.get_spindles(i, lo, hi, mavg_varr, mavg_iarr)
        
        # Check spindle duration
        self.duration_check()
        print('Spindle detection complete.')
        # combine dataframes
        print('Combining dataframes...')
        self.spMultiIndex()
        print('done.')



    def analyze_spindles(self):
        """ starting code for spindle statistics/visualizations 

        Parameters
        ----------
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

        spindles = {}
        for chan in self.spindle_events.keys():
            spindles[chan] = {}
            for i, spin in enumerate(self.spindle_events[chan]):
                # create individual df for each spindle
                spin_data = self.data[chan]['Raw'].loc[self.spindle_events[chan][i]]
                #spin_data_normed = (spin_data - min(spin_data))/(max(spin_data)-min(spin_data))
                
                # set new index so that each spindle is centered around zero
                half_length = len(spin)/2
                t_id = np.linspace(-half_length, half_length, int(2*half_length//1))
                # convert from samples to ms
                id_ms = t_id * (1/self.metadata['analysis_info']['s_freq']*1000)
                
                # create new dataframe
                spindles[chan][i] = pd.DataFrame(index=id_ms)
                spindles[chan][i].index.name='id_ms'
                spindles[chan][i]['time'] = spin_data.index
                spindles[chan][i]['Raw'] = spin_data.values
                #spindles[chan][i]['Raw_normed'] = spin_data_normed.values
        
        self.spindles = spindles
        print('Dataframes created. Spindle data stored in obj.spindles.')

    ## Slow Oscillation Detection Methods ##

