""" This file contains a class and methods for Non-REM EEG segments 
	
	Notes:
		- Analysis should output # of NaNs in the data
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
        	self.metadata['epoch'] = fname.split('_')[4]

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
        wn=np.asarray(self.sp_filtwindow)
        if np.any(wn <=0) or np.any(wn >=1):
            wn = wn/nyquist # must remake filter for each pt bc of differences in s_freq
   
        self.sos = butter(self.sp_filtorder, wn, btype='bandpass', output='sos')
        print("Zero phase butterworth filter successfully created: order =", self.sp_filtorder,"bandpass =", wn*nyquist)
    
    # step 2: filter channels
    def filt_EEG_singlechan(self, i):
        """ Apply Butterworth bandpass to signal by channel """
        filt_chan = sosfiltfilt(self.sos, self.data[i].to_numpy(), axis=0)
        self.spfiltEEG[i] = filt_chan
    
    # steps 3-4: calculate RMS & smooth   
    def rms_smooth(self, i):
        """ Calculate moving RMS (rectify) & smooth the EEG """
        mw = int(self.sp_RMSmw*self.s_freq) # convert moving window size from seconds to samples
    
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
        det_lo = float(mean_rms + self.sp_loSD*np.std(self.spRMSmavg[i]))
        det_hi = float(mean_rms + self.sp_hiSD*np.std(self.spRMSmavg[i]))
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
        sduration = [x*self.s_freq for x in self.sp_duration]
    
        for i in self.spindle_events:
            self.spindle_rejects[i] = []
            for j in self.spindle_events[i]:
                # if spindle is not within duration length range
                if not sduration[0] <= len(j) <= sduration[1]:
                    # add to reject list
                    self.spindle_rejects[i].append(j) 
                    # get index & remove from event list
                    ind = self.spindle_events[i].index(j)
                    del self.spindle_events[i][ind]
                    
    # set multiIndex
    def spMultiIndex(self):
        """ combine dataframes into a multiIndex dataframe"""
        # reset column levels
        self.spfiltEEG.columns = pd.MultiIndex.from_arrays([self.eeg_channels, np.repeat(('Filtered'), len(self.eeg_channels))],names=['Channel','datatype'])
        self.spRMS.columns = pd.MultiIndex.from_arrays([self.eeg_channels, np.repeat(('RMS'), len(self.eeg_channels))],names=['Channel','datatype'])
        self.spRMSmavg.columns = pd.MultiIndex.from_arrays([self.eeg_channels, np.repeat(('RMSmavg'), len(self.eeg_channels))],names=['Channel','datatype'])

        # list df vars for index specs
        dfs =[self.spfiltEEG, self.spRMS, self.spRMSmavg] # for > speed, don't store spinfilt_RMS as an attribute
        calcs = ['Filtered', 'RMS', 'RMSmavg']
        lvl0 = np.repeat(self.eeg_channels, len(calcs))
        lvl1 = calcs*len(self.eeg_channels)    
    
        # combine & custom sort
        self.spindle_calcs = pd.concat(dfs, axis=1).reindex(columns=[lvl0, lvl1])
        
        
    def detect_spindles(self, wn=[8, 16], order=4, sp_mw=0.2, loSD=0, hiSD=1.5, duration=[0.5, 3.0]):  
        """ Detect spindles by channel [Params/Returns] """
        self.sp_filtwindow = wn
        self.sp_filtorder = order
        self.sp_RMSmw = sp_mw
        self.sp_loSD = loSD
        self.sp_hiSD = hiSD
        self.sp_duration = duration
    
        # set attributes
        self.spindle_attributes()
        # Make filter
        self.make_butter()

        # For each EEG channel
        for i in self.eeg_channels:
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
