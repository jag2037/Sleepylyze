""" This file contains a class and methods for Non-REM EEG segments 
    
    Notes:
        - Analysis should output # of NaNs in the data

    TO DO: 
        - ** update analyze_spindles for psd_i removal
        - Optimize self.create_spindfs() method
        - Assign NREM attributes to slots on init
        - Update docstrings
        
        - ** Export SO and SPSO analyses
        - ** move EXCLUDE var into function calls
"""

from datetime import datetime
from datetime import timedelta
import glob
#import joblib
import json
import os
import math
import numpy as np
import pandas as pd
import warnings
import xlsxwriter
from pyedflib import highlevel

from mne.time_frequency import psd_array_multitaper
from scipy.signal import butter, sosfiltfilt, sosfreqz, find_peaks
from scipy.optimize import OptimizeWarning, curve_fit

class NREM:
    """ General class for nonREM EEG segments """
    def __init__(self, fname=None, fpath=None, match=None, in_num=None, epoched=False, batch=False, lowpass_freq=25, lowpass_order=4,
                laplacian_chans=None, replace_data=False, edf=False):
        """ Initialize NREM object
            
            Parameters
            ----------
            fname: str
                filename (if loading a single dataframe)
            fpath: str
                absolute path to file(s) directory
            match: str
                string to match within the filename of all files to load (Ex: '_s2_')
            in_num: str
                IN number, for batch loading
            epoched: bool (default: False)
                whether data has been epoched (if loading a single dataframe)
            batch: bool (default: True)
                whether to load all matching files from the fpath directory
            lowpass_freq: int or None (default: 25)
                lowpass filter frequency *Used in visualizations ONLY*. must be < nyquist
            lowpass_order: int (default: 4)
                Butterworth lowpass filter order (doubles for filtfilt)
            laplacian_chans: str, list, or None (default: None)
                channels to apply laplacian filter to [Options: 'all', list of channel names, None]
                For leading/lagging analysis, was using ['F3', 'F4', 'P3', 'P4']
            replace_data: bool (default: False)
                whether to replace primary data with laplcian filtered data
            edf: bool (default: False)
                whether the input file is in edf format. If false data is assumed to be in csv format
        """
        
        if batch:
            self.load_batch(fpath, match, in_num)
        else:
            filepath = os.path.join(fpath, fname)
            if edf == False:
                in_num, start_date, slpstage, cycle = fname.split('_')[:4]
                self.metadata = {'file_info':{'in_num': in_num, 'fname': fname, 'path': filepath,
                                            'sleep_stage': slpstage,'cycle': cycle} }
                if epoched is True:
                    self.metadata['file_info']['epoch'] = fname.split('_')[4]
            if edf == True:
                self.metadata = {'file_info':{'fname': fname, 'path': filepath} }
            # load the data
            self.load_segment(edf)

            # apply laplacian
            if laplacian_chans is not None:
                self.metadata['analysis_info']['spatial_filter'] = 'laplacian'
                data_lap = self.make_laplacian(laplacian_chans)
                # replace data
                if replace_data:
                    self.metadata['analysis_info']['RawData_replaced_wLaplacian'] = 'True'
                    self.data = data_lap
                else:
                    self.metadata['analysis_info']['RawData_replaced_wLaplacian'] = 'False'
                    self.data_lap = data_lap
            else:
                self.metadata['analysis_info']['spatial_filter'] = 'None'

            
            # apply lowpass filter
            if lowpass_freq:
                self.lowpass_raw(lowpass_freq, lowpass_order)



    def load_segment(self, edf):
        """ Load eeg segment and extract sampling frequency. 
            
            Parameters
            ----------
            edf: bool (default: False)
                whether the input file is in edf format. If false data is assumed to be in csv format
        """
        
        if edf == False:
            data = pd.read_csv(self.metadata['file_info']['path'], header = [0, 1], index_col = 0, parse_dates=True)
            
            # Check cycle length against 5 minute duration minimum
            cycle_len_secs = (data.index[-1] - data.index[0]).total_seconds()
            self.data = data
            
            diff = data.index.to_series().diff()[1:2]
            # use floor to round down if not an integer
            s_freq = math.floor(1000000/diff[0].microseconds)
            self.metadata['file_info']['start_time'] = str(data.index[0])
        if edf == True:
            signals, signal_headers, header = highlevel.read_edf(self.metadata['file_info']['path'])
            cycle_len_secs = len(signals[0])/int(signal_headers[0]['sample_frequency'])
            s_freq = math.floor(int(signal_headers[0]['sample_frequency']))
            startdate = header['startdate']
            self.metadata['file_info']['start_time']=startdate.strftime('%Y-%m-%d %H:%M:%S.%f')
            #set up to make dataframe
            starttime = startdate.strftime('%Y-%m-%d %H:%M:%S.%f') #get datetime object back
            labels = [] #make list of column names (channel names)
            for signal in signal_headers:
                labels.append(signal['label'])
            times = [] #make list of row names, all the time points
            time1 = startdate #start time
            diff = 1/s_freq #time in seconds between each sample
            for data in range(len(signals[0])): 
                times.append(time1)
                time1=time1+timedelta(seconds = diff) #get time of each sample
            #make dataframe
            data_pre = pd.DataFrame(signals)
            data_pre = pd.DataFrame.transpose(data_pre)
            times_idx = pd.Index(times)#make times an index object
            data= data_pre.set_index(times_idx) #set index as times
            data.columns = pd.MultiIndex.from_arrays([labels, np.repeat(('Raw'), len(labels))],names=['Channel','datatype']) #set column names
            self.data = data
        self.metadata['analysis_info'] = {'s_freq': s_freq, 'cycle_len_secs': cycle_len_secs}
        self.s_freq = s_freq

        print('EEG successfully imported.')

    def lowpass_raw(self, lowpass_freq=25, lowpass_order=4):
        """ Lowpass the raw data [for visualization only -- Removes high-frequency artifacts]

            Parameters
            ----------
            lowpass_freq: int (default: 25)
                lowpass frequency. must be < nyquist
            lowpass_order: int (default: 4)
                Butterworth lowpass filter order (doubles for filtfilt)

            Returns
            -------
            self.channels: list of str
                channel list (if not already created)
            self.data_lowpass

        """
        # set data
        data = self.data
        self.metadata['visualizations'] = {'lowpass_freq': lowpass_freq, 'lowpass_order_half': lowpass_order}

        # make butterworth lowpass filter
        nyquist = self.s_freq/2
        data_lowpass = pd.DataFrame(index = data.index)
        
        # adjust lowpass to nyquist
        if lowpass_freq >= 1:
            lowpass_freq = lowpass_freq/nyquist
        
        # make filter
        sos = butter(lowpass_order, lowpass_freq, btype='lowpass', output='sos')

        # create channel attribute
        channels = [x[0] for x in self.data.columns]
        self.channels = channels
        
        # filter the data
        for i in channels:
            # separate NaN and non-NaN values to avoid NaN filter output on cleaned data
            data_nan = data[i][data[i]['Raw'].isna()]
            data_notnan = data[i][data[i]['Raw'].isna() == False]
            # if the channel is not all NaN, lowpass filter
            if len(data_notnan) > 0:
                # filter notNaN data & add column to notNaN df
                data_notnan_filt = sosfiltfilt(sos, data_notnan.to_numpy(), axis=0)
                data_notnan['Filt'] = data_notnan_filt
                # merge NaN & filtered notNaN values, sort on index
                filt_chan = data_nan['Raw'].append(data_notnan['Filt']).sort_index()
                # add to dataframe
                data_lowpass[i] = filt_chan
            # otherwise add the nan column
            else:
                data_lowpass[i] = data[i]
            

        # set dataframe columns
        data_lowpass.columns = pd.MultiIndex.from_arrays([channels, np.repeat(('raw_lowpass'), len(channels))],names=['Channel','datatype'])
        # use the lowpassed data
        raw_lowpass_data = data_lowpass
        self.data_lowpass = data_lowpass

        # if calling a second time, replace zero-padded lowpass spindle values
        if hasattr(self, 'spindles_zpad_lowpass'):
            self.make_lowpass_zpad()
            print('Zero-padded lowpass spindle values recalculated')


    def load_batch(self, fpath, match, in_num):

        """ Load a batch of EEG segments & reset index from absolute to relative time 

            Parameters
            ----------
            fpath: str
                absolute path to file(s) directory
            match: str
                string to match within the filename of all files to load (Ex: '_s2_')
            in_num: str
                IN number, for batch loading
            TO DO: Throw error if IN doesn't match any files in folder
        """ 

        if in_num == None:
            in_num = input('Please specify IN number: ')

        if match == None:
            match = input('Please specify filename string to match for batch loading (ex. \'_s2_\'): ')

        # get a list of all matching files
        glob_match = f'{fpath}/*{match}*'
        files = glob.glob(glob_match)

        # load & concatenate files into a single dataframe
        data = pd.concat((pd.read_csv(file,  header = [0, 1], index_col = 0, parse_dates=True, low_memory=False) for file in files)).sort_index()

        # extract sampling frequency
        s_freq = 1/(data.index[1] - data.index[0]).total_seconds()

        # reset the index to continuous time
        ind_freq = str(int(1/s_freq*1000000))+'us'
        ind_start = '1900-01-01 00:00:00.000'
        ind = pd.date_range(start = ind_start, periods=len(data), freq=ind_freq)
        data.index = ind

        # set metadata & attributes
        self.metadata = {'file_info':{'in_num': in_num, 'files': files, 'dir': fpath,
                                        'match_phrase': match},
                        'analysis_info':{'s_freq': s_freq} }
        self.data = data
        self.s_freq = s_freq


    def make_laplacian(self, chans):
        """ Make laplacian spatial filter 
        
            Weights are determined by cartesian coordinate distance
            ref1: https://hal.inria.fr/hal-01055103/document
            ref2: https://arxiv.org/pdf/1406.0458.pdf
            
            NOTE: Weights are solely determined by vector distance, NOT geometric arrangement
        
            Parameters
            ----------
            chans: str or list
                channels to calculate laplacian for ('all' or list of channel names)
                
            Returns
            -------
            self.metadata.lapalcian_weights: dict
                dict of channels by 4 nearest neighbors and weights {chan: pd.Series(weight, index=neighbor_chan)}
            data_lap: pd.DataFrame
                laplacian filtered data for specified channels     
        """
        
        self.metadata['laplacian_weights'] = {}
            
        # set channel names if filtering all
        exclude = ['EOG_L','EOG_R', 'EKG', 'REF', 'FPZorEKG', 'A1', 'A2']
        channels = [x[0] for x in self.data.columns if x[0] not in exclude]
        if chans == 'all':
            chans = channels
        # set a dict to move between casefold and raw data cols
        cdict = {chan.casefold():chan for chan in channels}
        
        def dist(ref):
            """ calculate distance from reference channel """
            ref_coords = coords.loc[ref]
            rx = ref_coords['X']
            ry = ref_coords['Y']
            rz = ref_coords['Z']

            dist_dict = {}
            for chan in coords.index:
                # calculate distance
                cx, cy, cz = coords.loc[chan]['X'], coords.loc[chan]['Y'], coords.loc[chan]['Z']
                d = np.sqrt((cx-rx)**2 + (cy-ry)**2 + (cz-rz)**2)
                dist_dict[chan] = d

            # convert to series then sort
            dist_ser = pd.Series(dist_dict).sort_values()

            return dist_ser

        # load cartesian coords for all possible chans (10-5 montage)
        all_coords = pd.read_csv('cartesian_coords.txt')    
        # set all chans as lowercase & make index
        all_coords['Channel'] = [x.casefold() for x in all_coords.Channel]
        all_coords.set_index('Channel', inplace=True)

        # rename T3, T4, T5, T6 to T7, T8, P7, P8, to match change in 10-5 channel labels
        # ref: http://www.jichi.ac.jp/brainlab/download/TenFive.pdf
        rename = {'T3':'t7', 'T4':'t8', 'T5':'p7', 'T6':'p8'}
        channels_cf = [x.casefold() if x not in rename.keys() else rename[x] for x in channels]
        # grab cartesian coordinates
        coords = all_coords.loc[channels_cf]

        # undo renaming to revert to 10-20 conventions
        undo_rename = {val:key.casefold() for key, val in rename.items()}
        coords.rename(undo_rename, inplace=True)

        data_lap = pd.DataFrame(index=self.data.index)
        # calc nearest neighbors
        for chan in chans:
            c = chan.casefold()
            # get neighbor distance & set weights for 4 closest neighbors
            neighbors = dist(c)
            n_weights = 1 - neighbors[1:5]
            
            # calculate weighted neighbor data
            weighted_neighbors = pd.DataFrame(index=self.data.index)
            for neighbor, weight in n_weights.items():
                # calculated weighted values
                n_dat = self.data[cdict[neighbor]]*weight
                # add to weighted data dict
                weighted_neighbors[neighbor] = n_dat.values
            
            # get sum of weighted data
            weighted_sum = weighted_neighbors.sum(axis=1)
            weighted_sum.name='weighted_sum'

            # multiply ref chan by total weights
            c_dat = cdict[c] # match capitalization to column names
            c_weighted = self.data[c_dat]*n_weights.values.sum()

            # subtract weighted channel data from weighted neighbors
            lap = c_weighted.join(weighted_sum).diff(axis=1).weighted_sum
            lap.name = c
            data_lap[chan] = lap
            
            # set metadata
            self.metadata['laplacian_weights'][c] = n_weights
            
        # set columns to match non-montaged data
        data_lap.columns = pd.MultiIndex.from_arrays([chans, np.repeat(('Raw'), len(chans))],names=['Channel','datatype'])
        
        return data_lap


    ## Spindle Detection Methods ##

    # make attributes
    def spindle_attributes(self):
        """ Create attributes for spindle detection 

            Returns
            -------
            self.channels: list of str
                channel names
            self.spfiltEEG: pd.DataFrame
                filtered EEG data
            self.spRMS: pd.DataFrame
                root mean square of filtered data
            self.spRMSmavg: pd.DataFrame
                moving average of the root mean square of the filtered data
            self.spThresholds: pd.DataFrame
                spindle detection thresholds by channel
            self.spindle_events: dict
                spindle event detections
            self.spindle_rejects_t: dict
                spindle rejects based on time domain criteria. format {chan: [spindle reject indices,...]}
            self.spindle_rejects_f: dict
                spindle rejects based on frequency domain criteria. format {chan: [spindle reject indices,...]}

        """
        # check if channel list exists
        try:
            self.channels
        except AttributeError:
            # create if doesn't exist
            self.channels = [x[0] for x in self.data.columns]

        dfs =['spfiltEEG', 'spRMS', 'spRMSmavg'] # for > speed, don't store spRMS as an attribute
        [setattr(self, df, pd.DataFrame(index=self.data.index)) for df in dfs]
        self.spThresholds = pd.DataFrame(index=['Mean RMS', 'Low Threshold', 'High Threshold'])
        self.spindle_events = {}
        self.spindle_rejects_t = {}
        self.spindle_rejects_f = {}

    
    # step 1: make filter
    def make_butter_sp(self, wn, order):
        """ Make Butterworth bandpass filter [Parameters/Returns]
            wn: list of int (default: [8, 16])
                butterworth bandpass filter window
            order: int (default: 4)
                butterworth 1/2 filter order (applied forwards + backwards)
        """
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
    def get_spindles(self, i, min_sep):
        # vectorize data for detection looping
        lo, hi = self.spThresholds[i]['Low Threshold'], self.spThresholds[i]['High Threshold'] 
        mavg_varr, mavg_iarr = np.asarray(self.spRMSmavg[i]), np.asarray(self.spRMSmavg[i].index)
        
        # initialize spindle event list & set pointer to 0
        #self.spindle_events[i] = []
        spindle_events = []
        x=0

        while x < len(self.data):
            # if value crosses high threshold, start a fresh spindle
            if mavg_varr[x] >= hi:
                spindle = []

                # count backwards to find previous low threshold crossing
                for h in range(x, -1, -1):
                    # if a nan is encountered before the previous low crossing, break
                    if np.isnan(mavg_varr[h]):
                        break
                    elif mavg_varr[h] >= lo:
                        spindle.insert(0, mavg_iarr[h]) # add value to the beginning of the spindle
                    else:
                        break
       
                # count forwards to find next low threshold crossing
                for h in range(x+1, len(self.data), 1):
                    # if a nan is encountered before the next low crossing, break
                    if np.isnan(mavg_varr[h]):
                        break
                    # if above low threshold, add to current spindle
                    elif mavg_varr[h] >= lo and x < (len(self.data)-1):
                        spindle.append(mavg_iarr[h])
                    # if above low threshold and last value OR if nan, add to current spindle and add spindle to events list
                    elif (mavg_varr[h] >= lo and x == (len(self.data)-1)) or np.isnan(mavg_varr[h]): ## untested
                        spindle.append(mavg_iarr[h])
                        spindle_events.append(spindle)
                        #self.spindle_events[i].append(spindle)
                    # otherwise finish spindle & add to spindle events list
                    elif mavg_varr[h] < lo:
                        spindle_events.append(spindle)
                        #self.spindle_events[i].append(spindle)
                        break
        
                # advance the pointer to the end of the spindle
                x = h
            # if value doesn't cross high threshold, advance
            else:
                x += 1

        # combine spindles less than min_sep
        spindle_events_msep = []
        x = 0
        while x < len(spindle_events)-1:
            # if the following spindle is less than min_sep away
            if (spindle_events[x+1][0] - spindle_events[x][-1])/np.timedelta64(1, 's') < min_sep:
                # combine the two, append to list, and advance pointer by two
                spindle_comb = spindle_events[x] + spindle_events[x+1]
                spindle_events_msep.append(spindle_comb)
                x += 2
            else:
                # otherwise, append spindle to list, advance pointer by 1
                spindle_events_msep.append(spindle_events[x])
                # if this is the second-to-last spindle, also add final spindle to list (bc not combining)
                if x == (len(spindle_events)-2):
                    spindle_events_msep.append(spindle_events[x+1])
                x += 1
                
        self.spindle_events[i] = spindle_events_msep

    # step 7: apply time domain rejection criteria
    def reject_spins_t(self, min_chans_r, min_chans_d, duration):
        """ Reject spindles using time domain criteria:
                1. reject spindles that occur over fewer than 3 channels. 
                2. Apply min duration thresholding to spindles that occur over fewer than X channels. 
                3. Apply max duration thresholding to all remaining spindles
                [chans < min_chans_r = reject; min_chans_r < chans < min_chans_d = apply min duration threshold; x > max_dur = reject] 
            Parameters

            ----------
            min_chans_r: int
                minimum number of channels for spindles to occur accross concurrently to bypass
                automatic rejection
            min_chans_d: int
                minimum number of channels for spindles to occur across concurrently in order to 
                bypass duration criterion. performs best at 1/4 of total chans
            duration: list of float
                duration range (seconds) for spindle thresholding
            
            Returns
            -------
            modified self.spindle_events and self.spindle_rejects_t attributes
        """
        
        # convert duration from seconds to samples
        #sduration = [x*self.s_freq for x in duration]
        
        # make boolean mask for spindle presence
        bool_dict = {}
        for chan in self.spindle_events:
            if chan not in ['EOG_L', 'EOG_R', 'EKG', 'REF', 'FPZorEKG', 'A1', 'A2']:
                spins_flat = [time for spindle in self.spindle_events[chan] for time in spindle]
                bool_dict[chan] = np.isin(self.data.index.values, spins_flat)
        spin_bool = pd.DataFrame(bool_dict, index = self.data.index.values)
        spin_bool['chans_present'] = spin_bool.sum(axis=1)
        
        spindle_rejects_t = {}
        true_events = {}
        spindle_events = self.spindle_events
        # check individual spindles
        for chan in spindle_events:
            reject_idxs = []
            for e, spin in enumerate(spindle_events[chan]):
                spin_secs = (spin[-1] - spin[0])/np.timedelta64(1, 's')
                # reject if present over less than min_chans_r channels
                if not np.any(spin_bool['chans_present'].loc[spin] >= min_chans_r):
                    reject_idxs.append(e)

                # Apply min duration threshold if not present over more than minimum # of channels
                elif not np.any(spin_bool['chans_present'].loc[spin] >= min_chans_d):
                    # apply duration thresholding
                    if not duration[0] <= spin_secs <= duration[1]:
                        reject_idxs.append(e)
                # Apply max duration threshold to all spindles left (regardless of # of chans)
                else:
                    if spin_secs > duration[1]:
                        reject_idxs.append(e)
            # append spins to rejects
            spindle_rejects_t[chan] = [spindle_events[chan][idx] for idx in reject_idxs]
            true_events[chan] = [spin for e, spin in enumerate(spindle_events[chan]) if e not in reject_idxs]


        # replace values
        self.spindle_rejects_t = spindle_rejects_t
        self.spindle_events = true_events


    # set multiIndex
    def spMultiIndex(self, exclude):
        """ combine dataframes into a multiIndex dataframe"""
        channels = [x for x in self.channels if x not in exclude]

        # reset column levels
        self.spfiltEEG.columns = pd.MultiIndex.from_arrays([channels, np.repeat(('Filtered'), len(channels))],names=['Channel','datatype'])
        self.spRMS.columns = pd.MultiIndex.from_arrays([channels, np.repeat(('RMS'), len(channels))],names=['Channel','datatype'])
        self.spRMSmavg.columns = pd.MultiIndex.from_arrays([channels, np.repeat(('RMSmavg'), len(channels))],names=['Channel','datatype'])

        # list df vars for index specs
        dfs =[self.spfiltEEG, self.spRMS, self.spRMSmavg] # for > speed, don't store spinfilt_RMS as an attribute
        calcs = ['Filtered', 'RMS', 'RMSmavg']
        lvl0 = np.repeat(channels, len(calcs))
        lvl1 = calcs*len(channels)    
    
        # combine & custom sort
        self.spindle_calcs = pd.concat(dfs, axis=1).reindex(columns=[lvl0, lvl1])

    # step 8: create individual spindle dataframes                    
    def create_spindfs(self, zmethod, trough_dtype, buff, buffer_len):
        """ Create individual dataframes for individual spindles +/- a timedelta buffer 
            ** NOTE: buffer doesn't have spinso filter incorporated

            Parameters
            ----------
            zmethod: str (default: 'trough')
                method used to assign 0-center to spindles [options: 'trough', 'middle']. Trough assigns zero-center to
                the deepest negative trough. Middle assigns zero center to the midpoint in time.
            trough_dtype: str (default: 'spfilt')
                Which data to use for picking the most negative trough for centering [options: 'Raw', 'spfilt']
            buff: bool (default: False)
                    calculate spindle dataframes with buffer
            buffer_len: int
                length in seconds of buffer to calculate around 0-center of spindle
            self.spindle_events: dict
                dict of timestamps when spindles occur (created from self.detect_spindles())
            self.data: pd.DataFrame
                df containing raw EEG data

            Returns
            -------
            self.spindles: nested dict of dfs
                nested dict with spindle data by channel {channel: {spindle_num:spindle_data}}
            self.spindles_wbuffer: nested dict of dfs
                nested dict with spindle data w/ timedelta buffer by channel {channel: {spindle_num:spindle_data}}
        """

        ## create dict of dataframes for spindle analysis
        print('Creating individual spindle dataframes...')
        
        self.metadata['spindle_analysis']['zmethod'] = zmethod
        self.metadata['spindle_analysis']['trough_datatype'] = trough_dtype

        spindles = {}
        for chan in self.spindle_events.keys():
            spindles[chan] = {}
            for i, spin in enumerate(self.spindle_events[chan]):
                # create individual df for each spindle
                spin_data = self.data[chan]['Raw'].loc[self.spindle_events[chan][i]]
                spfilt_data = self.spfiltEEG[chan]['Filtered'].loc[self.spindle_events[chan][i]]
                # try:
                #     spsofilt_data = self.spsofiltEEG[chan]['Filtered'].loc[self.spindle_events[chan][i]]
                # # skip spsofilt if not yet calculated (if SO detections haven't been performed)
                # except AttributeError:
                #     pass
                
                # set new index so that each spindle is centered around zero
                if zmethod == 'middle':
                    # this method could use some work
                    half_length = len(spin)/2
                    t_id = np.linspace(-half_length, half_length, int(2*half_length//1))
                    # convert from samples to ms
                    id_ms = t_id * (1/self.metadata['analysis_info']['s_freq']*1000)
                elif zmethod == 'trough' and trough_dtype == 'Raw':
                    id_ms = (spin_data.index - spin_data.idxmin()).total_seconds()*1000
                elif zmethod == 'trough' and trough_dtype == 'spfilt':
                    id_ms = (spfilt_data.index - spfilt_data.idxmin()).total_seconds()*1000
                    
                # create new dataframe
                spindles[chan][i] = pd.DataFrame(index=id_ms)
                spindles[chan][i].index = [int(x) for x in spindles[chan][i].index]
                spindles[chan][i].index.name='id_ms'
                spindles[chan][i]['time'] = spin_data.index
                spindles[chan][i]['Raw'] = spin_data.values
                spindles[chan][i]['spfilt'] = spfilt_data.values
                try:
                    spindle[chan][i]['spsofilt'] = spsofilt_data.values
                # skip spsofilt if not yet calculated (if SO detections haven't been performed)
                except NameError:
                    pass
        
        self.spindles = spindles
        print('Spindle dataframes created. Spindle data stored in obj.spindles.')

        if buff:
            # now make buffered dataframes
            print(f'Creating spindle dataframes with {buffer_len}s buffer...')

            spindles_wbuffer = {}
            for chan in self.spindles.keys():
                spindles_wbuffer[chan] = {}
                for i in self.spindles[chan].keys():
                    # get +/- buffer length from zero-center of spindle
                    start = self.spindles[chan][i]['time'].loc[0] - pd.Timedelta(seconds=buffer_len)
                    end = self.spindles[chan][i]['time'].loc[0] + pd.Timedelta(seconds=buffer_len)
                    spin_buffer_data = self.data[chan]['Raw'].loc[start:end]

                    # assign the delta time index
                    id_ms = (spin_buffer_data.index - self.spindles[chan][i]['time'].loc[0]).total_seconds()*1000

                    # create new dataframe
                    spindles_wbuffer[chan][i] = pd.DataFrame(index=id_ms)
                    spindles_wbuffer[chan][i].index = [int(x) for x in spindles_wbuffer[chan][i].index]
                    spindles_wbuffer[chan][i].index.name='id_ms'
                    spindles_wbuffer[chan][i]['time'] = spin_buffer_data.index
                    spindles_wbuffer[chan][i]['Raw'] = spin_buffer_data.values
            
            self.spindles_wbuffer = spindles_wbuffer
            print('Spindle dataframes with buffer stored in obj.spindles_wbuffer.')

    def make_lowpass_zpad(self):
        """ Construct zero-padded spindle and spindle reject dictionaries for lowpass filtered data. 
            Needed for sleepyplot.spec_spins(). Called by self.lowpass_raw() and self.calc_spindle_psd_i

            Returns
            -------
            self.spindles_zpad_lowpass: nested dict
                dict of zero-padded spindle values from lowpass filtered data (format: {chan:{spin #: values}})
            self.spindles_zpad_rejects_lowpass: numpy.ndarray
                dict of zero-padded spindle frequency domain reject values from lowpass filtered data (format: {chan:{spin #: values}})
        """
        
        def create_zpad(spin, chan, x, zpad_len):
            """ Create the zero-padded spindle from raw data

                Parameters
                ----------
                spin: np.array
                    spindle mV values
                zpad_len: float
                    length to zero-pad the data to (in seconds) """
                # subtract mean to zero-center spindle for zero-padding
            sf = self.s_freq
            data = spin.values - np.mean(spin.values)
            zpad_samples=0
            zpad_seconds=0
            tx=0

            total_len = zpad_len*sf
            zpad_samples = total_len - len(data)
            zpad_seconds = zpad_samples/sf
            if zpad_samples > 0:
                padding = np.repeat(0, zpad_samples)
                data_pad = np.append(data, padding)
            else:
                spin_len = len(data)/sf
                print(f'Spindle {chan}:{x} length {spin_len} seconds longer than pad length {zpad_len}')
                data_pad = data
            # return the zero-padded spindle
            return data_pad
        
        # grab attributes
        spindles = self.spindles
        data_lowpass = self.data_lowpass
        spindle_rejects_f = self.spindle_rejects_f
        spindles_zpad_rejects = self.spindles_zpad_rejects
        
        # get length of zero-padding
        zpad_len = self.metadata['spindle_analysis']['zeropad_len_sec']
        spindles_zpad_lowpass = {}
        spindles_zpad_rejects_lowpass = {}
        
        for chan in spindles:
            spindles_zpad_lowpass[chan] = {}
            spindles_zpad_rejects_lowpass[chan] = {}
            
            # if there are spindles on that channel
            if len(spindles[chan]) > 0:
                # for each true spindle
                for x in spindles[chan]:
                    # get the time index & low-pass values
                    spin_idx = [np.datetime64(t) for t in spindles[chan][x].time.values]
                    spin = data_lowpass[chan].loc[spin_idx]
                    # make the zero-padding
                    data_pad = create_zpad(spin, chan, x, zpad_len)
                    # add to dict
                    spindles_zpad_lowpass[chan][x] = data_pad
            if len(spindle_rejects_f[chan]) > 0:
                reject_dict = {key:idxs for key, idxs in zip(spindles_zpad_rejects[chan].keys(), spindle_rejects_f[chan])}
                # for each rejected spindle
                for x, spin_idx in reject_dict.items():
                    # get the low-pass values
                    spin = data_lowpass[chan].loc[spin_idx]
                    # make the zero-padding
                    data_pad = create_zpad(spin, chan, x, zpad_len)
                    # add to dict
                    spindles_zpad_rejects_lowpass[chan][x] = data_pad
        
        # save as attributes 
        self.spindles_zpad_lowpass = spindles_zpad_lowpass
        self.spindles_zpad_rejects_lowpass = spindles_zpad_rejects_lowpass

    # step 9. calculate power spectrum for each spindle        
    def calc_spindle_psd_i(self, psd_bandwidth, zpad, zpad_len, pwr_prune, pwr_thres, spin_range, prune_range, min_peaks, pk_width_hz):
        """ Calculate multitaper power spectrum for individual spindles across all channels
            Option to threshold spindle detections based on a % power threshold. 

            Params
            ------
            psd_bandwidth: float
                frequency resolution in Hz
            zpad: bool (default: True)
                whether to zeropad the data (for increased spectral resolution)
            zpad_len: float
                length to zero-pad the data to (in seconds)
            pwr_prune: bool
                Whether to reject spindles using frequency-domain criterion: power in spindle range must = >X% of total power in prune range
                    Ex. spindle power must be >30% of total power between 4-25Hz
            pwr_thres: float
                % of power >4Hz that must be in the spindle range for a spindle to avoid rejection
            spin_range: list of int
                spindle frequency range (inclusive) to be used for spindle analysis and power thresholding
            prune_range: list of float
                frequency range for denominator of % power threshold calculation
            min_peaks: int (default: 1)
                minimum number of spectral peaks in the spindle range for a spindle to be accepted
            pk_width_hz: float (default: 0.5)
                minimum width (in Hz) for a peak to be considered a peak


            Returns
            -------
            self.spindles_zpad: dict
                zero-padded spindle values
            self.spindles_zpad_rejects: dict
                zero-padded spindle values for spins rejected in frequency domain
            self.spindle_psd_i: dict
                power spectra of individual spindles. format {channel: pd.Series} with index = frequencies and values = power (uV^2/Hz)
            self.spindle_psd_i_rejects: dict
                power spectra of individual spindles rejected in frequency domain. format {channel: pd.Series} with index = frequencies and values = power (uV^2/Hz)
            self.spindle_multitaper_calcs: dict of pd.DataFrame
                calculations used to calculated multitaper power spectral estimates for each spindle by channel
            self.spindle_multitaper_calcs_rejects: dict of pd.DataFrame
                calculations used to calculated multitaper power spectral estimates for spindles rejected in frequency domain
        """
        
        print('Calculating power spectra (this may take a few minutes)...')
        
        # update metadata
        analysis_metadata = {'psd_dtype': 'raw_individual', 'psd_method':'multitaper', 'psd_bandwidth':psd_bandwidth, 
                            'zeropad': zpad, 'zeropad_len_sec': zpad_len, 'pwr_prune': pwr_prune, 'pwr_thres': pwr_thres,
                            'prune_range': prune_range, 'min_peaks': min_peaks, 'pk_width_hz': pk_width_hz}
        self.metadata['spindle_analysis'].update(analysis_metadata)
        sf = self.metadata['analysis_info']['s_freq']
        spin_range = self.metadata['spindle_analysis']['spin_range']
        
        rmv_spins = {}
        spindle_rejects_f = {}

        spindles_zpad = {}
        spindles_zpad_rejects = {}
        
        spindle_psd = {}
        spindle_psd_rejects = {}
        
        spindle_multitaper_calcs = {}
        spindle_multitaper_calcs_rejects = {}
        
        for chan in self.spindles:
            spindles_zpad[chan] = {}
            spindles_zpad_rejects[chan] = {}
            
            spindle_psd[chan] = {}
            spindle_psd_rejects[chan] = {}
            spindle_rejects_f[chan] = []
            rmv_spins[chan] = []
            
            # set up multitaper_calcs df
            # waveform resolution is dependent on length of signal, regardless of zero-padding
            spindle_multitaper_calcs[chan] = pd.DataFrame(columns=['spin_samples', 'spin_seconds', 'zpad_samples', 'zpad_seconds', 'waveform_resoultion_Hz', 
                                                                   'psd_resolution_Hz', 'N_taper_len', 'W_bandwidth', 'K_tapers', f'perc_{prune_range[0]}-{prune_range[1]}Hzpwr_in_spin_range'])
            spindle_multitaper_calcs[chan].index.name = 'spindle_num'
            # for spindle rejects
            spindle_multitaper_calcs_rejects[chan] = pd.DataFrame(columns=['spin_samples', 'spin_seconds', 'zpad_samples', 'zpad_seconds', 'waveform_resoultion_Hz', 
                                                                   'psd_resolution_Hz', 'N_taper_len', 'W_bandwidth', 'K_tapers', f'perc_{prune_range[0]}-{prune_range[1]}Hzpwr_in_spin_range'])
            spindle_multitaper_calcs_rejects[chan].index.name = 'spindle_num'
            
            if len(self.spindles[chan]) > 0:
                for x in self.spindles[chan]:
                    # subtract mean to zero-center spindle for zero-padding
                    data = self.spindles[chan][x].Raw.values - np.mean(self.spindles[chan][x].Raw.values)
                    zpad_samples=0
                    zpad_seconds=0
                    tx=0
                    
                    # option to zero-pad the spindle
                    if zpad:
                        total_len = zpad_len*sf
                        zpad_samples = total_len - len(data)
                        zpad_seconds = zpad_samples/sf
                        if zpad_samples > 0:
                            padding = np.repeat(0, zpad_samples)
                            data_pad = np.append(data, padding)
                        else:
                            spin_len = len(data)/sf
                            print(f'Spindle {chan}:{x} length {spin_len} seconds longer than pad length {zpad_len}')
                            data_pad = data
                    
                    # or leave as-is
                    else:
                        data_pad = data
                        
                    # record PS params [K = 2NW-1]
                    spin_samples = len(data)
                    spin_seconds = len(data)/sf
                    waveform_res = 1/spin_seconds
                    psd_res = 1/(len(data_pad)/sf)
                    N_taper_len = len(data_pad)/sf
                    W_bandwidth = psd_bandwidth
                    K_tapers = int((2*N_taper_len*W_bandwidth)-1)

                    # calculate power spectrum
                    try:
                        pwr, freqs = psd_array_multitaper(data_pad, sf, adaptive=True, bandwidth=psd_bandwidth, fmax=25, 
                                                          normalization='full', verbose=0)
                    except ValueError:
                        print(f'Specified bandwidth too small for data length. Skipping spindle {chan}:{x}.')
                        continue
                    
                    # convert to series & add to dict
                    psd = pd.Series(pwr, index=freqs)
                    # set spindle status for rejection checks
                    status = True
                    
                    # check for minimum spectral peaks
                    # set minimum distance between peaks equal to psd_bandwidth 
                    samp_per_hz = len(psd)/(psd.index[-1]-psd.index[0])
                    bw_hz = self.metadata['spindle_analysis']['psd_bandwidth']
                    distance = samp_per_hz*bw_hz
                    # set minimum width in samples for a peak to be considered a peak
                    width = samp_per_hz*pk_width_hz
                    # get peaks
                    spindle_power = psd[(psd.index >= spin_range[0]) & (psd.index <= spin_range[1])]
                    p_idx, props = find_peaks(spindle_power, distance=distance, width=width, prominence=0.0)
                    # reject if < min peaks
                    if len(p_idx) < min_peaks:
                        # add to self.spindle_rejects_f
                        spindle_rejects_f[chan].append(self.spindle_events[chan][x])
                        # record params for removal from self.spindles & self.spindle_events after loop is complete
                        rmv_spins[chan].append(x)
                        # add to rejects psd dicts
                        spin_perc = 'not_calculated'
                        spindle_psd_rejects[chan][x] = psd
                        spindles_zpad_rejects[chan][x] = data_pad
                        spindle_multitaper_calcs_rejects[chan].loc[x] = [spin_samples, spin_seconds, zpad_samples, zpad_seconds, waveform_res, psd_res, N_taper_len, W_bandwidth, K_tapers, spin_perc]
                        # set status to false
                        status = False


                    # if spindle wasn't rejected by min_peaks criterion
                    if status == True:
                        # if not applying power % threshold
                        if pwr_prune == False:
                            # add to psd dicts
                            spindle_psd[chan][x] = psd
                            spindles_zpad[chan][x] = data_pad
                            spin_perc = 'not_calculated'

                        # otherwise apply power % threshold
                        elif pwr_prune:
                            # calculate total power > 4Hz
                            psd_subset = psd[(psd.index >= prune_range[0]) & (psd.index <= prune_range[1])]
                            # power in spindle range
                            psd_spins = psd[(psd.index >= spin_range[0]) & (psd.index <= spin_range[1])]
                            # percent of power > 4Hz in spindle range
                            spin_perc = int(psd_spins.sum()/psd_subset.sum()*100)

                            if spin_perc <= pwr_thres:
                                # add to self.spindle_rejects_f
                                spindle_rejects_f[chan].append(self.spindle_events[chan][x])
                                # record params for removal from self.spindles & self.spindle_events after loop is complete
                                rmv_spins[chan].append(x)
                                # add to rejects psd dicts
                                spindle_psd_rejects[chan][x] = psd
                                spindles_zpad_rejects[chan][x] = data_pad
                                spindle_multitaper_calcs_rejects[chan].loc[x] = [spin_samples, spin_seconds, zpad_samples, zpad_seconds, waveform_res, psd_res, N_taper_len, W_bandwidth, K_tapers, spin_perc]
                            else:
                                # add to psd dicts
                                spindle_psd[chan][x] = psd
                                spindles_zpad[chan][x] = data_pad
                                spindle_multitaper_calcs[chan].loc[x] = [spin_samples, spin_seconds, zpad_samples, zpad_seconds, waveform_res, psd_res, N_taper_len, W_bandwidth, K_tapers, spin_perc]


        # remove rejects from self.spindles & self.spindle_events
        for chan, spin_list in rmv_spins.items():
            # iterate backwards so that list indices for spindle_events don't change for subsequent items
            for spin in reversed(spin_list):
                del self.spindles[chan][spin]
                del self.spindle_events[chan][spin]

        
        self.spindles_zpad = spindles_zpad
        self.spindles_zpad_rejects = spindles_zpad_rejects
        self.spindle_multitaper_calcs = spindle_multitaper_calcs
        self.spindle_multitaper_calcs_rejects = spindle_multitaper_calcs_rejects
        self.spindle_psd_i = spindle_psd
        self.spindle_psd_i_rejects = spindle_psd_rejects
        self.spindle_rejects_f = spindle_rejects_f
        print('Spectra stored in obj.spindle_psd_i. Calculations stored in obj.spindle_multitaper_calcs. Zero-padded spindle data in obj.spindles_zpad.\n')    

        # calculate zero-padded lowpass filtered spindles if data has been lowpassed
        if hasattr(self, 'data_lowpass'):
            self.make_lowpass_zpad()
            print('Zero-padded lowpass filtered tabulated. Stored in obj.spindles_zpad_lowpass.')

        
        
    def detect_spindles(self, wn=[8, 16], order=4, sp_mw=0.2, loSD=0, hiSD=1.5, min_sep=0.2, duration=[0.5, 3.0], min_chans_r=3, min_chans_d=9,
                        zmethod='trough', trough_dtype='spfilt', buff=False, buffer_len=3, psd_bandwidth=1.0, zpad=True, zpad_len=3.0, pwr_prune=True,
                        pwr_thres=30, spin_range=[9, 16], prune_range=[4, 25], min_peaks=1, pk_width_hz=0.5, 
                        exclude=['EOG_L','EOG_R', 'EKG', 'REF', 'FPZorEKG', 'A1', 'A2', 'CZ', 'FP1', 'FP2', 'FZ', 'PZ']):  
        """ Detect spindles by channel
            
            Parameters
            ----------
            wn: list of int (default: [8, 16])
                butterworth bandpass filter window
            order: int (default: 4)
                butterworth 1/2 filter order (applied forwards + backwards)
            sp_mw: float (default: 0.2)
                moving window size for RMS & moving average calcs (seconds)
            loSD: float (default: 0)
                standard deviations above the average RMS that the spindle envelope must drop below to signify beginning/end of spindle
            hiSD: float (default: 1.5)
                standard deviations above the average RMS that the spindle envelope must exceed for a detection to be initiated
            min_sep: float (default: 0.1)
                minimum separation (in seconds) for spindles to be considered distinct, otherwise combine
            min_chans_r: int (default: 3)
                minimum number of channels for spindles to occur accross concurrently to bypass
                automatic rejection
            min_chans_d: int (default: 9)
                minimum number of channels for spindles to occur across concurrently in order to 
                bypass duration criterion. performs best at 1/4 of total chans
            duration: list of float
                duration range (seconds) for spindle thresholding
            zmethod: str (default: 'trough')
                method used to assign 0-center to spindles [options: 'trough', 'middle']. Trough assigns zero-center to
                the deepest negative trough. Middle assigns zero center to the midpoint in time.
            trough_dtype: str (default: 'spfilt')
                    Which data to use for picking the most negative trough for centering [options: 'Raw', 'spfilt']
            buff: bool (default: False)
                calculate spindle data dataframes with a delta time buffer around center of spindle
            buffer_len: int
                length in seconds of buffer to calculate around 0-center of spindle
            psd_bandwidth: float (default: 1.0)
                frequency resolution in Hz
            zpad: bool (default: False)
                whether to zeropad the data (for increased spectral resolution)
            zpad_len: float
                length to zero-pad the data to (in seconds)
            pwr_prune: bool (default: True)
                Whether to reject spindles using frequency-domain criterion: power in spindle range must = >X% of total power in prune range
                    Ex. spindle power must be >30% of total power between 4-25Hz
            pwr_thres: float (default: 30)
                % of power >4Hz that must be in the spindle range for a spindle to avoid rejection
            spin_range: list of int (default: [9, 16])
                spindle frequency range (inclusive) to be used for spindle analysis and power thresholding
            prune_range: list of float
                frequency range for denominator of % power threshold calculation
            min_peaks: int (default: 1)
                minimum number of spectral peaks in the spindle range for a spindle to be accepted
            pk_width_hz: float (default: 0.5)
                minimum width (in Hz) for a peak to be considered a peak
            exlcude: list of string or None (default: ['EOG_L','EOG_R', 'EKG', 'REF', 'FPZorEKG', 'A1', 'A2'])
                list of channels to exclude from analysis. NOTE: capital 'FZ' & 'PZ' are MOBEE 32 nan channels. 
                Sentence case 'Fz' and 'Pz' are valid FS128 channels.

            Returns
            -------
            ## incomplete ##

            self.spindle_psd_i: nested dict
                power spectra for individual spindles by channel (Only if psd_type == 'i')
                format {channel: {spindle: pd.Series}} with index = frequencies and values = power (uV^2/Hz)

        """

        self.metadata['spindle_analysis'] = {'sp_filtwindow': wn, 'sp_filtorder_half': order, 
            'sp_RMSmw': sp_mw, 'sp_loSD': loSD, 'sp_hiSD': hiSD, 'min_sep': min_sep, 'sp_duration': duration,
            'sp_minchans_toskipautoreject': min_chans_r, 'sp_minchans_toskipduration': min_chans_d, 'spin_range':spin_range}

        #self.s_freq = self.metadata['analysis_info']['s_freq']
    
        # set attributes
        self.spindle_attributes()
        # Make filter
        self.make_butter_sp(wn, order)

        print('Detecting spindles...')
        # loop through channels (all channels for plotting ease)
        for i in self.channels:
            if i not in exclude:
                #print(f'Detecting spindles on {i}...')
                # Filter
                self.spfilt(i)
                # Calculate RMS & smooth
                self.rms_smooth(i, sp_mw)
                # Set detection thresholds
                self.set_thres(i)
                # Detect spindles
                self.get_spindles(i, min_sep)

        # combine dataframes
        print('Combining dataframes...')
        self.spMultiIndex(exclude)
        
        # Apply time-domain rejection criteria
        print('Pruning spindle detections...')
        self.reject_spins_t(min_chans_r, min_chans_d, duration)
        # create individual datframes for each spindle
        self.create_spindfs(zmethod, trough_dtype, buff, buffer_len)
        # calculate power for individual spindles & prune in frequency domain
        self.calc_spindle_psd_i(psd_bandwidth, zpad, zpad_len, pwr_prune, pwr_thres, spin_range, prune_range, min_peaks, pk_width_hz)

        print('Spindle detection complete.')
        
        print('done.\n')

    

    def calc_spindle_means(self):
        """ Calculate mean, std, and sem at each timedelta from negative spindle peak per channel 

            Returns
            -------
            self.spindle_means: nested dict
                dictionary of raw and filtered spindle means by channel 
                format: {'Raw':{channel:pd.DataFrame}}, 'spfilt':{channel:pd.DataFrame}}
        """

        print('Aligning spindles...')
        # align spindles accoridng to timedelta & combine into single dataframe
        spindle_aggregates = {}
        datatypes = ['Raw', 'spfilt']
        for chan in self.spindles.keys():
            # only use channels that have spindles
            if self.spindles[chan]:
                spindle_aggregates[chan] = {}
                for datatype in datatypes:
                    # set the base df
                    first_spin = list(self.spindles[chan].keys())[0]
                    first_spin_colname = f'spin_{first_spin}'
                    agg_df = pd.DataFrame(self.spindles[chan][first_spin][datatype])
                    agg_df = agg_df.rename(columns={datatype:first_spin_colname})
                    rsuffix = list(self.spindles[chan].keys())[1:]
                    # join on the index for each spindle
                    agg_df = agg_df.join([self.spindles[chan][x][datatype].rename('spin_'+str(x)) for x in rsuffix], how='outer')
                    spindle_aggregates[chan][datatype] = agg_df
            
        print('Calculating spindle statistics...')
        # create a new multiindex dataframe for calculations
        spindle_means = {}
        calcs = ['count', 'mean', 'std' ,'sem']
        tuples = [(chan, calc) for chan in spindle_aggregates.keys() for calc in calcs]
        columns = pd.MultiIndex.from_tuples(tuples, names=['channel', 'calc'])
        for datatype in datatypes:
            spindle_means[datatype] = pd.DataFrame(columns=columns)
            # fill the dataframe
            for chan in spindle_aggregates.keys():
                spindle_means[datatype][(chan, 'count')] = spindle_aggregates[chan][datatype].notna().sum(axis=1)
                spindle_means[datatype][(chan, 'mean')] = spindle_aggregates[chan][datatype].mean(axis=1)
                spindle_means[datatype][(chan, 'std')] = spindle_aggregates[chan][datatype].std(axis=1)
                spindle_means[datatype][(chan, 'sem')] = spindle_aggregates[chan][datatype].sem(axis=1)
            
        self.spindle_aggregates = spindle_aggregates
        self.spindle_means = spindle_means
        print('Done. Spindles aggregated by channel in obj.spindle_aggregates dict. Spindle statisics stored in obj.spindle_means dataframe.\n')


    def calc_spindle_buffer_means(self):
        """ Calculate mean, std, and sem at each timedelta from negative spindle peak per channel 
            NOTE: This needs to be updated to include datatype parameter to stay aligned with calc_spin_means
             Also fix the join command for speed (see above)
        """
        
        print('Aligning spindles...')
        # align spindles accoridng to timedelta & combine into single dataframe
        spindle_buffer_aggregates = {}
        for chan in self.spindles.keys():
            # only use channels that have spindles
            if self.spindles_wbuffer[chan]:
                # set the base df
                agg_df = pd.DataFrame(self.spindles_wbuffer[chan][0]['Raw'])
                rsuffix = list(range(1, len(self.spindles_wbuffer[chan])))
                # join on the index for each spindle
                for x in range(1, len(self.spindles_wbuffer[chan])):
                    mean_df = agg_df.join(self.spindles_wbuffer[chan][x]['Raw'], how='outer', rsuffix=rsuffix[x-1])
                spindle_buffer_aggregates[chan] = mean_df
            
        print('Calculating statistics...')
        # create a new multiindex dataframe for calculations
        calcs = ['mean', 'std' ,'sem']
        tuples = [(chan, calc) for chan in spindle_buffer_aggregates.keys() for calc in calcs]
        columns = pd.MultiIndex.from_tuples(tuples, names=['channel', 'calc'])
        spindle_buffer_means = pd.DataFrame(columns=columns)
        
        # fill the dataframe
        for chan in spindle_buffer_aggregates.keys():
            spindle_buffer_means[(chan, 'mean')] = spindle_buffer_aggregates[chan].mean(axis=1)
            spindle_buffer_means[(chan, 'std')] = spindle_buffer_aggregates[chan].std(axis=1)
            spindle_buffer_means[(chan, 'sem')] = spindle_buffer_aggregates[chan].sem(axis=1)
            
        self.spindle_buffer_aggregates = spindle_buffer_aggregates
        self.spindle_buffer_means = spindle_buffer_means
        print('Done. Spindles aggregated by channel in obj.spindle_buffer_aggregates dict. Spindle statisics stored in obj.spindle_buffer_means dataframe.')


    def calc_spin_tstats(self):
        """ calculate time-domain spindle feature statistics 
            
            Returns
            -------
            self.spindle_tstats: pd.DataFrame
                MultiIndex dataframe with calculated spindle time statistics
        """
        
        spin_range = self.metadata['spindle_analysis']['spin_range']

        print('Calculating spindle time-domain statistics...')
        
        # create multi-index dataframe
        lvl1 = ['Count', 'Duration', 'Duration', 'Amplitude_raw', 'Amplitude_raw', 'Amplitude_spfilt', 'Amplitude_spfilt', 'Density', 'ISI', 'ISI']
        lvl2 = ['total', 'mean', 'sd', 'rms', 'sd', 'rms', 'sd', 'spin_per_min', 'mean', 'sd']
        columns = pd.MultiIndex.from_arrays([lvl1, lvl2])
        spindle_stats = pd.DataFrame(columns=columns)
        
        #exclude non-EEG channels
        exclude = ['EOG_L', 'EOG_R', 'EKG', 'REF', 'FPZorEKG', 'A1', 'A2']

        # fill dataframe
        for chan in self.spindles:
            if chan not in exclude:
                # calculate spindle count
                count = len(self.spindles[chan])
                
                if count == 0:
                    spindle_stats.loc[chan] = [count, None, None, None, None, None, None, None, None, None]
                
                else:
                    # calculate spindle duration
                    durations = np.array([(self.spindles[chan][spin].time.iloc[-1] - self.spindles[chan][spin].time.iloc[0]).total_seconds() for spin in self.spindles[chan]])
                    duration_mean = durations.mean()
                    duration_sd = durations.std()

                    # calculate amplitude
                    amplitudes_raw = np.concatenate([self.spindles[chan][x].Raw.values for x in self.spindles[chan]])
                    amp_rms_raw = np.sqrt(np.array([x**2 for x in amplitudes_raw]).mean())
                    amp_sd_raw = amplitudes_raw.std()
                    amplitudes_spfilt = np.concatenate([self.spindles[chan][x].spfilt.values for x in self.spindles[chan]])
                    amp_rms_spfilt = np.sqrt(np.array([x**2 for x in amplitudes_spfilt]).mean())
                    amp_sd_spfilt = amplitudes_spfilt.std()

                    # calculate density
                    #density = count/((self.data.index[-1] - self.data.index[0]).total_seconds()/60)
                    data_notnan = self.data[chan][self.data[chan]['Raw'].isna() == False]
                    minutes = (len(data_notnan)/self.s_freq)/60
                    density = count/(minutes)


                    # calculate inter-spindle-interval (ISI)
                    if len(self.spindles[chan]) > 1:
                        spin_keys = list(self.spindles[chan].keys())
                        # make a list of tuples of ISI start and end timestamps
                        isi_ranges = [(self.spindles[chan][spin_keys[x]].time.iloc[-1], self.spindles[chan][spin_keys[x+1]].time.iloc[0]) for x in range(len(spin_keys)) if x < len(spin_keys)-1]
                        # keep the ISI tuple only if there are no NaNs in the data (no missing data)
                        notNaN_isi_ranges = [i for i in isi_ranges if np.any(np.isnan(self.data[chan].loc[i[0]:i[1]])) == False]
                        # calculate the total seconds for each tuple
                        isi_arr = np.array([(isi[1]-isi[0]).total_seconds() for isi in notNaN_isi_ranges])
                        isi_mean = isi_arr.mean()
                        isi_sd = isi_arr.std()
                    else:
                        isi_mean = None
                        isi_sd = None

                    spindle_stats.loc[chan] = [count, duration_mean, duration_sd, amp_rms_raw, amp_sd_raw, amp_rms_spfilt, amp_sd_spfilt, density, isi_mean, isi_sd]
                    # spindle_stats.loc[chan] = [count, duration_mean, duration_sd, amp_rms_raw, amp_sd_raw, amp_rms_spfilt, amp_sd_spfilt, density, isi_mean, isi_sd, center_freq, total_pwr]

        self.spindle_tstats = spindle_stats   
        
        print('Spindle time stats stored in obj.spindle_tstats.\n')


    def calc_spindle_psd_concat(self, psd_bandwidth):
        """ Calculate multitaper power spectrum of concated spindles for each channel

            Params
            ------
            psd_bandwidth: float
                frequency resolution in Hz

            Returns
            -------
            self.spindle_psd_concat: dict
                format {channel: pd.Series} with index = frequencies and values = power (uV^2/Hz)
            self.spindle_multitaper_calcs_concat: pd.DataFrame
                calculations used to calculated concatenated multitaper power spectral estimates for each channel
        """
        
        print('Calculating power spectra (this may take a few minutes)...')
        self.metadata['spindle_analysis']['psd_dtype'] = 'raw_concat'
        self.metadata['spindle_analysis']['psd_method'] = 'multitaper'
        self.metadata['spindle_analysis']['psd_bandwidth'] = psd_bandwidth
        sf = self.metadata['analysis_info']['s_freq']
        
        spindle_psd = {}
        spindle_multitaper_calcs_concat = pd.DataFrame(index=['data_len', 'N', 'W', 'NW', 'K'])
        for chan in self.spindles:
            #print(f'Calculating spectra for {chan}...')
            if len(self.spindles[chan]) > 0:
                # concatenate spindles
                spindles = [self.spindles[chan][x].Raw.values for x in self.spindles[chan]]
                data = np.concatenate(spindles)
                
                # calculate power spectrum
                try:
                    pwr, freqs = psd_array_multitaper(data, sf, adaptive=True, bandwidth=psd_bandwidth, fmax=25, 
                                                      normalization='full', verbose=0)
                except ValueError as e:
                    print(e)
                    min_bw = float((str(e)).split(' ')[-1])
                    # round up to the nearest hundredth bc using exact # can still throw occasional errors
                    psd_bandwidth = math.ceil(min_bw*100)/100
                    print(f'Setting psd_bandwidth to {psd_bandwidth}')
                    pwr, freqs = psd_array_multitaper(data, sf, adaptive=True, bandwidth=psd_bandwidth, fmax=25, 
                                                      normalization='full', verbose=0)

                # convert to series & add to dict
                psd = pd.Series(pwr, index=freqs)
                spindle_psd[chan] = psd

                # record PS params [K = 2NW-1]
                N = len(data)/sf
                W = psd_bandwidth
                K = int((2*N*W)-1)
                spindle_multitaper_calcs_concat[chan] = [len(data), N, W, N*W, K] 
        
        self.spindle_multitaper_calcs_concat = spindle_multitaper_calcs_concat
        self.spindle_psd_concat = spindle_psd
        print('Done. Spectra stored in obj.spindle_psd_concat. Calculations stored in obj.spindle_multitaper_calcs_concat.\n')

    def calc_gottselig_norm(self, norm_range):
        """ calculated normalized spindle power on EEG channels (from Gottselig et al., 2002). works with
            calc_spindle_psd_concat.
        
            TO DO: change p0 value if optimize warning
        
            Parameters
            ----------
            norm_range: list of tuple
                frequency ranges for gottselig normalization
            
            Returns
            -------
            self.spindle_psd_concat_norm: nested dict
                format {chan: pd.Series(normalized power, index=frequency)}
        
        """
        
        print('Calculating Gottselig normalization...')

        def exponential_func(x, a, b, c):
            return a*np.exp(-b*x)+c
        
        self.metadata['spindle_analysis']['gottselig_range'] = norm_range
        exclude = ['EOG_L', 'EOG_R', 'EKG', 'REF', 'FPZorEKG', 'A1', 'A2']
        
        spindle_psd_norm = {}
        chans_norm_failed = []
        for chan in self.spindle_psd_concat:
            if chan not in exclude:
                spindle_psd_norm[chan] = {}

                # specify data to be fit (only data in norm range)
                incl_freqs = np.logical_or(((self.spindle_psd_concat[chan].index >= norm_range[0][0]) & (self.spindle_psd_concat[chan].index <= norm_range[0][1])),
                                            ((self.spindle_psd_concat[chan].index >= norm_range[1][0]) & (self.spindle_psd_concat[chan].index <= norm_range[1][1])))
                pwr_fit = self.spindle_psd_concat[chan][incl_freqs] 

                # set x and y values (convert y to dB)
                x_pwr_fit = pwr_fit.index
                y_pwr_fit = 10 * np.log10(pwr_fit.values)

                # fit exponential -- try second fit line if first throws infinite covariance
                try:
                    popt, pcov = curve_fit(exponential_func, xdata=x_pwr_fit, ydata=y_pwr_fit, p0=(1, 0, 1))
                except (OptimizeWarning, RuntimeError, TypeError):
                    try:
                        popt, pcov = curve_fit(exponential_func, xdata=x_pwr_fit, ydata=y_pwr_fit, p0=(1, 1e-6, 1))
                    except (OptimizeWarning, RuntimeError, TypeError) as e:
                        popt = np.full(3, np.nan)
                        chans_norm_failed.append(chan)
                        print(f'scipy.optimize.curvefit encountered error "{e}" on channel {chan}. Normalization skipped for this channel.')
                        pass

                xx = self.spindle_psd_concat[chan].index
                yy = exponential_func(xx, *popt)

                # subtract the fit line
                psd_norm = pd.Series(10*np.log10(self.spindle_psd_concat[chan].values) - yy, index=self.spindle_psd_concat[chan].index)

                # save the values
                spindle_psd_norm[chan]['normed_pwr'] = psd_norm
                spindle_psd_norm[chan]['values_to_fit'] = pd.Series(y_pwr_fit, index=x_pwr_fit)
                spindle_psd_norm[chan]['exp_fit_line'] = pd.Series(yy, index=xx) 

        self.spindle_psd_concat_norm = spindle_psd_norm
        self.metadata['spindle_analysis']['chans_concat_norm_failed'] = chans_norm_failed
        print('Gottselig normalization data stored in obj.spindle_psd_concat_norm.\n')


    def calc_spin_stats_i(self):
        """ Calculate statistics for individual spindles """

        print('\nCalculating individual spindle statistics...')

        # pull minimum width (in Hz) for a peak to be considered a peak
        pk_width_hz = self.metadata['spindle_analysis']['pk_width_hz']
        
        # create list of rows to be converted into dataframe
        stats_i_rows = []
        
        # create column names for dict keys to build rows
        cols = ['AP', 'RL', 'chan', 'spin', 'dur_ms', 'amp_raw_rms', 'amp_spfilt_rms', 
            'dominant_freq_Hz', 'total_peaks', 'peak_freqs_Hz', 'peak_ratios', 'peak2_freq', 'peak2_ratio', 'total_pwr_ms2']

        # assign anterior-posterior characters
        a_chars = ['f']
        p_chars = ['p', 'o', 't']
        c_chans = ['a1', 't9', 't3', 'c5', 'c3', 'c1', 'cz', 'c2', 'c4', 'c6', 't4', 't10', 'a2']
        
        # exclude non-EEG channels
        exclude = ['EKG', 'EOG_L', 'EOG_R', 'REF', 'FPZorEKG', 'A1', 'A2']
        # loop through all channels
        for chan in self.spindles.keys():
            if chan not in exclude:
                
                # assign anterior-posterior
                if chan.casefold() in c_chans:
                    ap = 'C' 
                elif any((c.casefold() in a_chars) for c in chan):
                    ap = 'A'
                elif any((c.casefold() in p_chars) for c in chan):
                    ap = 'P'
                # assign RL
                if chan[-1].casefold() == 'z':
                    rl = 'C'
                elif int(chan[-1]) % 2 == 0:
                    rl = 'R'
                else:
                    rl = 'L'

                # analyze individual spindles
                for spin in self.spindles[chan]:
                    # set individual spindle data
                    spindle = self.spindles[chan][spin]

                    # get time stats
                    dur_ms = np.abs(spindle.index[0]) + spindle.index[-1]
                    amp_raw_rms = np.sqrt(np.mean(spindle.Raw.values**2))
                    amp_spfilt_rms = np.sqrt(np.mean(spindle.spfilt.values**2))

                    # get frequency stats
                    psd_i = self.spindle_psd_i[chan][spin]
                    spin_range = self.metadata['spindle_analysis']['spin_range']
                    spindle_power = psd_i[(psd_i.index >= spin_range[0]) & (psd_i.index <= spin_range[1])]
                    total_pwr = spindle_power.sum()


                    # set minimum distance between peaks equal to psd_bandwidth 
                    samp_per_hz = len(psd_i)/(psd_i.index[-1]-psd_i.index[0])
                    bw_hz = self.metadata['spindle_analysis']['psd_bandwidth']
                    distance = samp_per_hz*bw_hz
                    # set minimum width in samples for a peak to be considered a peak
                    width = samp_per_hz*pk_width_hz
                    # get peaks
                    p_idx, props = find_peaks(spindle_power, distance=distance, width=width, prominence=0.0)
                    peaks = spindle_power.iloc[p_idx]
                    # get dominant frequency [major peak] (to 2 decimal points)
                    dominant_freq = round(peaks.idxmax(), 2)
                    total_peaks = len(peaks)
                    peak_freqs_hz = [round(idx, 2) for idx in peaks.index]
                    # ratio of peak amplitudes as a fraction of the dominant amplitude
                    peak_ratios = {np.round(key, 1):np.round((val/peaks.values.max()), 3) for key, val in peaks.items()}
                    # get 2nd most prominent peak as fraction of dominant peak power
                    if len(peak_ratios) > 1:
                        ratios_sorted = sorted(peak_ratios.items(), key=lambda x: x[1], reverse=True)
                        peak2_freq, peak2_ratio  = ratios_sorted[1][0], ratios_sorted[1][1]
                    else:
                        peak2_freq, peak2_ratio = None, None

                    vals = [ap, rl, chan, spin, dur_ms, amp_raw_rms, amp_spfilt_rms, dominant_freq, total_peaks, peak_freqs_hz, 
                            peak_ratios, peak2_freq, peak2_ratio, total_pwr]
                    row = {c:v for c, v in zip(cols, vals)}

                    # add row to stats_i list
                    stats_i_rows.append(row)
            
        # convert row list into dataframe
        stats_i_df = pd.DataFrame(stats_i_rows)
        self.spindle_stats_i = stats_i_df
        print('Done. Stats stored in obj.spindle_stats_i.')


    def calc_spin_fstats_concat(self):
        """ Calculate frequency statistics on concatenated spindles
            To do: determine statistics to calculate for individual spindles

            To calculate peaks on concatenated data, the spectrum is:
                1. smoothed with an RMS window length equal to the psd_bandwidth
                2. peaks must have a minimum horizontal distance equal to psd_bandwidth
                3. peaks must have a minimum frequency width (set by width_hz)

        """

        print('Calculating concatenated frequency-domain statistics...')

        # skip if no spindles detected
        if len(self.spindle_psd_concat) == 0:
            print('No spindles detected. Done.')
        else:
            spin_range = self.metadata['spindle_analysis']['spin_range']
            # pull minimum width (in Hz) for a peak to be considered a peak
            pk_width_hz = self.metadata['spindle_analysis']['pk_width_hz']
             #exclude non-EEG channels
            exclude = ['EOG_L', 'EOG_R', 'EKG', 'REF', 'FPZorEKG', 'A1', 'A2']

            # create fstats dataframe & peaks dict
            cols = ['dominant_freq_Hz', 'total_pwr_dB', 'total_peaks', 'peak_freqs_Hz', 'peak_ratios']
            spindle_fstats = pd.DataFrame(columns=cols)
            psd_concat_norm_peaks = {}

            # set the parameters for picking peaks
            # set minimum distance between adjacent peaks equal to spectral resolution
            psd = self.spindle_psd_concat[list(self.spindle_psd_concat.keys())[0]]
            samp_per_hz = len(psd)/(psd.index[-1]-psd.index[0])
            bw_hz = self.metadata['spindle_analysis']['psd_bandwidth']
            distance = samp_per_hz*bw_hz
            # distance must be >= 1
            if distance < 1:
                distance = 1
            # set minimum width in samples for a peak to be considered a peak
            width = samp_per_hz*pk_width_hz
            # set the moving window sample length equal to the psd bandwidth
            mw_samples = int(distance)

            # calculate stats for each channel
            for chan in self.spindle_psd_concat.keys():
                if chan not in exclude:
                    # smooth the signal
                    datsq = np.power(self.spindle_psd_concat_norm[chan]['normed_pwr'], 2)
                    window = np.ones(mw_samples)/float(mw_samples)
                    rms = np.sqrt(np.convolve(datsq, window, 'same'))
                    smoothed_data = pd.Series(rms, index=self.spindle_psd_concat[chan].index)
                    smoothed_spindle_power = smoothed_data[(smoothed_data.index >= spin_range[0]) & (smoothed_data.index <= spin_range[1])]

                    #calculate total spindle power (to 2 decimal points)
                    total_pwr = round(smoothed_spindle_power.sum(), 2)

                    # get peaks
                    p_idx, props = find_peaks(smoothed_spindle_power, distance=distance, width=width, prominence=0.0)
                    peaks = smoothed_spindle_power.iloc[p_idx]
                    # set dominant frequency to major peak
                    total_peaks = len(peaks)
                    if total_peaks > 0:
                        dominant_freq = round(peaks.idxmax(), 2)
                        peak_freqs_hz = [round(idx, 2) for idx in peaks.index]
                        # ratio of peak amplitudes as a fraction of the dominant amplitude
                        peak_ratios = {np.round(key, 1):np.round((val/peaks.values.max()), 2) for key, val in peaks.items()}
                    else:
                        dominant_freq, peak_freqs_hz, peak_ratios = None, None, None

                                
                    # add row to dataframe
                    spindle_fstats.loc[chan] = [dominant_freq, total_pwr, total_peaks, peak_freqs_hz, peak_ratios]

                    # save values to peaks dict
                    psd_concat_norm_peaks[chan] = {'smoothed_data':smoothed_data, 'peaks':peaks, 'props':props}

            self.psd_concat_norm_peaks = psd_concat_norm_peaks
            self.spindle_fstats_concat = spindle_fstats
            print('Done. Concat frequency stats stored in obj.spindle_fstats_concat.')


    def analyze_spindles(self, psd_type='concat', psd_bandwidth=1.0, zpad=True, zpad_len=3.0, norm_range=[(4,6), (18, 25)], buff=False, 
                        gottselig=True, fstats_concat=True):
        """ 
            Starting code for spindle statistics/visualizations 

            Parameters
            ----------
            psd_type: str (default: 'concat')
                What data to use for psd calculations [Options: 'i' (individual spindles), 'concat' (spindles concatenated by channel)]
                **this parameter is redundant now that 'i' is auto-calculated in spindle detection step -- can be hard-coded to 'concat'
            psd_bandwidth: float
                frequency bandwidth for power spectra calculations (Hz)
            zpad: bool (default: False)
                    whether to zeropad the spindle data (for increased spectral resolution)
            zpad_len: float
                length to zero-pad spindles to (in seconds)
            norm_range: list of tuple
                frequency ranges for gottselig normalization
            buff: bool (default: False)
                whether to calculate means with a time buffer around spindle center
            gottselig: bool (default: False)
                whether to calculate gottselig normalization on concatenated spectrum
            fstats_concat: bool (default: True)
                whether to calculate concatenated spindle frequency statistics
        
            Returns
            -------
            self.spindles: nested dict of dfs
                nested dict with spindle data by channel {channel: {spindle_num:spindle_data}}
            self.spindles_wbuffer: nested dict of dfs
                nested dict with spindle data w/ timedelta buffer by channel {channel: {spindle_num:spindle_data}}
            self.spindle_psd_concat: dict
                power spectra for concatenated spindles by channel (Only if psd_type == 'concat')
                format {channel: pd.Series} with index = frequencies and values = power (uV^2/Hz)
            self.spindle_psd_concat_norm: nested dict (Only if psd_type == 'concat')
                format {chan: pd.Series(normalized power, index=frequency)}
            self.spindle_psd_i: nested dict
                power spectra for individual spindles by channel (Only if psd_type == 'i')
                format {channel: {spindle: pd.Series}} with index = frequencies and values = power (uV^2/Hz)
            self.spindle_multitaper_calcs: pd.DataFrame
                calculations used to calculated multitaper power spectral estimates for each channel
            self.spindle_multitaper_calcs_concat: pd.DataFrame
                calculations used to calculated concatenated multitaper power spectral estimates for each channel
            self.spindle_features: pd.DataFrame
                MultiIndex dataframe with calculated spindle statistics
        """
        


        # calculate spindle & spindle buffer means
        self.calc_spindle_means()
        if buff:
            self.calc_spindle_buffer_means()

        # run time-domain spindle statistics by channel
        self.calc_spin_tstats()

        # calculate power spectra
        if psd_type == 'concat': # this is redundant (should always be 'concat')
            # calc psd on concated spindles
            self.calc_spindle_psd_concat(psd_bandwidth)

            if gottselig:
                # normalize power spectra for quantification
                self.calc_gottselig_norm(norm_range)
        
        # calculate individual spindle stats
        self.calc_spin_stats_i()
        # calculate frequency stats
        if fstats_concat:
            self.calc_spin_fstats_concat()



    def export_spindles(self, export_dir, raw=True, psd_concat=True, psd_i=True, stats=True):
        """ Export spindle analyses
        
            Parameters
            ----------
            export_dir: str
                path to export directory
            raw: bool (default: True)
                export raw EEG spindle detection tracings
            psd_concat: bool (default: True)
                export psd calculations and series for concatenated spindles
            psd_i: bool (default: True)
                export psd calculations and series for individual spindles
            stats: bool (default: True)
                export spindle time and frequency statistics
            
            Returns
            -------
            *To be completed
            
        """
        
        print(f'Spindle export directory: {export_dir}\n')    
        # make export directory if doesn't exit
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
        
        # set base for savename
        fname = self.metadata['file_info']['fname'].split('.')[0]
        
        # dump metadata into main directory
        filename = f'{fname}_spindle_metadata.txt'
        savename = os.path.join(export_dir, filename)
        with open(savename, 'w') as f:
            json.dump(self.metadata, f, indent=4)
        
        # export raw spindle tracings
        if raw:
            raw_dir = export_dir + '/spindle_tracings'
            if not os.path.exists(raw_dir):
                os.makedirs(raw_dir)
            
            # export spindle tracings for each channel
            print('Exporting spindle tracings...')
            for chan in self.spindles.keys():
                filename = f'{fname}_{chan}_spindle_tracings.txt'
                savename = os.path.join(raw_dir, filename)
                ## use json dict dump to save space
                spin_export = {}
                for spin, series in self.spindles[chan].items():
                    # convert time from datetime to str
                    s = series.astype({'time': str})
                    spin_export[spin] = s.to_dict()
                with open(savename, 'w') as f:
                    json.dump(spin_export, f, indent=4)
                ## for exporting into an excel workbook instead
    #           writer = pd.ExcelWriter(savename, engine='xlsxwriter')
    #           for spin in self.spindles[chan].keys():
    #               tab = f{'Spindle_{spin}'}
    #               self.spindles[chan][spin].to_excel(writer, sheet_name=tab)
            
            # export spindle aggregates
            print('Exporting spindle aggregates...')
            filename = f'{fname}_spindle_aggregates.xlsx'
            savename = os.path.join(raw_dir, filename)
            writer = pd.ExcelWriter(savename, engine='xlsxwriter')
            for chan in self.spindle_aggregates.keys():
                for dtype in self.spindle_aggregates[chan].keys():
                    tab = '_'.join([chan, dtype])
                    self.spindle_aggregates[chan][dtype].to_excel(writer, sheet_name=tab)
            writer.save()

            # export spindle means
            print('Exporting spindle means...\n')
            for dtype in self.spindle_means.keys():
                filename = f'{fname}_spindle_means_{dtype}.csv'
                savename = os.path.join(raw_dir, filename)
                self.spindle_means[dtype].to_csv(savename)


        # export concated spectra
        if psd_concat:
            # set subdirectory
            psd_concat_dir = export_dir + '/psd_concat'
            if not os.path.exists(psd_concat_dir):
                os.makedirs(psd_concat_dir)
            # export multitaper calcs (concat)
            print('Exporting concatenated spindle spectra calcs...')
            filename = f'{fname}_spindle_mt_calcs_concat.csv'
            savename = os.path.join(psd_concat_dir, filename)
            self.spindle_multitaper_calcs_concat.to_csv(savename)
            # export psd series
            # convert series to dicts for json dump
            psd_export = {}
            for name, series in self.spindle_psd_concat.items():
                psd_export[name] = series.to_dict()
            filename = f'{fname}_spindle_psd_concat.txt'
            savename = os.path.join(psd_concat_dir, filename)
            with open(savename, 'w') as f:
                json.dump(psd_export, f, indent=4)

            # export psd norm
            print('Exporting concatenated spindle norm spectra...\n')
            # convert series to dicts for json dump
            psd_norm_export = {}
            for chan in self.spindle_psd_concat_norm.keys():
                psd_norm_export[chan]={}
                for name, series in self.spindle_psd_concat_norm[chan].items():
                    psd_norm_export[chan][name] = series.to_dict()
            filename = f'{fname}_spindle_psd_norm.txt'
            savename = os.path.join(psd_concat_dir, filename)
            with open(savename, 'w') as f:
                json.dump(psd_norm_export, f, indent=4)
                
            
        if psd_i:
            # export individual spindle spectra
            print('Exporting individual spindle spectra...\n')
            psd_i_dir = export_dir + '/psd_individual'
            if not os.path.exists(psd_i_dir):
                os.makedirs(psd_i_dir)
            # export a file for each channel
            for chan in self.spindle_psd_i.keys():
                filename = f'{fname}_spindle_psd_i_{chan}.txt'
                savename = os.path.join(psd_i_dir, filename)
                # convert to dict for json dump
                psd_export = {}
                for spin, series in self.spindle_psd_i[chan].items():
                    psd_export[spin] = series.to_dict()
                with open(savename, 'w') as f:
                    json.dump(psd_export, f, indent=4)
                
            
        if stats:
            print('Exporting spindle statistics...\n')
            stats_dir = export_dir + '/statistics'
            if not os.path.exists(stats_dir):
                os.makedirs(stats_dir)
            # export spindle time stats
            filename = f'{fname}_spindle_tstats.csv'
            savename = os.path.join(stats_dir, filename)
            self.spindle_tstats.to_csv(savename)
            # export spindle individual stats
            filename = f'{fname}_spindle_stats_i.csv'
            savename = os.path.join(stats_dir, filename)
            self.spindle_stats_i.to_csv(savename)
            # export spindle frequency stats
            filename = f'{fname}_spindle_fstats_concat.csv'
            savename = os.path.join(stats_dir, filename)
            self.spindle_fstats_concat.to_csv(savename)
            
        print('Done.')


    ## Slow Oscillation Detection Methods ##

    def so_attributes(self):
        """ make attributes for slow oscillation detection """
        try:
            self.channels
        except AttributeError:
            # create if doesn't exist
            self.channels = [x[0] for x in self.data.columns]
        
        dfs = ['sofiltEEG', 'spsofiltEEG']
        [setattr(self, df, pd.DataFrame(index=self.data.index)) for df in dfs]
        self.so_events = {}
        self.so_rejects = {}

    def make_butter_so(self, wn, order):
        """ Make Butterworth bandpass filter [Parameters/Returns]

            Parameters
            ----------
            wn: list of int (default: [8, 16])
                butterworth bandpass filter window
            order: int (default: 4)
                butterworth 1/2 filter order (applied forwards + backwards)
        """
        nyquist = self.s_freq/2
        wn_arr = np.asarray(wn)
        
        if np.any(wn_arr <=0) or np.any(wn_arr >=1):
            wn_arr = wn_arr/nyquist # must remake filter for each pt bc of differences in s_freq
   
        self.so_sos = butter(order, wn_arr, btype='bandpass', output='sos')
        print(f"Zero phase butterworth filter successfully created: order = {order}x{order}, bandpass = {wn}")

    def make_butter_spso(self, spso_wn_pass, spso_wn_stop, spso_order):
        """ Make Butterworth bandpass and bandstop filter

            Parameters
            ----------
            spso_wn_pass: list (default: [0.1, 17])
            spso_wn_stop: list (default: [4.5, 7.5])
            spso_order: int (default: 8)

        """


        nyquist = self.s_freq/2
        wn_pass_arr = np.asarray(spso_wn_pass)
        wn_stop_arr = np.asarray(spso_wn_stop)
        
        # must remake filter for each pt bc of differences in s_freq
        if np.any(wn_pass_arr <=0) or np.any(wn_pass_arr >=1):
            wn_pass_arr = wn_pass_arr/nyquist 
            
        if np.any(wn_stop_arr <=0) or np.any(wn_stop_arr >=1):
            wn_stop_arr = wn_stop_arr/nyquist

        self.spso_sos_bandstop = butter(spso_order, wn_stop_arr, btype='bandstop', output='sos')
        self.spso_sos_bandpass = butter(spso_order, wn_pass_arr, btype='bandpass', output='sos')
        print(f"Zero phase butterworth filter successfully created: order = {spso_order}x{spso_order} bandpass = {spso_wn_pass}")
        print(f"Zero phase butterworth filter successfully created: order = {spso_order}x{spso_order} bandstop = {spso_wn_stop}")

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

    def spsofilt(self, i):
        """ Apply Butterworth bandpass-bandstop to signal by channel

            Parameters
            ----------
            i : str
                channel to filter
        """
        # separate NaN and non-NaN values to avoid NaN filter output on cleaned data
        data_nan = self.data[i][self.data[i]['Raw'].isna()]
        data_notnan = self.data[i][self.data[i]['Raw'].isna() == False]

        # filter notNaN data & add column to notNaN df
        ## bandpass
        data_notnan_bandpassed = sosfiltfilt(self.spso_sos_bandpass, data_notnan.to_numpy(), axis=0)
        ## now bandstop
        data_notnan_filt = sosfiltfilt(self.spso_sos_bandstop, data_notnan_bandpassed, axis=0)
        data_notnan['Filt'] = data_notnan_filt

        # merge NaN & filtered notNaN values, sort on index
        filt_chan = data_nan['Raw'].append(data_notnan['Filt']).sort_index()

        # add channel to main dataframe
        self.spsofiltEEG[i] = filt_chan

    def get_so(self, i, method, posx_thres, negposx_thres, npeak_thres, negpos_thres):
        """ Detect slow oscillations. Based on detection algorithm from Molle 2011 & Massimini 2004. 
            
            Parameters
            ----------
            i : str
                channel to filter
            method: str (default: 'absolute')
                SO detection method. [Options: 'absolute', 'ratio'] 
                'absolute' employs absolute voltage values for npeak_thres and negpos_thres. 
                'ratio' sets npeak_thres to None and negpos_thres to 1.75x the negative peak 
                voltage for a given detection (ratio derived from Massimini 2004)
                * NOTE: the idea was to use 'ratio' if reference is a single scalp electrode (e.g. FCz), which would
                result in variable absolute amplitudes according to electrode location. In practice this doesn't 
                seem to pull accurate SOs. Needs a minimum threshold for the negative peak
            posx_thres: list of float (default: [0.9, 2])
                threshold of consecutive positive-negative zero crossings in seconds. Equivalent to Hz range
                for slow oscillations
            negposx_thres: int (default: 300)
                minimum time (in milliseconds) between positive-to-negative and negative-to-positive zero crossing
            npeak_thres: int (default: -80)
                negative peak threshold in microvolts
            negpos_thres: int (default: 140)
                minimum amplitude threshold for negative to positive peaks
        """

        so_events = {}
        nx = 0
        
        # convert thresholds
        posx_thres_td = [pd.Timedelta(s, 's') for s in posx_thres]
        npeak_mv = npeak_thres*(10**-3)
        negpos_mv = negpos_thres*(10**-3)

        # convert channel data to series
        chan_dat = self.sofiltEEG[i]

        # get zero-crossings
        mask = chan_dat > 0
        # shift pos/neg mask by 1 and compare
        ## insert a false value at position 0 on the mask shift
        mask_shift = np.insert(np.array(mask), 0, None)
        ## remove the last value of the shifted mask and set the index 
        ## to equal the original mask 
        mask_shift = pd.Series(mask_shift[:-1], index=mask.index)
        # neg-pos are True; pos-neg are False
        so_zxings = mask[mask != mask_shift]
        
        # make empty lists for start and end times of vetted SO periods
        pn_pn_starts = []
        pn_pn_ends = []
        cols = ['start', 'end']

        # for each zero-crossing
        for e, (idx, xing) in enumerate(so_zxings.items()):
            # if it's not the last or second-to-last crossing
            if e not in [len(so_zxings)-1, len(so_zxings)-2]:
                # if it's positive-to-negative
                if xing == False:
                    # check the distance to the next negative-to-positive
                    pn_np_intvl = so_zxings.index[e+1] - idx
                    # if it's >= 300ms
                    if pn_np_intvl >= pd.to_timedelta(negposx_thres, 'ms'):
                        # if it's not the last or second-to-last crossing
                        if e not in [len(so_zxings)-1, len(so_zxings)-2]:
                            # if the next positive-to-negative crossing is within threshold
                            pn_pn_intvl = so_zxings.index[e+2] - idx
                            if posx_thres_td[0] <= pn_pn_intvl <= posx_thres_td[1]:
                                # grab the next positive to negative crossing that completes the SO
                                # period and add values to lists
                                pn_pn_starts.append(idx)
                                pn_pn_ends.append(so_zxings.index[e+2])

        # turn start and end lists into dataframe
        so_periods = pd.DataFrame(list(zip(pn_pn_starts, pn_pn_ends)), columns=cols)

        # loop through so_periods df
        for idx, row in so_periods.iterrows():
            # find negative & positive peaks
            npeak_time = chan_dat.loc[row.start:row.end].idxmin()
            npeak_val = chan_dat.loc[npeak_time]
            ppeak_time = chan_dat.loc[row.start:row.end].idxmax()
            ppeak_val = chan_dat.loc[ppeak_time]

            # check absolute value thresholds if method is absolute
            if method == 'absolute':
                # if negative peak is < than threshold
                if npeak_val < npeak_mv:
                    # if negative-positive peak amplitude is >= than threshold
                    if np.abs(npeak_val) + np.abs(ppeak_val) >= negpos_mv:
                        so_events[nx] = {'pn_zcross1': row.start, 'pn_zcross2': row.end, 'npeak': npeak_time, 
                                           'ppeak': ppeak_time, 'npeak_minus2s': npeak_time - datetime.timedelta(seconds=2), 
                                           'npeak_plus2s': npeak_time + datetime.timedelta(seconds=2)}
                        nx += 1
            
            # otherwise check ratio thresholds
            elif method == 'ratio':
                # npeak_val can be anything
                # if negative-positive peak amplitude is >= 1.75x npeak_val
                if np.abs(npeak_val) + np.abs(ppeak_val) >= 1.75*np.abs(npeak_val):
                    so_events[nx] = {'pn_zcross1': row.start, 'pn_zcross2': row.end, 'npeak': npeak_time, 
                                       'ppeak': ppeak_time, 'npeak_minus2s': npeak_time - datetime.timedelta(seconds=2), 
                                       'npeak_plus2s': npeak_time + datetime.timedelta(seconds=2)}

                    nx += 1
                    
        self.so_zxings = so_zxings
        self.so_events[i] = so_events

    def soMultiIndex(self):
        """ combine dataframes into a multiIndex dataframe"""
        # reset column levels
        self.sofiltEEG.columns = pd.MultiIndex.from_arrays([self.channels, np.repeat(('Filtered'), len(self.channels))],names=['Channel','datatype'])
        self.spsofiltEEG.columns = pd.MultiIndex.from_arrays([self.channels, np.repeat(('Filtered'), len(self.channels))],names=['Channel','datatype'])

        # list df vars for index specs
        # dfs =[self.sofiltEEG] # for > speed, don't store spinfilt_RMS as an attribute
        # calcs = ['Filtered']
        # lvl0 = np.repeat(self.channels, len(calcs))
        # lvl1 = calcs*len(self.channels)    
    
        # # combine & custom sort --> what was this for??
        # self.so_calcs = pd.concat(dfs, axis=1).reindex(columns=[lvl0, lvl1])

    def detect_so(self, wn=[0.1, 4], order=2, method='absolute', posx_thres = [0.9, 2], negposx_thres = 300, npeak_thres = -80, 
        negpos_thres = 140, spso_wn_pass = [0.1, 17], spso_wn_stop = [4.5, 7.5], spso_order=8):
        """ Detect slow oscillations by channel
        
            TO DO: Update docstring

            Parameters
            ----------
            wn: list (default: [0.1, 4])
                Butterworth filter window
            order: int (default: 2)
                Butterworth filter order (default of 2x2 from Massimini et al., 2004)
            method: str (default: 'ratio')
                SO detection method. [Options: 'absolute', 'ratio'] 
                'absolute' employs absolute voltage values for npeak_thres and negpos_thres. 
                'ratio' sets npeak_thres to None and negpos_thres to 1.75x the negative peak 
                voltage for a given detection (ratio derived from Massimini 2004)
                * NOTE: 'ratio' should be used if reference is a single scalp electrode (e.g. FCz), which would
                result in variable absolute amplitudes according to electrode location
            posx_thres: list of float (default: [0.9, 2])
                threshold of consecutive positive-negative zero crossings in seconds
            negposx_thres: int (default: 300)
                minimum time (in milliseconds) between positive-to-negative and negative-to-positive zero crossing
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
        
        self.metadata['so_analysis'] = {'so_filtwindow': wn, 'so_filtorder_half': order, 'method': method, 
                                        'posx_thres': posx_thres, 'negposx_thres': negposx_thres, 'npeak_thres': npeak_thres, 
                                        'negpos_thres': negpos_thres}
        
        # set attributes
        self.so_attributes()
        
        # make butterworth filter
        self.make_butter_so(wn, order)
        self.make_butter_spso(spso_wn_pass, spso_wn_stop, spso_order)

        # loop through channels (all channels for plotting ease)
        for i in self.channels:
                # Filter
                self.sofilt(i)
                self.spsofilt(i)

                # Detect SO
                self.get_so(i, method, posx_thres, negposx_thres, npeak_thres, negpos_thres)
                
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
                # create individual df for each SO
                start = self.so_events[chan][i]['npeak_minus2s']
                end = self.so_events[chan][i]['npeak_plus2s']
                so_data = self.data[chan]['Raw'].loc[start:end]
                so_filtdata = self.sofiltEEG[chan]['Filtered'].loc[start:end]
                spso_filtdata = self.spsofiltEEG[chan]['Filtered'].loc[start:end]
                
                # find & drop any NaN insertions at 1ms sfreq (likely artifact from blocking data)
                nan_idx = [e for e, x in enumerate(np.diff(so_data.index)) if int(x) == 3000000]
                if len(nan_idx) > 0:
                    so_data = so_data.drop(so_data.index[nan_idx])
                    so_filtdata = so_filtdata.drop(so_filtdata.index[nan_idx])
                    spso_filtdata = spso_filtdata.drop(spso_filtdata.index[nan_idx])

                # set new index so that each SO is zero-centered around the negative peak
                ms1 = list(range(-2000, 0, int(1/self.metadata['analysis_info']['s_freq']*1000)))
                ms2 = [-x for x in ms1[::-1]]
                id_ms = ms1 + [0] + ms2
                
                # create new dataframe
                so[chan][i] = pd.DataFrame(index=id_ms)
                so[chan][i].index.name='id_ms'
                
                # if the SO is not a full 2s from the beginning OR if there's a data break < 2seconds before the peak
                if (start < self.data.index[0]) or (start < so_data.index[0]):
                    # extend the df index to the full 2s
                    time_freq = str(int(1/self.metadata['analysis_info']['s_freq']*1000000))+'us'
                    time = pd.date_range(start=start, end=end, freq=time_freq)
                    so[chan][i]['time'] = time
                    # append NaNs onto the end of the EEG data
                    nans = np.repeat(np.NaN, len(time)-len(so_data))
                    data_extended = list(nans) + list(so_data.values)
                    so[chan][i]['Raw'] = data_extended
                    filtdata_extended = list(nans) + list(so_filtdata.values)
                    so[chan][i]['sofilt'] = filtdata_extended
                    spsofiltdata_extended = list(nans) + list(spso_filtdata.values)
                    so[chan][i]['spsofilt'] = spsofiltdata_extended

                # if the SO is not a full 2s from the end OR if there's a data break < 2seconds after the peak
                elif (end > self.data.index[-1]) or (end > so_data.index[-1]):
                    # extend the df index to the full 2s
                    time_freq = str(int(1/self.metadata['analysis_info']['s_freq']*1000000))+'us'
                    time = pd.date_range(start=start, end=end, freq=time_freq)
                    so[chan][i]['time'] = time
                    # append NaNs onto the end of the EEG data
                    nans = np.repeat(np.NaN, len(time)-len(so_data))
                    data_extended = list(so_data.values) + list(nans)
                    so[chan][i]['Raw'] = data_extended
                    filtdata_extended = list(so_filtdata.values) + list(nans)
                    so[chan][i]['sofilt'] = filtdata_extended
                    spsofiltdata_extended = list(spso_filtdata.values) + list(nans)
                    so[chan][i]['spsofilt'] = spsofiltdata_extended
                else:
                    so[chan][i]['time'] = so_data.index
                    so[chan][i]['Raw'] = so_data.values
                    so[chan][i]['sofilt'] = so_filtdata.values
                    so[chan][i]['spsofilt'] = spso_filtdata.values
        
        self.so = so
        print('Dataframes created. Slow oscillation data stored in obj.so.')

    
    def align_spindles(self):
        """ Align spindles along slow oscillations """
        print('Aligning spindles to slow oscillations...')
        so = self.so
        data = self.data
        spindles = self.spindles

        # create a dictionary of SO indices
        so_dict = {}
        for chan in so:
            so_dict[chan] = [so[chan][i].time.values for i in so[chan]]
        
        # flatten the dictionary into a boolean df
        so_bool_dict = {}
        for chan in so_dict:
            if chan not in ['EOG_L', 'EOG_R', 'EKG', 'REF', 'FPZorEKG', 'A1', 'A2']:
                so_flat = [time for so in so_dict[chan] for time in so]
                so_bool_dict[chan] = np.isin(data.index.values, so_flat)
        so_bool = pd.DataFrame(so_bool_dict, index=data.index)
        
        # create a spindle boolean df
        spin_bool_dict = {}
        for chan in spindles.keys():
            if chan not in ['EOG_L', 'EOG_R', 'EKG', 'REF', 'FPZorEKG', 'A1', 'A2']:
                spins_tlist = [df.time.values for df in spindles[chan].values()]
                spins_flat = [time for spindle in spins_tlist for time in spindle]
                spin_bool_dict[chan] = np.isin(data.index.values, spins_flat)
        spin_bool = pd.DataFrame(spin_bool_dict, index=data.index)
                
        # create a map of slow oscillations to spindles
        so_spin_map = {}
        for chan in spindles.keys():
            so_spin_map[chan] = {}
            so_flat = [time for so in so_dict[chan] for time in so]
            # for each spindle
            for e_spin, spin in spindles[chan].items():
                # grab the trough of the filtered spindle
                spin_trough = np.datetime64(spin.loc[0].time)
                # if spindle trough overlaps w/ SO +/- 2s:
                if spin_trough in so_flat:
                        for e_so, so_times in enumerate(so_dict[chan]):
                            if spin_trough in so_times:
                                try:
                                    so_spin_map[chan][e_so].append(e_spin)
                                except KeyError:
                                    so_spin_map[chan][e_so] = [e_spin]

        print('Compiling aggregate dataframe...')
        # Make aggregate dataframe
        spso_aggregates = {}
        for chan in so.keys():
            if chan not in ['EOG_L', 'EOG_R', 'EKG', 'REF', 'FPZorEKG', 'A1', 'A2']:
                spso_aggregates[chan] = {}
                for so_idx, spins in so_spin_map[chan].items(): 
                    spso_agg = so[chan][so_idx]
                    for s in spins:
                        # add spindle filtered and spso filtered data for each spindle
                        spso_agg = spso_agg.join(self.spfiltEEG[(chan, 'Filtered')].loc[spindles[chan][s].time.values].rename('spin_'+str(s)+'_spfilt'), 
                            on='time', how='outer')
                    spso_aggregates[chan][so_idx] = spso_agg

        self.so_bool = so_bool
        self.spin_bool = spin_bool
        self.so_spin_map = so_spin_map
        self.spso_aggregates = spso_aggregates

        print('Alignment complete. Aggregate data stored in obj.spso_aggregates.\n')

    def spso_distribution(self):
        """ get distribution of spindles along slow oscillations by cluster """

        print('Calculating spindle distribution along slow oscillations...')
        # create dicts to hold result
        spin_dist_bool = {'all':{'0':{}, '1':{}}, 'by_chan':{}}
        spin_dist = {'all':{'0':{}, '1':{}}, 'by_chan':{}}
        
        # Make boolean arrays of spindle distribution
        for chan in self.spso_aggregates.keys():
            spin_dist_bool['by_chan'][chan] = {'0':{}, '1':{}}
            # iterrate over individual SO dataframes
            for so_id, df in self.spso_aggregates[chan].items():
                # grab spindle columns
                spin_cols = [x for x in df.columns if x.split('_')[0] == 'spin']
                for spin in spin_cols:
                    # get index & cluster of spindle
                    spin_idx = int(spin_cols[0].split('_')[1])
                    clust = int(self.spindle_stats_i[(self.spindle_stats_i.chan == chan) & (self.spindle_stats_i.spin == spin_idx)].cluster.values)
                    # set spindle column & idx labels, save boolean values to dict
                    spin_label = chan + '_' + str(spin_idx)
                    spin_dist_bool['all'][str(clust)][spin_label] = df[df.index.notna()][spin].notna().values
                    spin_dist_bool['by_chan'][chan][str(clust)][spin_idx] = df[df.index.notna()][spin].notna().values
        idx = df[df.index.notna()].index

        # create series & normalize from dataframe
        for clust, dct in spin_dist_bool['all'].items():
             # calculate # of spindles at each timedelta
            bool_df = pd.DataFrame(dct, index=idx)
            dist_ser = bool_df.sum(axis=1)
            # normalize the values to total # of spindles in that cluster
            dist_norm = dist_ser/len(bool_df.columns)
            spin_dist['all'][str(clust)]['dist'] = dist_ser
            spin_dist['all'][str(clust)]['dist_norm'] = dist_norm

        # Get distribution by channel
        for chan, clst_dict in spin_dist_bool['by_chan'].items():
            spin_dist['by_chan'][chan] = {'0':{}, '1':{}}
            for clust, dct in clst_dict.items():
                 # calculate # of spindles at each timedelta
                bool_df = pd.DataFrame(dct, index=idx)
                dist_ser = bool_df.sum(axis=1)
                # normalize the values to total # of spindles in that cluster
                dist_norm = dist_ser/len(bool_df.columns)
                spin_dist['by_chan'][chan][str(clust)]['dist'] = dist_ser
                spin_dist['by_chan'][chan][str(clust)]['dist_norm'] = dist_norm


        # use channel distributions to get distirbutions by location
        # assign anterior-posterior characters
        a_chars = ['f']
        p_chars = ['p', 'o', 't']
        c_chans = ['a1', 't9', 't3', 'c5', 'c3', 'c1', 'cz', 'c2', 'c4', 'c6', 't4', 't10', 'a2']

        spin_dist_bool['AP'] = {'A':{}, 'P':{}}
        spin_dist_bool['LR'] = {'L':{}, 'R':{}}
        spin_dist_bool['quads'] = {'al':{}, 'ar':{}, 'pl':{}, 'pr':{}}

        # recategorize channels into AP/RL/quads dicts
        for chan, spso_dict in spin_dist_bool['by_chan'].items():
            # assign anterior-posterior
            if chan.casefold() in c_chans:
                ap = 'C' 
            elif any((c.casefold() in a_chars) for c in chan):
                ap = 'A'
            elif any((c.casefold() in p_chars) for c in chan):
                ap = 'P'
            # assign RL
            if chan[-1].casefold() == 'z':
                rl = 'C'
            elif int(chan[-1]) % 2 == 0:
                rl = 'R'
            else:
                rl = 'L'

            for clust, clust_dict in spso_dict.items():
                for spin, dct in clust_dict.items():
                    # give dict a new name
                    dname = chan + '_' + clust + '_' + str(spin)
                    # move item into proper dicts
                    if ap == 'A':
                        spin_dist_bool['AP']['A'][dname] = dct
                        if rl == 'R':
                            spin_dist_bool['LR']['R'][dname] = dct
                            spin_dist_bool['quads']['ar'][dname] = dct
                        elif rl == 'L':
                            spin_dist_bool['LR']['L'][dname] = dct
                            spin_dist_bool['quads']['al'][dname] = dct
                    elif ap == 'P':
                        spin_dist_bool['AP']['P'][dname] = dct
                        if rl == 'R':
                            spin_dist_bool['LR']['R'][dname] = dct
                            spin_dist_bool['quads']['pr'][dname] = dct
                        elif rl == 'L':
                            spin_dist_bool['LR']['L'][dname] = dct
                            spin_dist_bool['quads']['pl'][dname] = dct

        # git distributions for dicts
        dicts = ['AP', 'LR', 'quads']
        for d in dicts:
            spin_dist[d] = {}
            for group, bool_dict in spin_dist_bool[d].items():
                spin_dist[d][group] = {}
                bool_df = pd.DataFrame(bool_dict, index=idx)
                dist_ser = bool_df.sum(axis=1)
                # normalize the values to total # of spindles in that dict
                dist_norm = dist_ser/len(bool_df.columns)
                spin_dist[d][group]['dist'] = dist_ser
                spin_dist[d][group]['dist_norm'] = dist_norm

        self.spin_dist_bool = spin_dist_bool
        self.spin_dist = spin_dist
        print('Done. Distributions (overall and by channel) stored in obj.spin_dist_bool & obj.spin_dist\n')


    def analyze_spso(self):
        """ starter code to calculate placement of spindles on the slow oscillation """
        
        # align spindles & SOs
        self.align_spindles()

        # calculate spindle distribution along SOs
        self.spso_distribution()


    def export_spso(self, export_dir, spso_aggregates=True, spin_dist=True, spin_dist_bychan=False):
        """ Export spindle-S0 analyses
            TO DO: 
            * Add support for exporting spindle-SO distribution by channel
            * Add stats

            Parameters
            ----------
            export_dir: str
                path to export directory
            spso_aggregates: bool (default: True)
                export aligned aggregates of spindles and slow oscillations
            spin_dist: bool (default: True)
                export % distribution of spindles along zero-centered slow oscillations by cluster
            spin_dist_bychan: bool (default: False)
                export % distribution of spindles along zero-centered slow oscillations by cluster by channel
                * not completed *

            Returns
            -------
            *To be completed

        """

        spso_dir = os.path.join(export_dir, 'spindle_SO')
        print(f'SPSO export directory: {spso_dir}\n')    
        # make export directory if doesn't exit
        if not os.path.exists(spso_dir):
            os.makedirs(spso_dir)

        # set base for savename
        fname = self.metadata['file_info']['fname'].split('.')[0]

        # export spindle-SO aggregates
        if spso_aggregates:      
            print('Exporting spso aggregates...')
            filename = f'{fname}_spso_aggregates.xlsx'
            savename = os.path.join(spso_dir, filename)
            writer = pd.ExcelWriter(savename, engine='xlsxwriter')
            for chan in self.spso_aggregates.keys():
                for so in self.spso_aggregates[chan].keys():
                    tab = '_SO'.join([chan, str(so)])
                    self.spso_aggregates[chan][so].to_excel(writer, sheet_name=tab)
            writer.save()

        # export spindle distribution along SO
        if spin_dist:
            print('Exporting spindle-SO distribution...')
            # 'all' is by cluster here
            comps = ['all', 'AP', 'LR', 'quads']
            for c in comps:
                if c == 'all':
                    filename = f'{fname}_spso_distribution_clust.xlsx'
                else:
                    filename = f'{fname}_spso_distribution_{c}.xlsx'
                savename = os.path.join(spso_dir, filename)
                writer = pd.ExcelWriter(savename, engine='xlsxwriter')
                for group in self.spin_dist[c].keys():
                    for dtype in self.spin_dist[c][group].keys():
                        if c == 'all':
                            tab = f'clust{group}_SO_{dtype}'
                        else:
                            tab = f'{group}_SO_{dtype}'
                        self.spin_dist[c][group][dtype].to_excel(writer, sheet_name=tab)
                writer.save()
            
        print('Done.')