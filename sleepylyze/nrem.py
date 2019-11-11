""" This file contains a class and methods for Non-REM EEG segments 
    
    Notes:
        - Analysis should output # of NaNs in the data

    TO DO: 
        - For self.detect_spindles(), move attributes into metadata['analysis_info'] dict
        - Optimize self.create_spindfs() method
        - Assign NREM attributes to slots on init
        - Update docstrings
"""

import datetime
import glob
import joblib
import json
import os
import numpy as np
import pandas as pd
import warnings
import xlsxwriter

from mne.time_frequency import psd_array_multitaper
from scipy.signal import butter, sosfiltfilt, sosfreqz
from scipy.optimize import OptimizeWarning, curve_fit

class NREM:
    """ General class for nonREM EEG segments """
    def __init__(self, fname=None, fpath=None, match=None, in_num=None, epoched=False, batch=False):
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
        """
        if batch:
            self.load_batch(fpath, match, in_num)
        else:
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

        self.metadata['file_info']['start_time'] = str(data.index[0])
        self.metadata['analysis_info'] = {'s_freq': s_freq, 'cycle_len_secs': cycle_len_secs}
        self.s_freq = s_freq

        print('EEG successfully imported.')

    def load_batch(self, fpath, match, in_num):
        """ Load a batch of EEG segments & reset index from absolute to relative time 
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
                x += 1
                
        self.spindle_events[i] = spindle_events_msep

    # step 7: apply rejection criteria
    def reject_spins(self, min_chans, duration):
        """ Reject spindles that occur over fewer than 3 channels. Apply duration thresholding to 
            spindles that occur over fewer than X channels. 
            [chans < 3 = reject; 3 < chans < X = apply duration threshold; X < chans = keep]
        
            Parameters
            ----------
            min_chans: int
                minimum number of channels for spindles to occur across concurrently in order to 
                bypass duration criterion. performs best at 1/4 of total chans
            duration: list of float
                duration range (seconds) for spindle thresholding
            
            Returns
            -------
            modified self.spindle_events and self.spindle_rejects attributes
        """
        
        # convert duration from seconds to samples
        sduration = [x*self.s_freq for x in duration]
        
        # make boolean mask for spindle presence
        spin_bool = pd.DataFrame(index = self.data.index)
        for chan in self.spindle_events:
            if chan not in ['EOG_L', 'EOG_R', 'EKG']:
                spins_flat = [time for spindle in self.spindle_events[chan] for time in spindle]
                spin_bool[chan] = np.isin(self.data.index.values, spins_flat)
        spin_bool['chans_present'] = spin_bool.sum(axis=1)
        
        # check individual spindles
        for chan in self.spindle_events:
            self.spindle_rejects[chan] = []
            for spin in self.spindle_events[chan]:
                # reject if present over less than 3 channels
                if not np.any(spin_bool['chans_present'].loc[spin] >= 3):
                        self.spindle_rejects[chan].append(spin)
                        self.spindle_events[chan].remove(spin)
                # Apply duration threshold if not present over more than minimum # of channels
                elif not np.any(spin_bool['chans_present'].loc[spin] >= min_chans):
                    # apply duration thresholding
                    if not sduration[0] <= len(spin) <= sduration[1]:
                        self.spindle_rejects[chan].append(spin)
                        self.spindle_events[chan].remove(spin)
                # Apply max duration threshold to all spindles left (regardless of # of chans)
                else:
                    if len(spin) > sduration[1]:
                        self.spindle_rejects[chan].append(spin)
                        self.spindle_events[chan].remove(spin)
                    
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
        
        
    def detect_spindles(self, wn=[8, 16], order=4, sp_mw=0.2, loSD=0, hiSD=1.5, min_sep=0.2, duration=[0.5, 3.0], min_chans=9):  
        """ Detect spindles by channel [Params/Returns] 
            
            Parameters
            ----------

            min_sep: float (default: 0.1)
                minimum separation (in seconds) for spindles to be considered distinct, otherwise combine

            Returns
            -------

        """

        self.metadata['spindle_analysis'] = {'sp_filtwindow': wn, 'sp_filtorder_half': order, 
            'sp_RMSmw': sp_mw, 'sp_loSD': loSD, 'sp_hiSD': hiSD, 'min_sep': min_sep, 'sp_duration': duration,
            'sp_minchans_toskipduration': min_chans}

        #self.s_freq = self.metadata['analysis_info']['s_freq']
    
        # set attributes
        self.spindle_attributes()
        # Make filter
        self.make_butter_sp(wn, order)

        print('Detecting spindles...')
        # loop through channels (all channels for plotting ease)
        for i in self.channels:
           # if i not in ['EOG_L', 'EOG_R', 'EKG']:
                #print(f'Detecting spindles on {i}...')
                # Filter
                self.spfilt(i)
                # Calculate RMS & smooth
                self.rms_smooth(i, sp_mw)
                # Set detection thresholds
                self.set_thres(i)
                # Detect spindles
                self.get_spindles(i, min_sep)
        
        # Apply rejection criteria
        print('Pruning spindle detections...')
        self.reject_spins(min_chans, duration)
        print('Spindle detection complete.')
        # combine dataframes
        print('Combining dataframes...')
        self.spMultiIndex()
        print('done.\n')

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

    def calc_spindle_means(self, datatype):
        """ Calculate mean, std, and sem at each timedelta from negative spindle peak per channel 
        
            Parameters
            ----------
            datatype: str (default: 'spfilt')
                Which data to use for averaging (options: 'Raw', 'spfilt', 'spsofilt')
        """
        self.metadata['analysis_info']['stats_dtype'] = datatype

        print('Aligning spindles...')
        # align spindles accoridng to timedelta & combine into single dataframe
        spindle_aggregates = {}
        for chan in self.spindles.keys():
            # only use channels that have spindles
            if self.spindles[chan]:
                # set the base df
                agg_df = pd.DataFrame(self.spindles[chan][0][datatype])
                agg_df = agg_df.rename(columns={datatype:'spin_0'})
                rsuffix = list(range(1, len(self.spindles[chan])))
                # join on the index for each spindle
                agg_df = agg_df.join([self.spindles[chan][x][datatype].rename('spin_'+str(x)) for x in rsuffix], how='outer')
                spindle_aggregates[chan] = agg_df
            
        print('Calculating spindle statistics...')
        # create a new multiindex dataframe for calculations
        calcs = ['mean', 'std' ,'sem']
        tuples = [(chan, calc) for chan in spindle_aggregates.keys() for calc in calcs]
        columns = pd.MultiIndex.from_tuples(tuples, names=['channel', 'calc'])
        spindle_means = pd.DataFrame(columns=columns)
        
        # fill the dataframe
        for chan in spindle_aggregates.keys():
            spindle_means[(chan, 'mean')] = spindle_aggregates[chan].mean(axis=1)
            spindle_means[(chan, 'std')] = spindle_aggregates[chan].std(axis=1)
            spindle_means[(chan, 'sem')] = spindle_aggregates[chan].sem(axis=1)
            
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

    def calc_spindle_psd(self, psd_bandwidth):
        """ Calculate multitaper power spectrum for all channels

            Params
            ------
            bandwidth: float
                frequency resolution in Hz

            Returns
            -------
            self.spindle_psd: dict
                format {channel: pd.Series} with index = frequencies and values = power (uV^2/Hz)
            self.spindle_multitaper_calcs: pd.DataFrame
                calculations used to calculated multitaper power spectral estimates for each channel
        """
        
        print('Calculating power spectra (this may take a few minutes)...')
        self.metadata['spindle_analysis']['psd_dtype'] = 'raw'
        self.metadata['spindle_analysis']['psd_method'] = 'multitaper'
        self.metadata['spindle_analysis']['psd_bandwidth'] = psd_bandwidth
        sf = self.metadata['analysis_info']['s_freq']
        
        spindle_psd = {}
        spindle_multitaper_calcs = pd.DataFrame(index=['data_len', 'N', 'W', 'NW', 'K'])
        for chan in self.spindles:
            #print(f'Calculating spectra for {chan}...')
            if len(self.spindles[chan]) > 0:
                # concatenate spindles
                spindles = [self.spindles[chan][x].Raw.values for x in self.spindles[chan]]
                data = np.concatenate(spindles)
                
                # record PS params [K = 2NW-1]
                N = len(data)/sf
                W = psd_bandwidth
                K = int((2*N*W)-1)
                spindle_multitaper_calcs[chan] = [len(data), N, W, N*W, K] 
                
                # calculate power spectrum
                pwr, freqs = psd_array_multitaper(data, sf, adaptive=True, bandwidth=psd_bandwidth, fmax=25, 
                                                  normalization='full', verbose=0)
                # convert to series & add to dict
                psd = pd.Series(pwr, index=freqs)
                spindle_psd[chan] = psd
        
        self.spindle_multitaper_calcs = spindle_multitaper_calcs
        self.spindle_psd = spindle_psd
        print('Done. Spectra stored in obj.spindle_psd. Calculations stored in obj.spindle_multitaper_calcs.\n')

    def calc_gottselig_norm(self, norm_range):
        """ calculated normalized spindle power on EEG channels (from Gottselig et al., 2002)
        
            TO DO: change p0 value if optimize warning
        
            Parameters
            ----------
            norm_range: list of tuple
                frequency ranges for gottselig normalization
            
            Returns
            -------
            self.spindle_psd_norm: nested dict
                format {chan: pd.Series(normalized power, index=frequency)}
        
        """
        
        print('Calculating Gottselig normalization...')

        def exponential_func(x, a, b, c):
            return a*np.exp(-b*x)+c
        
        self.metadata['spindle_analysis']['gottselig_range'] = norm_range
        exclude = ['EOG_L', 'EOG_R', 'EKG']
        
        spindle_psd_norm = {}
        chans_norm_failed = []
        for chan in self.spindle_psd:
            if chan not in exclude:
                spindle_psd_norm[chan] = {}

                # specify data to be fit (only data in norm range)
                incl_freqs = np.logical_or(((self.spindle_psd[chan].index >= norm_range[0][0]) & (self.spindle_psd[chan].index <= norm_range[0][1])),
                                            ((self.spindle_psd[chan].index >= norm_range[1][0]) & (self.spindle_psd[chan].index <= norm_range[1][1])))
                pwr_fit = self.spindle_psd[chan][incl_freqs] 

                # set x and y values (convert y to dB)
                x_pwr_fit = pwr_fit.index
                y_pwr_fit = 10 * np.log10(pwr_fit.values)

                # fit exponential -- try second fit line if first throws infinite covariance
                with warnings.catch_warnings():
                    warnings.simplefilter("error", OptimizeWarning)
                    try:
                        popt, pcov = curve_fit(exponential_func, xdata=x_pwr_fit, ydata=y_pwr_fit, p0=(1, 0, 1))
                    except (OptimizeWarning, RuntimeError):
                        try:
                            popt, pcov = curve_fit(exponential_func, xdata=x_pwr_fit, ydata=y_pwr_fit, p0=(1, 1e-6, 1))
                        except (OptimizeWarning, RuntimeError):
                            popt = np.full(3, np.nan)
                            chans_norm_failed.append(chan)
                            print(f'scipy.optimize.curvefit encountered RuntimeError on channel {chan}. Normalization skipped for this channel.')
                            pass

                xx = self.spindle_psd[chan].index
                yy = exponential_func(xx, *popt)

                # subtract the fit line
                psd_norm = pd.Series(10*np.log10(self.spindle_psd[chan].values) - yy, index=self.spindle_psd[chan].index)

                # save the values
                spindle_psd_norm[chan]['normed_pwr'] = psd_norm
                spindle_psd_norm[chan]['values_to_fit'] = pd.Series(y_pwr_fit, index=x_pwr_fit)
                spindle_psd_norm[chan]['exp_fit_line'] = pd.Series(yy, index=xx) 

        self.spindle_psd_norm = spindle_psd_norm
        self.metadata['spindle_analysis']['chans_norm_failed'] = chans_norm_failed
        print('Gottselig normalization data stored in obj.spindle_psd_norm.\n')

    def calc_spinstats(self, spin_range):
        """ calculate spindle feature statistics 
        
            Parameters
            ----------
            spin_range: list of int
                spindle frequency range to be used for calculating center frequency
            
            Returns
            -------
            self.spindle_features: pd.DataFrame
                MultiIndex dataframe with calculated spindle statistics
        """
        
        print('Calculating spindle statistics...')
        
        # create multi-index dataframe
        lvl1 = ['Count', 'Duration', 'Duration', 'Amplitude_raw', 'Amplitude_raw', 'Amplitude_spfilt', 'Amplitude_spfilt', 'Density', 'ISI', 'ISI', 'Power', 'Power']
        lvl2 = ['total', 'mean', 'sd', 'rms', 'sd', 'rms', 'sd', 'spin_per_min', 'mean', 'sd', 'center_freq', 'total_pwr']
        columns = pd.MultiIndex.from_arrays([lvl1, lvl2])
        spindle_stats = pd.DataFrame(columns=columns)
        
        #exclude non-EEG channels
        exclude = ['EOG_L', 'EOG_R', 'EKG']

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
                    density = count/((self.data.index[-1] - self.data.index[0]).total_seconds()/60)

                    # calculate inter-spindle-interval (ISI)
                    isi_arr = np.array([(self.spindles[chan][x+1].time.iloc[0] - self.spindles[chan][x].time.iloc[-1]).total_seconds() for x in self.spindles[chan] if x < len(self.spindles[chan])-1])
                    isi_mean = isi_arr.mean()
                    isi_sd = isi_arr.std()

                    # calculate center frequency & total spindle power
                    spindle_power = self.spindle_psd_norm[chan]['normed_pwr'][(self.spindle_psd[chan].index >= spin_range[0]) & (self.spindle_psd[chan].index <= spin_range[1])]
                    center_freq = spindle_power.idxmax()
                    total_pwr = spindle_power.sum()

                    spindle_stats.loc[chan] = [count, duration_mean, duration_sd, amp_rms_raw, amp_sd_raw, amp_rms_spfilt, amp_sd_spfilt, density, isi_mean, isi_sd, center_freq, total_pwr]

        self.spindle_stats = spindle_stats   
        
        print('Spindle stats stored in obj.spindle_stats.\n')

    def analyze_spindles(self, zmethod='trough', trough_dtype='spfilt', buff=False, buffer_len=3, psd_bandwidth=1.0, 
                        norm_range=[(4,6), (18, 25)], spin_range=[9, 16], datatype = 'spfilt'):
        """ starting code for spindle statistics/visualizations 

        Parameters
        ----------
        zmethod: str (default: 'trough')
            method used to assign 0-center to spindles [options: 'trough', 'middle']. Trough assigns zero-center to
            the deepest negative trough. Middle assigns zero center to the midpoint in time.
        trough_dtype: str (default: 'spfilt')
                Which data to use for picking the most negative trough for centering [options: 'Raw', 'spfilt']
        buff: bool (default: False)
            calculate spindle data dataframes with a delta time buffer around center of spindle
        buffer_len: int
            length in seconds of buffer to calculate around 0-center of spindle
        psd_bandwidth: float
            frequency bandwidth for power spectra calculations (Hz)
        norm_range: list of tuple
            frequency ranges for gottselig normalization
        spin_range: list of int
            spindle frequency range to be used for calculating center frequency
        datatype: str (default: 'spfilt')
                Which data to use for aggregation & averaging (options: 'Raw', 'spfilt', 'spsofilt')

        Returns
        -------
        self.spindles: nested dict of dfs
            nested dict with spindle data by channel {channel: {spindle_num:spindle_data}}
        self.spindles_wbuffer: nested dict of dfs
            nested dict with spindle data w/ timedelta buffer by channel {channel: {spindle_num:spindle_data}}
        self.spindle_psd: dict
                format {channel: pd.Series} with index = frequencies and values = power (uV^2/Hz)
        self.spindle_multitaper_calcs: pd.DataFrame
                calculations used to calculated multitaper power spectral estimates for each channel
        self.spindle_psd_norm: nested dict
            format {chan: pd.Series(normalized power, index=frequency)}
        self.spindle_features: pd.DataFrame
            MultiIndex dataframe with calculated spindle statistics
        """
        
        # create individual datframes for each spindle
        self.create_spindfs(zmethod, trough_dtype, buff, buffer_len)

        # calculate spindle & spindle buffer means
        self.calc_spindle_means(datatype)
        if buff:
            self.calc_spindle_buffer_means()

        # calculate power spectra
        self.calc_spindle_psd(psd_bandwidth)

        # normalize power spectra for quantification
        self.calc_gottselig_norm(norm_range)

        # run spindle statistics by channel
        self.calc_spinstats(spin_range)


    def export_spindles(self, export_dir):
        """ Export spindle analyses
        
            Parameters
            ----------
            export_dir: str
                Directory to save exported files
            
            Returns
            -------
            export_dir/fname_metadata.txt: json dump file
            export_dir/calcs/fname_multitaper_calcs.csv: csv file
            export_dir/calcs/fname_spindle_psd.txt: json dump file
            export_dir/calcs/fname_spindle_psd_norm.txt: json dump file
            export_dir/fname_spindle_aggregates.csv: multi-tab excel file
            export_dir/fname_spindle_means.csv: csv file
            export_dir/fname_spindle_stats.csv: csv file
              
        """
        
        print('Exporting spindle analyses..')

        # make export directory if doesn't exit
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
        
        # make subdirectory for calculations
        calc_dir = os.path.join(export_dir, 'calcs')
        if not os.path.exists(calc_dir):
            os.makedirs(calc_dir)
        
        # set base for savename
        fname = self.metadata['file_info']['fname'].split('.')[0]
        
        # dump metadata
        filename = f'{fname}_spindle_metadata.txt'
        savename = os.path.join(export_dir, filename)
        with open(savename, 'w') as f:
            json.dump(self.metadata, f, indent=4)
        
        # export multitaper calcs
        filename = f'{fname}_spindle_mt_calcs.csv'
        savename = os.path.join(calc_dir, filename)
        self.spindle_multitaper_calcs.to_csv(savename)
        
        # export psd
        # convert series to dicts for json dump
        psd_export = {}
        for name, series in self.spindle_psd.items():
            psd_export[name] = series.to_dict()
        filename = f'{fname}_spindle_psd.txt'
        savename = os.path.join(calc_dir, filename)
        with open(savename, 'w') as f:
            json.dump(psd_export, f, indent=4)
        
        # export psd norm
        # convert series to dicts for json dump
        psd_norm_export = {}
        for chan in self.spindle_psd_norm.keys():
            psd_norm_export[chan]={}
            for name, series in self.spindle_psd_norm[chan].items():
                psd_norm_export[chan][name] = series.to_dict()
        filename = f'{fname}_spindle_psd_norm.txt'
        savename = os.path.join(calc_dir, filename)
        with open(savename, 'w') as f:
            json.dump(psd_norm_export, f, indent=4)
        
        # export spindle aggregates
        filename = f'{fname}_spindle_aggregates.xlsx'
        savename = os.path.join(export_dir, filename)
        writer = pd.ExcelWriter(savename, engine='xlsxwriter')
        for chan_tab, df in self.spindle_aggregates.items():
            df.to_excel(writer, sheet_name=chan_tab)
        writer.save() 
        
        # export spindle means
        filename = f'{fname}_spindle_means.csv'
        savename = os.path.join(export_dir, filename)
        self.spindle_means.to_csv(savename)
        
        # export spindle stats
        filename = f'{fname}_spindle_stats.csv'
        savename = os.path.join(export_dir, filename)
        self.spindle_stats.to_csv(savename)

        print(f'Export complete. Analyses stored in {export_dir}')


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
        """ Make Butterworth bandpass filter [Parameters/Returns]"""
        
        nyquist = self.s_freq/2
        wn_arr = np.asarray(wn)
        
        if np.any(wn_arr <=0) or np.any(wn_arr >=1):
            wn_arr = wn_arr/nyquist # must remake filter for each pt bc of differences in s_freq
   
        self.so_sos = butter(order, wn_arr, btype='bandpass', output='sos')
        print(f"Zero phase butterworth filter successfully created: order = {order}x{order}, bandpass = {wn}")

    def make_butter_spso(self, spso_wn_pass, spso_wn_stop, spso_order):
        """ Make Butterworth bandpass and bandstop filter [Parameters/Returns]"""
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
        """ Apply Butterworth bandpass-bandstop to signal by channel """

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
        self.spsofiltEEG.columns = pd.MultiIndex.from_arrays([self.channels, np.repeat(('Filtered'), len(self.channels))],names=['Channel','datatype'])

        # list df vars for index specs
        # dfs =[self.sofiltEEG] # for > speed, don't store spinfilt_RMS as an attribute
        # calcs = ['Filtered']
        # lvl0 = np.repeat(self.channels, len(calcs))
        # lvl1 = calcs*len(self.channels)    
    
        # # combine & custom sort --> what was this for??
        # self.so_calcs = pd.concat(dfs, axis=1).reindex(columns=[lvl0, lvl1])

    def detect_so(self, wn=[0.1, 4], order=2, posx_thres = [0.9, 2], npeak_thres = -80, negpos_thres = 140,
                    spso_wn_pass = [0.1, 17], spso_wn_stop = [4.5, 7.5], spso_order=8):
        """ Detect slow oscillations by channel
        
            TO DO: Update docstring

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
        
        self.metadata['so_analysis'] = {'so_filtwindow': wn, 'so_filtorder_half': order, 'posx_thres': posx_thres,
                                        'npeak_thres': npeak_thres, 'negpos_thres': negpos_thres}
        
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
                so_filtdata = self.sofiltEEG[chan]['Filtered'].loc[start:end]
                spso_filtdata = self.spsofiltEEG[chan]['Filtered'].loc[start:end]
                
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
                    filtdata_extended = list(nans) + list(so_filtdata.values)
                    so[chan][i]['sofilt'] = filtdata_extended
                    spsofiltdata_extended = list(nans) + list(spso_filtdata.values)
                    so[chan][i]['spsofilt'] = spsofiltdata_extended

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

    def analyze_spso(self):
        """ starter code to calculate placement of spindles on the slow oscillation """
        
        print('Aligning spindles to slow oscillations...')
        # create a dictionary of SO indices
        so_dict = {}
        for chan in self.so:
            so_dict[chan] = [self.so[chan][i].time.values for i in self.so[chan]]
        
        # flatten the dictionary into a boolean df
        so_bool = pd.DataFrame(index = self.data.index)
        for chan in so_dict:
            if chan not in ['EOG_L', 'EOG_R', 'EKG']:
                so_flat = [time for so in so_dict[chan] for time in so]
                so_bool[chan] = np.isin(self.data.index.values, so_flat)
        
        # create a spindle boolean df
        spin_bool = pd.DataFrame(index = self.data.index)
        for chan in self.spindle_events:
            if chan not in ['EOG_L', 'EOG_R', 'EKG']:
                spins_flat = [time for spindle in self.spindle_events[chan] for time in spindle]
                spin_bool[chan] = np.isin(self.data.index.values, spins_flat)
                
        # create a map of slow oscillations to spindles
        so_spin_map = {}
        for chan in self.spindle_events:
            so_spin_map[chan] = {}
            so_flat = [time for so in so_dict[chan] for time in so]
            # for each spindle
            for e_spin, spin in enumerate(self.spindle_events[chan]):
                spin_trough = np.datetime64(self.data[chan]['Raw'].loc[self.spindle_events[chan][e_spin]].idxmin())
                # if spindle trough overlaps w/ SO +/- 2s:
                if spin_trough in so_flat:
                        for e_so, so in enumerate(so_dict[chan]):
                            if spin_trough in so:
                                try:
                                    so_spin_map[chan][e_so].append(e_spin)
                                except KeyError:
                                    so_spin_map[chan][e_so] = [e_spin]

        # Make aggregate dataframe
        spso_aggregates = {}
        for chan in self.so.keys():
            if chan not in ['EOG_L', 'EOG_R', 'EKG']:
                spso_aggregates[chan] = {}
                for so, spins in so_spin_map[chan].items(): 
                    spso_agg = self.so[chan][so]
                    for s in spins:
                        spso_agg = spso_agg.join(self.spfiltEEG[(chan, 'Filtered')].loc[self.spindle_events[chan][s]].rename('spin_'+str(s)), 
                            on='time', how='outer')
                    spso_aggregates[chan][so] = spso_agg

        self.so_bool = so_bool
        self.spin_bool = spin_bool
        self.so_spin_map = so_spin_map
        self.spso_aggregates = spso_aggregates

        print('Alignment complete. Aggregate data stored in obj.spso_aggregates.')