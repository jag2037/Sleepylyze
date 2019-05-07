""" import this class by using the __init__.py file 'from .ioeeg import Dataset' 

    To Do:
        Remove spindle detection functions --> to be moved to new class for cut segments 
"""

import datetime
import math 
import numpy as np 
import os
import pandas as pd
import scipy.io as io
from scipy.signal import buttord, butter, sosfiltfilt, sosfreqz

class Dataset:
    """ General class containing EEG recordings

    NOTE: This assumes a continuous record. If record breaks are present they WILL NOT
    be detected and timestamps will be inaccurate
    
    Parameters
    ----------
    fname: str
        xltek .txt filename
    fpath: str
        absolute path to file
    
    Attributes
    ----------
    fname: str
        xltek .txt filename
    fpath: str
        absolute path to file
    filepath: str
        absolute path to file
    in_num: str
    s_freq: int
        sampling frequency
    chans: int
        number of expected channels based on headbox
    hbsn: int
        headbox serial number 
    start_time: str
        time of first recording (hh:mm:ss)
    hbid: str
        headbox id
    channels: list
        channel names
    data: pandas.DataFrame
        raw EEG/EKG data
    """
    
    def __init__(self, fname, fpath=None):
        if fpath is not None:
            filepath = os.path.join(fpath, fname)
            #filepath = fpath + fname
        else:
            filepath = fname
        
        self.fname = fname
        self.fpath = fpath
        self.filepath = filepath
        
        self.get_info()
        self.get_chans()
        self.load_eeg()

    
    # io methods #
    def get_info(self):
        """ Read in the header block and extract useful info """
        # extract IN
        self.in_num = self.fname.split("_")[0]
        print("Patient identifier:", self.in_num)
        
        # extract sampling freq, # of channels, headbox sn
        with open(self.filepath, 'r') as f: # pull out the header w/ first recording
            header = []
            for i in range(1, 267): # this assumes s_freq<=250
                header.append(f.readline()) # use readline (NOT READLINES) here to save memory for large files
        self.s_freq = int(float(header[7].split()[3])) # extract sampling frequency
        self.chans = int(header[8].split()[2]) # extract number of channels as integer
        self.hbsn = int(header[10].split()[3]) # extract headbox serial number
        self.start_date = (header[15].split()[0]).replace('/', '-')
        self.start_time = header[15].split()[1] # extract start time in hh:mm:ss
        
        # Get starting time in usec for index
        firstsec = []
        for i in range(15, len(header)):
            firstsec.append(header[i].split()[1])
        firstsec = pd.Series(firstsec)
        # Find the timestamp cut between seconds 1 & 2, convert to usec
        first_read = self.s_freq - (firstsec.eq(self.start_time).sum())
        self.start_us = 1/self.s_freq*first_read
         
    def get_chans(self):
        """ Define the channel list for the detected headbox """
        chans = self.chans
        hbsn = self.hbsn
        
        def check_chans(chans, expected_chans):
            print(chans, ' of ', expected_chans, ' expected channels found')
            if chans != expected_chans: # this may not be a proper check
                print('WARNING: Not all expected channels for this headbox were detected. Proceeding with detected channels')
        
        if hbsn == 125:
            hbid = "MOBEE 32" # EKG only used for one pt (IN343C??)
            print('Headbox:', hbid)
            expected_chans = 35
            check_chans(chans, expected_chans)
            channels_all = ['REF','FP1','F7','T3','A1','T5','O1','F3','C3','P3','FPZorEKG','FZ','CZ',
                'PZ','FP2','F8','T4','A2','T6','O2','F4','C4','P4','AF7','AF8','FC5','FC6','FC1','FC2',
                'CP5','CP6','CP1','CP2','OSAT','PR']
            channels = channels_all[:-2]
            channels_rmvd = channels_all[-2:]
            print('Removed the following channels: \n', channels_rmvd)

        elif hbsn == 65535 and self.chans == 40: # for IN346B. need to adjust this for other EMU40s
            hbid = "EMU40 DB+18"
            print('Headbox:', hbid)
            expected_chans = 40 
            check_chans(chans, expected_chans) # this check does nothing here
            channels_all = ['Fp1','F7','T3','T5','O1','F3','C3','P3','EMGref','Fz','Cz','Fp2','F8','T4',
                'T6','O2','F4','C4','P4','EMG','FPz','Pz','AF7','AF8','FC5','FC6','FC1','FC2','CP5','CP6',
                'CP1','CP2','PO7','PO8','F1','F2','CPz','POz','Oz','EKG']
            channels = channels_all # remove EMGs?
            
        elif hbsn == 65535:
            hbid = "FS128"
            print('Headbox:', hbid)
            expected_chans = 128
            check_chans(chans, expected_chans) 
            channels_all = ['Fp1','F3','FC1','C3','CP1','P3','O1','AF7','F7','FC5','T3','CP5',
                'T5','PO7','FPz','Fz','Cz','CPz','Pz','POz','Oz','FP2','F4','FC2','C4','CP2','P4','O2',
                'AF8','F8','FC6','T4','CP6','T6','PO8','F1','F2','EOG_L','EOG_R','EKG']
                # remove any chans here?
            channels = channels_all

        self.hbsn = hbsn
        self.hbid = hbid
        self.channels = channels
        self.eeg_channels = [e for e in self.channels if e not in ('EOG_L', 'EOG_R', 'EKG')]
    
    def load_eeg(self):
        """ docstring here """
        # set the last column of data to import
        end_col = len(self.channels) + 3 # works for MOBEE32, check for other headboxes
        
        # read in only the data
        print('Importing EEG data...')
        # setting dtype to float will speed up load, but crashes if there is anything wrong with the record
        data = pd.read_csv(self.filepath, delim_whitespace=True, header=None, skiprows=15, usecols=range(3,end_col),
                               dtype = float, na_values = ['AMPSAT', 'SHORT'])
        
        # create DateTimeIndex
        ind_freq = str(int(1/self.s_freq*1000000))+'us'
        ind_start = self.start_date + ' ' + self.start_time + str(self.start_us)[1:]
        ind = pd.date_range(start = ind_start, periods=len(data), freq=ind_freq)

        # make a new dataframe with the proper index & column names
        data.columns = pd.MultiIndex.from_arrays([self.channels, np.repeat(('Raw'), len(self.channels))],names=['Channel','datatype'])
        data.index = ind

        self.data = data
        print('Data successfully imported')


    def load_hyp(self, scorefile, date):
        """ Loads hypnogram .txt file and produces DateTimeIndex by sleep stage and 
        stage cycle for cutting
        
        Parameters
        ----------
        scorefile: .txt file
            plain text file with 30-second epoch sleep scores, formatted [hh:mm:ss score]
            NOTE: Time must be in 24h time & scores in consistent format (int or float)
        date: str
            start date of sleep scoring, formatted 'MM-DD-YYYY'

        Returns:
        --------
        stage_cuts: dict
            Nested dict of sleep stages, cycles, and DateTimeIndex
            Ex. {'awake': {1: DateTimeIndex[(...)], 2: DateTimeIndex[(...)]}}
        
        NOTES
        -----
        To Do: Require .txt file to include start date?
        """
                # read the first 8 characters to get the starting time
        with open(scorefile, 'r') as f:
                start_sec = f.read(8)
                
        # read in sleep scores & resample to EEG/EKG frequency
        print('Importing sleep scores...')
        scores = pd.read_csv(scorefile, delimiter='\t', header=None, names=['Score'], usecols=[1], dtype=float)
        scores = pd.DataFrame(np.repeat(scores.values, self.s_freq*30,axis=0), columns=scores.columns)
        
        # reindex to match EEG/EKG data
        ## --> add warning here if index start date isn't found in EEG data
        ind_freq = str(int(1/self.s_freq*1000000))+'us'
        ind_start = date + ' ' + start_sec + '.000'
        ind = pd.date_range(start = ind_start, periods=len(scores), freq=ind_freq)
        scores.index = ind

        # check that scorefile and data overlap
        print('Veryifying timestamp overlap (>5s runtime most likely means no overlap) ...')
        for i in ind:
            if i in self.data.index:
                overlap = i
                self.hyp = pd.Series(scores['Score'])
                print(('Scorefile and Data timestamp match detected at {}.').format(i))
                break
        try:
            overlap
        except:
            raise Exception(('Scorefile times do not overlap with data times. Check "Date" parameter and sleep scoring./n Scorefile start:{}  Data start:{}').format(ind[0], self.data.index[0]))
        
        # add hypnogram column to dataframe (tested against join, concat, merge; this is fastest) 
        #self.data[('Hyp', 'Stage')] = scores

        # get cycle blocks for each stage (detect non-consecutive indices & assign cycles)
        stages = {'awake': 0.0, 'rem': 1.0, 's1': 2.0, 's2': 3.0, 'ads': 4.0, 'sws': 5.0, 'rbrk': 6.0}
        ms = pd.Timedelta(1/self.s_freq*1000, 'ms')
        self.sleepstages = stages

        print('Assigning sleep stages & cycles...')
        stage_cuts = {}
        for stage, val in stages.items():
            stage_dat = self.hyp[self.hyp == val]
            # if the sleep stage is present
            if len(stage_dat) != 0:
                # detect non-consecutive indices to determine cycle breaks
                breaks = stage_dat.index.to_series().diff() != ms
                # assign cycle # for each consecutive block
                cycles = breaks.cumsum()
                
                # get index values for each cycle block, add entry to cyc dict
                cyc = {}
                for c in range(1, max(cycles)+1):
                    idx = cycles[cycles == c].index   
                    cyc[c] = idx
                # add cyc dict to nested stage_cuts dict
                stage_cuts[stage] = cyc
            else:
                stage_cuts[stage] = None

        self.stage_cuts = stage_cuts
        print('Done.')

    def cut_EEG(self, sleepstage='all'):
        """ cut dataset based on loaded hypnogram 
        Parameters
        ----------
        stage: str or list of str
            sleep stages to cut. Options: {'awake', 'rem', s1', 's2', 'ads', 'sws', 'rcbrk'} """
        if sleepstage == 'all':
            stages = self.stage_cuts.keys()
        else:
            stage = sleepstage

        cut_data = {}
        for stage in stages:
            if self.stage_cuts[stage] is not None:
                data = {}
                cycs = len(self.stage_cuts[stage])
                for c in range(1, cycs+1):
                    data[c] = self.data.loc[(self.stage_cuts[stage][c])]
                cut_data[stage] = data

        self.cut_data = cut_data


    def export_csv(self, data=None, stages = 'all', savedir=None):
        """ Export data to csv 
        
        Parameters
        ----------
        data: df [Optional]
            Single dataframe to export. Defaults to self.cut_data dict of dfs produced by Dataset.cut_eeg())
        stages: str or list
            stages to export (for use in combination with dict data type)
            Options: [awake, rem, s1, s2, ads, sws, rcbrk]

        Returns
        -------
        csv files w/ EEG data
        
        """

        # set save dirctory
        if savedir is None:
            savedir = os.getcwd()
            chngdir = input('Files will be saved to ' + savedir + '. Change save directory? [Y/N] ')
            if chngdir == 'Y':
                savedir = input('New save directory: ')
                if not os.path.exists(savedir):
                    createdir = input(savedir + ' does not exist. Create directory? [Y/N] ')
                    if createdir == 'Y':
                        os.makedirs(savedir)
                    else:
                        savedir = input('Try again. Save directory: ')
                        if not os.path.exists(savedir):
                            print(savedir + ' does not exist. Aborting. ')
                            return
        elif not os.path.exists(savedir):
            print(savedir + ' does not exist. Creating directory...')
            os.makedirs(savedir)
        else:
            print('Files will be saved to ' + savedir)
        
        # Option to export a single dataframe
        if data is not None:
            # abort if data is not a single df
            if type(data) != pd.core.frame.DataFrame:
                print(('If data parameter is specified, type must be pd.core.frame.DataFrame. Specified data is type {}').format(type(data)))
                return
            # set savename base & elements for modification
            df = data
            savename_base = self.in_num + '_' + str(df.index[0]).replace(' ', '_').replace(':', '')
            savename_elems = savename_base.split('_')
            spec_stg =  input('Specify sleep stage? [Y/N] ')
            if spec_stg == 'N':
                savename = savename_base
            if spec_stg == 'Y':
                stg = input('Sleep stage: ')
                spec_cyc = input('Specify cycle? [Y/N] ')
                if spec_cyc == 'N':
                    savename = '_'.join(savename_elems[:2]) + '_' + stg + '_' + savename_elems[2]
                elif spec_cyc == 'Y':
                    cyc = input('Sleep stage cycle: ')
                    savename = '_'.join(savename_elems[:2]) + '_' + stg + '_cycle' + cyc + '_' + savename_elems[2]
            
            print('Exporting file...\n')
            data.to_csv(os.path.join(savedir, savename))
            print(('{} successfully exported.\n*If viewing in Excel, remember to set custom date format to  "m/d/yyyy h:mm:ss.000".').format(savename))
     
        # Export a nested dict of dataframes [Default]
        elif data is None:
            print('Exporting files...\n')
            if stages == 'all':
                for stg in self.cut_data.keys():
                    for cyc in self.cut_data[stg].keys():
                        df = self.cut_data[stg][cyc]
                        savename_base = self.in_num + '_' + str(df.index[0]).replace(' ', '_').replace(':', '')
                        savename_elems = savename_base.split('_')
                        savename = '_'.join(savename_elems[:2]) + '_' + stg + '_cycle' + str(cyc) + '_' + savename_elems[2]
                        df.to_csv(os.path.join(savedir, savename))
                        print(('{} successfully exported.').format(savename)) 
            elif type(stages) == list:
                for stg in stages:
                    if stg not in self.cut_data.keys():
                        stg = input('"'+ stg+'" is not a valid sleep stage code or is not present in this dataset. Options: awake rem s1 s2 ads sws rcbrk\nSpecify correct code from options or [skip]: ')
                    if stg == 'skip':
                        continue
                    for cyc in self.cut_data[stg].keys():
                        df = self.cut_data[stg][cyc]
                        savename_base = self.in_num + '_' + str(df.index[0]).replace(' ', '_').replace(':', '')
                        savename_elems = savename_base.split('_')
                        savename = '_'.join(savename_elems[:2]) + '_' + stg + '_cycle' + str(cyc) + '_' + savename_elems[2]
                        df.to_csv(os.path.join(savedir, savename))
                        print(('{} successfully exported.').format(savename))
            elif type(stages) == str:
                stg = stages
                if stg not in self.cut_data.keys():
                    stg = input('"'+ stg+'" is not a valid sleep stage code or is not present in this dataset. Options: awake rem s1 s2 ads sws rcbrk\nSpecify correct code from options or [abort]: ')
                if stg == 'abort':
                    return
                for cyc in self.cut_data[stg].keys():
                    df = self.cut_data[stg][cyc]
                    savename_base = self.in_num + '_' + str(df.index[0]).replace(' ', '_').replace(':', '')
                    savename_elems = savename_base.split('_')
                    savename = '_'.join(savename_elems[:2]) + '_' + stg + '_cycle' + str(cyc) + '_' + savename_elems[2]
                    df.to_csv(os.path.join(savedir, savename))
                    print(('{} successfully exported.').format(savename))
            print('\nDone.')

        else:
            print(('Abort: Data must be type pd.core.frame.DataFrame or dict. Input data is type {}.').format(type(data)))


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
                
                   