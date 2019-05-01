""" import this class by using the __init__.py file 'from .ioeeg import Dataset' """

import datetime
import math 
import numpy as np 
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
            filepath = fpath + fname
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
        data = pd.read_csv(self.filepath, delim_whitespace=True, header=None, skiprows=15, usecols=range(3,end_col),
                               dtype = np.float64)
        
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
        """ Loads hypnogram .txt file and aligns with Dataset
        
        Parameters
        ----------
        scorefile: .txt file
            plain text file with 30-second epoch sleep scores, formatted [hh:mm:ss score]
        date: str
            start date of sleep scoring, formatted 'MM-DD-YYYY'
        
        NOTES
        -----
        To Do: Require .txt file to include start date?
        """
        # read the first 8 characters to get the starting time
        with open(scorefile, 'r') as f:
                start_sec = f.read(8)
                
        # read in sleep scores & resample to EEG/EKG frequency
        scores = pd.read_csv(scorefile, delimiter='\t', header=None, names=['Score'], usecols=[1], dtype=float)
        scores = pd.DataFrame(np.repeat(scores.values, self.s_freq*30,axis=0), columns=scores.columns)
        
        # reindex to match EEG/EKG data
        ind_freq = str(int(1/self.s_freq*1000000))+'us'
        ind_start = date + ' ' + start_sec + '.000'
        ind = pd.date_range(start = ind_start, periods=len(scores), freq=ind_freq)
        scores.index = ind
        scores = pd.Series(scores['Score'])
        
        # add hypnogram column to dataframe (tested against join, concat, merge; this is fastest) 
        self.data[('Hyp', 'Score')] = scores


    # EKG Analysis Methods #
    # --> want to preface self.mw params, etc. with ekg so not confused with spindle params
    # --> want to create a new df for threshold (don't add to existing)
    def set_Rthres(self, mw_size=0.2, upshift=1.05):
        """ set R peak detection threshold based on moving average + %signal upshift """
        print('Calculating moving average with {} sec window and a {} upshift...'.format(mw_size, upshift))
        s_freq = self.s_freq
        data = self.data
        
        mw = int(mw_size*s_freq) # moving window size in number of samples (must be an integer)
        mavg = data.EKG.rolling(mw).mean() # calculate rolling average on column "EKG"

        # replace edge nans with overall average
        ekg_avg = np.mean(data['EKG'])
        mov_avg = [ekg_avg if math.isnan(x) else x for x in mavg]

        det_thres = [x*upshift for x in mov_avg] # set detection threshold as +5% of moving average
        data['EKG_thres'] = det_thres # add a column onto the EEG dataframe

    def detect_Rpeaks(self):
        """ detect R peaks from raw signal """
        print('Detecting R peaks...')
        data = self.data
        
        window = []
        peaklist = []
        listpos = 0 # use a counter to move over the different data columns
        for datapoint in data.EKG:
            m_avg = data.EKG_thres[listpos] # get the moving average at a given position
            if (datapoint <= m_avg) and (len(window) < 1): # If signal has not crossed m_avg -> do nothing
                listpos += 1
            elif (datapoint > m_avg): # If signal crosses moving average, mark ROI
                window.append(datapoint)
                listpos += 1
            else: #If signal drops below moving average -> find local maxima within window
                peak = max(window)
                beatposition = listpos - len(window) + (window.index(max(window))) #Notate the position of the point on the X-axis
                peaklist.append(beatposition) #Add detected peak to list
                window = [] #Clear marked ROI
                listpos += 1
            
            self.r_times = [data.index[x] for x in peaklist] # get peak times           
            self.r_vals = [data.EKG[x] for x in peaklist] # get peak values
        print('R peak detection complete')

    def calc_RR(self):
        """ Calculate the intervals between successive R-R peaks """
        r_times = self.r_times
        rr = []
        for i in range(len(r_times)-1):
            rr.append(r_times[i+1]-r_times[i]) # gives you a timedelta object
        rr_us = np.array([x.microseconds for x in rr]) # convert timedelta to microseconds
        self.rr_int = rr_us/1e6 # convert to seconds

    def calc_RRdiff(self):
        """ Calculate the difference between successive R-R intervals, as the difference squared """
        rr_int = self.rr_int
        rr_diff = []
        rr_diffsq = []
        for i in range(len(rr_int)-1):
            diff = abs(rr_int[i+1]-rr_int[i])
            rr_diff.append(diff)
            rr_diffsq.append(diff**2)
        
        self.rr_int_diff = rr_diff 
        self.rr_int_diffsq = rr_diffsq

    def calc_RRstats(self):
        """ Calculate commonly used HRV statistics """   
        # heartrate in bpm
        self.heartrate = np.mean(self.rr_int)*60
        print('Average heartrate = {}bpm'.format(int(self.heartrate)))

        # inter-beat interval & SD
        self.ibi = np.mean(self.rr_int)
        self.sdnn = np.std(self.rr_int)
        print('Average IBI (sec) = {0:.2f} SD = {0:.2f}'.format(self.ibi, self.sdnn))

        # SD & RMS of differences between successive RR intervals
        self.sdsd = np.std(self.rr_int_diff)
        self.rmssd = np.sqrt(self.rr_int_diffsq)

        # nn20 & nn50

        # pnn20 & pnn50
        print('Call dataset.__dict__ for additional statistics')

    def ekgstats(self):
        self.set_Rthres()
        self.detect_Rpeaks()
        self.calc_RR()
        self.calc_RRdiff()
        self.calc_RRstats()



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
                
                   