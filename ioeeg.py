""" import this class by using the __init__.py file 'from .ioeeg import Dataset' """

import datetime
import math 
import numpy as np 
import pandas as pd
import scipy.io as io

class Dataset:
    """ General class containing EEG recordings
    
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

    
    def get_info(self):
        """ Read in the header block and extract useful info """
        # extract IN
        self.in_num = self.fname.split("_")[0]
        print("Patient identifier:", self.in_num)
        
        # extract sampling freq, # of channels, headbox sn
        with open(self.filepath, 'r') as f:	# pull out the header w/ first recording
            header = []
            for i in range(1,17):
                header.append(f.readline())	# use readline (NOT READLINES) here to save memory for large files
        self.s_freq = int(float(header[7].split()[3])) # extract sampling frequency
        self.chans = int(header[8].split()[2]) # extract number of channels as integer
        self.hbsn = int(header[10].split()[3]) # extract headbox serial number
        self.start_time = header[15].split()[1] # extract start time
         
    def get_chans(self):
        """ Define the channel list for the detected headbox """
        chans = self.chans
        hbsn = self.hbsn
        
        def check_chans(chans, expected_chans):
            print(chans, ' of ', expected_chans, ' expected channels found')
            if chans != expected_chans:	# this may not be a proper check
                print('WARNING: Not all expected channels for this headbox were detected. Proceeding with detected channels')
        
        if hbsn == 125:
            hbid = "MOBEE 32"
            print('Headbox:', hbid)
            expected_chans = 35
            check_chans(chans, expected_chans)
            channels_all = ['REF','FP1','F7','T3','A1','T5','O1','F3','C3','P3','FPZorEKG','FZ','CZ',
                'PZ','FP2','F8','T4','A2','T6','O2','F4','C4','P4','AF7','AF8','FC5','FC6','FC1','FC2',
                'CP5','CP6','CP1','CP2','OSAT','PR']
            channels = channels_all[:-2]
            channels_rmvd = channels_all[-2:]
            print('Removed the following channels: \n', channels_rmvd)
            
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
    
    def load_eeg(self):
        """ docstring here """
        # set vars (since this code was already written as standalone)
        s_freq = self.s_freq
        start_time = self.start_time
        filepath = self.filepath
        channels = self.channels
        
        # set the last column of data to import
        end_col = len(channels) + 3 # works for MOBEE32, check for other headboxes
        
        print('Importing EEG data...')
        data_full = pd.read_csv(filepath, delim_whitespace=True, header=None, skiprows=15, 
        usecols=range(0,end_col)) #.transpose() # read in the data and transpose the matrix --> maybe transpose later instead

        # Determine microsecond values & append to datetime
        step_size = int(1000000/s_freq)
        readings = range(1, s_freq+1) # add one bc range is non-inclusive
        usecs = range(0, 1000000, step_size)
        usec_dict = dict(zip(readings, usecs))

        first_us = s_freq - (data_full[1].eq(start_time).sum())
        reps, rem = divmod((len(data_full) - first_us), s_freq) # get the number of full repeates and the remainder
        last_us = s_freq - rem
        # make the list of usecs as strings
        usec_col = list(map(str,((list(usecs[first_us:]) + list(np.tile(usecs, reps)) + list(usecs[:last_us])))))

        new_time = [] # make the new column
        for i in range(0, len(data_full)): # fill it with values
            new_time.append(data_full[1][i] + ':' + usec_col[i])
        data_full[1] = new_time # replace the old column

        # combine date & time columns to a datetime object
        dtime = []
        for i, (d, t) in enumerate(zip(data_full[0], data_full[1])):
            mo, d, yr = [int(d) for d in data_full[0][i].split('/')]
            h, m, s, us = [int(t) for t in data_full[1][i].split(':')]
            dtime.append(datetime.datetime(yr, mo, d, h, m, s, us))
            #start_time = datetime.datetime(yr, mo, d, h, m, s, us) # replace the start_time variable 

        # make a new dataframe with the proper index & column names
        data = data_full.loc[:,3:]
        data.columns = channels
        data.index = dtime

        self.data = data
        print('Data successfully imported')



class Rpeaks(Dataset):
    """ Class containing R-peak detections 
    
    Attributes
    ----------
    to be completed....
    
    Notes
    ------
    To be completed:
        - Add additional pertinent HRV stats (pnn20, etc)
    """
    def __init__(self, fname, fpath):
        super().__init__(fname, fpath)
        self.set_Rthres()
        self.detect_rpeaks()
        self.calc_RR()
        self.calc_RRdiff()
        self.RRstats()
    
    def set_Rthres(self, mw_size=0.2, upshift=1.05):
        """ set R peak detection threshold based on moving average + %signal upshift """
        s_freq = self.s_freq
        data = self.data
        
        mw = int(mw_size*s_freq) # moving window size in number of samples (must be an integer)
        mavg = data.EKG.rolling(mw).mean() # calculate rolling average on column "EKG"
    
        # replace edge nans with overall average
        ekg_avg = np.mean(data['EKG'])
        mov_avg = [ekg_avg if math.isnan(x) else x for x in mavg]
    
        det_thres = [x*upshift for x in mov_avg] # set detection threshold as +5% of moving average
    
        # make a new dataframe with just the EKG & detection threshold
        df = pd.DataFrame(data['EKG'])
        df['det_thres'] = det_thres
    
        self.ekg = df
    
    def detect_rpeaks(self):
        """ detect R peaks from raw signal """
        window = []
        peaklist = []
        listpos = 0 # use a counter to move over the different data columns
        for datapoint in self.ekg.EKG:
            m_avg = self.ekg.det_thres[listpos] # get the moving average at a given position
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
            
            self.r_times = [self.ekg.index[x] for x in peaklist] # get peak times           
            self.r_vals = [self.ekg.EKG[x] for x in peaklist] # get peak values
            
    
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
    
    def RRstats(self):
        """ Calculate commonly used HRV statistics """   
        # heartrate
        self.heartrate = np.mean(self.rr_int)*60
    
        # inter-beat interval & SD
        self.ibi = np.mean(self.rr_int)
        self.sdnn = np.std(self.rr_int)
    
        # SD & RMS of differences between successive RR intervals
        self.sdsd = np.std(self.rr_int_diff)
        self.rmssd = np.sqrt(self.rr_int_diffsq)
    
        # nn20 & nn50
    
        # pnn20 & pnn50
