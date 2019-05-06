""" This file contains the EKG class and helper functions for batch loading """

import datetime
import pandas as pd 
import math
import numpy as np 
import scipy.io as io

class EKG:
    """ General class containing EKG analyses
    
    Parameters
    ----------
    df: pd.DataFrame
        Dataframe containing 'EKG' column data
    
    Attributes
    ----------
    """

    def __init__(self, fname, fpath):
        
        self.filename = fname
        self.filepath = os.path.join(fpath, fname)
        self.in_num, self.start_date, self.slpstage, self.cycle = fname.split('_')[:-1]
        
        self.load_ekg()
        

    def load_ekg(self):
        """ Load ekg data and extract sampling frequency """
        data = pd.read_csv(self.filepath, header = [0, 1], index_col = 0)['EKG']
        data.index = pd.to_datetime(data.index)
        self.data = data

        diff = data.index.to_series().diff()[1:2]
        self.s_freq = 1000000/diff[0].microseconds

    ### EKG analysis methods
    # --> want to create a new df for threshold (don't add to existing)
    def set_Rthres(self, mw_size=0.2, upshift=1.05):
        """ set R peak detection threshold based on moving average + %signal upshift """
        print('Calculating moving average with {} sec window and a {} upshift...'.format(mw_size, upshift))
        
        mw = int(mw_size*self.s_freq) # moving window size in number of samples (must be an integer)
        mavg = self.data.Raw.rolling(mw).mean() # calculate rolling average on column "EKG"

        # replace edge nans with overall average
        ekg_avg = np.mean(self.data['Raw'])
        mov_avg = [ekg_avg if math.isnan(x) else x for x in mavg]

        det_thres = [x*upshift for x in mov_avg] # set detection threshold as +5% of moving average
        self.data['EKG_thres'] = det_thres # add a column onto the EEG dataframe

    def detect_Rpeaks(self):
        """ detect R peaks from raw signal """
        print('Detecting R peaks...')
        
        window = []
        peaklist = []
        listpos = 0 # use a counter to move over the different data columns
        for datapoint in self.data.Raw:
            m_avg = self.data.EKG_thres[listpos] # get the moving average at a given position
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
            
            self.r_times = [self.data.index[x] for x in peaklist] # get peak times           
            self.r_vals = [self.data.Raw[x] for x in peaklist] # get peak values
        print('R peak detection complete')

    def calc_RR(self):
        """ Calculate the intervals between successive R-R peaks, as well as first order derivative """
        rr = []
        for i in range(len(self.r_times)-1):
            rr.append(self.r_times[i+1]-self.r_times[i]) # gives you a timedelta object
        rr_us = np.array([x.microseconds for x in rr]) # convert timedelta to microseconds
        
        self.rr_int = rr_us/1e6 # convert to seconds
        self.rr_int_diff = np.diff(self.rr_int)
        self.rr_int_diffsq = self.rr_int_diff**2

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