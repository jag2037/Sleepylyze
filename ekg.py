""" Descriptor here """

import datetime
import pandas as pd 
import math
import numpy as np 
import scipy.io as io

class EKG:
    """ General class containing EKG analyses
    
    Parameters
    ----------
    dataset: instance of Dataset class
    
    Attributes
    ----------
    """

    def __init__(self, dataset):
        
        self.datafile = dataset.filepath
        self.in_num = dataset.in_num
        self.s_freq = dataset.s_freq
        
        #self.get_info()
        #self.get_chans()
        #self.load_eeg()

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