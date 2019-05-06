""" This file contains the EKG class and helper functions for batch loading 

    TO DO:
        1. Add helper function for batch loading -- Completed 5-6-19
        2. Optimize detect_Rpeaks() for speed
    """

import datetime
import numpy as np 
import os
import pandas as pd 
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
        print(('EKG successfully imported:\n\tPatient Identifier: {}\n\tStart Date:{}\n\tStage/cycle: {} {}\n').format(self.in_num, self.start_date, self.slpstage, self.cycle))
        

    def load_ekg(self):
        """ Load ekg data and extract sampling frequency """
        data = pd.read_csv(self.filepath, header = [0, 1], index_col = 0, parse_dates=True)['EKG']
        #data.index = pd.to_datetime(data.index)
        self.data = data

        diff = data.index.to_series().diff()[1:2]
        self.s_freq = 1000000/diff[0].microseconds

    def set_Rthres(self, mw_size, upshift):
        """ set R peak detection threshold based on moving average + %signal upshift """
        print('Calculating moving average with {} sec window and a {} upshift...'.format(mw_size, upshift))
        
        # convert moving window to sample & calc moving average over window
        mw = int(mw_size*self.s_freq)
        mavg = self.data.Raw.rolling(mw).mean()

        # replace edge nans with overall average
        ekg_avg = np.mean(self.data['Raw'])
        mavg = mavg.fillna(ekg_avg)

        # set detection threshold as +5% of moving average
        det_thres = mavg*upshift
        self.data['EKG_thres'] = det_thres # can remove this for speed, just keep as series

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
            
            self.rpeaks = self.data['Raw'].loc[self.data.index[peaklist]]
        print('R peak detection complete')

    def calc_RR(self):
        """ Calculate the intervals between successive R-R peaks, as well as first order derivative """
        # get time between peaks and convert to seconds
        self.rr_int = np.diff(self.rpeaks.index)/np.timedelta64(1, 's')
        # get difference between intervals & squared
        self.rr_int_diff = np.diff(self.rr_int)
        self.rr_int_diffsq = self.rr_int_diff**2

    def calc_RRstats(self):
        """ Calculate commonly used HRV statistics """   
        # heartrate in bpm
        secs = int((self.data.index[-1]-self.data.index[0])/np.timedelta64(1, 's'))
        self.heartrate = (len(self.rpeaks)/secs) *60
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
        print('Call ekg.__dict__ for additional statistics')

    def ekgstats(self, mw_size = 0.2, upshift = 1.05):
        """ Calculate all statistics on EKG object """
        self.set_Rthres(mw_size, upshift)
        self.detect_Rpeaks()
        self.calc_RR()
        self.calc_RRstats()



def loadEKG_batch(path, stage=None):
    """ Batch import all raw data from a given directory 
    
    Parameters
    ----------
    dirc: str
        Directory containing raw files to import
    stage: str {Default: None}
        Sleep stage to import [Options: awake, rem, s1, s2, ads, sws, rcbrk]

    Returns
    -------
    List of EKG class objects

    """

    files = [x for x in os.listdir(path) if stage in x]
    if len(files) == 0:
        print('"'+ stage +'" is not a valid sleep stage code or is not present in this dataset. Options: awake rem s1 s2 ads sws rcbrk. Aborting.')

    names = ['ekg'+ str(n) for n, m in enumerate(files)]
    ekg_set = []
    for file, name in zip(files, names):
        name = EKG(file, path)
        ekg_set.append(name)

    return ekg_set


