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

    def __init__(self, fname, fpath, min_dur=True):
        
        self.filename = fname
        self.filepath = os.path.join(fpath, fname)
        self.in_num, self.start_date, self.slpstage, self.cycle = fname.split('_')[:-1]
        
        self.load_ekg(min_dur)
        
        

    def load_ekg(self, min_dur):
        """ 
        Load ekg data and extract sampling frequency. 
        
        Parameters
        ----------
        min_dur: bool, default: True
            If set to True, will not load files shorter than 5 minutes long 
        """
        
        data = pd.read_csv(self.filepath, header = [0, 1], index_col = 0, parse_dates=True)['EKG']
        
        # Check cycle length against 5 minute duration minimum
        self.cycle_len_secs = (data.index[-1] - data.index[0]).total_seconds()
        if self.cycle_len_secs < 60*5:
            if min_dur == True:
                print(('{} {} {} is shorter than 5 minutes. Cycle will not be loaded.').format(self.in_num, self.slpstage, self.cycle))
                return
            else:
                print(('* WARNING: {} {} {} is shorter than 5 minutes.').format(self.in_num, self.slpstage, self.cycle))
                self.data = data
        else:
            self.data = data
        
        diff = data.index.to_series().diff()[1:2]
        self.s_freq = 1000000/diff[0].microseconds

        print(('EKG successfully imported:\n\tPatient Identifier: {}\n\tStart Date:{}\n\tStage/cycle: {} {}\n').format(self.in_num, self.start_date, self.slpstage, self.cycle))

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

        raw = pd.Series(self.data['Raw'])
        thres = pd.Series(self.data['EKG_thres'])
        
        peaks = []
        x = 0
        while x < len(self.data):
            if raw[x] > thres[x]:
                roi_start = x
                # count forwards to find down-crossing
                for h in range(x, len(self.data), 1):
                    if raw[h] < thres[h]:
                        roi_end = h
                        break
                # get maximum between roi_start and roi_end
                peak = raw[x:h].idxmax()
                peaks.append(peak)
                # advance the pointer
                x = h
            else:
                x += 1

        self.rpeaks = raw[peaks]
        print('R peak detection complete')

    def calc_RR(self):
        """ Calculate the intervals between successive R-R peaks, as well as first order derivative.
            Returns: rr_int, rr_int_diff, and rr_int_diffsq in milliseconds """
        # get time between peaks and convert to seconds
        self.rr_int = np.diff(self.rpeaks.index)/np.timedelta64(1, 'ms')
        # get difference between intervals & squared
        self.rr_int_diff = np.diff(self.rr_int)
        self.rr_int_diffsq = self.rr_int_diff**2

    def calc_RRstats(self):
        """ Calculate commonly used HRV statistics """   
        # heartrate in bpm
        secs = int((self.data.index[-1]-self.data.index[0])/np.timedelta64(1, 's'))
        self.heartrate = (len(self.rpeaks)/secs) *60
        print('Average heartrate = {}bpm'.format(int(self.heartrate)))

        # inter-beat interval & SD (ms)
        self.ibi = np.mean(self.rr_int)
        self.sdrr = np.std(self.rr_int)
        print('Average IBI (ms) = {0:.2f} SD = {0:.2f}'.format(self.ibi, self.sdnn))

        # SD & RMS of differences between successive RR intervals (ms)
        self.sdsd = np.std(self.rr_int_diff)
        self.rmssd = np.sqrt(np.mean(self.rr_int_diffsq))

        # rr20 & rr50
        # prr20 & prr50
        self.prr20 = sum(np.abs(ekg.rr_int_diff) >= 20.0)/len(ekg.rr_int_diff)*100
        self.prr50 = sum(np.abs(ekg.rr_int_diff) >= 50.0)/len(ekg.rr_int_diff)*100
        print('pRR20 = {0:.2f}% & pRR50 = {0:.2f}%'.format(self.prr20, self.prr50))

        # hrv triangular index
        stat, bin_edges, bin_num = stats.binned_statistic(ekg.rr_int, ekg.rr_int, bins = np.arange(min(ekg.rr_int), max(ekg.rr_int) + 7.8125, 7.8125), statistic='count')
        self.hti = sum(stat)/max(stat)
        self.tinn = bin_edges[-1] - bin_edges[0]
        print('HRV Triangular Index (HTI) = {}.\nTriangular Interpolation of NN Interval Histogram (TINN) = {}ms'.format(self.hti, self.tinn))
        print('Call ekg.__dict__ for all statistics')

    def ekgstats(self, mw_size = 0.2, upshift = 1.05):
        """ Calculate all statistics on EKG object """
        self.set_Rthres(mw_size, upshift)
        self.detect_Rpeaks()

        # divide cycles into 5-minute epochs
        #if self.cycle_len_secs > 60*5:

        self.calc_RR()
        self.calc_RRstats()



def loadEKG_batch(path, stage=None, min_dur=True):
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

    NOTE: this fails if stage not specified and folder does not contain any cycles
    of a given sleep stage. Can write in code to pull present stages from filenames
    to fix this.

    """

    files = [x for x in os.listdir(path) if stage in x]
    if len(files) == 0:
        print('"'+ stage +'" is not a valid sleep stage code or is not present in this dataset. Options: awake rem s1 s2 ads sws rcbrk. Aborting.')

    names = ['ekg'+ str(n) for n, m in enumerate(files)]
    ekg_set = []
    for file, name in zip(files, names):
        name = EKG(file, path, min_dur)
        ekg_set.append(name)

    print('\nDone.')
    return ekg_set


