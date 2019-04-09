""" descriptor here """

import datetime
import pandas as pd 
import math
import numpy as np 
import scipy.io as io

def set_Rthres(dataset, mw_size=0.2, upshift=1.05):
    """ set R peak detection threshold based on moving average + %signal upshift """
    s_freq = dataset.s_freq
    data = dataset.data
    
    mw = int(mw_size*s_freq) # moving window size in number of samples (must be an integer)
    mavg = data.EKG.rolling(mw).mean() # calculate rolling average on column "EKG"

    # replace edge nans with overall average
    ekg_avg = np.mean(data['EKG'])
    mov_avg = [ekg_avg if math.isnan(x) else x for x in mavg]

    det_thres = [x*upshift for x in mov_avg] # set detection threshold as +5% of moving average

    # make a new dataframe with just the EKG & detection threshold
    #df = pd.DataFrame(data['EKG'])
    data['EKG_thres'] = det_thres


def detect_Rpeaks(dataset):
    """ detect R peaks from raw signal """
    data = dataset.data
    
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
        
        dataset.r_times = [data.index[x] for x in peaklist] # get peak times           
        dataset.r_vals = [data.EKG[x] for x in peaklist] # get peak values

def calc_RR(dataset):
    """ Calculate the intervals between successive R-R peaks """
    r_times = dataset.r_times
    rr = []
    for i in range(len(r_times)-1):
        rr.append(r_times[i+1]-r_times[i]) # gives you a timedelta object
    rr_us = np.array([x.microseconds for x in rr]) # convert timedelta to microseconds
    dataset.rr_int = rr_us/1e6 # convert to seconds

def calc_RRdiff(dataset):
    """ Calculate the difference between successive R-R intervals, as the difference squared """
    rr_int = dataset.rr_int
    rr_diff = []
    rr_diffsq = []
    for i in range(len(rr_int)-1):
        diff = abs(rr_int[i+1]-rr_int[i])
        rr_diff.append(diff)
        rr_diffsq.append(diff**2)
    
    dataset.rr_int_diff = rr_diff 
    dataset.rr_int_diffsq = rr_diffsq

def calc_RRstats(dataset):
    """ Calculate commonly used HRV statistics """   
    # heartrate in bpm
    dataset.heartrate = np.mean(dataset.rr_int)*60

    # inter-beat interval & SD
    dataset.ibi = np.mean(dataset.rr_int)
    dataset.sdnn = np.std(dataset.rr_int)

    # SD & RMS of differences between successive RR intervals
    dataset.sdsd = np.std(dataset.rr_int_diff)
    dataset.rmssd = np.sqrt(dataset.rr_int_diffsq)

    # nn20 & nn50

    # pnn20 & pnn50

def ekgstats(dataset):
    set_Rthres(dataset)
    detect_Rpeaks(dataset)
    calc_RR(dataset)
    calc_RRdiff(dataset)
    calc_RRstats(dataset)