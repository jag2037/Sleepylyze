""" This file contains the EKG class and helper functions for batch loading """

import datetime
import numpy as np 
import os
import pandas as pd 
import scipy as sp
import scipy.io as io
import scipy.stats as stats
import pyhrv.nonlinear as nl

from mne.time_frequency import psd_array_multitaper
from scipy.signal import welch

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
                print('--> To load data, set min_dur to False')
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

    def time_stats(self):
        """ Calculate commonly used HRV statistics. Min/max HR is determined over 5 RR intervals 

            TO DO: Reformat as stats dictionary
        """   
        print('Calculating time domain statistics...')
        time_stats = {}
        # heartrate in bpm
        time_stats['HR_avg'] = 60/np.mean(self.rr_int)*1000
        #print('Average heartrate (bpm) = {}'.format(int(self.heartrate_avg)))
        
        rollmean_rr = pd.Series(self.rr_int).rolling(5).mean()
        mx_rr, mn_rr = np.nanmax(rollmean_rr), np.nanmin(rollmean_rr)
        time_stats['HR_max'] = 60/mn_rr*1000
        time_stats['HR_min'] = 60/mx_rr*1000
        #print('Maximum heartrate (bpm) = {}'.format(int(self.heartrate_max)))
        #print('Minimum heartrate (bpm) = {}'.format(int(self.heartrate_min)))


        # inter-beat interval & SD (ms)
        time_stats['ibi'] = np.mean(self.rr_int)
        time_stats['sdrr'] = np.std(self.rr_int)
        #print('Average IBI (ms) = {0:.2f} SD = {1:.2f}'.format(self.ibi, self.sdrr))

        # SD & RMS of differences between successive RR intervals (ms)
        time_stats['sdsd'] = np.std(self.rr_int_diff)
        time_stats['rmssd'] = np.sqrt(np.mean(self.rr_int_diffsq))

        # rr20 & rr50
        # prr20 & prr50
        time_stats['prr20'] = sum(np.abs(self.rr_int_diff) >= 20.0)/len(self.rr_int_diff)*100
        time_stats['prr50'] = sum(np.abs(self.rr_int_diff) >= 50.0)/len(self.rr_int_diff)*100
        #print('pRR20 = {0:.2f}% & pRR50 = {1:.2f}%'.format(self.prr20, self.prr50))

        # hrv triangular index
        bin_width = 7.8125
        stat, bin_edges, bin_num = stats.binned_statistic(self.rr_int, self.rr_int, bins = np.arange(min(self.rr_int), max(self.rr_int) + bin_width, bin_width), statistic='count')
        time_stats['hti'] = sum(stat)/max(stat)
        # triangular interpolation of NN interval
        # if 1st bin is max, can't calculatin TINN
        if stat[0] == max(stat):
            time_stats['tinn'] = None
        else:
            # this calculation is wrong
            time_stats['tinn'] = bin_edges[-1] - bin_edges[0]
        #print('HRV Triangular Index (HTI) = {0:.2f}.\nTriangular Interpolation of NN Interval Histogram (TINN) (ms) = {1}\n\t*WARNING: TINN calculation may be incorrect. Formula should be double-checked'.format(self.hti, self.tinn))
        #print('Call ekg.__dict__ for all statistics')
        print('Time domain stats stored in ekg.time_stats\n')

    
    def interpolateRR(self):
        """ Resample RR tachogram to original sampling frequency (since RRs are not evenly spaced)
            and interpolate for power spectral estimation 
            *Note: adapted from pyHRV
        """
        fs = self.s_freq
        t = np.cumsum(self.rr_int)
        t -= t[0]
        f_interp = sp.interpolate.interp1d(t, self.rr_int, 'cubic')
        t_interp = np.arange(t[0], t[-1], 1000./fs)
        self.rr_interp = f_interp(t_interp)
        self.fs_interp = self.s_freq


    def psd_welch(self, window='hamming'):
        """ Calculate welch power spectral density """
        # set nfft to guidelines of power of 2 above len(data), min 256 (based on MATLAB guidelines)
        nfft = max(256, 2**(int(np.log2(len(self.rr_interp))) + 1))
        
        # Adapt 'nperseg' according to the total duration of the NNI series (5min threshold = 300000ms)
        if max(np.cumsum(self.rr_int)) < 300000:
            nperseg = nfft
        else:
            nperseg = 300
        
        # default overlap = 50%
        f, Pxx = welch(self.rr_interp, fs=4, window=window, scaling = 'density', nfft=nfft, 
                        nperseg=nperseg)
        self.psd_welch =psd_welch = {'freqs':f, 'pwr': Pxx, 'nfft': nfft, 'nperseg': nperseg}

    def psd_mt(self, bandwidth=0.02):
        """ Calculate multitaper power spectrum 

            Params
            ------
            bandwidth: float
                frequency resolution (NW)

            Returns
            -------
            psd_mt: dict
                'freqs': ndarray
                'psd': ndarray. power spectral density in (V^2/Hz). 10log10 to convert to dB.

        """
        pwr, freqs = psd_array_multitaper(self.rr_interp, self.fs_interp, adaptive=True, 
                                            bandwidth=bandwidth, normalization='full', verbose=0)
        self.psd_mt = {'freqs': freqs, 'pwr': pwr}

    def calc_fstats(self, method, bands):
        """ Calculate different frequency band measures 
            TO DO: add option to change bands
            Note: modified from pyHRV
            * normalized units are normalized to total lf + hf power, according to Heathers et al. (2014)
        """
        if method is None:
            method = input('Please enter PSD method (options: "welch", "mt"): ')
        if method == 'welch':
            psd = self.psd_welch
        elif method == 'mt':
            psd = self.psd_mt
        
        # set frequency bands
        if bands is None:
            ulf = None
            vlf = (0.000, 0.04)
            lf = (0.04, 0.15)
            hf = (0.15, 0.4)
            args = (ulf, vlf, lf, hf)
            names = ('ulf', 'vlf', 'lf', 'hf')
        freq_bands = dict(zip(names, args))
        self.freq_bands = freq_bands
        
        # get indices and values for frequency bands in calculated spectrum
        fband_vals = {}
        for key in freq_bands.keys():
            fband_vals[key] = {}
            if freq_bands[key] is None:
                fband_vals[key]['idx'] = None
                fband_vals[key]['pwr'] = None
            else:
                fband_vals[key]['idx'] = np.where((freq_bands[key][0] <= psd['freqs']) & (psd['freqs'] <= freq_bands[key][1]))[0]
                fband_vals[key]['pwr'] = psd['pwr'][fband_vals[key]['idx']]
                
        self.psd_fband_vals = fband_vals
        
        # calculate stats 
        freq_stats = {}
        freq_stats['method'] = method
        freq_stats['total_pwr'] = sum(filter(None, [np.sum(fband_vals[key]['pwr']) for key in fband_vals.keys()]))
        # by band
        for key in freq_bands.keys():
            freq_stats[key] = {}
            freq_stats[key]['freq_range'] = freq_bands[key]
            if freq_bands[key] is None:
                freq_stats[key]['pwr_ms2'] = None
                freq_stats[key]['pwr_peak'] = None
                freq_stats[key]['pwr_log'] = None
                freq_stats[key]['pwr_%'] = None
                freq_stats[key]['pwr_nu'] = None
            else:
                freq_stats[key]['pwr_ms2'] = np.sum(fband_vals[key]['pwr'])
                peak_idx = np.where(fband_vals[key]['pwr'] == max(fband_vals[key]['pwr']))[0][0]
                freq_stats[key]['pwr_peak'] = psd['freqs'][fband_vals[key]['idx'][peak_idx]]
                freq_stats[key]['pwr_log'] = np.log(freq_stats[key]['pwr_ms2'])
                freq_stats[key]['pwr_%'] = freq_stats[key]['pwr_ms2']/freq_stats['total_pwr']*100
        
        # add normalized units to lf & hf bands
        for key in ['lf', 'hf']:
            freq_stats[key]['pwr_nu'] = freq_stats[key]['pwr_ms2']/(freq_stats['lf']['pwr_ms2'] + freq_stats['hf']['pwr_ms2'])*100
        # add lf/hf ratio
        freq_stats['lf/hf'] = freq_stats['lf']['pwr_ms2']/freq_stats['hf']['pwr_ms2']
       
        self.freq_stats = freq_stats


    def freq_stats(self, fs=4, method='mt', bandwidth=0.02, window='hamming', bands=None):
        """ Calculate frequency domain statistics 

        Parameters
        ----------
        method: str, optional (default: 'mt')
            Method to compute power spectra. options: 'welch', 'mt' (multitaper)
        bandwith: float, optional (default: 0.02)
            Bandwidth for multitaper power spectral estimation
        window: str, optional (default: 'hamming')
            Window to use for welch FFT. See mne.time_frequency.psd_array_multitaper for options
        bands: Nonetype
            Frequency bands of interest. Leave as none for default. To do: update for custom bands
        """
        # resample & interpolate tachogram
        print('Interpolating and resampling RR tachogram...')
        self.interpolateRR()
       
       # calculate power spectrum
        print('Calculating power spectrum...')
        if method == 'mt':
            self.psd_mt(bandwidth)
        elif method == 'welch':
            self.psd_welch(window)
        
        #calculate frequency domain statistics
        print('Calculating frequency domain measures...')
        self.calc_fstats(method, bands)
        print('Frequency measures stored in ekg.freq_stats\n')

    
    def nonlinear_stats(self):
        """ calculate nonlinear dynamics poincare & sample entropy 
            Note: From pyhrv non-linear module """
        print('Calculating nonlinear statistics...')
        nonlinear_stats = {}

        # poincare
        pc = nl.poincare(self.rr_int)
        nonlinear_stats['poincare'] = {'sd1': pc[1], 'sd2': pc[2], 'sd_ratio':pc[3], 'ellipse_area':pc[4], 
                        'plot':pc[0]}
        # sample entropy (tolerance and dim set to match Riganello et al. 2018)
        nonlinear_stats['sampEn'] = nl.sample_entropy(self.rr_int, dim=2, tolerance=0.15)

        # detrended fluctuation analysis
        dfa = nl.dfa(self.rr_int)
        nonlinear_stats['dfa'] = {'alpha1': dfa[1], 'alpha2': dfa[2], 'plot': dfa[0]}

        self.nonlinear_stats = nonlinear_stats
        print('Nonlinear stats stored in ekg.nonlinear_stats\n')


    def hrv_stats(self, mw_size = 0.2, upshift = 1.05):
        """ Calculate all statistics on EKG object 

            TO DO: Add freq_stats arguments to hrv_stats params? 
                   Divide into 5-min epochs 
        """
        # detect R peaks
        self.set_Rthres(mw_size, upshift)
        self.detect_Rpeaks()

        # divide cycles into 5-minute epochs
        #if self.cycle_len_secs > 60*5:

        # make RR tachogram
        self.calc_RR()
        self.time_stats()
        self.freq_stats()
        self.nonlinear_stats()
        print('Done.')



def loadEKG_batch(path, stage=None, min_dur=True):
    """ Batch import all raw data from a given directory 
    
    Parameters
    ----------
    dirc: str
        Directory containing raw files to import
    stage: str (Default: None)
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


