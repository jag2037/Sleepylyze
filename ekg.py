""" This file contains the EKG class and helper functions for batch loading 

    TO DO: Update stats calculation to run on NN intervals instead of RR intervals
        -- add artifact_auto to __init__ call
"""

import datetime
import numpy as np 
import os
import pandas as pd 
import scipy as sp
import scipy.io as io
import scipy.stats as stats
import statistics
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

    def __init__(self, fname, fpath, min_dur=True, epoched=True, mw_size=0.08, upshift=1.02, rm_artifacts=False):
        """ Initialize raw EKG object

        Parameters
        ----------
        fname: str
            filename
        fpath: str
            path to file
        min_dur: bool (default:True)
            only load files that are >= 5 minutes long
        epoched: bool (default: True)
            Whether file was epoched using ioeeg
        mw_size: float (default: 0.08)
            Moving window size for R peak detection (seconds)
        upshift: float (default: 1.02)
            Detection threshold upshift for R peak detection (% of signal)
        rm_artifacts: bool (default: False)
            Apply IBI artifact removal algorithm

        Returns
        -------
        EKG object w/ R peak detections and calculated inter-beat intervals
         """

        # set metadata
        filepath = os.path.join(fpath, fname)
        if epoched == False:
            in_num, start_date, slpstage, cycle = fname.split('_')[:4]
        elif epoched == True:
            in_num, start_date, slpstage, cycle, epoch = fname.split('_')[:5]
        self.metadata = {'file_info':{'in_num': in_num,
                                'fname': fname,
                                'path': filepath,
                                'start_date': start_date,
                                'sleep_stage': slpstage,
                                'cycle': cycle
                                }
                        }
        if epoched == True:
            self.metadata['file_info']['epoch'] = epoch
        
        # load the ekg
        self.load_ekg(min_dur)

        # detect R peaks & calculate inter-beat intevals
        self.calc_RR(mw_size, upshift, rm_artifacts)
        
        
    def load_ekg(self, min_dur):
        """ 
        Load ekg data and extract sampling frequency. 
        
        Parameters
        ----------
        min_dur: bool, default: True
            If set to True, will not load files shorter than 5 minutes long 
        """
        
        data = pd.read_csv(self.metadata['file_info']['path'], header = [0, 1], index_col = 0, parse_dates=True)['EKG']
        
        # Check cycle length against 5 minute duration minimum
        cycle_len_secs = (data.index[-1] - data.index[0]).total_seconds()
        if cycle_len_secs < 60*5-1:
            if min_dur == True:
                print('Data is shorter than 5 minutes. Cycle will not be loaded.')
                print('--> To load data, set min_dur to False')
                return
            else:
                print('* WARNING: Data is shorter than 5 minutes.')
                self.data = data
        else:
            self.data = data
        
        diff = data.index.to_series().diff()[1:2]
        s_freq = 1000000/diff[0].microseconds
        nans = len(data) - data['Raw'].count()

        self.metadata['file_info']['start_time'] = data.index[0]
        self.metadata['analysis_info'] = {'s_freq': s_freq, 'cycle_len_secs': cycle_len_secs, 
                                        'NaNs(samples)': nans, 'NaNs(secs)': nans/s_freq}

        print('EKG successfully imported.')

    def set_Rthres(self, mw_size, upshift):
        """ set R peak detection threshold based on moving average + %signal upshift """
        print('Calculating moving average with {} sec window and a {} upshift...'.format(mw_size, upshift))
        
        # convert moving window to sample & calc moving average over window
        mw = int(mw_size*self.metadata['analysis_info']['s_freq'])
        mavg = self.data.Raw.rolling(mw).mean()

        # replace edge nans with overall average
        ekg_avg = np.mean(self.data['Raw'])
        mavg = mavg.fillna(ekg_avg)

        # set detection threshold as +5% of moving average
        det_thres = mavg*upshift
        self.data['EKG_thres'] = det_thres # can remove this for speed, just keep as series

        self.metadata['analysis_info']['mw_size'] = mw_size
        self.metadata['analysis_info']['upshift'] = upshift

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

        # get time between peaks and convert to mseconds
        self.rr = np.diff(self.rpeaks.index)/np.timedelta64(1, 'ms')
        print('R-R intervals calculated')


    def rm_artifacts(self):
        """ Replace ectopic and misplaced beats with NaNs. Based on Kubios automatic artefact correction algorithm 
            Note: This function does not interpolate RR intervals or R peaks to replace detected artifacts.

            *IMPORTANT: This algorithm assumes normal distribution of RR intervals. It will wash out datasets
            that are not normally distributed (NOT for use with ultra-short recordings)

            *NOTE: Not reliable as of 6-20-19. Exit pipeline w/ exported IBI list & rm artificacts using 
            Artiifact IBI module

            Returns
            -------
            self.nn: np.array
                nn-intervals after artifact rejection, including NaNs
            self.arts: list of tuple (index, rr_int, rejection type)
                detected artifacts 
        """
        print('Removing ectopic and misplaced beats...')

        th_wn_len = 91
        md_wn_len = 11
        nn = []
        arts = []

        for i, r in enumerate(self.rr):

            # if i is on left edge, keep first time threshold window stationary
            if i < (0.5 * th_wn_len):
                # set window for th calculation & remove i from comparison
                th_wn = self.rr[0:th_wn_len]
                th_wn = th_wn[np.arange(len(th_wn))!= i]

                # get quartile deviation & time varying threshold
                qd = stats.iqr(th_wn, nan_policy='omit')/2
                th = qd * 5.2

                # make median window
                if i > 0.5 * md_wn_len:
                    # if i is not on left edge, make moving median window
                    md_wn_start = int(i-md_wn_len/2)
                    md_wn_stop = int(i+md_wn_len/2)
                    md_wn = self.rr[md_wn_start:md_wn_stop]
                    md_wn = md_wn[np.arange(len(md_wn)) != i]
                else:
                    # if i is on left edge, keep first median window stationary
                    md_wn = self.rr[0:md_wn_len]
                    md_wn = md_wn[np.arange(len(md_wn)) != i]

                md = statistics.median(md_wn)
                # check for missed beats
                if np.abs(r/2 - md) < 2*th == True:
                    nn.append(np.NaN)
                    arts.append(i, r, 'missed')
                # check for extra beats
                elif i < len(self.rr) - 1 == True:
                    if np.abs(r + self.rr[i+1] - md) < 2*th == True:
                        nn.append(np.NaN)
                        arts.append(i, r, 'extra')
                else:
                    # if none detected, add r to nn array
                    nn.append(r)
            
            # use moving window for center of array
            elif (0.5 * th_wn_len) <= i < (len(self.rr) - (0.5 * th_wn_len)):
                #print('hit middle')
                th_wn_start = int(i-th_wn_len/2)
                th_wn_stop = int(i+th_wn_len/2)
                th_wn = self.rr[th_wn_start:th_wn_stop]
                th_wn = th_wn[np.arange(len(th_wn)) != i]

                # get quartile deviation & time varying threshold
                qd = stats.iqr(th_wn, nan_policy='omit')/2
                th = qd * 5.2

                # make median moving window
                wn_start = int(i-md_wn_len/2)
                wn_stop = int(i+md_wn_len/2)
                md_wn = self.rr[wn_start:wn_stop]
                md_wn = md_wn[np.arange(len(md_wn)) != i]

                md = statistics.median(md_wn)
                # check for missed beats
                if np.abs(r/2 - md) < 2*th == True:
                    nn.append(np.NaN)
                    arts.append(i, r, 'missed')
                # check for extra beats
                elif i < len(self.rr) - 1 == True:
                    if np.abs(r + self.rr[i+1] - md) < 2*th == True:
                        nn.append(np.NaN)
                        arts.append(i, r, 'extra')
                else:
                    # if none detected, add r to nn array
                    nn.append(r)

            # if i is on right edge, keep last time threshold window stationary
            elif i >= (len(self.rr) - (0.5 * th_wn_len)):
                # set window for th calculation & remove i from comparison
                th_wn = self.rr[-th_wn_len:]
                th_wn = th_wn[np.arange(len(th_wn))!= i]

                # get quartile deviation & time varying threshold
                qd = stats.iqr(th_wn, nan_policy='omit')/2
                th = qd * 5.2

                # if i is on right edge, keep last median window stationary
                if i > len(self.rr) - (0.5 * md_wn_len):
                    md_wn = self.rr[-md_wn_len:]
                    md_wn = md_wn[np.arange(len(md_wn)) != i]
                    md = statistics.median(md_wn)

                else:
                    # else make median moving window
                    wn_start = int(i-md_wn_len/2)
                    wn_stop = int(i+md_wn_len/2)
                    md_wn = self.rr[wn_start:wn_stop]
                    md_wn = md_wn[np.arange(len(md_wn)) != i]

                md = statistics.median(md_wn)
                # check for missed beats
                if np.abs(r/2 - md) < 2*th == True:
                    nn.append(np.NaN)
                    arts.append(i, r, 'missed')
                # check for extra beats
                elif i < len(self.rr) - 1 == True:
                    if np.abs(r + self.rr[i+1] - md) < 2*th == True:
                        nn.append(np.NaN)
                        arts.append(i, r, 'extra')
                else:
                    # if none detected, add r to nn array
                    nn.append(r)
                
        self.nn = np.array(nn)
        self.nn_diff = np.diff(self.nn)
        self.nn_diffsq = self.nn_diff**2

        self.rr_arts = np.array(arts)


    def calc_RR(self, mw_size, upshift, rm_artifacts):
        """ Detect R peaks and calculate R-R intervals """
        
        # set R peak detection parameters
        self.set_Rthres(mw_size, upshift)
        # detect R peaks & make RR tachogram
        self.detect_Rpeaks()
        # remove artifacts
        if rm_artifacts == True:
            self.rm_artifacts()

    def export_RR(self, savedir):
        """ Export R peaks and RR interval data to .txt files """

        # set save directory
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
        
        # set savename info
        # f_info1 = self.metadata['file_info']['fname'].split('_')[:5]
        # f_info2 = data['metadata']['file_info']['fname'].split('_')[5].split('.')[0]

        # save R peak detections
        savepeaks = self.metadata['file_info']['fname'].split('.')[0] + '_rpeaks.txt'
        peaks_file = os.path.join(savedir, savepeaks)
        #self.rpeaks.to_csv(peaks_file)
        np.savetxt(peaks_file, self.rpeaks, delimiter='\n')

        # save RR intervals
        saverr = self.metadata['file_info']['fname'].split('.')[0] + '_rr.txt'
        rr_file = os.path.join(savedir, saverr)
        np.savetxt(rr_file, self.rr, delimiter='\n')

        # save NN intervals, if exists
        try: 
            self.nn
        except AttributeError: 
            pass
        else:
            savenn = self.metadata['file_info']['fname'].split('.')[0] + '_nn.txt'
            nn_file = os.path.join(savedir, saverr)
            np.savetxt(nn_file, self.nn, delimiter='\n')

        print('Done.')


class IBI:
    """ Class for EKG inter-beat interval data. Can be cleaned data (nn intervals) or raw (rr intervals). """

    def __init__(self, fname, fpath, s_freq, start_time, epoched=True, itype='nn'):
        """ Initialize inter-beat interval object

        Parameters
        ----------
        fname: str
            filename
        fpath: str
            path to file
        s_freq: float
            sampling frequency of original EKG data
        epoched: bool (default: True)
            specify whether data was epoched with ioeeg.py
        itype: str (options: 'nn' or 'rr')
            interval type (cleaned = 'nn'; otherwise 'rr')

        Returns
        -------
        IBI object
        """

        # set metadata
        filepath = os.path.join(fpath, fname)
        if epoched == False:
            in_num, start_date, slpstage, cycle = fname.split('_')[:4]
        elif epoched == True:
            in_num, start_date, slpstage, cycle, epoch = fname.split('_')[:5]
        self.metadata = {'file_info':{'in_num': in_num,
                                    'fname': fname,
                                    'path': filepath,
                                    'start_date': start_date,
                                    'start_time': start_time,
                                    'sleep_stage': slpstage,
                                    'cycle': cycle
                                    },
                        'analysis_info':{'s_freq': s_freq,
                                    'itype': itype
                                    }
                        }
        if epoched == True:
            self.metadata['file_info']['epoch'] = epoch

        # load data
        self.load_ibi(itype)


    def load_ibi(self, itype):
        """ Load ibi .txt file """
        print('Loading interbeat interval data...')
        
        if itype == 'nn':
            nn_full = np.loadtxt(self.metadata['file_info']['path'])
            # remove NaNs
            self.nn = nn_full[~np.isnan(nn_full)]
            self.metadata['analysis_info']['rrArtifacts_rmvd'] = len(nn_full) - len(self.nn)
        elif itype == 'rr':
            self.rr = np.loadtxt(self.metadata['file_info']['path'])

        print('Done.')


    def calc_tstats(self, itype):
        """ Calculate commonly used time domain HRV statistics. Min/max HR is determined over 5 RR intervals 

            Params
            ------
            itype: str, 
                Interval type (Options:'rr', 'nn')
        """   
        print('Calculating time domain statistics...')

        if itype == 'rr':
            ii = self.rr
            ii_diff = np.diff(self.rr)
            ii_diffsq = ii_diff**2
            self.rr_diff = ii_diff
            self.rr_diffsq = ii_diffsq
        
        elif itype == 'nn':
            ii = self.nn
            ii_diff = np.diff(self.nn)
            ii_diffsq = ii_diff**2
            self.nn_diff = ii_diff
            self.nn_diffsq = ii_diffsq

        # heartrate in bpm
        hr_avg = 60/np.mean(ii)*1000
        
        rollmean_ii = pd.Series(ii).rolling(5).mean()
        mx_ii, mn_ii = np.nanmax(rollmean_ii), np.nanmin(rollmean_ii)
        hr_max = 60/mn_ii*1000
        hr_min = 60/mx_ii*1000


        # inter-beat interval & SD (ms)
        ibi = np.mean(ii)
        sdrr = np.std(ii)

        # SD & RMS of differences between successive II intervals (ms)
        sdsd = np.std(ii_diff)
        rmssd = np.sqrt(np.mean(ii_diffsq))

        # pNN20 & pNN50
        pxx20 = sum(np.abs(ii_diff) >= 20.0)/len(ii_diff)*100
        pxx50 = sum(np.abs(ii_diff) >= 50.0)/len(ii_diff)*100

        # hrv triangular index
        bin_width = 7.8125
        stat, bin_edges, bin_num = stats.binned_statistic(ii, ii, bins = np.arange(min(ii), max(ii) + bin_width, bin_width), statistic='count')
        hti = sum(stat)/max(stat)
        # triangular interpolation of NN interval
        # if 1st bin is max, can't calculatin TINN
        #if stat[0] == max(stat):
        #    tinn = None
        #else:
            # this calculation is wrong
        #    tinn = bin_edges[-1] - bin_edges[0]
        tinn = 'calc not programmed'
        #print('HRV Triangular Index (HTI) = {0:.2f}.\nTriangular Interpolation of NN Interval Histogram (TINN) (ms) = {1}\n\t*WARNING: TINN calculation may be incorrect. Formula should be double-checked'.format(self.hti, self.tinn))
        #print('Call ekg.__dict__ for all statistics')
        self.time_stats = {'linear':{'HR_avg': hr_avg, 'HR_max': hr_max, 'HR_min': hr_min, 'IBI_mean': ibi,
                                    'SDRR': sdrr, 'RMSSD': rmssd, 'pXX20': pxx20, 'pXX50': pxx50},
                            'geometric': {'hti':hti, 'tinn':tinn}
                            }
        print('Time domain stats stored in obj.time_stats\n')

    
    def interpolateII(self, itype):
        """ Resample tachogram to original sampling frequency (since RRs are not evenly spaced)
            and interpolate for power spectral estimation 
            *Note: adapted from pyHRV

            Params
            ------
            itype: str
                interval type (options: 'rr', 'nn')
        """
        # specify data
        if itype == 'rr':
            ii = self.rr
        elif itype == 'nn':
            ii = self.nn

        # interpolate
        fs = self.metadata['analysis_info']['s_freq']
        t = np.cumsum(ii)
        t -= t[0]
        f_interp = sp.interpolate.interp1d(t, ii, 'cubic')
        t_interp = np.arange(t[0], t[-1], 1000./fs)
        self.ii_interp = f_interp(t_interp)
        self.metadata['analysis_info']['s_freq_interp'] = self.metadata['analysis_info']['s_freq']


    def calc_psd_welch(self, itype, window):
        """ Calculate welch power spectral density 

            Params
            ------
            itype: str
                interval type (options: 'rr', 'nn')
            window: str
                windowing function. options from scipy.signal welch. (wrapper default: 'hamming')
        """
        
        self.metadata['analysis_info']['psd_method'] = 'welch'
        self.metadata['analysis_info']['psd_window'] = window

        # specify data
        if itype == 'rr':
            ii = self.rr
        elif itype == 'nn':
            ii = self.nn
        
        # set nfft to guidelines of power of 2 above len(data), min 256 (based on MATLAB guidelines)
        nfft = max(256, 2**(int(np.log2(len(self.ii_interp))) + 1))
        
        # Adapt 'nperseg' according to the total duration of the II series (5min threshold = 300000ms)
        if max(np.cumsum(ii)) < 300000:
            nperseg = nfft
        else:
            nperseg = 300
        
        # default overlap = 50%
        f, Pxx = welch(self.ii_interp, fs=4, window=window, scaling = 'density', nfft=nfft, 
                        nperseg=nperseg)
        self.psd_welch = {'freqs':f, 'pwr': Pxx, 'nfft': nfft, 'nperseg': nperseg}


    def calc_psd_mt(self, bandwidth):
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
        self.metadata['analysis_info']['psd_method'] = 'multitaper'
        self.metadata['analysis_info']['psd_bandwidth'] = bandwidth
        sf_interp = self.metadata['analysis_info']['s_freq_interp']

        pwr, freqs = psd_array_multitaper(self.ii_interp, sf_interp, adaptive=True, 
                                            bandwidth=bandwidth, normalization='full', verbose=0)
        self.psd_mt = {'freqs': freqs, 'pwr': pwr}
        self.metadata['analysis_info']['psd_method'] = 'multitaper'

    def calc_fbands(self, method, bands):
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
        #self.freq_bands = freq_bands
        
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
        total_pwr = sum(filter(None, [np.sum(fband_vals[key]['pwr']) for key in fband_vals.keys()]))
        freq_stats = {'totals':{'total_pwr': total_pwr}}
        # by band
        for key in freq_bands.keys():
            freq_stats[key] = {}
            freq_stats[key]['freq_range'] = str(freq_bands[key])
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
                freq_stats[key]['pwr_%'] = freq_stats[key]['pwr_ms2']/freq_stats['totals']['total_pwr']*100
        
        # add normalized units to lf & hf bands
        for key in ['lf', 'hf']:
            freq_stats[key]['pwr_nu'] = freq_stats[key]['pwr_ms2']/(freq_stats['lf']['pwr_ms2'] + freq_stats['hf']['pwr_ms2'])*100
        # add lf/hf ratio
        freq_stats['totals']['lf/hf'] = freq_stats['lf']['pwr_ms2']/freq_stats['hf']['pwr_ms2']
        
        self.freq_stats = freq_stats


    def calc_fstats(self, itype, method, bandwidth, window='hamming', bands=None):
        """ Calculate frequency domain statistics 

        Parameters
        ----------
        itype: str
            interval type (options: 'rr', 'nn')
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
        print('Interpolating and resampling tachogram...')
        self.interpolateII(itype)
       
       # calculate power spectrum
        print('Calculating power spectrum...')
        if method == 'mt':
            self.calc_psd_mt(bandwidth)
        elif method == 'welch':
            self.calc_psd_welch(itype, window)
        
        #calculate frequency domain statistics
        print('Calculating frequency domain measures...')
        self.calc_fbands(method, bands)
        print('Frequency measures stored in obj.freq_stats\n')

    
    def calc_nlstats(self, itype, calc_dfa=False, save_plots=False):
        """ calculate nonlinear dynamics poincare & sample entropy 
            Note: From pyhrv non-linear module 

            Params
            ------
            itype: str
                Interval type (options: 'rr', 'nn')
            calc_dfa: bool (default: False)
                Option to calculate detrended fluctuation analysis. Appropriate
                for data several hours long
        """
        
        # specify data
        if itype == 'rr':
            ii = self.rr
        elif itype == 'nn':
            ii = self.nn

        print('Calculating nonlinear statistics...')
        nonlinear_stats = {}

        # poincare
        pc = nl.poincare(ii)
        nonlinear_stats['poincare'] = {'sd1': pc[1], 'sd2': pc[2], 'sd_ratio':pc[3], 'ellipse_area':pc[4]}
        
        # sample entropy (tolerance and dim set to match Riganello et al. 2018)
        sampEN = nl.sample_entropy(ii, dim=2, tolerance=0.15)
        nonlinear_stats['entropy'] = {'sampEN':sampEN[0]}

        # detrended fluctuation analysis
        if calc_dfa is True:
            dfa = nl.dfa(ii)
            nonlinear_stats['dfa'] = {'alpha1': dfa[1], 'alpha2': dfa[2]}

        if save_plots == True:
            nonlinear_stats['poincare']['plot'] = pc[0]
            if calc_dfa is True:
                nonlinear_stats['dfa']['plot'] = dfa[0]

        self.nonlinear_stats = nonlinear_stats
        print('Nonlinear stats stored in obj.nonlinear_stats\n')


    def hrv_stats(self, method='mt', bandwidth=0.01):
        """ Calculate all statistics on IBI object 

            TO DO: Add freq_stats arguments to hrv_stats params? 
                   Divide into 5-min epochs 
        """

        # set itype
        itype = self.metadata['analysis_info']['itype']

        # calculate statistics
        self.calc_tstats(itype)
        self.calc_fstats(itype, method, bandwidth)
        self.calc_nlstats(itype)
        print('Done.')

    def to_spreadsheet(self, spreadsheet, savedir):
        """ Append calculations as a row in master spreadsheet. Creates new spreadsheet
            if output file does not exist. 
            
            Parameters
            ----------
            ekg: EKG object
            spreadsheet: .csv
                output file (new or existing)
        """
        
        # this is from before division to two classes. 'data' and 'rpeaks' arrays shouldn't exist in IBI object.
        arrays = ['data', 'rpeaks', 'rr', 'rr_diff', 'rr_diffsq', 'nn', 'nn_diff', 'nn_diffsq', 'rr_arts', 'ii_interp', 'psd_mt', 'psd_fband_vals']
        data = {k:v for k,v in vars(self).items() if k not in arrays}
        
        reform = {(level1_key, level2_key, level3_key): values
                    for level1_key, level2_dict in data.items()
                    for level2_key, level3_dict in level2_dict.items()
                    for level3_key, values      in level3_dict.items()}
        
        df = pd.DataFrame(reform, index=[0])
        df.set_index([('metadata', 'file_info', 'in_num'), ('metadata', 'file_info', 'start_time')], inplace=True)
        savename = os.path.join(savedir, spreadsheet)
        
        if os.path.exists(spreadsheet):
            with open(savename, 'a') as f:
                df.to_csv(f, header=False, line_terminator='\n')
            print('Data added to {}'.format(spreadsheet))
        else:
            with open(savename, 'a') as f:
                df.to_csv(f, header=True, line_terminator='\n')
            print('{} does not exist. Data saved to new spreadsheet'.format(spreadsheet))

    def to_report(self, savedir=None, json=False):
        """ export statistics as a csv report 
            TO DO: add line to check if nn exists
        """
        # set save directory
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

        # set savename info
        # save_info1 = '_'.join(data['metadata']['file_info']['fname'].split('_')[:5])
        # save_info2 = data['metadata']['file_info']['fname'].split('_')[5].split('.')[0]
        
        # export everything that isn't a dataframe, series, or array    
        arrays = ['data', 'rpeaks', 'rr', 'rr_diff', 'rr_diffsq', 'nn', 'nn_diff', 'nn_diffsq', 'rr_arts', 'ii_interp', 'psd_mt', 'psd_fband_vals']
        data = {k:v for k,v in vars(self).items() if k not in arrays}
        
        # set savename info
        if 'epoch' in self.metadata['file_info'].keys():
            saveinfo = '_'.join((self.metadata['file_info']['fname'].split('_')[:6]))
        else:
            saveinfo = '_'.join((self.metadata['file_info']['fname'].split('_')[:5]))

        # save calculations
        if json is False:
            savename = saveinfo + '_HRVstats.txt'
            file = os.path.join(savedir, savename)
            with open(file, 'w') as f:
                for k, v in data.items():
                    if type(v) is not dict:
                        line = k+' '+str(v) + '\n'
                        f.write(line)
                    elif type(v) is dict:
                        line = k + '\n'
                        f.write(line)
                        for kx, vx in v.items():
                            if type(vx) is not dict:
                                line = '\t'+ kx + ' ' + str(vx) + '\n'
                                f.write(line)
                            else:
                                line = '\t' + kx + '\n'
                                f.write(line)
                                for kxx, vxx in vx.items():
                                    line = '\t\t' + kxx + ' ' + str(vxx) + '\n'
                                    f.write(line)
        else:
            savename = saveinfo + '_HRVstats_json.txt'
            file = os.path.join(savedir, savename)
            with open(file, 'w') as f:
                json.dump(data, f, indent=4)   

        # re-save nn intervals (w/o NaNs) w/ set naming convention
        savenn = saveinfo + '_nn.txt'
        nn_file = os.path.join(savedir, savenn)
        np.savetxt(nn_file, self.nn, delimiter='\n')

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

