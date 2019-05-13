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

        #self.filename = fname
        filepath = os.path.join(fpath, fname)
        in_num, start_date, slpstage, cycle = fname.split('_')[:-1]
        self.metadata = {'file_info':{'in_num': in_num,
                                'fname': fname,
                                'path': filepath,
                                #'start_date': start_date,
                                'sleep_stage': slpstage,
                                'cycle': cycle
                                }
                    }
        self.load_ekg(min_dur)
        
        

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
        if cycle_len_secs < 60*5:
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

        self.metadata['file_info']['start_time'] = data.index[0]
        self.metadata['analysis_info'] = {'s_freq': s_freq, 'cycle_len_secs': cycle_len_secs}

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
        # heartrate in bpm
        hr_avg = 60/np.mean(self.rr_int)*1000
        #print('Average heartrate (bpm) = {}'.format(int(self.heartrate_avg)))
        
        rollmean_rr = pd.Series(self.rr_int).rolling(5).mean()
        mx_rr, mn_rr = np.nanmax(rollmean_rr), np.nanmin(rollmean_rr)
        hr_max = 60/mn_rr*1000
        hr_min = 60/mx_rr*1000
        #print('Maximum heartrate (bpm) = {}'.format(int(self.heartrate_max)))
        #print('Minimum heartrate (bpm) = {}'.format(int(self.heartrate_min)))


        # inter-beat interval & SD (ms)
        ibi = np.mean(self.rr_int)
        sdrr = np.std(self.rr_int)
        #print('Average IBI (ms) = {0:.2f} SD = {1:.2f}'.format(self.ibi, self.sdrr))

        # SD & RMS of differences between successive RR intervals (ms)
        sdsd = np.std(self.rr_int_diff)
        rmssd = np.sqrt(np.mean(self.rr_int_diffsq))

        # prr20 & prr50
        prr20 = sum(np.abs(self.rr_int_diff) >= 20.0)/len(self.rr_int_diff)*100
        prr50 = sum(np.abs(self.rr_int_diff) >= 50.0)/len(self.rr_int_diff)*100
        #print('pRR20 = {0:.2f}% & pRR50 = {1:.2f}%'.format(self.prr20, self.prr50))

        # hrv triangular index
        bin_width = 7.8125
        stat, bin_edges, bin_num = stats.binned_statistic(self.rr_int, self.rr_int, bins = np.arange(min(self.rr_int), max(self.rr_int) + bin_width, bin_width), statistic='count')
        hti = sum(stat)/max(stat)
        # triangular interpolation of NN interval
        # if 1st bin is max, can't calculatin TINN
        if stat[0] == max(stat):
            tinn = None
        else:
            # this calculation is wrong
            tinn = bin_edges[-1] - bin_edges[0]
        #print('HRV Triangular Index (HTI) = {0:.2f}.\nTriangular Interpolation of NN Interval Histogram (TINN) (ms) = {1}\n\t*WARNING: TINN calculation may be incorrect. Formula should be double-checked'.format(self.hti, self.tinn))
        #print('Call ekg.__dict__ for all statistics')
        self.time_stats = {'linear':{'HR_avg': hr_avg, 'HR_max': hr_max, 'HR_min': hr_min, 'IBI': ibi,
                                    'SDRR': sdrr, 'RMSSD': rmssd, 'pRR20': prr20, 'pRR50': prr50},
                            'geometric': {'hti':hti, 'tinn':tinn}
                            }
        print('Time domain stats stored in ekg.time_stats\n')

    
    def interpolateRR(self):
        """ Resample RR tachogram to original sampling frequency (since RRs are not evenly spaced)
            and interpolate for power spectral estimation 
            *Note: adapted from pyHRV
        """
        fs = self.metadata['analysis_info']['s_freq']
        t = np.cumsum(self.rr_int)
        t -= t[0]
        f_interp = sp.interpolate.interp1d(t, self.rr_int, 'cubic')
        t_interp = np.arange(t[0], t[-1], 1000./fs)
        self.rr_interp = f_interp(t_interp)
        self.metadata['analysis_info']['s_freq_interp'] = self.metadata['analysis_info']['s_freq']


    def psd_welch(self, window='hamming'):
        """ Calculate welch power spectral density """
        
        self.metadata['analysis_info']['psd_method'] = 'welch'
        self.metadata['analysis_info']['psd_window'] = window
        
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
        self.psd_welch = {'freqs':f, 'pwr': Pxx, 'nfft': nfft, 'nperseg': nperseg}


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
        self.metadata['analysis_info']['psd_method'] = 'multitaper'
        self.metadata['analysis_info']['psd_bandwidth'] = bandwidth
        sf_interp = self.metadata['analysis_info']['s_freq_interp']

        pwr, freqs = psd_array_multitaper(self.rr_interp, sf_interp, adaptive=True, 
                                            bandwidth=bandwidth, normalization='full', verbose=0)
        self.psd_mt = {'freqs': freqs, 'pwr': pwr}
        self.metadata['analysis_info']['psd_method'] = 'multitaper'

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

    
    def nonlinear_stats(self, save_plots=False):
        """ calculate nonlinear dynamics poincare & sample entropy 
            Note: From pyhrv non-linear module """
        print('Calculating nonlinear statistics...')
        nonlinear_stats = {}

        # poincare
        pc = nl.poincare(self.rr_int)
        nonlinear_stats['poincare'] = {'sd1': pc[1], 'sd2': pc[2], 'sd_ratio':pc[3], 'ellipse_area':pc[4]}
        # sample entropy (tolerance and dim set to match Riganello et al. 2018)
        sampEN = nl.sample_entropy(self.rr_int, dim=2, tolerance=0.15)
        nonlinear_stats['entropy'] = {'sampEN':sampEN[0]}

        # detrended fluctuation analysis
        dfa = nl.dfa(self.rr_int)
        nonlinear_stats['dfa'] = {'alpha1': dfa[1], 'alpha2': dfa[2]}

        if save_plots == True:
            nonlinear_stats['dfa']['plot'] = dfa[0]
            nonlinear_stats['poincare']['plot'] = pc[0]

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

    def to_spreadsheet(self, spreadsheet):
        """ Append calculations as a row in master spreadsheet. Creates new spreadsheet
            if output file does not exist. 
            
            Parameters
            ----------
            ekg: EKG object
            spreadsheet: .csv
                output file (new or existing)
        """
        
        arrays = ['data', 'rpeaks', 'rr_int', 'rr_int_diff', 'rr_int_diffsq', 'rr_interp', 'psd_mt', 'psd_fband_vals']
        data = {k:v for k,v in vars(self).items() if k not in arrays}
        
        reform = {(level1_key, level2_key, level3_key): values
                    for level1_key, level2_dict in data.items()
                    for level2_key, level3_dict in level2_dict.items()
                    for level3_key, values      in level3_dict.items()}
        
        df = pd.DataFrame(reform, index=[0])
        df.set_index([('metadata', 'file_info', 'in_num'), ('metadata', 'file_info', 'start_time')], inplace=True)
        savename = spreadsheet
        
        if os.path.exists(spreadsheet):
            with open(savename, 'a') as f:
                df.to_csv(f, header=False, line_terminator='\n')
            print('Data added to {}'.format(spreadsheet))
        else:
            with open(savename, 'a') as f:
                df.to_csv(f, header=True, line_terminator='\n')
            print('{} does not exist. Data saved to new spreadsheet'.format(spreadsheet))

    def to_report(self, json=False):
        """ export statistics as a csv report """
        
        # export everything that isn't a dataframe, series, or array    
        arrays = ['data', 'rpeaks', 'rr_int', 'rr_int_diff', 'rr_int_diffsq', 'rr_interp', 'psd_mt', 'psd_fband_vals']
        data = {k:v for k,v in vars(self).items() if k not in arrays}
        
        # save calculations
        if json is False:
            savename = data['metadata']['file_info']['fname'] + '_HRVstats.txt'
            with open(savename, 'w') as f:
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
            savename = data['metadata']['file_info']['fname'] + '_HRVstats_json.txt'
            with open(savename, 'w') as f:
                json.dump(data, f, indent=4)   

        # save detections
        savepeaks = data['metadata']['file_info']['fname'] + '_rpeaks.txt'
        self.rpeaks.to_csv(savepeaks)
        saverr = data['metadata']['file_info']['fname'] + '_rri.txt'
        np.savetxt(saverr, self.rr_int, delimiter='\n')


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

