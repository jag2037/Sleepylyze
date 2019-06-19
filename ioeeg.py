""" import this class by using the __init__.py file 'from .ioeeg import Dataset' 

    To Do:
        Remove spindle detection functions --> to be moved to new class for cut segments 
        Add modified calculate_transitions code
"""

import datetime
import math 
import numpy as np 
import os
import pandas as pd
import re
import scipy.io as io
from scipy.signal import buttord, butter, sosfiltfilt, sosfreqz
import statistics

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
    trim:
    start:
    end:
    noise_log:
    rm_chans:
    
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
    
    def __init__(self, fname, fpath=None, trim=True, start='22:00:00', end='07:00:00', noise_log=None, rm_chans=None):
        if fpath is not None:
            filepath = os.path.join(fpath, fname)
            #filepath = fpath + fname
        else:
            filepath = fname
        
        self.fname = fname
        self.fpath = fpath
        self.filepath = filepath
        
        self.get_info()
        self.get_chans()
        self.load_eeg()
        
        if trim == True:
            self.trim_eeg(start, end)
        
        if noise_log is not None or rm_chans is not None: 
            self.clean_eeg(noise_log, rm_chans)

        print('\nDone.')

    
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
        # setting dtype to float will speed up load, but crashes if there is anything wrong with the record
        data = pd.read_csv(self.filepath, delim_whitespace=True, header=None, skiprows=15, usecols=range(3,end_col),
                               dtype = float, na_values = ['AMPSAT', 'SHORT'])
        
        # create DateTimeIndex
        ind_freq = str(int(1/self.s_freq*1000000))+'us'
        ind_start = self.start_date + ' ' + self.start_time + str(self.start_us)[1:]
        ind = pd.date_range(start = ind_start, periods=len(data), freq=ind_freq)

        # make a new dataframe with the proper index & column names
        data.columns = pd.MultiIndex.from_arrays([self.channels, np.repeat(('Raw'), len(self.channels))],names=['Channel','datatype'])
        data.index = ind

        self.data = data
        print('Data successfully imported')

    def trim_eeg(self, start, end):
        """ Trim excess time off the ends of the raw data

        Parameters
        ----------
        start: str (format: hh:mm:ss)
            New start time of data (inclusive). Any previous times will be removed
        end: str (format: hh:mm:ss)
            New end time of data (exclusive). This second + any later seconds will be removed
        """
        
        print('Trimming EEG...')
        # specify indices NOT between start and end time
        rm_idx = self.data.between_time(end, start, include_start=True, include_end=False).index
        # drop specified indices
        self.data.drop(rm_idx, axis=0, inplace=True)
        
        print('Data trimmed to {} (inclusive) - {} (exclusive).'.format(start, end))

    def clean_eeg(self, noise_log, rm_chans):
        """ Replace artifact with NaN 
        
        Parameters
        ----------
        noise_log: .txt file (optional)
            file containing list of artifact seconds to remove (format: YYYY-MM-DD hh:mm:ss)
        rm_chans: str or list of string (optional)
            entire channels to remove

        Returns
        -------
        self.data pd.DataFrame of raw EEG with artifact times replaced with NaNs
        """

        print('Cleaning raw EEG data...')

        channel_list = [col[0] for col in self.data.columns]
        
        # check chans against data channel list (case insensitive) & convert str to list
        if rm_chans is not None:
            print('Removing noisy channels...')
            if type(rm_chans) == str:
                rm_chans_list = [x for x in channel_list if rm_chans.casefold() == x.casefold()]
            elif type(rm_chans) == list:
                rm_chans_list = [x for x in channel_list for r in rm_chans if r.casefold() == x.casefold()]
            # replace channels with NaNs
            for chan in rm_chans_list:
                self.data[(chan, 'Raw')] = np.NaN
            print('Noisy channels removed.')
        
        if noise_log is not None:
            noise = pd.read_csv(noise_log, header=None, names=[0, 1, 'channels'], sep = '\t', index_col=0, parse_dates=[[0, 1]])
            noise.index.name = 'time'
            # split channel strings into lists
            noise['channels'] = [re.findall(r"[\w'\*]+", n) for n in noise['channels']]
            # unpack to dict
            unpacked_noise = {(y, mo, d, h, m, s): chans for (y, mo, d, h, m, s, chans) in zip(noise.index.year, noise.index.month, 
                                                        noise.index.day, noise.index.hour, noise.index.minute, noise.index.second, 
                                                        noise['channels'])}
            # case correct channel names
            unpacked_noise_corr = {}
            channel_list_cf = [x.casefold() for x in channel_list]
            for key, val in unpacked_noise.items():
                new_val = []
                for i in range(len(val)):
                    if val[i] == '*':
                        new_val.append(val[i])
                    else:
                        new_val.append(channel_list[channel_list_cf.index(val[i].casefold())])
                unpacked_noise_corr[key] = new_val

            # compare to data index
            print('Specifying noise indices...')
            noise_idx = {}
            for i in self.data.index:
                for idx, chan in unpacked_noise_corr.items():
                    if (i.year, i.month, i.day, i.hour, i.minute, i.second) == idx:
                        # make a dict of indices marked as noise w/ channels to apply to
                        noise_idx[i] = chan

            # replace noise with NaNs
            print('Removing noisy time points...')
            for t, c in noise_idx.items():
                for cx in c:
                    if cx == '*':
                        for x in self.eeg_channels:
                            #self.data[(x, 'Raw')].loc[t] = np.NaN # much slower
                            #self.data.at[t, (x, 'Raw')] = np.NaN # not depricated, slightly slower
                            self.data.set_value(t, (x, 'Raw'), np.NaN)
                    else:
                        #self.data[(cx, 'Raw')].loc[t] = np.NaN
                        #self.data.at[t, (cx, 'Raw')] = np.NaN
                        self.data.set_value(t, (cx, 'Raw'), np.NaN)

        print('Data cleaned.')

    def load_hyp(self, scorefile):
        """ Loads hypnogram .txt file and produces DateTimeIndex by sleep stage and 
        stage cycle for cutting
        
        TO DO: update docstring

        Parameters
        ----------
        scorefile: .txt file
            plain text file with 30-second epoch sleep scores, formatted [MM/DD/YYYY hh:mm:ss score]
            NOTE: Time must be in 24h time & scores in consistent format (int or float)

        Returns:
        --------
        stage_cuts: dict
            Nested dict of sleep stages, cycles, and DateTimeIndex
            Ex. {'awake': {1: DateTimeIndex[(...)], 2: DateTimeIndex[(...)]}}
        
        """
        # read the first line to get starting date & time
        with open(scorefile, 'r') as f:
                first_epoch = f.readline()
                start_date = first_epoch.split(' ')[0]
                start_sec = first_epoch.split(' ')[1].split('\t')[0]
                
        # read in sleep scores & resample to EEG/EKG frequency
        print('Importing sleep scores...')
        hyp = pd.read_csv(scorefile, delimiter='\t', header=None, names=['Score'], usecols=[1], dtype=float)
        scores = pd.DataFrame(np.repeat(hyp.values, self.s_freq*30,axis=0), columns=hyp.columns)

        # reindex to match EEG/EKG data
        ## --> add warning here if index start date isn't found in EEG data
        ind_freq = str(int(1/self.s_freq*1000000))+'us'
        ind_start = start_date + ' ' + start_sec + '.000'
        ind = pd.date_range(start = ind_start, periods=len(scores), freq=ind_freq)
        scores.index = ind

        # check that scorefile and data overlap
        print('Veryifying timestamp overlap (>5s runtime most likely means no overlap) ...')
        for i in ind:
            if i in self.data.index:
                overlap = i
                self.hyp = pd.Series(scores['Score'])
                print(('Scorefile and data timestamp match detected at {}.').format(i))
                break
        try:
            overlap
        except:
            raise Exception(('Scorefile times do not overlap with data times. Check "Date" parameter and sleep scoring./n Scorefile start:{}  Data start:{}').format(ind[0], self.data.index[0]))
        
        # add hypnogram column to dataframe (tested against join, concat, merge; this is fastest) 
        #self.data[('Hyp', 'Stage')] = scores

        # get cycle blocks for each stage (detect non-consecutive indices & assign cycles)
        stages = {'awake': 0.0, 'rem': 1.0, 's1': 2.0, 's2': 3.0, 'ads': 4.0, 'sws': 5.0, 'rbrk': 6.0}
        ms = pd.Timedelta(1/self.s_freq*1000, 'ms')
        self.sleepstages = stages

        print('Assigning sleep stages & cycles...')
        stage_cuts = {}
        for stage, val in stages.items():
            stage_dat = self.hyp[self.hyp == val]
            # if the sleep stage is present
            if len(stage_dat) != 0:
                # detect non-consecutive indices to determine cycle breaks
                breaks = stage_dat.index.to_series().diff() != ms
                # assign cycle # for each consecutive block
                cycles = breaks.cumsum()
                
                # get index values for each cycle block, add entry to cyc dict
                cyc = {}
                for c in range(1, max(cycles)+1):
                    idx = cycles[cycles == c].index   
                    cyc[c] = idx
                # add cyc dict to nested stage_cuts dict
                stage_cuts[stage] = cyc
            else:
                stage_cuts[stage] = None

        self.stage_cuts = stage_cuts

        # Sleep structure analyses
        print('Calculating sleep structure statistics...')
        hyp_stats = {}
        # sleep efficiency (removing record breaks)
        hyp_stats['sleep_efficiency'] = (len(hyp) - len(hyp[hyp.values == 0.0]) - len(hyp[hyp.values == 6.0]))/(len(hyp) - len(hyp[hyp.values == 6.0])) * 100
        # sleep stage % (removing record breaks)
        for stage, code in stages.items():
            if stage is not 'rbrk':
                percent = len(hyp[hyp.values == code])/(len(hyp)-len(hyp[hyp.values == 6.0])) * 100
            elif stage is 'rbrk':
                percent = len(hyp[hyp.values == code])/len(hyp)
            hyp_stats[stage]= {'%night': percent}
        # cycle stats for each sleep stage
        for stage, cycles in self.stage_cuts.items():
            if cycles is None:
                hyp_stats[stage]['n_cycles'] = 0
            else:
                # number of cycles
                hyp_stats[stage]['n_cycles'] = len(cycles)
                # length of each cycle
                cycle_lens = {}
                for cycle, idx in cycles.items():
                    length = (idx[-1] - idx[0]).seconds + 1
                    cycle_lens[cycle] = length
                hyp_stats[stage]['cycle_lengths'] = cycle_lens
                # average cycle length
                hyp_stats[stage]['avg_len'] = sum(cycle_lens.values())/len(cycle_lens.values())
                hyp_stats[stage]['stdev'] = np.std(list(cycle_lens.values()))
                hyp_stats[stage]['median'] = statistics.median(list(cycle_lens.values())) 

        self.hyp_stats = hyp_stats

        print('Done.')

    def export_hypstats(self, savedir=None, export_hyp=True):
        """ Save a report of sleep structure statistics 
        
        Params
        ------
        savedir: str
            path to directory where files will be saved
        export_hyp: bool (default: True)
            export resampled hypnogram
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
        else:
            if not os.path.exists(savedir):
                createdir = input(savedir + ' does not exist. Create directory? [Y/N] ')
                if createdir == 'Y':
                    os.makedirs(savedir)
                else:
                    savedir = input('Try again. Save directory: ')
                    if not os.path.exists(savedir):
                        print(savedir + ' does not exist. Aborting. ')
                        return
        
        # export hypnogram stats
        stats_savename = d.in_num + '_SleepStats_' + d.start_date + '.txt'
        stats_save = os.path.join(savedir, stats_savename)
        with open(stats_save, 'w') as f:
            json.dump(self.hyp_stats, f, indent=4)

        # export the resampled hypnogram @ the original frequency (for downstream speed)
        new_hyp = d.hyp[(d.hyp.index.microsecond == 0) & ((d.hyp.index.second == 0) | (d.hyp.index.second == 30))]
        hyp_savename = d.in_num + '_Hypnogram_' + d.start_date + '.txt'
        hyp_save = os.path.join(savedir, hyp_savename)
        new_hyp.to_csv(hyp_save, header=False)

    def cut_EEG(self, sleepstage='all', epoch_len=None):
        """ cut dataset based on loaded hypnogram 
        Parameters
        ----------
        stage: str or list of str
            sleep stages to cut. Options: {'awake', 'rem', s1', 's2', 'ads', 'sws', 'rcbrk'} 
        epoch_len: int (Optional)
            length (in seconds) to epoch the data by 
        """

        if sleepstage == 'all':
            stages = self.stage_cuts.keys()
        else:
            stage = sleepstage

        if epoch_len is not None:
            # cut data by sleep stage and epoch

            # convert from seconds to samples
            #epoch_len = self.s_freq*epoch_len
            epoch_cuts = {}
            epoch_data = {}

            for stage in stages:
                # create epoch_cuts dict entry
                if self.stage_cuts[stage] is None:
                    epoch_cuts[stage] = None
                else:
                    epoch_cuts[stage] = {}
                    # loop through each cycle
                    for cycle, idx in self.stage_cuts[stage].items():
                        cycle_len = (idx[-1] - idx[0]).seconds
                        # if cycle is at least one epoch long, create new epoch
                        if cycle_len >= epoch_len:
                            epoch_cuts[stage][cycle] = {}
                            epochs = range(int(cycle_len/epoch_len))
                            for e in epochs:
                                epoch_cuts[stage][cycle][e] = idx[epoch_len*self.s_freq*e : epoch_len*self.s_freq*(e+1)]
                    # cut epoch data
                    if len(epoch_cuts[stage]) >0:
                        epoch_data[stage] = {}
                        cycs = [x for x in epoch_cuts[stage]]
                        for cyc in cycs:
                            epoch_data[stage][cyc] = {}
                            for epoch, idx in epoch_cuts[stage][cyc].items():
                                epoch_data[stage][cyc][epoch] = self.data.loc[(epoch_cuts[stage][cyc][epoch])]

            self.epoch_cuts = epoch_cuts
            self.epoch_data = epoch_data
        
        else:
            # cut data by sleep stage w/o epochs
            cut_data = {}
            for stage in stages:
                if self.stage_cuts[stage] is not None:
                    data = {}
                    cycs = len(self.stage_cuts[stage])
                    for c in range(1, cycs+1):
                        data[c] = self.data.loc[(self.stage_cuts[stage][c])]
                    cut_data[stage] = data

            self.cut_data = cut_data


    def export_csv(self, data=None, stages ='all', epoched=False, savedir=None):
        """ Export data to csv 
        
        Parameters
        ----------
        data: df [Optional]
            Single dataframe to export. Defaults to  self.cut_data dict of dfs produced by Dataset.cut_eeg())
        stages: str or list
            stages to export (for use in combination with dict data type)
            Options: [awake, rem, s1, s2, ads, sws, rcbrk]
        epoched: bool (default:False)
            Specify whether data is epoched

        Returns
        -------
        csv files w/ EEG data
        
        """

        # set save dirctory
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
        
        # Option to export a single dataframe
        if data is not None:
            # abort if data is not a single df
            if type(data) != pd.core.frame.DataFrame:
                print(('If data parameter is specified, type must be pd.core.frame.DataFrame. Specified data is type {}').format(type(data)))
                return
            # set savename base & elements for modification
            df = data
            savename_base = self.in_num + '_' + str(df.index[0]).replace(' ', '_').replace(':', '')
            savename_elems = savename_base.split('_')
            spec_stg =  input('Specify sleep stage? [Y/N] ')
            if spec_stg == 'N':
                savename = savename_base + '.csv'
            if spec_stg == 'Y':
                stg = input('Sleep stage: ')
                spec_cyc = input('Specify cycle? [Y/N] ')
                if spec_cyc == 'N':
                    savename = '_'.join(savename_elems[:2]) + '_' + stg + '_' + savename_elems[2] + '.csv'
                elif spec_cyc == 'Y':
                    cyc = input('Sleep stage cycle: ')
                    savename = '_'.join(savename_elems[:2]) + '_' + stg + '_cycle' + cyc + '_' + savename_elems[2] + '.csv'
            
            print('Exporting file...\n')
            data.to_csv(os.path.join(savedir, savename))
            print(('{} successfully exported.\n*If viewing in Excel, remember to set custom date format to  "m/d/yyyy h:mm:ss.000".').format(savename))
     
        # Export a nested dict of dataframes w/o epochs [Default]
        elif data is None and epoched is False:
            print('Exporting files...\n')
            if stages == 'all':
                for stg in self.cut_data.keys():
                    for cyc in self.cut_data[stg].keys():
                        df = self.cut_data[stg][cyc]
                        savename_base = self.in_num + '_' + str(df.index[0]).replace(' ', '_').replace(':', '')
                        savename_elems = savename_base.split('_')
                        savename = '_'.join(savename_elems[:2]) + '_' + stg + '_cycle' + str(cyc) + '_' + savename_elems[2] + '.csv'
                        df.to_csv(os.path.join(savedir, savename))
                        print(('{} successfully exported.').format(savename)) 
            elif type(stages) == list:
                for stg in stages:
                    if stg not in self.cut_data.keys():
                        stg = input('"'+ stg+'" is not a valid sleep stage code or is not present in this dataset. Options: awake rem s1 s2 ads sws rcbrk\nSpecify correct code from options or [skip]: ')
                    if stg == 'skip':
                        continue
                    for cyc in self.cut_data[stg].keys():
                        df = self.cut_data[stg][cyc]
                        savename_base = self.in_num + '_' + str(df.index[0]).replace(' ', '_').replace(':', '')
                        savename_elems = savename_base.split('_')
                        savename = '_'.join(savename_elems[:2]) + '_' + stg + '_cycle' + str(cyc) + '_' + savename_elems[2] + '.csv'
                        df.to_csv(os.path.join(savedir, savename))
                        print(('{} successfully exported.').format(savename))
            elif type(stages) == str:
                stg = stages
                if stg not in self.cut_data.keys():
                    stg = input('"'+ stg+'" is not a valid sleep stage code or is not present in this dataset. Options: awake rem s1 s2 ads sws rcbrk\nSpecify correct code from options or [abort]: ')
                if stg == 'abort':
                    return
                for cyc in self.cut_data[stg].keys():
                    df = self.cut_data[stg][cyc]
                    savename_base = self.in_num + '_' + str(df.index[0]).replace(' ', '_').replace(':', '')
                    savename_elems = savename_base.split('_')
                    savename = '_'.join(savename_elems[:2]) + '_' + stg + '_cycle' + str(cyc) + '_' + savename_elems[2] + '.csv'
                    df.to_csv(os.path.join(savedir, savename))
                    print(('{} successfully exported.').format(savename))
            print('\nDone.')

        # Export a nested dict of dataframes w/ epochs
        elif data is None and epoched is True:
            print('Exporting files...\n')
            if stages == 'all':
                for stg in self.epoch_data.keys():
                    for cyc in self.epoch_data[stg].keys():
                        for epoch in self.epoch_data[stg][cyc].keys():
                            df = self.epoch_data[stg][cyc][epoch]
                            savename_base = self.in_num + '_' + str(df.index[0]).replace(' ', '_').replace(':', '')
                            savename_elems = savename_base.split('_')
                            savename = '_'.join(savename_elems[:2]) + '_' + stg + '_cycle' + str(cyc) + '_epoch' + str(epoch) + '_' + savename_elems[2] + '.csv'
                            df.to_csv(os.path.join(savedir, savename))
                            print(('{} successfully exported.').format(savename)) 
            elif type(stages) == list:
                for stg in stages:
                    if stg not in self.cut_data.keys():
                        stg = input('"'+ stg+'" is not a valid sleep stage code or is not present in this dataset. Options: awake rem s1 s2 ads sws rcbrk\nSpecify correct code from options or [skip]: ')
                    if stg == 'skip':
                        continue
                    for cyc in self.cut_data[stg].keys():
                        for epoch in self.epoch_data[stg][cyc].keys():
                            df = self.epoch_data[stg][cyc][epoch]
                            savename_base = self.in_num + '_' + str(df.index[0]).replace(' ', '_').replace(':', '')
                            savename_elems = savename_base.split('_')
                            savename = '_'.join(savename_elems[:2]) + '_' + stg + '_cycle' + str(cyc) + '_epoch' + str(epoch) + '_' + savename_elems[2] + '.csv'
                            df.to_csv(os.path.join(savedir, savename))
                            print(('{} successfully exported.').format(savename)) 
            elif type(stages) == str:
                stg = stages
                if stg not in self.cut_data.keys():
                    stg = input('"'+ stg+'" is not a valid sleep stage code or is not present in this dataset. Options: awake rem s1 s2 ads sws rcbrk\nSpecify correct code from options or [abort]: ')
                if stg == 'abort':
                    return
                for cyc in self.cut_data[stg].keys():
                    for epoch in self.epoch_data[stg][cyc].keys():
                            df = self.epoch_data[stg][cyc][epoch]
                            savename_base = self.in_num + '_' + str(df.index[0]).replace(' ', '_').replace(':', '')
                            savename_elems = savename_base.split('_')
                            savename = '_'.join(savename_elems[:2]) + '_' + stg + '_cycle' + str(cyc) + '_epoch' + str(epoch) + '_' + savename_elems[2] + '.csv'
                            df.to_csv(os.path.join(savedir, savename))
                            print(('{} successfully exported.').format(savename)) 
            print('\nDone.\n\t*If viewing in Excel, remember to set custom date format to  "m/d/yyyy h:mm:ss.000".')

        else:
            print(('Abort: Data must be type pd.core.frame.DataFrame or dict. Input data is type {}.').format(type(data)))