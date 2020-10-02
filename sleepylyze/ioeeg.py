""" 
    This file contains a Dataset class for raw EEG files collected from the Natus XLTEK
    clinical EEG system

    To Do:
        Update docstrings
        Add support for earlier headboxes
        Incorporate metadata pSQL metadata table update
    NOTE:
        For new headboxes, add HBSN from exported XLTEK file to the correct montage
"""

import datetime
from datetime import timedelta
from io import StringIO
import json
import math 
import numpy as np 
import os
import pandas as pd
import psycopg2
import re
import statistics
from sqlalchemy import *


class Dataset:
    """ General class containing EEG recordings

    NOTE: This assumes a continuous record. If record breaks are present they WILL NOT
    be detected and timestamps will be inaccurate
    
    Attributes
    ----------
    metadata: dict
        import import & analysis metadata
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
    
    Methods
    -------
    get_info
    get_chans
    load_eeg
    trim_eeg
    to_psql
    clean_eeg
    clean_eeg_psql
    load_hyp
    export_hypstats
    cut_EEG
    export_csv
    """
    
    def __init__(self, fname, fpath=None, trim=True, start='22:00:00', end='07:00:00', noise_log=None, rm_chans=None, 
                psql=True, psql_user = None, psql_password=None, psql_database='raw_eeg', 
                data_dir='D:\Jackie\RawEEG'):
        """ Initialize Dataset object and scrub noise

            Parameters
            ----------
            fname: str
                filename
            fpath: str
                path to file
            trim: bool (default: True)
                trim EEG file
            start: str (default: '22:00:00')
                time to begin trim (inclusive)
            end: str (default: '07:00:00')
                time to end trim (exclusive)
            noise_log: str
                noise log file (including path)
            rm_chans: list of str (default: None)
                bad channels to remove
            psql: bool (default: True)
                upload data to local postgres server & scrub through psql
            psql_user: str
                local postgres server username
            psql_pass: str
                local postgres server password
            psql_database: str (default: 'raw_eeg')
                postgres database to upload to
            data_dir: str (default: 'D:\Jackie\RawEEG')
                directory to save cleaned and condensed raw EEG csv files

        Returns
        -------
        ioeeg.Dataset object containing raw EEG data, optionally cleaned and trimmed

        """
        
        if fpath is not None:
            filepath = os.path.join(fpath, fname)
        else:
            filepath = fname
        
        self.metadata = {'in_num': fname.split("_")[0],
                        'fname': fname,
                        'filepath': filepath}

        self.get_info()
        self.get_chans()
        self.load_eeg()
        
        if trim == True:
            self.trim_eeg(start, end)

        if psql == True:
            self.to_psql(psql_user, psql_password, psql_database, data_dir)
        
        if noise_log is not None or rm_chans is not None:
            if psql == True:
                self.clean_eeg_psql(noise_log, rm_chans, psql_user, psql_password, psql_database, data_dir)
            else:
                self.clean_eeg(noise_log, rm_chans)

        print('\nDone.')

    
    # io methods #
    def get_info(self):
        """ Read in the header block and extract useful info """

        print("Patient identifier:", self.metadata['in_num'])

        # read the first line to check encoding
        with open(self.metadata['filepath'], 'r') as f:
            line = f.readline()
        if line[3] == '\x00':
            f_encoding = 'utf-16-le'
        else:
            f_encoding = 'ascii' #might want to set this to None (used with open and pd.read_csv)
        self.metadata['encoding'] = f_encoding
        
        # extract sampling freq, # of channels, headbox sn
        with open(self.metadata['filepath'], 'r', encoding=f_encoding) as f: # pull out the header w/ first recording
            header = []
            for i in range(1, 267): # this assumes s_freq<=250
                header.append(f.readline()) # use readline (NOT READLINES) here to save memory for large files
        self.metadata['s_freq'] = int(float(header[7].split()[3])) # extract sampling frequency
        self.metadata['chans'] = int(header[8].split()[2]) # extract number of channels as integer
        self.metadata['hbsn'] = int(header[10].split()[3]) # extract headbox serial number
        self.metadata['start_date'] = (header[15].split()[0]).replace('/', '-')
        self.metadata['start_time'] = header[15].split()[1] # extract start time in hh:mm:ss
        
        # set metadata if there's an extra 'Stamp' column inserted before the data (exists for some versions of XLTEK)
        if header[13].split('\t')[3] == 'Stamp':
        	self.metadata['stamp_col'] = True
        else:
        	self.metadata['stamp_col'] = False

        # Get starting time in usec for index
        s_freq = self.metadata['s_freq']
        start_time = self.metadata['start_time']
        firstsec = []
        for i in range(15, len(header)):
            firstsec.append(header[i].split()[1])
        firstsec = pd.Series(firstsec)
        # Find the timestamp cut between seconds 1 & 2, convert to usec
        first_read = s_freq - (firstsec.eq(start_time).sum())
        self.metadata['start_us'] = 1/s_freq*first_read
         
    def get_chans(self):
        """ Define the channel list for the detected headbox """
        chans = self.metadata['chans']
        hbsn = self.metadata['hbsn']
        
        def check_chans(chans, expected_chans):
            print(chans, ' of ', expected_chans, ' expected channels found')
            if chans != expected_chans: # this may not be a proper check
                print('WARNING: Not all expected channels for this headbox were detected. Proceeding with detected channels')
        
        if hbsn in [125, 254, 245]:
            hbid = "MOBEE 32" # EKG used for IN363Mv2 (& IN343C??). FPZ is ground. A2 is EOG2.
            print('Headbox:', hbid)
            expected_chans = 35
            check_chans(chans, expected_chans)
            channels_all = ['REF','FP1','F7','T3','A1','T5','O1','F3','C3','P3','FPZorEKG','FZ','CZ',
                'PZ','FP2','F8','T4','A2','T6','O2','F4','C4','P4','AF7','AF8','FC5','FC6','FC1','FC2',
                'CP5','CP6','CP1','CP2','OSAT','PR']
            channels = channels_all[:-2]
            channels_rmvd = channels_all[-2:]
            print('Removed the following channels: \n', channels_rmvd)

        elif hbsn == 65535 and self.metadata['chans'] == 40: # for IN346B. need to adjust this for other EMU40s
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
                'T5','PO7','FPz','Fz','Cz','CPz','Pz','POz','Oz','Fp2','F4','FC2','C4','CP2','P4','O2',
                'AF8','F8','FC6','T4','CP6','T6','PO8','F1','F2','EOG_L','EOG_R','EKG']
                # remove any chans here?
            channels = channels_all

        elif hbsn == 207:
            hbid = "EMU40"
            print('Headbox:', hbid)
            expected_chans = 40
            check_chans(chans, expected_chans)
            channels_all = ['Fp1','F7','T3','T5','O1','F3','C3','P3','EMGref','Fz','Cz','Fp2','F8','T4',
                'T6','O2','F4','C4','P4','EMG','FPz','Pz','EMG1','EMG2','FC5','FC6','FC1','FC2','CP5',
                'CP6','CP1','CP2','PO7','PO8','F1','F2','CPz','POz','Oz','EKG']
            channels = channels_all

        self.metadata['hbid'] = hbid
        self.metadata['channels'] = channels
            
    def load_eeg(self):
        """ Import raw EEG data """

        # set the first column of data to import (depends on presence of 'Stamp' col)
        if self.metadata['stamp_col']:
        	start_col = 4
        else:
        	start_col = 3

        # set the last column of data to import
        end_col = start_col + len(self.metadata['channels'])
        
        # read in only the data
        print('Importing EEG data...')
        # setting dtype to float will speed up load, but crashes if there is anything wrong with the record
        data = pd.read_csv(self.metadata['filepath'], encoding=self.metadata['encoding'], delim_whitespace=True, header=None, skiprows=15, usecols=range(start_col,end_col),
                               dtype = float, na_values = ['AMPSAT', 'SHORT'])
        
        # create DateTimeIndex
        ind_freq = str(int(1/self.metadata['s_freq']*1000000))+'us'
        ind_start = self.metadata['start_date'] + ' ' + self.metadata['start_time'] + str(self.metadata['start_us'])[1:]
        ind = pd.date_range(start = ind_start, periods=len(data), freq=ind_freq)

        # make a new dataframe with the proper index & column names
        data.columns = pd.MultiIndex.from_arrays([self.metadata['channels'], np.repeat(('Raw'), len(self.metadata['channels']))],names=['Channel','datatype'])
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

        Returns
        -------
        Dataset.data attribute trimmed to start:end times
        """
        
        print('Trimming EEG...')
        # specify indices NOT between start and end time
        rm_idx = self.data.between_time(end, start, include_start=True, include_end=False).index
        # drop specified indices
        self.data.drop(rm_idx, axis=0, inplace=True)
        
        print('Data trimmed to {} (inclusive) - {} (exclusive).'.format(start, end))

    def to_psql(self, psql_user, psql_password, psql_database, data_dir):
        """ Upload raw EEG data to local postgreSQL server & save out to condensed csv file """
        
        print('Uploading dataframe to postgreSQL database...')
        
        # create StringIO object
        in_memory_csv = StringIO()
        
        # write dataframe to stringio object without headers & reset pointer
        self.data.to_csv(in_memory_csv, header=False)
        in_memory_csv.seek(0)
        
        # create column names string for SQL table creation
        cols = [x.lower() for x in self.metadata['channels']]
        column_str = []
        for col in cols:
            column_str.append(f'{col} NUMERIC(16,6),')
        column_str = "".join(column_str)
        
        # create string for SQL table creation (this overwrites if exists)
        table_name = self.metadata['fname'].split('.')[0].lower()
        table_name = re.sub('-', '_', table_name)
        self.psql_tablename = table_name
        base_table = f"""
        DROP TABLE IF EXISTS {table_name};
        CREATE TABLE {table_name} (
            time TIMESTAMP WITHOUT TIME ZONE,
            {column_str}
            PRIMARY KEY (time)
        );
        """
        # create a link to the psql database
        database = 'postgresql://' + psql_user + ':' + psql_password + '@localhost/' + psql_database
        engine = create_engine(database)
        conn = engine.connect()
        conn.execute(base_table)
        
        # create table
        meta = MetaData(engine)
        data_table = Table(table_name, meta, autoload=True)
        
        # load data
        with engine.begin() as conn2:
            cur = conn2.connection.cursor()
            cur.copy_from(in_memory_csv, table_name, sep=',', columns= ['time'] + cols, null='')
        
        # save to condensed csv file
        print('Saving condensed csv file..')
        raw_csv = os.path.join(data_dir, (table_name + '_raw.csv'))
        copy = f"""COPY {table_name} TO '{raw_csv}' WITH CSV HEADER;"""
        conn.execute(copy)
        
        print('PostgreSQL upload complete.')

    def clean_eeg_psql(self, noise_log, rm_chans, psql_user, psql_password, psql_database, data_dir):
        """ Replace artifact with NaN in the postgreSQL raw EEG table 

        Parameters
        ----------
        noise_log: .txt file (optional)
            file containing list of artifact seconds to remove (format: YYYY-MM-DD hh:mm:ss)
        rm_chans: str or list of string (optional)
            entire channels to remove
        data_dir
        psql_user: str
            postgreSQL username
        psql_password: str
            postgreSQL password
        psql_database: str
            postgreSQL database where raw eeg file is located

        Returns
        -------
        self.data pd.DataFrame of raw EEG with artifact times replaced with NaNs
        Condensed cleaned dataframe as csv output
        """
    
        # create a link to the sql database
        database = 'postgresql://' + psql_user + ':' + psql_password + '@localhost/' + psql_database
        engine = create_engine(database)
        conn = engine.connect()    
        
        # specify the table
        meta = MetaData(engine)
        table_name = self.psql_tablename
        data_table = Table(table_name, meta, autoload=True)
        
        # set channel list
        channel_list = [col[0] for col in self.data.columns]
            
        # remove noisy channels
        if rm_chans is not None:
            print('Removing noisy channels...')
            # check chans against data channel list (case insensitive) & convert str to list
            if type(rm_chans) == str:
                rm_chans_list = [x.lower() for x in channel_list if rm_chans.casefold() == x.casefold()]
            elif type(rm_chans) == list:
                rm_chans_list = [x.lower() for x in channel_list for r in rm_chans if r.casefold() == x.casefold()]
            
            # create dict of chans to remove
            noise_chans = {}
            for col in rm_chans_list:
                noise_chans[col] = None
            # replace noise channels w/ null values
            data_table.update().values(**noise_chans).execute()
            print('Noisy channels removed.')
        
        # remove noisy indices
        if noise_log is not None:
            print('Removing noisy indices...')
            noise = pd.read_csv(noise_log, header=None, names=[0, 1, 'channels'], sep = '\t', index_col=0, parse_dates=[[0, 1]])
            noise.index.name = 'time'
            # split channel strings into lists
            noise['channels'] = [re.findall(r"[\w'\*]+", n.lower()) for n in noise['channels']]
            
            # convert noise df to dictionary
            noise_dict = noise.to_dict()
            # create lowercase & eeg channel lists
            channels_lower = [col[0].lower() for col in self.data.columns]
            eeg_chans = [x for x in channels_lower if x != 'ekg']

            for t, chans in noise_dict['channels'].items():
                # create dictionary for channel:value updating
                chan_noise_dict = {}
                for c in chans:
                    if c == '*':
                        for cx in eeg_chans:
                            chan_noise_dict[cx] = None
                    else:
                        chan_noise_dict[c] = None

                # update values
                data_table.update().values(**chan_noise_dict).where(and_(
                    data_table.c.time >= t,
                    data_table.c.time < t + timedelta(seconds=1)
                )).execute()
            print('Noisy indices removed.')
            
        # save to csv file
        print('Saving cleaned EEG file...')
        clean_csv = os.path.join(data_dir, (table_name + '_clean.csv'))
        copy = f"""COPY {table_name} TO '{clean_csv}' WITH CSV HEADER;"""
        conn.execute(copy)
                
        # load back in & reformat column names
        print('Reloading pandas dataframe...')
        self.data = pd.read_csv(clean_csv, index_col=0, parse_dates=True)
        self.data.columns = pd.MultiIndex.from_arrays([self.metadata['channels'], np.repeat(('Raw'), len(self.metadata['channels']))],names=['Channel','datatype'])
        self.data.index.name = None
        
        print('EEG cleaning complete.')

    def clean_eeg(self, noise_log, rm_chans):
        """ Replace artifact with NaN in the pandas df
            NOTE: this takes ~16x longer than clean_eeg_psql
        
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
            eeg_channels = [e for e in self.metadata['channels'] if e not in ('EOG_L', 'EOG_R', 'EKG')]
            for t, c in noise_idx.items():
                for cx in c:
                    if cx == '*':
                        for x in eeg_channels:
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
        self.stage_cuts: dict
            Nested dict of sleep stages, cycles, and DateTimeIndex
            Ex. {'awake': {1: DateTimeIndex[(...)], 2: DateTimeIndex[(...)]}}
        self.hyp_stats
        
        """
        # read the first line to get starting date & time
        with open(scorefile, 'r') as f:
                first_epoch = f.readline()
                start_date = first_epoch.split(' ')[0]
                start_sec = first_epoch.split(' ')[1].split('\t')[0]
                
        # read in sleep scores & resample to EEG/EKG frequency
        print('Importing sleep scores...')
        hyp = pd.read_csv(scorefile, delimiter='\t', header=None, names=['Score'], usecols=[1], dtype=float)
        scores = pd.DataFrame(np.repeat(hyp.values, self.metadata['s_freq']*30,axis=0), columns=hyp.columns)

        # reindex to match EEG/EKG data
        ## --> add warning here if index start date isn't found in EEG data
        ind_freq = str(int(1/self.metadata['s_freq']*1000000))+'us'
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
        ms = pd.Timedelta(1/self.metadata['s_freq']*1000, 'ms')
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
            if stage != 'rbrk':
                percent = len(hyp[hyp.values == code])/(len(hyp)-len(hyp[hyp.values == 6.0])) * 100
            elif stage == 'rbrk':
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
        stats_savename = self.metadata['in_num'] + '_SleepStats_' + self.metadata['start_date'] + '.txt'
        stats_save = os.path.join(savedir, stats_savename)
        with open(stats_save, 'w') as f:
            json.dump(self.hyp_stats, f, indent=4)

        # export the resampled hypnogram @ the original frequency (for downstream speed)
        new_hyp = self.hyp[(self.hyp.index.microsecond == 0) & ((self.hyp.index.second == 0) | (self.hyp.index.second == 30))]
        hyp_savename = self.metadata['in_num'] + '_Hypnogram_' + self.metadata['start_date'] + '.txt'
        hyp_save = os.path.join(savedir, hyp_savename)
        new_hyp.to_csv(hyp_save, header=False)

    def cut_EEG(self, sleepstage='all', epoch_len=None):
        """ cut dataset based on loaded hypnogram 

        	*NOTE: non-epoched data updated for Pandas KeyError future warning. Epoched data to be updated.

        Parameters
        ----------
        stage: str or list of str
            sleep stages to cut. Options: {'awake', 'rem', s1', 's2', 'ads', 'sws', 'rcbrk'} 
        epoch_len: int (Optional)
            length (in seconds) to epoch the data by 
        """

        # set data start and end idxs (for determining valid cut indices)
        data_start, data_end = np.sort(self.data.index)[0], np.sort(self.data.index)[-1]

        # set sleep stages to cut
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
                                epoch_cuts[stage][cycle][e] = idx[epoch_len*self.metadata['s_freq']*e : epoch_len*self.metadata['s_freq']*(e+1)]
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
                        # set the indices loaded from the noise log
                        idxs = self.stage_cuts[stage][c]
                        # exclude any indices that are outside of the loaded data
                        valid_idxs = idxs[(idxs >= data_start) & (idxs <= data_end)]
                        data[c] = self.data.loc[valid_idxs]
                    cut_data[stage] = data

            self.cut_data = cut_data


    def export_csv(self, data=None, stages ='all', epoched=False, savedir=None):
        """ Export data to csv (single df or cut data)

        TO DO: Add 'NA' values to cycle and epoch if exporting a single df without specs
        
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
            savename_base = self.metadata['in_num'] + '_' + str(df.index[0]).replace(' ', '_').replace(':', '').split('.')[0]
            savename_elems = savename_base.split('_')
            spec_stg =  input('Specify sleep stage? [Y/N] ')
            if spec_stg.casefold() == 'n':
                savename = savename_base + '.csv'
            elif spec_stg.casefold() == 'y':
                stg = input('Sleep stage: ')
                spec_cyc = input('Specify cycle? [Y/N] ')
                if spec_cyc.casefold() == 'n':
                    savename = '_'.join(savename_elems[:2]) + '_' + stg + '_' + savename_elems[2] + '.csv'
                elif spec_cyc.casefold() == 'y':
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
                        savename_base = self.metadata['in_num'] + '_' + str(df.index[0]).replace(' ', '_').replace(':', '').split('.')[0]
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
                        savename_base = self.metadata['in_num'] + '_' + str(df.index[0]).replace(' ', '_').replace(':', '').split('.')[0]
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
                    savename_base = self.metadata['in_num'] + '_' + str(df.index[0]).replace(' ', '_').replace(':', '').split('.')[0]
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
                            savename_base = self.metadata['in_num'] + '_' + str(df.index[0]).replace(' ', '_').replace(':', '').split('.')[0]
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
                            savename_base = self.metadata['in_num'] + '_' + str(df.index[0]).replace(' ', '_').replace(':', '').split('.')[0]
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
                            savename_base = self.metadata['in_num'] + '_' + str(df.index[0]).replace(' ', '_').replace(':', '')
                            savename_elems = savename_base.split('_')
                            savename = '_'.join(savename_elems[:2]) + '_' + stg + '_cycle' + str(cyc) + '_epoch' + str(epoch) + '_' + savename_elems[2] + '.csv'
                            df.to_csv(os.path.join(savedir, savename))
                            print(('{} successfully exported.').format(savename)) 
            print('\nDone.\n\t*If viewing in Excel, remember to set custom date format to  "m/d/yyyy h:mm:ss.000".')

        else:
            print(('Abort: Data must be type pd.core.frame.DataFrame or dict. Input data is type {}.').format(type(data)))