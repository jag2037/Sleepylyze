""" this file contains a class for transitions macrostructure data and associated methods """

import json
import os
import pandas as pd
from collections import Counter

class HypStats:
    """ General class for analyzing sleep state transitions """

    def __init__(self, fhyp_stats, fhyp=None):
        """ fhypstats: json dump file from ioeeg.Dataset export_hypstats method """
        
        s = fhyp_stats.split('_')
        self.in_num = s[0]
        self.start_date = s[-1].split('.')[0]
        
        with open(fhyp_stats, 'r') as f:
            self.hyp_stats = json.load(f)
        
        if fhyp is not None:
            self.hyp = pd.Series.from_csv(fhyp)
            # note: Series.from_csv is deprecated. if needed, can use:
            ## self.hyp = pd.read_csv(hyp_file, index_col=0, header=None, parse_dates=True, squeeze=True)


    def calc_transitions(self):
        """ calculate sleep stage transitions """

        stages = {'Awake': 0., 'REM': 1., 'Stage 1': 2., 'Stage 2': 3., 'Alpha-Delta': 4., 'SWS': 5., 'Record Break': 6.}

        # Count number of epochs present
        epoch_dict = {}
        for x, c in Counter(self.hyp.values).items():
            stg = list(stages.keys())[list(stages.values()).index(x)]
            epoch_dict[stg] = c

        # add entries for absent stages
        for stg in stages.keys():
            if stg not in epoch_dict.keys():
                epoch_dict[stg] = 0

        # count number of transitions
        count_df = pd.DataFrame(columns = list(stages.keys())[:-1], index = list(stages.keys())[:-1])
        for (x, y), c in Counter(zip(self.hyp.values, self.hyp.values[1:])).items():
            # don't count record breaks of self-self 'transitions'
             if (x != 6.0) & (y != 6.0) & (x != y):
                col = list(stages.keys())[list(stages.values()).index(y)]
                row = list(stages.keys())[list(stages.values()).index(x)]
                count_df[col][row] = c

        # fill NaNs with zeros
        count_df.fillna(value=0, inplace=True)
        # set column and index labels
        count_df.columns = list(stages.keys())[:-1]

        # calculate percentages
        percent_df = count_df / sum(count_df.sum(axis=0)) * 100
        
        self.epoch_counts = epoch_dict
        self.transition_counts = count_df
        self.transition_perc = percent_df


    def export_transitions(self, savedir=None):
        """ export epoch counts dict, transition counts and percents dfs """
        
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
                    
        # export epoch_counts dict
        savename = self.in_num + '_' + self.start_date + '_epoch_counts.txt'
        save = os.path.join(savedir, savename)
        with open(save, 'w') as f:
            json.dump(self.epoch_counts, f, indent=4)
        
        # export transition counts df
        savename = self.in_num + '_' + self.start_date + '_transition_counts.csv'
        save = os.path.join(savedir, savename)
        self.transition_counts.to_csv(save, header=True, index=True)
        
        # export transition percents df
        savename = self.in_num + '_' + self.start_date + '_transition_percents.csv'
        save = os.path.join(savedir, savename)
        self.transition_perc.to_csv(save, header=True, index=True)
