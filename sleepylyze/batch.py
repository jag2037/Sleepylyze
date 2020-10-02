""" This module contains functions for batch analyses 
	
	To do: 
		- Fix hacky imports
        - test calc_elapsed_sleep

"""

import glob
import os
import nrem
import pandas as pd
import sleepyplot as slp 

def spindle_analysis(fname, fpath, export_dir, wn, order, sp_mw, loSD, hiSD, min_sep, 
    				duration, min_chans, zmethod, trough_dtype, buff, buffer_len, 
    				psd_bandwidth, norm_range, spin_range, datatype):
    """ Full spindle analysis """
    
    # load data
    n = nrem.NREM(fname, fpath) #check other params
    # detect & analyze spindles
    n.detect_spindles(wn, order, sp_mw, loSD, hiSD, min_sep, duration, min_chans)
    n.analyze_spindles(zmethod, trough_dtype, buff, buffer_len, psd_bandwidth, 
    					norm_range, spin_range, datatype)
    # export data
    n.export_spindles(export_dir)

    # Plot analyses & save plots
    plot_dir = os.path.join(export_dir, 'plots')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    basename = os.path.join(plot_dir, fname.split('.')[0])
    print('Plotting analyses & saving plots...')
    
    # spindle overlay
    spin_fig_raw = slp.plot_spins(n, datatype='Raw')
    savename = f'{basename}_SpinOverlay_raw.png'
    spin_fig_raw.savefig(savename, dpi=300)
    
    spin_fig_filt = slp.plot_spins(n, datatype='spfilt')
    savename = f'{basename}_SpinOverlay_spfilt.png'
    spin_fig_filt.savefig(savename, dpi=300)
    
    # spindle means
    spin_means_fig = slp.plot_spin_means(n)
    savename = f'{basename}_spinmeans.png'
    spin_means_fig.savefig(savename, dpi=300)
    
    # power spectra
    spec_fig = slp.plot_spindlepower(n)
    savename = f'{basename}_spectra.png'
    spec_fig.savefig(savename, dpi=300)
    
    # gottselig norm
    gott_fig = slp.plot_gottselig(n)
    savename = f'{basename}_gottselig.png'
    gott_fig.savefig(savename, dpi=300)
    print('Done.\n\n')


def calc_elapsed_sleep(in_num, hyp_file, fpath, savedir, export=True):
    """ 
    Calculate minutes of elapsed sleep from a hypnogram file & concatenate stage 2 sleep files

    Parameters
    ----------
    in_num: str
        patient identifier
    hyp_file: str (format: *.txt)
        file with hypnogram at 30-second intervals
    fpath: str
        path to EEG files cut by sleep stage
    savedir: str
        path to save EEG files cut by hrs elapsed sleep
    export: bool (default: True)
        whether to export blocked dataframes

    Returns
    -------
    .csv files with EEG data blocked in two-hour chunks (according to Purcell et al. 2017)
        OR
    pd.dataframes blocked in two-hour chunks (according to Purcell et al. 2017)

    """
    
    # calculate elapsed sleep for each 30-second time interval
    print('Loading hypnogram...')
    sleep_scores = [1, 2, 3, 4, 5] # exclude 0 and 6 for awake and record break
    hyp = pd.read_csv(hyp_file, header=None, index_col=[0], sep='\t', names=['time', 'score'], parse_dates=True)
    mins_elapsed = hyp.score.isin(sleep_scores).cumsum()/2
    
    # get a list of all matching files
    glob_match = f'{fpath}/{in_num}*_s2_*'
    files = glob.glob(glob_match)
    
    # make list of dfs for concat
    print('Reading data...')
    data = [pd.read_csv(file, header = [0, 1], index_col = 0, parse_dates=True) for file in files]

    # add NaN to the end of each df
    data_blocked = [df.append(pd.Series(name=df.iloc[-1].name + pd.Timedelta(milliseconds=1))) for df in data]

    # concatenate the dfs
    print('Concatenating data...')
    s2_df = pd.concat(data_blocked).sort_index()
    
    # assign indices to hours elapsed sleep
    print('Assigning minutes elapsed...')
    idx0_2 = mins_elapsed[mins_elapsed.between(0, 120)].index
    idx2_4 = mins_elapsed[mins_elapsed.between(120.5, 240)].index
    idx4_6 = mins_elapsed[mins_elapsed.between(240.5, 360)].index
    idx6_8 = mins_elapsed[mins_elapsed.between(360.5, 480)].index
    
    dfs = []
    df_names = []
    # cut dataframe into blocks by elapsed sleep (0-2, 2-4, 4-6, 6-8)
    df_two = s2_df[(s2_df.index > idx0_2[0]) & (s2_df.index < idx0_2[-1])]
    dfs.append(df_two)
    df_names.append('0-2hrs')
    try:
        df_four = s2_df[(s2_df.index > idx2_4[0]) & (s2_df.index < idx2_4[-1])]
    except IndexError:
        print('<2 hrs sleep. Passing block 2-4hrs.')
        pass
    else:
        dfs.append(df_four)
        df_names.append('2-4hrs')
    try:
        df_six = s2_df[(s2_df.index > idx4_6[0]) & (s2_df.index < idx4_6[-1])]
    except IndexError:
        print('<4 hrs sleep. Passing block 4-6hrs.')
        pass
    else:
        dfs.append(df_six)
        df_names.append('4-6hrs')
    try:
        df_eight = s2_df[(s2_df.index > idx6_8[0]) & (s2_df.index < idx6_8[-1])]
    except IndexError:
        print('<6 hrs sleep. Passing block 6-8hrs.')
        pass
    else:
        dfs.append(df_eight)
        df_names.append('6-8hrs')
    
    if export:
        # export blocked data
        if not os.path.exists(savedir):
            print(savedir + ' does not exist. Creating directory...')
            os.makedirs(savedir)

        print('Saving files...')
        for df, hrs in zip(dfs, df_names):
            try:
                date = df.index[0].strftime('%Y-%m-%d')
            # if the df is empty, pass
            except IndexError:
                print(f'No stage 2 sleep during {hrs} of sleep.')
                pass
            savename = in_num + '_' + date + '_s2_' + hrs + '.csv'
            df.to_csv(os.path.join(savedir, savename))

        print(f'Files saved to {savedir}')
    else:
        return df_two, df_four, df_six, df_eight

    print('Done')



def batch_spindle_analysis(fpath, match, export_dir, wn=[8, 16], order=4, sp_mw=0.2, loSD=0, hiSD=1.5, min_sep=0.2, 
    				duration=[0.5, 3.0], min_chans=9, zmethod='trough', trough_dtype='spfilt', buff=False, buffer_len=3, 
    				psd_bandwidth=1.0, norm_range=[(4,6), (18, 25)], spin_range=[9, 16], datatype = 'spfilt'):
	""" Run spindle analysis on all matching files in a directory """ 

	# get a list of all matching files
	glob_match = f'{fpath}/*{match}*'
	fnames = [os.path.basename(x) for x in glob.glob(glob_match)]

	for fname in fnames:
		print(f'Analyzing {fname}...\n')
		spindle_analysis(fname, fpath, export_dir, wn, order, sp_mw, loSD, hiSD, min_sep, 
    				duration, min_chans, zmethod, trough_dtype, buff, buffer_len, 
    				psd_bandwidth, norm_range, spin_range, datatype)

	print('Batch complete.')
