""" This module contains functions for batch analyses 
	
	To do: 
		- Fix hacky imports

"""

import glob
import os
import nrem
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


def batch_spindle_analysis(fpath, match, export_dir, wn=[8, 16], order=4, sp_mw=0.2, loSD=0, hiSD=1.5, min_sep=0.2, 
    				duration=[0.5, 3.0], min_chans=9, zmethod='trough', trough_dtype='spfilt', buff=False, buffer_len=3, 
    				psd_bandwidth=1.0, norm_range=[(4,6), (18, 25)], spin_range=[9, 16], datatype = 'spfilt'):
	""" Run spindle analysis on all matching files in a directory """ 

	# get a list of all matching file
	glob_match = f'{fpath}/*{match}*'
	fnames = [os.path.basename(x) for x in glob.glob(glob_match)]

	for fname in fnames:
		print(f'Analyzing {fname}...\n')
		spindle_analysis(fname, fpath, export_dir, wn, order, sp_mw, loSD, hiSD, min_sep, 
    				duration, min_chans, zmethod, trough_dtype, buff, buffer_len, 
    				psd_bandwidth, norm_range, spin_range, datatype)

	print('Batch complete.')
