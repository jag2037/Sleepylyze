""" wrapper codes for lfp spindle analysis """

def analyze_spindles(seg):
    """ run spindle analysis on a data segment 

		Params
		------
		seg: str
			full path to lfp data segment

		Returns
		-------
		n: nrem.NREM object 
			with spindles detected & analyzed
		segname: str
			name of data segment
		savedir: str
			directory where all segment analyses are saved
    """
    
    # set new save directory for each segment 
    savedir = seg.replace('.csv', '')
    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    # set path params & load data
    fpath, fname = os.path.split(seg)
    segname = fname.replace('.csv', '') # for setting savenames later
    n = NREM(fname, fpath)
    # run spindle detection. Remove minimum channel criteria
    n.detect_spindles(min_chans_r=0, min_chans_d=0, spin_range=[8, 16], pwr_thres=25)
    # calculate stats
    # set 'concat' to calculate spectra for concatenated spindles + individual spindles
    n.analyze_spindles(psd_type='concat', psd_bandwidth=1.5, fstats_concat=False)

    # export time-domain stats
    savename = f'{segname}_SpindleTimeStats.csv'
    save = os.path.join(savedir, savename)
    n.spindle_tstats.to_csv(save)
    
    # export metadata --> remove once export method is updated
    savename = f'{segname}_Metadata.txt'
    save = os.path.join(savedir, savename)
    with open(save, 'w') as f:
        json.dump(n.metadata, f, indent=4)
    
    return n, segname, savedir


def get_xlims(n, xaxis_seconds=30):
    """ get xlimits for x-axis windowing in plotLFP. Allows saving multiple figures at same x-scale for long time series 
        
        Parameters
        ----------
        xaxis_seconds: int (default: 30)
            length of x-axis window in seconds
        *see sleepyplot.plotLFP for details on additional parameters that can be passed

        Returns
        -------
        xlims: list of tuple
        	tuple of xlims for each figure
        xaxis_windows: int
        	number of figures to produce
    """
    
    # create xlim tuples for windowing
    total_seconds = (n.data.index[-1] - n.data.index[0]).total_seconds()
    
    # if the data is longer than the set x-axis window
    if total_seconds > xaxis_seconds:
        # calculate samples per window
        xaxis_samples = n.s_freq*xaxis_seconds
        # calculate total windows (always round up to keep consistent scale & not truncate the data)
        xaxis_windows = math.ceil(len(n.data)/xaxis_samples)
        # if the windows divide evenly, pull timestamps from data
        if len(n.data) == xaxis_samples*xaxis_windows:
            xlims = [(n.data.index[w*xaxis_samples], n.data.index[(w+1)*xaxis_samples -1]) for w in range(xaxis_windows)]
        # otherwise create the last window manually
        else:
            xlims = [(n.data.index[w*xaxis_samples], n.data.index[(w+1)*xaxis_samples -1]) for w in range(xaxis_windows-1)]
            last_window = (n.data.index[xaxis_windows-1*xaxis_samples], n.data.index[xaxis_windows-1*xaxis_samples] + datetime.timedelta(0, xaxis_seconds))
            xlims.append(last_window)
    else:
        xlims, xaxis_windows = None, None
    
    return xlims, xaxis_windows


def make_figures(n, segname, savedir):
    """ make & save spindle figures for a data segment 

		Params
		------
		n: nrem.NREM object
			with spindles detected & analyzed
		segname: str
			name of data segment
		savedir: str
			directory where all segment analyses are saved

		Returns
		-------
		*lots of figures. To be listed...

    """
    
    # make xlims if data is longer than window length
    xlims, xaxis_windows = get_xlims(n, xaxis_seconds=30)
    # if xlimits returned plot the windows
    if xlims is not None:
        for e, x in enumerate(xlims):
            win_frac = str(e+1) + '/' + str(xaxis_windows)
            fig_detection = slp.plotLFP(n, win_frac=win_frac, xlim=x)
            # save the figure & close
            win_frac_sub = win_frac.replace('/', 'of')
            savename = f'{segname}_SpindleDetection_{win_frac_sub}.png'
            save = os.path.join(savedir, savename)
            fig_detection.savefig(save)
            plt.close(fig_detection)  

    # otherwise don't window
    else:
        win_frac = '1/1'
        fig_detection = slp.plotLFP(n, win_frac=win_frac, xlim=xlims)
        # save the figure & close
        win_frac_sub = win_frac.replace('/', 'of')
        savename = f'{segname}_SpindleDetection_{win_frac_sub}.png'
        save = os.path.join(savedir, savename)
        fig_detection.savefig(save)
        plt.close(fig_detection)  
    
    # spectra for individual spindles (& rejects) for each channel
    fig_psdi_lfp1 = slp.plot_spindlepower_chan_i(n, 'LFP1')
    fig_psdi_rejects_lfp1 = slp.plot_spindlepower_chan_i(n, 'LFP1', spin_type='rejects')
    fig_psdi_lfp2 = slp.plot_spindlepower_chan_i(n, 'LFP2')
    fig_psdi_rejects_lfp2 = slp.plot_spindlepower_chan_i(n, 'LFP2', spin_type='rejects')
    # spectra for spindles concatenated by channel
    fig_psdconcat = slp.plot_spindlepower(n, dB=False)

    # save figs
    fig_dict = {'LFP1_SpindleSpectra.png':fig_psdi_lfp1, 'LFP1_SpindleSpectra_Rejects.png':fig_psdi_rejects_lfp1, 
                'LFP2_SpindleSpectra.png':fig_psdi_lfp2, 'LFP2_SpindleSpectra_Rejects.png':fig_psdi_rejects_lfp2, 'SpindleSpectra_Concat.png':fig_psdconcat}
    for label, fig in fig_dict.items():
        # if figure was created
        if fig:
            savename = f'{segname}_{label}'
            save = os.path.join(savedir, savename)
            fig.savefig(save)
            plt.close(fig)   
    
    # save tracings + psd for individual spindles
    def save_spinfig(n, chan, spin, savename, savedir):
        save = os.path.join(savedir, savename)
        fig = slp.spec_spins(n, chan, spin)
        fig.savefig(save)
        plt.close(fig)

    for chan in n.spindles:
        for spin in n.spindle_psd_i[chan]:
            savename = f'{segname}_{chan}_spindle{spin}_true.png'
            save_spinfig(n, chan, spin, savename, savedir)
        for spin in n.spindle_psd_i_rejects[chan]:
            savename = f'{segname}_{chan}_spindle{spin}_reject.png'
            save_spinfig(n, chan, spin, savename, savedir)



def batch_analyze_lfps(path):
    """ Run batch spindle analysis on LFP data 
		
		Params
		------
		path: str
			path to directory w/ subdirectories containing data segments (path > subdir > seg)
			(ex. 'E:/Jackie/Dropbox (Schiff Lab)/UH3_SpindleAnalysis/P102/CTN/SLEEP')

    """

    # check path
    if not os.path.exists(path):
    	print('Path not found.')
    	return
    
    # get names of all data directories for pt
    #path = f'E:/Jackie/Dropbox (Schiff Lab)/UH3_SpindleAnalysis/{pt_id}/CTN/SLEEP'
    dirs  = [name for name in os.listdir(path) if '.' not in name]
    dirs.sort(key=lambda x: int(x.split(' ')[1]))

    # create directory paths
    dir_paths = [os.path.join(path, seg_dir) for seg_dir in dirs]
    
    # analyze each directory
    for d in dir_paths:
        # get a list of segments in the directory
        seg_list = [os.path.join(d, seg) for seg in os.listdir(d) if '.csv' in seg]

        # for each segment, run spindle analysis
        for seg in seg_list:
            print(f'\n** Analyzing segment {seg}')
            # run spindle detection/analysis
            n, segname, savedir = analyze_spindles(seg)
            # Make & save figures
            make_figures(n, segname, savedir)


def clear_analyses(path):
	""" Clear previous spindle analysis files

		CAREFUL! This deletes all files without '.csv' extension in
		segment subdirectories

		Params
		------
		path: str
			path to directory w/ subdirectories containing data segments (path > subdir > seg)
			(ex. 'E:/Jackie/Dropbox (Schiff Lab)/UH3_SpindleAnalysis/P102/CTN/SLEEP')
	"""

	dirs  = [name for name in os.listdir(path) if '.' not in name]
	dirs.sort(key=lambda x: int(x.split(' ')[1]))

	# create directory paths
	dir_paths = [os.path.join(path, seg_dir) for seg_dir in dirs]

	# delete directories
	for d in dir_paths:
	    rm_dirs = [x for x in os.listdir(d) if '.csv' not in x]
	    rm_paths = [os.path.join(d, rm) for rm in rm_dirs]
	    for r in rm_paths:
	         shutil.rmtree(r)