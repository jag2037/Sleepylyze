""" Module for clustering of individual spindle power spectra """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from scipy.spatial.distance import cdist
from scipy.stats import ttest_ind, levene, bartlett
from sklearn.metrics import calinski_harabasz_score
from sklearn.cluster import KMeans
from mne.connectivity import phase_slope_index, spectral_connectivity


def fmt_kmeans(n):
    """ reformat spidnle data for kmeans clustering 
    
        Params
        ------
        n: nrem.NREM object
            post-spindle detections and analysis
        
        Returns
        -------
        psd_1d: 1-dimensional np.array
            spindle spectra
        f_idx: np.array
            spindle frequency indices
    """
    
    spin_range = n.metadata['spindle_analysis']['spin_range']
    
    # specify the data
    exclude = ['EOG_L', 'EOG_R', 'EKG', 'REF', 'FPZorEKG']
    chan_arr_list = []
    chan_names = []
    for chan in n.spindle_psd_i.keys():
        if chan not in exclude:
            # set channel spindle data
            spin_psd_chan = n.spindle_psd_i[chan]
            first_spin_idx = list(spin_psd_chan.keys())[0]
            spin_idxs = (spin_psd_chan[first_spin_idx].index >= spin_range[0]) & (spin_psd_chan[first_spin_idx].index <= spin_range[1])
            # this fails if a spindle is longer than zpad_len (bc then not all spindles are the same length)
            chan_arr = np.array([spin_psd_chan[x][spin_idxs].values for x in spin_psd_chan])
            chan_arr_list.append(chan_arr)
            chan_names.append(chan)

    # stack all of the channels into a single array
    psd_arr = np.vstack(chan_arr_list)
    # get frequency index from first spindle
    first_chan = chan_names[0]
    first_spin = list(n.spindle_psd_i[first_chan].keys())[0]
    first_psd = n.spindle_psd_i[first_chan][first_spin]
    f_idx = first_psd[(first_psd.index >= spin_range[0]) & (first_psd.index <= spin_range[1])].index
    
    # reshape the data to a 1-dimensional array
    psd_1d = np.reshape(psd_arr, (psd_arr.shape[0], psd_arr.shape[1], 1))
    
    return psd_1d, f_idx


def calc_kmeans(n, n_clusters, train_split = None):
    """ calc k-means clustering on spindle spectra

        Params
        ------
        n: nrem.NREM object
            post-spindle detections and analysis
        n_clusters: int
            number of clusters
        train_split: int or None (default: None)
            percent of the data to use for training the kmeans model.
            If none, use all of the data
            
        Returns
        -------
        TimeSeriesScalerMeanVariance(): scaler object fit to training data
        km_psd: sklearn.cluster.KMeans model
        psd: dict of results
    """

    psd_1d, f_idx = fmt_kmeans(n)
    # shuffle the array to prevent grabbing a time-biased set
    np.random.shuffle(psd_1d)

    if train_split is not None:
        # specify the training data
        train_len = int(psd_1d.shape[0]*(train_split/100))
        X_train_psd = TimeSeriesScalerMeanVariance().fit_transform(psd_1d[:train_len])
        X_test_psd = TimeSeriesScalerMeanVariance().transform(psd_1d[train_len:])
        
        # create the model and predict
        ### sklearn kMeans (first convert to 2d)
        X_test_2d = np.array([x.ravel() for x in X_test_psd])
        X_train_2d = np.array([x.ravel() for x in X_train_psd])
        km_psd = KMeans(n_clusters, random_state=0).fit(X_train_2d)
        y_train_pred = km_psd.predict(X_train_2d)
        y_test_pred = km_psd.predict(X_test_2d)
        
        if n_clusters > 1:
            # calculate calinski_harabasz_score(X, labels)
            ch_score_train = calinski_harabasz_score(X_train_2d, y_train_pred)
            ch_score_test = calinski_harabasz_score(X_test_2d, y_test_pred)
        else:
            ch_score_train, ch_score_test = None, None
        result = {'X_train':X_train_psd, 'y_train_pred':y_train_pred, 'ch_score_train':ch_score_train, 'X_test':X_test_psd, 'y_test_pred':y_test_pred, 'ch_score_test':ch_score_test}

    elif train_split is None:
        # use all of the data
        X_psd = TimeSeriesScalerMeanVariance().fit_transform(psd_1d)
        # create the model and predict
        ### sklearn kMeans (first convert to 2d)
        X_2d = np.array([x.ravel() for x in X_psd])
        km_psd = KMeans(n_clusters, random_state=0).fit(X_2d)
        y_pred = km_psd.predict(X_2d)

        # set test-train comparison scores to None
        ch_score_train, ch_score_test = None, None

        result = {'X':X_psd, 'y_pred':y_pred}

    
    return TimeSeriesScalerMeanVariance(), km_psd, f_idx, result



def plot_kmeans(n, km_psd, f_idx, X, y_pred, raw=False, std=True):
    """ plot k-means clusters """
    
    spin_range = n.metadata['spindle_analysis']['spin_range']
    n_clusters = km_psd.cluster_centers_.shape[0]
    centroids = km_psd.cluster_centers_
    
    if n_clusters == 2:
        fig, axs = plt.subplots(1, n_clusters, figsize=(8, 7), sharey=True)
        fig.subplots_adjust(hspace=0.1)
    else:
        fig, axs = plt.subplots(1, n_clusters, figsize=(2*n_clusters, 3), sharey=True)
    plt.rcParams["font.family"] = "Arial"

    if n_clusters == 1:
        axs = [axs]
    else: 
        axs = axs.flatten()
    for e, (label, ax) in enumerate(zip(range(n_clusters), axs)):
        if raw:
            # plot raw tracings
            for psd in X[y_pred == label]:
                 ax.plot(pd.Series(psd.ravel(), f_idx), alpha=0.4, color='grey')

        if std:
            # plot standard deviation of raw tracings
            clust = {e:x.ravel() for e, x in enumerate(X[y_pred == label])}
            clust_df = pd.DataFrame(clust, index=f_idx).T
            ax.fill_between(f_idx, y1=(clust_df.mean() - clust_df.std()).values, y2=(clust_df.mean() + clust_df.std()).values, color='grey', alpha=0.3)
        # plot centroids
        ax.plot(pd.Series(centroids[label], f_idx), color='black', lw=3) #"r-")
        # set axis params
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.set_ylim(-3, 4)
        ax.set_xlim(spin_range[0], spin_range[1])
        ax.set_title(f'Cluster {label}', size=30)
        ax.set_xlabel('Frequency (Hz)', size=30)
        ax.tick_params(labelsize=20)

        if e == 0:
            ax.set_ylabel('Normalized Power', size=30)

    #fig.suptitle(f'KMeans, clusters = {n_clusters}')

    return fig


def pick_clusters(n, clusters, train_split, plot_clusts=True, plot_scree=True):
    """ Make scree plot for different cluster numbers 
    
        Params
        ------
        n: nrem.NREM object
        clusters: int
            length of clusters to loop through
        train_split: int
            percentage of data to use for training
        plot_clusts: bool (default: True)
            whether to plot clusters
        plot_scree: bool (default: True)
            whether to plot scree plot
    """
    distortions = []
    clust_figs = []
    for n_clusters in range(1, clusters):
        # make and fit the model to training data
        scaler, km_psd, f_idx, psd = calc_kmeans(n, n_clusters, train_split)

        # set x and y variables
        if train_split is not None:
            X = psd['X_train']
            y_pred = psd['y_train_pred']    
        elif train_split is None:
            X = psd['X']
            y_pred = psd['y_pred']
        
        if plot_clusts:
            # plot the kmeans for each cluster
            fig = plot_kmeans(n, km_psd, f_idx, X=X, y_pred=y_pred)
            clust_figs.append(fig)

        # calculate distortions for the scree plot
        cents = np.reshape(km_psd.cluster_centers_, (n_clusters, len(f_idx)))
        x = np.reshape(X, (len(X), len(f_idx)))
        distortions.append(sum(np.min(cdist(x, cents, 'euclidean'), axis=1)) / x.shape[0])


    if plot_scree:
        ## plot the scree plot
        scree_fig, ax = plt.subplots()
        dists = pd.Series(distortions, index=range(1, clusters))
        ax.plot(dists, marker='o')
        ax.set_title('KMeans Cluster Dispersion')
        ax.set_xlabel('# Clusters')
        ax.set_ylabel('Distortion (SSE)')

    return clust_figs, scree_fig


def run_kmeans(n, n_clusters, train_split, plot_clusts=True):
    """ run kmeans clustering 
        
        Params
        ------
        n: nrem.NREM object
        n_clusters: int
            number of clusters to use
        train_split: int or None (default: None)
            percent of the data to use for training the kmeans model.
            If none, use all of the data
        plot_clusts: bool (default: True)
            whether to return a figure of cluster centers

    """
    
    print('Fitting the kmeans model...')
    scaler, km, f_idx, result = calc_kmeans(n, n_clusters, train_split)
    if plot_clusts:
        # set x and y variables
        if train_split is not None:
            X = result['X_train']
            y_pred = result['y_train_pred']    
        elif train_split is None:
            X = result['X']
            y_pred = result['y_pred']
        # plot the kmeans for each cluster
        fig = plot_kmeans(n, km, f_idx, X=X, y_pred=y_pred)

    print('Formatting data...')
    # scale the data & reformat
    psd_1d, f_idx = fmt_kmeans(n)
    psd_1d_scaled = scaler.transform(psd_1d)
    psd_2d = np.array([x.ravel() for x in psd_1d_scaled])
    print('Assigning labels...')
    # predict labels
    labels = km.predict(psd_2d)
    # add labels to the stats df
    n.spindle_stats_i['cluster'] = labels
    print('Labels assigned to cluster column in n.spindle_stats_i.\nDone.')

    return fig


def calc_spin_clust_means(n, savedir, export=True):
    """ calculate cluster means 
    
        Parameters
        ----------
        savedir: str
            location to save dataframes
        export: bool (default: True)
            whether to export dataframes
        
        Returns
        -------
        n.spindle_aggregates_clust
        n.spindle_clust_means
    """
    
    print('Aligning spindles...')
    # align spindles accoridng to timedelta & combine into single dataframe
    spindle_aggregates_clust = {}
    datatypes = ['Raw', 'spfilt']
    
    for clust in [0, 1]:
        # get spindle assignments
        clust_df = n.spindle_stats_i[n.spindle_stats_i.cluster == clust].loc[:, ['chan', 'spin']]
        #for e, (chan, spin) in clust_df.iterrows():
        spindle_aggregates_clust[clust] = {}
        for datatype in datatypes:
            prefix = list(clust_df.chan)
            suffix = list(clust_df.spin)
            spin_suffix = ['spin_'+ str(x) for x in suffix]
            dfs = [n.spindle_aggregates[chan][datatype][spin].rename(chan+'_'+spin) for chan, spin in zip(prefix, spin_suffix)]
            # create a new df from the series'
            df_combined = pd.DataFrame(dfs).T
            spindle_aggregates_clust[clust][datatype] = df_combined
    
    print('Calculating spindle cluster statistics...')
    # create a new multiindex dataframe for calculations
    spindle_clust_means = {}
    calcs = ['count', 'mean', 'std' ,'sem']
    tuples = [(clust, calc) for clust in spindle_aggregates_clust.keys() for calc in calcs]
    columns = pd.MultiIndex.from_tuples(tuples, names=['cluster', 'calc'])
    for datatype in datatypes:
        spindle_clust_means[datatype] = pd.DataFrame(columns=columns)
        # fill the dataframe
        for clust in spindle_aggregates_clust.keys():
            spindle_clust_means[datatype][(clust, 'count')] = spindle_aggregates_clust[clust][datatype].notna().sum(axis=1)
            spindle_clust_means[datatype][(clust, 'mean')] = spindle_aggregates_clust[clust][datatype].mean(axis=1)
            spindle_clust_means[datatype][(clust, 'std')] = spindle_aggregates_clust[clust][datatype].std(axis=1)
            spindle_clust_means[datatype][(clust, 'sem')] = spindle_aggregates_clust[clust][datatype].sem(axis=1)
        
    n.spindle_aggregates_clust = spindle_aggregates_clust
    n.spindle_clust_means = spindle_clust_means
    
    if export:
        print('Exporting dataframes...')
        fname = n.metadata['file_info']['fname'].split('.')[0]
        filename = f'{fname}_spindle_aggregates_clust.xlsx'
        savename = os.path.join(savedir, 'spindle_tracings', filename)
        writer = pd.ExcelWriter(savename, engine='xlsxwriter')
        for clust in n.spindle_aggregates_clust.keys():
            for dtype in n.spindle_aggregates_clust[clust].keys():
                tab = '_'.join([str(clust), dtype])
                n.spindle_aggregates_clust[clust][dtype].to_excel(writer, sheet_name=tab)
        writer.save()

        # export spindle means
        print('Exporting spindle means...\n')
        for dtype in n.spindle_clust_means.keys():
            filename = f'{fname}_spindle_means_clust_{dtype}.csv'
            savename = os.path.join(savedir, 'spindle_tracings', filename)
            n.spindle_clust_means[dtype].to_csv(savename)

    print('Done. Spindles aggregated by cluster in obj.spindle_aggregates_clust dict. Spindle statisics stored in obj.spindle_clust_means dataframe.\n')

def zpad(spin, sf, zpad_len, zpad_mult):
    """ zero-pad individual raw spindle data. for use with cluster.calc_cohpsi. 
        
        Parameters
        ----------
        spin: np.array
            spindle EEG mV values
        sf: float
            EEG sampling frequency
        zpad_len: float or None
            length in seconds to zero-pad the spindle to
        zpad_mult: float
            multiple of spindle length to zpad out to (or this OR zpad_len)
        
    """
    # subtract mean to zero-center spindle for zero-padding
    data = spin - np.mean(spin)
    zpad_samples=0
    zpad_seconds=0
    tx=0

    if (zpad_len is not None) & (zpad_mult is not None):
        print('Only zpad_len or zpad_mult can be used. Please pick one. Abort.')
        return 
    
    if zpad_len is not None:
        # zeropad the spindle
        total_len = zpad_len*sf
        zpad_samples = total_len - len(data)
        zpad_seconds = zpad_samples/sf
        if zpad_samples > 0:
            padding = np.repeat(0, zpad_samples)
            data_pad = np.append(data, padding)
        else:
            spin_len = len(data)/sf
            data_pad = data
    elif zpad_mult is not None:
        zpad_samples = len(data)*zpad_mult
        zpad_seconds = zpad_samples/sf
        padding = np.repeat(0, zpad_samples)
        data_pad = np.append(data, padding)
    else:
        data_pad = data

    return data_pad


def calc_cohpsi(n, zpad_len=None, zpad_mult=None, bw=1.5, adapt=True):
    """ Calculate coherence and phase slope index between F3-P3 and F4-P4 for clusters 0 and 1 
        Notes: 
            - F3, F4, P3, and P4 should be laplacian filtered
            - zpad_len typically needs to be longer than in nrem module bc we're looking @ union of overlapping
                spindle events


        Parameters
        ----------
        zpand_len: int or float (default: 5)
            length to zpad spindle events to
        zpad_mult: float
            multiple of spindle length to zpad out to (or this OR zpad_len)
        bw: float (default: 1.5)
            multitaper bandwidth
        adapt: bool (default: True)
            whether to adaptively combine tapers for multitaper

        Returns
        -------
        coh_params: dict
            parameters used to calculate values
        coh_df: pd.DataFrame
            dataframe of calculated values for each spindle event
    """

    # subset spindle # by channel and cluster
    # left
    f3_c0 = n.spindle_stats_i[(n.spindle_stats_i.chan=='F3') & (n.spindle_stats_i.cluster == 0)]
    f3_c1 = n.spindle_stats_i[(n.spindle_stats_i.chan=='F3') & (n.spindle_stats_i.cluster == 1)]
    p3_c0 = n.spindle_stats_i[(n.spindle_stats_i.chan=='P3') & (n.spindle_stats_i.cluster == 0)]
    p3_c1 = n.spindle_stats_i[(n.spindle_stats_i.chan=='P3') & (n.spindle_stats_i.cluster == 1)]

    # right
    f4_c0 = n.spindle_stats_i[(n.spindle_stats_i.chan=='F4') & (n.spindle_stats_i.cluster == 0)]
    f4_c1 = n.spindle_stats_i[(n.spindle_stats_i.chan=='F4') & (n.spindle_stats_i.cluster == 1)]
    p4_c0 = n.spindle_stats_i[(n.spindle_stats_i.chan=='P4') & (n.spindle_stats_i.cluster == 0)]
    p4_c1 = n.spindle_stats_i[(n.spindle_stats_i.chan=='P4') & (n.spindle_stats_i.cluster == 1)]

    # pull the spindle data corresponding to the indiviudal spindles from each cluster
    #left
    f3_c0_spinevents = {k:v for k, v in n.spindles['F3'].items() if k in f3_c0.spin.values}
    f3_c1_spinevents = {k:v for k, v in n.spindles['F3'].items() if k in f3_c1.spin.values}
    p3_c0_spinevents = {k:v for k, v in n.spindles['P3'].items() if k in p3_c0.spin.values}
    p3_c1_spinevents = {k:v for k, v in n.spindles['P3'].items() if k in p3_c1.spin.values}

    # right
    f4_c0_spinevents = {k:v for k, v in n.spindles['F4'].items() if k in f4_c0.spin.values}
    f4_c1_spinevents = {k:v for k, v in n.spindles['F4'].items() if k in f4_c1.spin.values}
    p4_c0_spinevents = {k:v for k, v in n.spindles['P4'].items() if k in p4_c0.spin.values}
    p4_c1_spinevents = {k:v for k, v in n.spindles['P4'].items() if k in p4_c1.spin.values}

    # create lists to zip and loop through
    spin_events_f = [f3_c0_spinevents, f3_c1_spinevents, f4_c0_spinevents, f4_c1_spinevents]
    spin_events_p = [p3_c0_spinevents, p3_c1_spinevents, p4_c0_spinevents, p4_c1_spinevents]
    spin_maps = dict([('lc0', {}), ('lc1', {}), ('rc0', {}), ('rc1', {})])
    fp_spin_events = {'f3p3_c0':{}, 'f3p3_c1':{}, 'f4p4_c0':{}, 'f4p4_c1':{}}

    # create a map of indices between frontal and parietal overlapping spindles
    for spin_map, spin_event_f, spin_event_p in zip(spin_maps.values(), spin_events_f, spin_events_p):
        # for each frontal spindle, check for overlap with each parietal spindle
        for f_spin, f_dat in spin_event_f.items():
            # pull timestamps for the f spindle
            f_time = f_dat.time.values
            p_overlap = []
            # check the timestamps against all p spindles
            for p_spin, p_dat in spin_event_p.items():
                p_time = p_dat.time.values
                # if there are any common timestamps
                if any(t in p_time for t in f_time):
                    # append the spindle index
                    p_overlap.append(p_spin)
            spin_map[f_spin] = p_overlap

    # get the union of timestamps for overlapping spindles
    for (fp_label, fp_spins), (sp_label, spin_map), spin_event_f, spin_event_p in zip(fp_spin_events.items(), spin_maps.items(), spin_events_f, spin_events_p):
        for f_spin, p_spin_list in spin_map.items():
            # get timestamps for the frontal spindle
            f_timestamps = spin_event_f[f_spin].time.values
            # get a list of timestamps for the parietal spindle (in case of >1 overlap)
            p_timestamps_list = [spin_event_p[p].time.values for p in p_spin_list]
            # flatten the list of p_timestamps
            p_timestamps = np.array([t for sublist in p_timestamps_list for t in sublist])
            # merge the union of the timestamps
            f_df = pd.DataFrame(f_timestamps, columns=['timestamp'])
            p_df = pd.DataFrame(p_timestamps, columns=['timestamp'])
            fp_df = pd.merge(f_df, p_df, how='outer')
            fp_timestamps = sorted(fp_df.timestamp.values)
            # use f_spin as key to preserve relationship between other spindle data
            fp_spins[f_spin] = fp_timestamps 

    ## run coherence and PSI for each spindle

    # set data, min/max frequencies of interest to min/max spindle range
    fmin, fmax = n.metadata['spindle_analysis']['spin_range']
    data = n.data

    psi_dicts = {'f3p3_c0':{}, 'f3p3_c1':{}, 'f4p4_c0':{}, 'f4p4_c1':{}}
    data_chans = [('F3', 'P3'), ('F3', 'P3'), ('F4', 'P4'), ('F4', 'P4')]
    
    # create columns and empty rows list for coh_df
    cols=['label', 'fspin_event', 'chans', 'f_chan', 'p_chan', 'cluster', 'clust_text', 'coherence', 'coh_freqs', 'psi', 'plv', 'plv_freqs', 'pli', 'pli_freqs']
    rows = []

    # create dict to save out events
    f_psi_data = {'f3':{}, 'f4':{}}
    p_psi_data = {'p3':{}, 'p4':{}}

    # run calculations for each spindle event
    sf = n.s_freq
    for chans, (events_label, events_dict), (psi_label, psi_dict) in zip(data_chans, fp_spin_events.items(), psi_dicts.items()):
        for spin, ts in events_dict.items():
            # set channel names
            f_chan = psi_label[:2]
            p_chan = psi_label[2:4]
            # pull and zero-pad the data
            f_dat = data[(chans[0], 'Raw')].loc[ts].values
            f_dat_zpad = zpad(f_dat, sf, zpad_len=zpad_len, zpad_mult=zpad_mult)
            p_dat = data[(chans[1], 'Raw')].loc[ts].values
            p_dat_zpad = zpad(p_dat, sf, zpad_len=zpad_len, zpad_mult=zpad_mult)
            # save to dicts
            f_psi_data[f_chan][spin] = f_dat_zpad
            p_psi_data[p_chan][spin] = p_dat_zpad

            # reformat for calculations
            spin_dat = np.array([f_dat_zpad, p_dat_zpad])
            psi_dat = spin_dat.reshape(1, spin_dat.shape[0], spin_dat.shape[1])
            # calc psi
            try:
                psi_arr, psi_freqs, times, n_epochs, n_tapers = phase_slope_index(psi_dat, sfreq=n.s_freq, 
                                                                              mt_bandwidth=bw, mt_adaptive=adapt, fmin=fmin, fmax=fmax)
                psi = psi_arr[1][0]
            except ValueError:
                psi = None
                pass
            # calc coherence
            try:
                coh_arr, coh_freqs, times, n_epochs, n_tapers = spectral_connectivity(psi_dat, method='coh', sfreq=n.s_freq, 
                                                                                  mt_bandwidth=bw, mt_adaptive=adapt, fmin=fmin, fmax=fmax)
                coh = coh_arr[1][0]
            except ValueError:
                coh, coh_freqs = None, None
                pass
            # calc phase lag index
            try:
                pli_arr, pli_freqs, times, n_epochs, n_tapers = spectral_connectivity(psi_dat, method='pli', sfreq=n.s_freq, 
                                                                                  mt_bandwidth=bw, mt_adaptive=adapt, fmin=fmin, fmax=fmax)
                pli = pli_arr[1][0]
            except ValueError:
                pli = None
                pass
            # calc phase locking value
            try:
                plv_arr, plv_freqs, times, n_epochs, n_tapers = spectral_connectivity(psi_dat, method='plv', sfreq=n.s_freq, 
                                                                                  mt_bandwidth=bw, mt_adaptive=adapt, fmin=fmin, fmax=fmax)
                plv = plv_arr[1][0]
            except ValueError:
                plv = None
                pass
            
            # create the row & append to rows list
            c = psi_label.split('_')[0]
            clust = psi_label.split('_')[1][1]
            if int(clust) == 0:
                clust_txt = 'zero'
            elif int(clust) == 1:
                clust_txt = 'one'
            vals = [psi_label, spin, c, f_chan, p_chan, clust, clust_txt, coh, coh_freqs, psi, plv, plv_freqs, pli, pli_freqs]
            row = {c:v for c, v in zip(cols, vals)}
            rows.append(row)

    # convert rows into dataframe
    coh_df = pd.DataFrame(rows)

    # save params
    coh_params = {'data_chans': data_chans, 'zpad_len': zpad_len, 'bw':bw, 'adapt':adapt}

    return coh_params, coh_df, f_psi_data, p_psi_data


def psi_stats(coh_df, equal_var):
    """ run equal variance and t-tests on psi by cluster """
    stats = {}
    clusters = ['0', '1']
    clust_psi = {}

    for c in clusters:
        # pull psi values and remove Nones
        psi = np.array([x[0] for x in coh_df[coh_df.cluster == c].psi if x is not None])
        clust_psi[c] = psi
        # get desciptive statistics
        mean = psi.mean()
        std = psi.std()
        stats[c] = {'mean': mean, 'std':std}
    
    # test for unequal variance
    # levene = not normal distributions; bartlett = normal distributions
    lev = levene(clust_psi['0'], clust_psi['1'])
    bart = bartlett(clust_psi['0'], clust_psi['1'])
    stats['levene_notnormdist'] = lev
    stats['bartlett_normdist'] = bart
    
    stats['t_test'] = {'equal_var':equal_var}
    ttest = ttest_ind(clust_psi['0'], clust_psi['1'], equal_var = True)
    stats['t_test']['result'] = ttest
    
    return stats


def plot_coh(coh_df):
    """ Plot mean coherence for each label in coh_df (f3-p3_c0, f3-p3_c1, f4-p4_c0, f4-p4_c1) 
        Note: This will throw an error if all spindles are not the same length (e.g. no zero-padding)
    """
    # mean coherence
    fig, axs = plt.subplots(4, 1, figsize =(8, 12))

    labels = set(coh_df.label.values)
    for label, ax in zip(labels, axs.flatten()):
        data = coh_df[coh_df.label == label].coherence
        coh_mean = data.mean()
        coh_std = np.array(data).std()
        
        coh_freqs = coh_df[coh_df.label == label].coh_freqs.iloc[0]
        ser = pd.Series(coh_mean, coh_freqs)
        ax.plot(ser)
        ax.fill_between(x=coh_freqs, y1=(coh_mean-coh_std), y2=(coh_mean+coh_std), alpha= 0.2)
        ax.set_title(label)

    fig.tight_layout()
    
    return fig

def plot_psi(coh_df, group='label'):
    """ Plot phase slope index

        Parameters
        ----------
        coh_df: pd.DataFarme
            df of coherence/psi values
        group: str (default: 'label')
            how to group the data [options: 'label', 'clust_text', 'chans']
    """

    fig, ax = plt.subplots()

    df_notna = coh_df[coh_df.psi.notna()]
    ax = sns.swarmplot(y=group, x='psi', data=df_notna)
    ax.axvline(x=0, color='black', linestyle=':')

    return fig


### histograms ###

def make_quadrants(n):
    """ make left-right/anterior-posterior quadrants for plotting """
    stats = n.spindle_stats_i
    
    # quads overall
    al = stats[(stats.AP == 'A') & (stats.RL == 'L')]
    ar = stats[(stats.AP == 'A') & (stats.RL == 'R')]
    pl = stats[(stats.AP == 'P') & (stats.RL == 'L')]
    pr = stats[(stats.AP == 'P') & (stats.RL == 'R')]
    
    quads = {'al':al, 'ar':ar, 'pl':pl, 'pr':pr}
    
    # make quads by cluster
    al_c0 = al[al['cluster']==0]
    al_c1 = al[al['cluster']==1]
    ar_c0 = ar[ar['cluster']==0]
    ar_c1 = ar[ar['cluster']==1]

    pl_c0 = pl[pl['cluster']==0]
    pl_c1 = pl[pl['cluster']==1]
    pr_c0 = pr[pr['cluster']==0]
    pr_c1 = pr[pr['cluster']==1]
    
    quads_c0 = {'al_c0':al_c0, 'ar_c0':ar_c0, 'pl_c0':pl_c0, 'pr_c0':pr_c0}
    quads_c1 = {'al_c1':al_c1, 'ar_c1':ar_c1, 'pl_c1':pl_c1, 'pr_c1':pr_c1}
    
    return quads, (quads_c0, quads_c1)


def plot_dist(in_num, quads):
    """ plot cluster distribution """
    
    fig, axs = plt.subplots(2, 2, sharey=True)
    fig.subplots_adjust(hspace=0.5)

    # get cluter assignments for each spindle
    quads_c = [q.cluster for q in quads.values()]
    for ax, quad in zip(axs.flatten(), quads_c):
        ax.hist(quad[quad==0], bins=[0,1,2], align='left', alpha=0.5, label = 'Cluster 0')
        ax.hist(quad[quad==1], bins=[0,1,2], align='left', alpha=0.5, label = 'Cluster 1')
        clust_ratio = np.round((quad==0).sum()/len(quad), 2)*100
        ax.set_title(f'{clust_ratio}% cluster 0')
        ax.set_xlabel('Cluster')
        ax.set_ylabel('spindle count')
    fig.suptitle(f'{in_num} Cluster Distribution')
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    
    return fig


def plot_peakfreq(in_num, quads_clust):
    """ Plot peak frequency distribution by cluster """
    fig, axs = plt.subplots(2, 2, sharey=True)
    fig.subplots_adjust(hspace=0.5)

    quads_c0 = [q.dominant_freq_Hz for q in quads_clust[0].values()]
    quads_c1 = [q.dominant_freq_Hz for q in quads_clust[1].values()]
    for ax, quad_c0, quad_c1 in zip(axs.flatten(), quads_c0, quads_c1):
        ax.hist(quad_c0, align='left', alpha=0.5, label= 'Cluster 0')
        ax.hist(quad_c1, align='left', alpha=0.5, label = 'Cluster 1')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('spindle count')

    fig.suptitle(f'{in_num} Dominant Spectral Peak Frequency')
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    
    return fig


def plot_totalpeaks(in_num, quads_clust):
    """ Plot # of total peaks by cluster """
    fig, axs = plt.subplots(2, 2, sharey=True)
    fig.subplots_adjust(hspace=0.5)

    quads_c0 = [q.total_peaks for q in quads_clust[0].values()]
    quads_c1 = [q.total_peaks for q in quads_clust[1].values()]
    for ax, quad_c0, quad_c1 in zip(axs.flatten(), quads_c0, quads_c1):
        ax.hist(quad_c0, align='left', bins=[0,1,2,3,4,5], alpha=0.5, label= 'Cluster 0')
        ax.hist(quad_c1, align='left', bins=[0,1,2,3,4,5], alpha=0.5, label = 'Cluster 1')
        ax.set_xlabel('Peaks')
        ax.set_ylabel('spindle count')

    fig.suptitle('# of Spectral Peaks')
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    
    return fig

def plot_duration(in_num, quads_clust):
    """ Plot spindle duration by cluster """
    fig, axs = plt.subplots(2, 2, sharey=True)
    fig.subplots_adjust(hspace=0.5)

    quads_c0 = [q.dur_ms for q in quads_clust[0].values()]
    quads_c1 = [q.dur_ms for q in quads_clust[1].values()]
    for ax, quad_c0, quad_c1 in zip(axs.flatten(), quads_c0, quads_c1):
        ax.hist(quad_c0, align='left', alpha=0.5, label= 'Cluster 0')
        ax.hist(quad_c1, align='left', alpha=0.5, label = 'Cluster 1')
        ax.set_xlabel('ms')
        ax.set_ylabel('spindle count')
    fig.suptitle(f'{in_num} Spindle Duration')
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    
    return fig

def plot_peakratio(in_num, quads_clust):
    """ Plot second peak pwr as fraction of dominant peak power by cluster """
    fig, axs = plt.subplots(2, 2, sharey=True)
    fig.subplots_adjust(hspace=0.5)

    quads_c0 = [q.peak2_ratio for q in quads_clust[0].values()]
    quads_c1 = [q.peak2_ratio for q in quads_clust[1].values()]
    for ax, quad_c0, quad_c1 in zip(axs.flatten(), quads_c0, quads_c1):
        ax.hist(quad_c0, align='left', alpha=0.5, label= 'Cluster 0')
        ax.hist(quad_c1, align='left', alpha=0.5, label = 'Cluster 1')
        ax.set_xlabel('Power Ratio\n(% of primary peak)')
        ax.set_ylabel('spindle count')
    fig.suptitle(f'{in_num} secondary peak power ')
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    
    return fig