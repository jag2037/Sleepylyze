""" Module for clustering of individual spindle power spectra """

import numpy as np
import pandas as pd
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from scipy.spatial.distance import cdist
from sklearn.metrics import calinski_harabasz_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


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
    exclude = ['EOG_L', 'EOG_R', 'EKG']
    chan_arr_list = []
    for chan in n.spindle_psd_i.keys():
        if chan not in exclude:
            # set channel spindle data
            spin_psd_chan = n.spindle_psd_i[chan]
            first_spin_idx = list(spin_psd_chan.keys())[0]
            spin_idxs = (spin_psd_chan[first_spin_idx].index >= spin_range[0]) & (spin_psd_chan[first_spin_idx].index <= spin_range[1])
            # this fails if a spindle is longer than zpad_len (bc then not all spindles are the same length)
            chan_arr = np.array([spin_psd_chan[x][spin_idxs].values for x in spin_psd_chan])
            chan_arr_list.append(chan_arr)

    # stack all of the channels into a single array
    psd_arr = np.vstack(chan_arr_list)
    # get frequency index from first spindle
    first_chan = list(n.spindle_psd_i.keys())[0]
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
    
    fig, axs = plt.subplots(1, n_clusters, figsize=(2*n_clusters, 3), sharey=True)
    if n_clusters == 1:
        axs = [axs]
    else: 
        axs = axs.flatten()
    for label, ax in zip(range(n_clusters), axs):
        if raw:
            # plot raw tracings
            for psd in X[y_pred == label]:
                 ax.plot(pd.Series(psd.ravel(), f_idx), alpha=0.4, color='grey')

        if std:
            # plot standard deviation of raw tracings
            clust = {e:x.ravel() for e, x in enumerate(X[y_pred == label])}
            clust_df = pd.DataFrame(clust, index=f_idx).T
            ax.fill_between(f_idx, y1=(clust_df.mean() - clust_df.std()).values, y2=(clust_df.mean() + clust_df.std()).values, color='grey', alpha=0.5)
        # plot centroids
        ax.plot(pd.Series(centroids[label], f_idx), "r-")
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.set_ylim(-3, 4)
        ax.set_xlim(spin_range[0], spin_range[1])
        ax.set_title(f'Cluster {label}')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Normalized Power')

    fig.suptitle(f'KMeans, clusters = {n_clusters}')

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