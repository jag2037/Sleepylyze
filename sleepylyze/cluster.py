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


def calc_kmeans(n, n_clusters, train_split = 30):
    """ calc k-means clustering on spindle spectra

        Params
        ------
        n: nrem.NREM object
            post-spindle detections and analysis
        n_clusters: int
            number of clusters
        train_split: int (default: 30)
            percent of the data to use for training the kmeans model
            
        Returns
        -------
        TimeSeriesScalerMeanVariance(): scaler object fit to training data
        km_psd: sklearn.cluster.KMeans model
        psd: dict of results
    """

    psd_1d, f_idx = fmt_kmeans(n)
    # shuffle the array to prevent grabbing a time-biased set
    np.random.shuffle(psd_1d)

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
    
    return TimeSeriesScalerMeanVariance(), km_psd, f_idx, {'X_train':X_train_psd, 'y_train_pred':y_train_pred, 'ch_score_train':ch_score_train, 'X_test':X_test_psd, 'y_test_pred':y_test_pred, 'ch_score_test':ch_score_test}



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
        # make and fit the model to trianing data
        scaler, km_psd, f_idx, psd = calc_kmeans(n, n_clusters, train_split)
        
        if plot_clusts:
            # plot the kmeans for each cluster
            fig = plot_kmeans(n, km_psd, f_idx, X=psd['X_train'], y_pred=psd['y_train_pred'])
            clust_figs.append(fig)

        # calculate distortions for the scree plot
        cents = np.reshape(km_psd.cluster_centers_, (n_clusters, len(f_idx)))
        x = np.reshape(psd['X_train'], (len(psd['X_train']), len(f_idx)))
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