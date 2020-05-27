""" PCA for NREM spindle detections """


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn.decomposition import PCA
from .cluster import fmt_kmeans


def fmt_pca(n):
	""" format data for PCA """
	psd_1d, f_idx = fmt_kmeans(n)

	# scale the data
	psd_scaled = TimeSeriesScalerMeanVariance().fit_transform(psd_1d)
	# format back to 1d
	psd_data = psd_data.reshape(psd_data.shape[0], psd_data.shape[1])

	return psd_data, f_idx


def calc_pca(psd_data, n_components):
    """ calc PCA on spindle spectra """
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(psd_data)

    # Plot the 1st principal component aginst the 2nd and use the 3rd for color
    fig, ax = plt.subplots(1, 2, figsize=(6, 4))
    ax[0].scatter(pca_result[:, 0], pca_result[:, 1], c=pca_result[:, 2])
    ax[0].set_xlabel('1st principal component')
    ax[0].set_ylabel('2nd principal component')
    ax[0].set_title('first 3 principal components')

    # plot components w/ variance explained
    ax[1].plot(pca.explained_variance_, marker='o')
    ax[1].set_xticks(np.arange(pca.n_components))
    ax[1].set_title('Scree Plot')
    ax[1].set_ylabel('Explained Variance')
    ax[1].set_xlabel('Principle Component')
    
    return pca, pca_result, fig


def plot_components(in_num, pca, f_idx):
    """ plot PCA components """
    
    fig, axs = plt.subplots(pca.n_components_, 1, figsize=(3, 8), sharex=True)
    
    # tracings
    for e, (c, ax) in enumerate(zip(pca.components_, axs.flatten())):
        ax.plot(f_idx, c)
        ax.set_ylabel(e)
        ax.set_xlim(f_idx[0], f_idx[-1])
    plt.xlabel('Frequency (Hz)', ha='center')
    #plt.ylabel('Principle Component', rotation='vertical')
    #fig.text(0.5, 0, 'Frequency (Hz)', ha='center')
    fig.text(0, 0.5, 'Principle Component', va='center', rotation='vertical')
    #plt.suptitle(f'{in_num} Principle Components')
    fig.tight_layout()
    
    return fig

def pca_heatmap(in_num, pca, f_idx):
    """ plot heatmap of components """
    
    fig, ax = plt.subplots()
    im = ax.matshow(pca.components_,cmap='viridis')
    cbar = ax.figure.colorbar(im, ax=ax, orientation='horizontal')
    # set xtick labels based on length of x-ticks
    a = ax.get_xticks().tolist()
    xticks = [round(f_idx[e],2) for e, idx in enumerate(f_idx) if e%(len(a)-2) == 0]
    xticks.insert(0, '')
    ax.set_xticklabels(xticks)
    
    ax.set_ylabel('Principle Component')
    ax.set_xlabel('Frequency (Hz)')
    fig.suptitle(f'{in_num} Principle Components')
    
    return fig