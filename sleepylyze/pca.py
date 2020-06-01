""" PCA for NREM spindle detections """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn.decomposition import PCA
from cluster import fmt_kmeans


def fmt_pca(n):
	""" format data for PCA 
		
		Parameters
		----------
		n: sleepylyze.nrem.NREM object

		Returns
		-------
		psd_data: 2 dimensional numpy.ndarray (x, y)
			dimensions (# of spindles, length of spindle frequency range)
		f_idx: pandas.core.indexes.numeric.Float64Index
			frequency indices for power spectra in the spindle range

	"""
	psd_1d, f_idx = fmt_kmeans(n)

	# scale the data
	psd_scaled = TimeSeriesScalerMeanVariance().fit_transform(psd_1d)
	# format back to 1d
	psd_data = psd_scaled.reshape(psd_scaled.shape[0], psd_scaled.shape[1])

	return psd_data, f_idx


def calc_pca(in_num, psd_data, n_components):
    """ calc PCA on spindle spectra 
		
		Parameters
		----------
		psd_data: 2 dimensional numpy.ndarray (x, y)
			dimensions (# of spindles, length of spindle frequency range)
		n_components: int
			number of principle components to calculate

		Returns
		-------
		pca: sklearn.decomposition.pca.PCA object
			PCA model
		pca_result: numpy.ndarray
			PCA coefficients for each frequency value
		fig: matplotlib.figure.Figure
			subplots of 1/2/3 principle components and scree plot

    """
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(psd_data)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    # Plot the 1st principal component aginst the 2nd and use the 3rd for color
    # ax[0].scatter(pca_result[:, 0], pca_result[:, 1], c=pca_result[:, 2])
    # ax[0].set_xlabel('1st principal component')
    # ax[0].set_ylabel('2nd principal component')
    # ax[0].set_title('first 3 principal components')

    # plot components w/ variance explained
    ax.plot(pca.explained_variance_, marker='o')
    ax.set_xticks(np.arange(pca.n_components))
    ax.set_title(f'{in_num} Scree Plot')
    ax.set_ylabel('Explained Variance')
    ax.set_xlabel('Principle Component')
    
    return pca, pca_result, fig


def plot_components(in_num, pca, f_idx, n_components):
    """ plot PCA components 

		Parameters
		----------
		n_components: int
			number of components to plot
    """
    
    fig, axs = plt.subplots(n_components, 1, figsize=(3, n_components*2), sharex=True)
    
    # tracings
    for e, (c, ax) in enumerate(zip(pca.components_[:n_components], axs.flatten())):
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

def plot_heatmap(in_num, pca, f_idx):
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

def plot_c1c2(in_num, pca_result):
	""" plot component 1 vs component 2 """
	fig, ax = plt.subplots(figsize=(8, 8))

	ax.scatter(x=pca_result[:,0], y=pca_result[:,1], alpha=0.2)
	ax.set_xlabel('Component 1')
	ax.set_ylabel('Component 2')
	ax.set_title(f'{in_num}')

	return fig