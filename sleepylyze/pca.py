""" PCA for NREM spindle detections """

import numpy as np
import pandas as pd
import matplotlib
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


def calc_pca(n, n_components):
    """ calc PCA on spindle spectra 
		
		Parameters
		----------
		n: sleepylyze.nrem.NREM object
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
    in_num = n.metadata['file_info']['in_num']
    psd_data, f_idx = fmt_pca(n)

    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(psd_data)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    # plot components w/ variance explained
    ax.plot(pca.explained_variance_, marker='o')
    ax.set_xticks(np.arange(pca.n_components))
    ax.set_title(f'{in_num} Scree Plot')
    ax.set_ylabel('Explained Variance')
    ax.set_xlabel('Principle Component')
    
    return pca, f_idx, pca_result, fig


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

def plot_clust(n, pca_result):
	""" plot component 1 vs 2 colored by cluster """
	fig, ax = plt.subplots(figsize=(8, 8))

	plot = ax.scatter(x=pca_result[:,0], y=pca_result[:,1], alpha=0.5, c=n.spindle_stats_i.cluster.values, cmap='RdYlBu')
	ax.set_xlabel('PCA 1')
	ax.set_ylabel('PCA 2')
	ax.set_title(n.metadata['file_info']['in_num'])

	# plot an empty scatter to create legend
	for clust in [0, 1]:
	    # 2 specifies # of discrete colors
	    c = plt.get_cmap('Spectral', 2)(clust)
	    hx = matplotlib.colors.rgb2hex(c[:-1])
	    legend = ax.scatter([], [], c=hx, alpha=1, cmap='Spectral', label=f'KMeans Cluster {clust}')
	ax.legend()

	return fig

def plot_location(n, pca_result):
	""" plot component 1 vs 2 colored by location """
	in_num = n.metadata['file_info']['in_num']
	fig, ax = plt.subplots(figsize=(8, 8))

	# convert anterior-posterior to numbers for color
	cdict = {'A':0, 'C':0.5, 'P':1}
	ap = [cdict[a] for a in n.spindle_stats_i.AP]
	ax.scatter(x=pca_result[:,0], y=pca_result[:,1], alpha=0.5, c=ap, cmap='RdYlBu')
	ax.set_xlabel('PCA 1')
	ax.set_ylabel('PCA 2')
	ax.set_title(f'{in_num} Spatial Distribution')


	cmap = plt.get_cmap('RdYlBu', 2)
	cols = cmap(np.linspace(0, 1, 3))
	# plot an empty scatter to create legend
	for e, label in enumerate(['Frontal', 'Central', 'Parieto-ocipital']):
	    # get hex color
	    hx = matplotlib.colors.rgb2hex(cols[e])
	    legend = ax.scatter([], [], c=hx, alpha=1, label=label)
	ax.legend()

	return fig

def plot_duration(n, pca_result):
	""" plot component 1 vs 2 colored by spindle duration """
	fig, ax = plt.subplots(figsize=(10, 8))

	plot = ax.scatter(x=pca_result[:,0], y=pca_result[:,1], alpha=1, c=n.spindle_stats_i.dur_ms.values, cmap='RdYlBu')
	ax.set_xlabel('PCA 1')
	ax.set_ylabel('PCA 2')
	ax.set_title(n.metadata['file_info']['in_num'])

	fig.colorbar(plot, label='Spindle Duration (ms)')

	return fig

def plot_peakfreq(n, pca_result):
	""" plot component 1 vs 2 colored by spindle peak frequency """
	fig, ax = plt.subplots(figsize=(10, 8))

	plot = ax.scatter(x=pca_result[:,0], y=pca_result[:,1], alpha=1, c=n.spindle_stats_i.dominant_freq_Hz, cmap='RdYlBu')
	ax.set_xlabel('PCA 1')
	ax.set_ylabel('PCA 2')
	ax.set_title(n.metadata['file_info']['in_num'])

	fig.colorbar(plot, label='Spindle Peak Frequency (Hz)')

	return fig



def feature_scatter(n, pca_result):
    """ Plot PC1 vs PC2 with points colored by feature (kmeans cluster, location, peak freq, duration)"""
    
    fig, axs = plt.subplots(1, 4, gridspec_kw={'width_ratios': [0.8, 0.8, 1, 1]}, figsize=(32, 8))

    for e, ax in enumerate(axs.flatten()):
        if e == 0:
            # plot cluster
            plot = ax.scatter(x=pca_result[:,0], y=pca_result[:,1], alpha=0.5, c=n.spindle_stats_i.cluster.values, cmap='RdYlBu')
            ax.set_xlabel('PC 1')
            ax.set_ylabel('PC 2')
            ax.set_title('Kmeans Cluster')
            # plot an empty scatter to create legend
            for clust in [0, 1]:
                # 2 specifies # of discrete colors
                c = plt.get_cmap('Spectral', 2)(clust)
                hx = matplotlib.colors.rgb2hex(c[:-1])
                legend = ax.scatter([], [], c=hx, alpha=1, cmap='Spectral', label=f'Cluster {clust}')
            ax.legend()

        elif e == 3:
            # plot duration
            plot = ax.scatter(x=pca_result[:,0], y=pca_result[:,1], alpha=1, c=n.spindle_stats_i.dur_ms.values, cmap='RdYlBu')
            ax.set_xlabel('PC 1')
            ax.set_ylabel('PC 2')
            ax.set_title('Duration')
            fig.colorbar(plot, ax=ax, label='Spindle Duration (ms)')

        elif e == 2:
            # plot frequency
            plot = ax.scatter(x=pca_result[:,0], y=pca_result[:,1], alpha=1, c=n.spindle_stats_i.dominant_freq_Hz, cmap='RdYlBu')
            ax.set_xlabel('PC 1')
            ax.set_ylabel('PC 2')
            ax.set_title('Peak Frequency')
            fig.colorbar(plot, ax=ax, label='Spindle Peak Frequency (Hz)')

        elif e == 1:
            # plot location
            # convert anterior-posterior to numbers for color
            cdict = {'A':0, 'C':0.5, 'P':1}
            ap = [cdict[a] for a in n.spindle_stats_i.AP]
            ax.scatter(x=pca_result[:,0], y=pca_result[:,1], alpha=0.5, c=ap, cmap='RdYlBu')
            ax.set_xlabel('PC 1')
            ax.set_ylabel('PC 2')
            ax.set_title(f'Spatial Distribution')


            cmap = plt.get_cmap('RdYlBu', 2)
            cols = cmap(np.linspace(0, 1, 3))
            # plot an empty scatter to create legend
            for e, label in enumerate(['Frontal', 'Central', 'Parieto-ocipital']):
                # get hex color
                hx = matplotlib.colors.rgb2hex(cols[e])
                legend = ax.scatter([], [], c=hx, alpha=1, label=label)
            ax.legend()

    params = {'legend.fontsize': 'medium',
                  'axes.titlesize' : 20,
                'axes.labelsize' : 16,
                'xtick.labelsize' : 16,
                'ytick.labelsize' : 16}
    plt.rcParams.update(params)

    fig.suptitle(n.metadata['file_info']['in_num'], size=22, weight='bold')
    fig.tight_layout()
    
    return fig


def plot_spectra(n, pca, pca_result, components):
    """ Plot individual spindle spectra colored by PCA component weight
    
        Parameters
        ----------
        n: nrem.NREM object
            w/ spindles detected and PCA run
        components: list of int
            components to plot
    """

    spin_range = n.metadata['spindle_analysis']['spin_range']
    
    psd_list = np.array([list(chan_dict.values()) for chan_dict in n.spindle_psd_i.values()])
    psd_flat = [item for sublist in psd_list for item in sublist]
    f_idx = psd_flat[0][(psd_flat[0].index >= spin_range[0]) & (psd_flat[0].index <= spin_range[1])].index

    fig, ax = plt.subplots(len(components), 2, figsize=(4*len(components),3*len(components)), gridspec_kw={'width_ratios': [0.8, 1]})

    for pc in components:
        row = pc-1
        # plot the components
        ## if len(components) <2, this will throw an error bc of the 2x2 calling of subplots
        ax[row, 0].plot(f_idx, pca.components_[row], color='black', lw=1.5)
        ax[row, 0].set_xlim(f_idx[0], f_idx[-1])
        ax[row, 0].set_ylabel(f'PC{pc}')
        ax[row, 0].set_xlabel('Frequency (Hz)')
        ax[row, 0].set_title('Principle Component')
                        
        # plot individual spindle spectra
        for spin, psd in enumerate(psd_flat):
            psd_spin = psd[(psd.index >= spin_range[0]) & (psd.index <= spin_range[1])]
            # subtract 1 from pc bc pc 1 = pca_result[:, 0]
            pc_val = pca_result[:,pc-1][spin]
            ax[row, 1].plot(psd_spin, c=plt.cm.RdYlBu(pc_val), alpha=0.5)

        # set labels
        ax[row, 1].set_ylabel('Power (mv$^2$/Hz)')
        ax[row, 1].set_xlabel('Frequency (Hz)')
        ax[row, 1].set_title('Spindle Spectra')
        # set y-axis to scientific notation
        ax[row, 1].ticklabel_format(axis='y', style='sci', scilimits=(0,0))

        # create colorbar
        pc_vals = np.array(pca_result[:,pc-1])
        colors = np.array([plt.cm.RdYlBu(pc_val) for pc_val in pc_vals])
        norm = matplotlib.colors.Normalize(vmin=pc_vals.min(), vmax=pc_vals.max())
        cmap = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.RdYlBu)
        cmap.set_array([])
        plt.colorbar(cmap, ax=ax[row, 1], label=f'PC Weight')
             
    params = {'legend.fontsize': 'medium',
              'axes.titlesize' : 16,
            'axes.labelsize' : 16,
            'xtick.labelsize' : 16,
            'ytick.labelsize' : 16}
    plt.rcParams.update(params)

    fig.suptitle(n.metadata['file_info']['in_num'], size=18, weight='semibold')
    fig.tight_layout()
    
    return fig