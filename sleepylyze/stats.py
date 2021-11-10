""" module for post-analysis statistics """

import numpy as np
import os
import pandas as pd
from scipy import stats

def calc_dist_stats(fname, fpath, in_num, visit, file_date):
    """ calculate descriptive stats and KS statistics from HCXXX_DATE_s2_0-2hrs_spindle_stats_i.csv file """

    file = os.path.join(fpath, fname)

    # load the csv file
    spindle_stats_i = pd.read_csv(file)
    
    # set the stats of interest and set multi-index columns
    stats_oi = ['total_peaks', 'dominant_freq_Hz', 'peak2_freq', 'peak2_ratio', 'dur_ms']
    stat_metrics = ['median_1', 'iqr_1', 'median_2', 'iqr_2', 'ks_val', 'pval', 'sided']
    lvl2 = [None, None, None, None] + list(np.tile(stat_metrics, len(stats_oi)))
    lvl1 = ['in_num', 'visit', 'group1', 'group2'] + list(np.repeat(stats_oi, len(stat_metrics)))
    
    # add SO factors
    so_metrics = ['max_perc', 'max', 'mean', 'med', 'iqr']
    groups = np.repeat(['1', '2'], len(so_metrics))
    lvl2_so = ['_'.join([n, g]) for g, n in zip(groups, np.tile(so_metrics, 2))]
    lvl2_so.extend(['ks_val', 'pval', 'sided'])
    lvl1_so = np.repeat('so_dist', len(lvl2_so))
                        
    # combine the sets
    lvl1.extend(lvl1_so)
    lvl2.extend(lvl2_so)
    col_idx = pd.MultiIndex.from_arrays([lvl1, lvl2])
    
    # make a row for each value
    rows = []
    comps = ['AP', 'LR', 'clust']
    for comp in comps:
        if comp == 'AP':
            g1, g2 = 'anterior', 'posterior'
            dat1 = spindle_stats_i[spindle_stats_i.AP == 'A']
            dat2 = spindle_stats_i[spindle_stats_i.AP == 'P']
            so_stats = np.repeat(None, 10)
            
        elif comp == 'LR':
            g1, g2 = 'left', 'right'
            dat1 = spindle_stats_i[spindle_stats_i.RL == 'L']
            dat2 = spindle_stats_i[spindle_stats_i.RL == 'R']
            so_stats = np.repeat(None, 10)
        
        elif comp =='clust':
            g1, g2 = 'clust0', 'clust1'
            dat1 = spindle_stats_i[spindle_stats_i.cluster ==0]
            dat2 = spindle_stats_i[spindle_stats_i.cluster ==1]

        row = [in_num, visit, g1, g2]
        # compare distributions
        for stat in stats_oi:
            med1 = np.nanmedian(dat1[stat])
            med2 = np.nanmedian(dat2[stat])
            # return 25th & 75th percentile values
            iqr1 = np.nanpercentile(dat1[stat], [25, 75])
            iqr2 = np.nanpercentile(dat2[stat], [25, 75])
            # return range
            #iqr1 = stats.iqr(dat1[stat], nan_policy='omit')
            #iqr2 = stats.iqr(dat2[stat], nan_policy='omit')
            if stat == 'dominant_freq_Hz' and (comp == 'clust' or comp =='AP'):
                # we expect cluster 1 spindles to be faster than cluster 0
                # we expect posterior spindles to be faster than anterior
                sided = 'greater'
            else:
                # for the other stats we have no directional expectation
                sided = 'two-sided'
            ks = stats.ks_2samp(dat1[stat].values, dat2[stat].values, alternative=sided)
            row.extend([med1, iqr1, med2, iqr2, ks[0], ks[1], sided])
            
        # compare SO distributions
        # load the normalized data
        so_fname = f'spindle_SO/{in_num}_{file_date}_s2_0-2hrs_spso_distribution_{comp}.xlsx'
        so_file = os.path.join(fpath, so_fname)
        if comp == 'clust':
            tabs = ['clust0_SO_dist_norm', 'clust1_SO_dist_norm']
        elif comp == 'AP':
            tabs = ['A_SO_dist_norm', 'P_SO_dist_norm']
        elif comp == 'LR':
            tabs = ['L_SO_dist_norm', 'R_SO_dist_norm']

        g0 = pd.read_excel(so_file, sheet_name=tabs[0], index_col = 'id_ms')
        g1 = pd.read_excel(so_file, sheet_name=tabs[1], index_col = 'id_ms')
        # get stats
        so_stats = []
        # get descriptive stats for each distribution
        for dat in [g0, g1]:
            # get max
            dat_max = np.max(dat[0])
            dat_xmax = dat[0].idxmax()

            # convert back to counts
            xlist = np.concatenate([np.repeat(idx, val*100) for (idx, val) in dat.iterrows()]).tolist()
            mean = np.mean(xlist) 
            med = np.median(xlist)
            iqr = np.nanpercentile(xlist, [25, 75])
            #iqr = stats.iqr(xlist)
            # add to stats
            so_stats.extend([dat_max, dat_xmax, mean, med, iqr])
        # run KS on overall percentage distributions
        ks = stats.ks_2samp(g0[0].values, g1[0].values, alternative=sided)
        so_stats.extend([ks[0], ks[1], sided])
        row.extend(so_stats)
        rows.append(row)
    
    # create df & apply column index
    stats_df = pd.DataFrame(rows)
    stats_df.columns = col_idx
    
    return stats_df