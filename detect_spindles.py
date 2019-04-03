# Filename: detect_spindles.py
# Created: 3-12-19
# Author: JLG
#
# Description:
#   First run read_xltek w/ the following:
#       >> s_freq, time, data, channels = read_xltek('filename')
#
#
#
#############################################################################################
#%matplotlib # run this in ipython console first

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import buttord, butter, sosfiltfilt, sosfreqz
import shapely.geometry as SG

def order_butter(s_freq, hi_pass=8., lo_pass=16., ws=None, trans_width=.05, gpass=0.001, gstop=60.0):
    """
    Determine order and 3dB attenuation frequency window for Butterworth filter
    
    Parameters
    ----------
    s_freq: int
        sampling frequency
    hi_pass: float
        high pass filter (bottom of spindle range)
    lo_pass: float
        low pass filter (top of spindle range)\
    trans_width: float
        distance between bandpass and bandstop frequencies (determines width of transition zone)
    gpass: float
        from scipy.signal.buttord. maximum ripple in passband (dB)
    gstop: float
        from scipy.signal.buttord. minimum attenuation in stopband (dB)

    Returns
    -------
    order: int
        minimum filter order to fit specified parameters
    wn: ndarray
        Butterworth natural frequency (the "3 dB frequency") for use with butter filter
    ws: ndarray
        stop band
    """
    nyquist = s_freq/2
    hi = hi_pass/nyquist
    lo = lo_pass/nyquist
    wp = [hi, lo] # set pass band
    if ws == None:
        ws = [(1-trans_width)*hi, (1+trans_width)*lo] # set stop band to 5% on each side
    else:
        ws = [x/nyquist for x in ws] 
    order, wn = buttord(wp, ws, gpass, gstop)
    return order, wn, wp, ws, nyquist

def make_butter(wn, order, s_freq):
    """ Make Butterworth bandpass filter [Paramters/Returns]"""
    nyquist = s_freq/2
    wn=np.asarray(wn)
    if np.any(wn <=0) or np.any(wn >=1):
        print("Passband not normalized to nyquist frequency. Normalizing now.")
        wn = wn/nyquist # must remake filter for each pt bc of differences in s_freq
        print("Nyqyuist normalized passband=", wn)
    sos = butter(order, wn, btype='bandpass', output='sos')
    print("Butterworth filter successfully created.\nOrder =", order,"\nBandpass=", wn*nyquist)
    return wn, order, sos

def plot_butter(s_freq, wn, sos, order, wp=None, ws=None, plot_filt=False, plot_filtfilt=True, 
    y_int=[0.5, 0.7, 0.99], xlim=(0, 20)):
    """ plot the frequency response of the filter [Parameters/Returns]
        TO DO: Find & plot intersects between graphs and y=1, y=0.7, y=0.5
    """
    nyquist=s_freq/2
    wn_freq = wn*nyquist
    
    plt.figure(1)
    plt.clf()
    
    filtphase = ['Filt', 'FiltFilt']
    linecolor = np.linspace(0, 1, len(wn)) # plot different windows w/ different colors
    # finish lcolor code here
    linestyle = ['-', '--', '-.', ':']     # plot different orders w/ different linestyles  
    alpha = [.5, .75]                      # plot different phase (filtertype) w/ different alpha vals 
    
    #line_col = ['red', 'blue', 'green']
    for s, k, a in zip(sos, order, wn):
        w, h = sosfreqz(s, fs=s_freq) # specify sampling frequency to put filter frequency in same units (Hz); otherwise will ouput in radians/sample    
        
        # set which phase filters to plot
        if plot_filt == True and plot_filtfilt == True:
            y_vals = [abs(h), abs(h)**2]
        elif plot_filt == True and plot_filtfilt == False:
            y_vals = [abs(h)]
            filtphase = filtphase[0]
        elif plot_filt == False and plot_filtfilt == True:
            y_vals = [abs(h)**2]
            filtphase = filtphase[1]

        #linestyle = ['solid', 'dashed']
        for i, (m, n, y) in enumerate(zip(filtphase, linestyle, y_vals)):
            plt.plot(w, y, label='%s wn = %s\norder = %d' %(filtphase[i], wn_freq, k), linestyle = linestyle[i], alpha=.75, color = line_col[order.index(k)])
            line = SG.LineString(list(zip(w, y)))
            for j in y_int:
                yline = SG.LineString([(min(w), j), (max(w), j)])
                coords = np.array(line.intersection(yline))
                label = '{} gain at {:.2f}Hz & {:.2f}Hz'.format(coords[0][1], coords[0][0], coords[1][0]) # print intersects w/ 2 decimal places
                plt.scatter(coords[:,0], coords[:,1], s=35, alpha=.5, c=line_col[order.index(k)], edgecolors=line_col[order.index(k)], 
                    linestyle=linestyle[i], label=label)
    
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.grid(True)
    plt.legend(loc='best')
    plt.xlim(xlim)

def plot_butter2(s_freq, wn, sos, order, wp=None, ws=None, plot_filt=False, plot_filtfilt=True, 
    y_int=[0.5, 0.7, 0.99], xlim=(0, 20)):
    """ plot the frequency response of the filter for a given order fwd vs backwards
    """
    nyquist=s_freq/2
    wn_freq = wn*nyquist
    
    plt.figure(1)
    plt.clf()
    
    filtphase = ['Filt', 'FiltFilt']
    linecolor = np.linspace(0, 1, len(wn)) # plot different windows w/ different colors
    # finish lcolor code here
    linestyle = ['-', '--', '-.', ':']     # plot different orders w/ different linestyles  
    alpha = [.5, .75]                      # plot different phase (filtertype) w/ different alpha vals 
    
    #line_col = ['red', 'blue', 'green']
    for s, k, a in zip(sos, order, wn):
        w, h = sosfreqz(s, fs=s_freq) # specify sampling frequency to put filter frequency in same units (Hz); otherwise will ouput in radians/sample    
        

        #linestyle = ['solid', 'dashed']
        for i, (m, n, y) in enumerate(zip(filtphase, linestyle, y_vals)):
            plt.plot(w, y, label='%s wn = %s\norder = %d' %(filtphase[i], wn_freq, k), linestyle = linestyle[i], alpha=.75, color = line_col[order.index(k)])
            line = SG.LineString(list(zip(w, y)))
            for j in y_int:
                yline = SG.LineString([(min(w), j), (max(w), j)])
                coords = np.array(line.intersection(yline))
                label = '{} gain at {:.2f}Hz & {:.2f}Hz'.format(coords[0][1], coords[0][0], coords[1][0]) # print intersects w/ 2 decimal places
                plt.scatter(coords[:,0], coords[:,1], s=35, alpha=.5, c=line_col[order.index(k)], edgecolors=line_col[order.index(k)], 
                    linestyle=linestyle[i], label=label)
    
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.grid(True)
    plt.legend(loc='best')
    plt.xlim(xlim)
            
        # if wp is not None: # if used buttord to define wp and ws --> this isn't up to date
        #     wp_freq = np.asarray(wp)*nyquist
        #     if ws is not None:
        #         ws_freq = np.asarray(ws)*nyquist
        #     # wp_freq, wn_freq, ws_freq = np.asarray(wp)*nyquist, wn*nyquist, np.asarray(ws)*nyquist
        #     plt.plot(w, abs(h), label="-- passband = %s\n-. cutoff (-3dB attenuation) = %s\n: stopband = %s\norder = %d" 
        #         % (wp_freq, wn_freq, ws_freq, k)) # linear scale
        #     [plt.axvline(x=f, linestyle='--') for f in wp_freq]
        #     [plt.axvline(x = f, linestyle=':', label = "Stop frequency at %d" % f) for f in ws_freq]
        #     [plt.axvline(x = f, linestyle='-.', label = "-3dB attenuation at %d" % f) for f in wn_freq]
        # else:


        #[plt.axvline(x = f, linestyle='-.', label = "-3dB attenuation at %d" % f) for f in wn_freq]
        # plt.plot(w, abs(h)**2, label='filt squared', linestyle='--') # included above
        # plt.plot(w, 20*np.log10(abs(h)), 'g') # log scale -- looks similar if you condense Y axis
            ## IF SAMPLING FREQUENCY NOT SPECIFIED BEFOREHAND, MUST CONVERT FREQUENCY UNITS ON PLOT BACK TO Hz:
        # plt.plot((0.5*s_freq/np.pi)*w, abs(h))

    # calculate 0.5, 0.7, 1.0 gain intersections
    #y_vals = [abs(h), abs(h)**2]
    #y_int = [0.5, 0.7]
    #for y in y_vals:
    #    line = SG.LineString(list(zip(w, y))) # interpolate y values for filter
    #    for i in y_int:
    #        yline = SG.LineString([(min(w), i), (max(w), i)])
    #        coords = np.array(line.intersection(yline))
    #        label = '{} gain at {:.2f}Hz and {:.2f}Hz'.format(coords[0][1], coords[0][0], coords[1][0]) # print intersects w/ 2 decimal places
    #        plt.scatter(coords[:,0], coords[:,1], s=50, c='black', alpha=.75, label=label)
            #plt.text(coords[:,0], coords[:,1]+.3, coords[:,0], fontsize=5)
            #filt_vals.extend(coords[:,0])



    # plot 0.5, 0.7, 1 gain lines --> don't need this if plotting intersections
    # plt.plot(w, np.repeat(0.5, len(w)),
    #   w, np.repeat(0.7, len(w)),
    #   w, np.repeat(1.0, len(w)), 
    #   linestyle =':', color='black')





def use_butter(data, sos):
    """ Apply Butterworth bandpass to signal [Parameters/Returns]"""
    dat = sosfiltfilt(sos, data) # defaults to odd padding --> see if you want this
    return dat

#def detect_spindles(s_freq, time, data, channels, hi_pass=8, lo_pass=16, rms_win_len=.2, 
#   sm_win_len=.2, thres_method='avg_rms+SD', thres_hi=3, thres_lo=1, min_dur=.5, max_dur=3, 
#   min_inter=.5):
    # """ 
    # Description

    # Parameters
    # ----------
    # s_freq: int
    #   sampling frequency
    # time: numpy.ndarray (datatype 'O')
    #   time corresponding to data in format 'hh:mm:ss:uuuuuu' (u=microsecond)
    # data: numpy.ndarray (datatype 'O')
    #   EEG signal of shape (channels, time)
    # channels: numpy.ndarray (datatype 'O')
    #   list of channels
    # hi_pass: int
    #   high pass filter (bottom of spindle range)
    # lo_pass: int
    #   low pass filter (top of spindle range)
    # rms_win_len: float
    #   length of moving window for root mean square calculations. Must be longer than
    #   one full cycle of the frequency band of the signal of interest (Ex. FOA = 8Hz,
    #   1 full cycle = 1/8 = .125sec)
    # sm_win_len: float
    #   length of moving window for smoothing function
    # thres_method: str
    #   method for setting the detection threshold. Options:
    #       'avg_rms+SD': set to standard deviations above the average of the RMS of
    #           the signal
    # thres_hi: int
    #   # of standard deviations above the RMS average to set the high detection threshold
    # thres_lo: int
    #   # of standard deviations above the RMS average to set the low detection threshold
    # min_dur: float (seconds)
    #   minimum length between thres_lo crossings for a detection to be considered a spindle
    # max_dur: float (seconds)
    #   maximum length between thres_lo crossings for a detection to be considered a spindle
    # min_inter: float (seconds)
    #   minimum interval between detected spindles to be considered separate events

    # Returns
    # -------
    # spin_events: multidimensional np array

    # """

    #for chan in channels:
        #bandpass signal between hi_pass and lo_pass
        # see wonambi transform_signal 'double_butter'
