def plot_butter1(s_freq, wn, sos, order, wp=None, ws=None, line_col = None, y_int=[0.5, 0.7, 0.99], xlim=(0, 20)):
     """ plot the frequency response of the filter [Parameters/Returns]
          Optimized for comparing filt to filtfilt
     """
     nyquist=s_freq/2
     wn_freq = wn*nyquist
     
     if line_col is None:
          line_col = ['red', 'blue', 'green']

     plt.figure(1)
     plt.clf()

     for s, k in zip(sos, order):
          w, h = sosfreqz(s, fs=s_freq) # specify sampling frequency to put filter frequency in same units (Hz); otherwise will ouput in radians/sample         
          y_vals = [abs(h), abs(h)**2]
          name = ['Filt', 'FiltFilt']
          #y_int = [0.5, 0.7, 0.95]
          #linestyle = ['solid', 'dashed']
          alpha = [1, .5]
          for i, (m, n, y) in enumerate(zip(name, alpha, y_vals)):
               plt.plot(w, y, label='%s wn = %s\norder = %d' %(name[i], wn_freq, k), linestyle = '-', alpha=alpha[i], color = line_col[order.index(k)])
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


def plot_butter2(s_freq, wn, sos, order, wp=None, ws=None, line_col = None, y_int=[0.5, 0.7, 0.99], xlim=(0, 20)):
     """ plot the frequency response of the filter [Parameters/Returns]
     """
     nyquist=s_freq/2
     wn_freq = wn*nyquist
     
     #linecolor = np.linspace(0, 1, len(wn)) # plot different windows w/ different colors
     # finish lcolor code here
     linecolor = ['red', 'blue', 'green'] # different windows w/ different colors
     linestyle = ['-', '--', '-.', ':']   # plot different orders w/ different linestyles 

     plt.figure(1)
     plt.clf()

     for s, k in zip(sos, order):
          w, h = sosfreqz(s, fs=s_freq) # specify sampling frequency to put filter frequency in same units (Hz); otherwise will ouput in radians/sample         
          y_vals = [abs(h), abs(h)**2]
          name = ['Filt', 'FiltFilt']
          #y_int = [0.5, 0.7, 0.95]
          linestyle = ['solid', 'dashed']
          for i, (m, n, y) in enumerate(zip(name, linestyle, y_vals)):
               plt.plot(w, y, label='%s wn = %s\norder = %d' %(name[i], wn_freq, k), linestyle = linestyle[i], alpha=.75, color = line_col[order.index(k)])
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
