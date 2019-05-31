""" This file contains a class and methods for Non-REM EEG segments 
	
	Notes:
		- Analysis should output # of NaNs in the data
"""

class NREM:
	""" General class for nonREM EEG segments """
	def __init__(self, fname, fpath, epoched=False):
		filepath = os.path.join(fpath, fname)

       	in_num, start_date, slpstage, cycle = fname.split('_')[:4]
        self.metadata = {'file_info':{'in_num': in_num, 'fname': fname, 'path': filepath,
                                	'sleep_stage': slpstage,'cycle': cycle} }
        if epoched is True:
        	self.metadata['epoch'] = fname.split('_')[4]

    def load_segment(self):
        """ Load eeg segment and extract sampling frequency. """
        
        data = pd.read_csv(self.metadata['file_info']['path'], header = [0, 1], index_col = 0, parse_dates=True)
        
        # Check cycle length against 5 minute duration minimum
        cycle_len_secs = (data.index[-1] - data.index[0]).total_seconds()
        self.data = data
        
        diff = data.index.to_series().diff()[1:2]
        s_freq = 1000000/diff[0].microseconds

        self.metadata['file_info']['start_time'] = data.index[0]
        self.metadata['analysis_info'] = {'s_freq': s_freq, 'cycle_len_secs': cycle_len_secs}

        print('EEG successfully imported.')

