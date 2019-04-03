""" Load EEG raw data exported from XLTEK """

import datetime
import pandas as pd 
import numpy as np 
import scipy.io as io


def check_chans(chans, expected_chans):
		print(chans, ' of ', expected_chans, ' expected channels found')
		if chans != expected_chans:	# this may not be a proper check
			print('WARNING: Not all expected channels for this headbox were detected. Proceeding with detected channels')


def read_xltek(fname=None, savemat=False):
	"""
	Load a .txt file exported from XLTEK and return arrays 

	Parameters
	----------
	datapath: string (default None)
		path to the xltek .txt raw data file
	savemat: BOOL
		option to save output as a matlab file (default False)
	
	Returns
	--------
	time:
	data:
	s_freq:
	channels:


	TO DO: Have option to export as arrays for matlab file, default to pandas dataframe
	"""
	
	if fname is None:
		fname = input("Path to file not found. Filepath? ")

	# extract IN
	in_num = fname.split("_")[0]
	print("Patient identifier:", in_num)

	# extract sampling freq, # of channels, headbox sn
	with open(fname, 'r') as f:	# pull out the header w/ first recording
		header = []
		for i in range(1,17):
			header.append(f.readline())	# use readline (NOT READLINES) here to save memory for large files
	s_freq = int(float(header[7].split()[3])) # extract sampling frequency
	chans = int(header[8].split()[2]) # extract number of channels as integer
	hbsn = int(header[10].split()[3]) # extract headbox serial number
	start_time = header[15].split()[1] # extract start time
			
	# can use either the number of channels or the hbsn to determine columns to extract

	# to define channels, probably want to make a new function that gets called to not clutter this fn
	#channels = np.array(['FC5']*chans, dtype=object).reshape(chans,1) # for now create an repeated list

	# define channels for different headboxes
	if hbsn == 125:
		hbid = "MOBEE 32"
		print('Headbox:', hbid)
		expected_chans = 35
		check_chans(chans, expected_chans)
		#print(chans, ' of ', expected_chans, ' expected channels found')
		#if chans != expected_chans:	# this may not be a proper check
		#	print('WARNING: Not all expected channels for this headbox were detected. Proceeding with detected channels')
		channels_all = np.array(['REF','FP1','F7','T3','A1','T5','O1','F3','C3','P3','FPZorEKG','FZ','CZ',
			'PZ','FP2','F8','T4','A2','T6','O2','F4','C4','P4','AF7','AF8','FC5','FC6','FC1','FC2',
			'CP5','CP6','CP1','CP2','OSAT','PR'], dtype=object).reshape(chans,1)
		channels = channels_all[:-2]
		channels_rmvd = channels_all[-2:]
		print('Removed the following channels: \n', np.array_str(channels_rmvd))
	elif hbsn == 65535:
		hbid = "FS128"
		print('Headbox:', hbid)
		expected_chans = 128
		check_chans(chans, expected_chans) 
		channels_all = np.array(['Fp1','F3','FC1','C3','CP1','P3','O1','AF7','F7','FC5','T3','CP5',
			'T5','PO7','FPz','Fz','Cz','CPz','Pz','POz','Oz','FP2','F4','FC2','C4','CP2','P4','O2',
			'AF8','F8','FC6','T4','CP6','T6','PO8','F1','F2','EOG_L','EOG_R','EKG'], 
			dtype=object).reshape(40,1) # probably want this as a list for downstream processign
		# remove any chans here?
		channels = channels_all

		
	# read in only the data from the file, only the channels of interest
	end_col = len(channels) + 3 # this works for MOBEE32, check for other headboxes

	print('Importing EEG data...')
	data_full = pd.read_csv(fname, delim_whitespace=True, header=None, skiprows=15, 
		usecols=range(0,end_col)) #.transpose() # read in the data and transpose the matrix --> maybe transpose later instead
		
	# Determine microsecond values & append to datetime
	step_size = int(1000000/s_freq)
	readings = range(1, s_freq+1) # add one bc range is non-inclusive
	usecs = range(0, 1000000, step_size)
	usec_dict = dict(zip(readings, usecs))
		
	first_us = s_freq - (data_full[1].eq(start_time).sum())
	reps, rem = divmod((len(data_full) - first_us), s_freq) # get the number of full repeates and the remainder
	last_us = s_freq - rem
	# make the list of usecs as strings
	usec_col = list(map(str,((list(usecs[first_us:]) + list(np.tile(usecs, reps)) + list(usecs[:last_us])))))

	new_time = [] # make the new column
	for i in range(0, len(data_full)): # fill it with values
		new_time.append(data_full[1][i] + ':' + usec_col[i])
	data_full[1] = new_time # replace the old column
		
	# combine date & time columns to a datetime object
	dtime = []
	for i, (d, t) in enumerate(zip(data_full[0], data_full[1])):
		mo, d, yr = [int(d) for d in data_full[0][i].split('/')]
		h, m, s, us = [int(t) for t in data_full[1][i].split(':')]
		dtime.append(datetime.datetime(yr, mo, d, h, m, s, us))
		#start_time = datetime.datetime(yr, mo, d, h, m, s, us) # replace the start_time variable 
	
	# transpose the dataframe & convert to np arrays for .mat saving
	data_t = data_full.transpose()
	time = pd.DataFrame.to_numpy(data_t.loc[1])
	data = pd.DataFrame.to_numpy(data_t.loc[3:])
	print('Data successfully imported')

	# make empty hypnogram file (for visbrain.Sleep package)
	#print('Creating empty hypnogram...')
	#hypno = np.zeros(data.shape[1])
	#print('Hypnogram successfully created')

	# save as .mat file (vars needed: data, channels, sf, hypno)
	# this returns TIME, not DATETIME
	if savemat==True:
		print('Saving MATLAB file...')
		axis = {'data':data, 'chan':channels, 'sf':float(s_freq), 'time': time, 'hypno':hypno} # create a dictionary w/ vars & values
		# axis = OrderedDict([('data',data), ('chan',channels), ('sf',float(s_freq)), ('time',time), ('hypno',hypno)])
		mat_fname = arg.split('.')[0] + '.mat' # set the name for the mat file
		io.savemat(mat_fname, axis) # save the new .mat file
		filepath = os.path.dirname(os.path.realpath(mat_fname))
		print('File successfully saved as', filepath+'\\'+mat_fname)

	return s_freq, dtime, data, channels 

