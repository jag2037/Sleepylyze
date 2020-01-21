from distutils.core import setup

setup(
	name='sleepylyze',
	version='0.1dev',
	author= 'J. Gottshall',
	author_email='jackie.gottshall@gmail.com',
	packages=['sleepylyze',],
	url='https://github.com/jag2037/sleepylyze',
	license='',
	description= 'Python analysis of EEG sleep architecture'
	long_description=open('README.txt').read(),
	install_requires=[
		'datetime',
		'io',
		'json',
		'math',
		'numpy',
		'os',
		'pandas',
		'psycopg2',
		're',
		'statistics',
		'sqlalchemy',
		'glob',
		'warnings',
		'xlsxwriter',
		'mne',
		'scipy'
	],
	)