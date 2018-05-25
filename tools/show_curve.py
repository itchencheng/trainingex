

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('/home/chen/caffe/tools/extra')
import parse_log

def ReadCSV(logtest):
	fi = open(logtest)
	data = []
	lines = fi.readlines()
	fi.close()
	''' <test> NumIters,Seconds,LearningRate,accuracy,loss '''
	''' <train> NumIters,Seconds,LearningRate,loss '''
	for line in lines:
		words = line.strip().split(',')
		data.append(words)
	return np.array(data[1:])

def main():
	logfile = sys.argv[1]
	print(logfile)
	logdir = os.path.dirname(logfile)
	print(logdir)	
	
	''' parse log ''' 
	train_dict_list, test_dict_list = parse_log.parse_log(logfile)
	
	''' save to file '''
	parse_log.save_csv_files(logfile, logdir, 
							 train_dict_list, test_dict_list)
	
	''' read csv '''
	logtest = logfile + '.test'
	logtrain = logfile + '.train'
	test_data = ReadCSV(logtest)
	train_data = ReadCSV(logtrain)
	print(test_data[0])
	print(train_data[0])
	
	''' plot '''
	fig, ax = plt.subplots(1, 1)
	
	ax.plot(test_data[:,0], test_data[:,-1], color='blue', label="test_loss")
	ax.plot(train_data[:,0], train_data[:,-1], color='black', label="train_loss")
	ax.set_ylabel('loss')
	ax.legend(loc=1)
	
	ax2 = ax.twinx()
	ax2.plot(test_data[:,0], test_data[:,3], color='green', label="accuracy#1" )
	#ax2.plot(test_data[:,0], test_data[:,4], color='red', label="accuracy#5" )
	ax2.set_ylabel('acurracy')
	ax2.legend(loc=2)
	
	plt.show()
	
	
if __name__ == "__main__":
	main()