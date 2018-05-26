

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
	fig, ax1 = plt.subplots(1,1, figsize=(15,10))
	fig.subplots_adjust(right=0.8)

	# ax1
	ax1.plot(test_data[:,0], test_data[:,-1], color='blue', label="test_loss")
	ax1.plot(train_data[:,0], train_data[:,-1], color='green', label="train_loss")
	ax1.set_ylabel('loss')
	ax1.set_xlabel('iteration')
	
	# ax2
	ax2 = ax1.twinx()
	lines = []
	if(len(test_data[0]) > 5):
		acc1, = ax2.plot(test_data[:,0], test_data[:,3], color='red', label="accuracy#1" )
		acc5, = ax2.plot(test_data[:,0], test_data[:,4], color='yellow', label="accuracy#5" )
		lines.append(acc1)
		lines.append(acc5)
	else:
		acc1, = ax2.plot(test_data[:,0], test_data[:,3], color='red', label="accuracy#1" )
		lines.append(acc1)
	ax2.set_ylabel('acurracy')
	
	# ax3
	ax3 = ax1.twinx()
	ax3.spines['right'].set_position( ('axes', 1.1) )
	lr, = ax3.plot(train_data[:,0], train_data[:,2], color='black', label="LearningRate")
	lines.append(lr)
	ax3.set_ylabel('LearningRate')
	
	# legend
	ax1.legend()
	ax2.legend(lines, [l.get_label() for l in lines], loc="upper center")
	
	plt.show()
	
	
if __name__ == "__main__":
	main()