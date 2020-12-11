import numpy as np
import numpy.ma as ma
import h5py
from scipy.interpolate import splprep, splev
import os

def splprep_predict(x,y,prob,maxTime):

	orig_x = x
	orig_y = y

	#Create x,y, and time arrays
	x = np.asarray(x)
	y = np.asarray(y)
	t = np.asarray(range(maxTime))
	  
	#Remove missing data using a mask
	#A masked array allows the use of arrays with missing data  
	x = ma.masked_where(prob <= p_cutoff, x)
	y = ma.masked_where(prob <= p_cutoff, y)
	  
	if(np.any(x.mask)):
		x = x[~x.mask]
	  
	if(np.any(y.mask)):
		y = y[~y.mask]

	#As long as all the points aren't masked...
	if len(x) > 0:

		newX = [x[0]]
		newY = [y[0]]

		#So this goes from the second data point to the end
		#And this function only appends the new data if there is a change between them
		#splprep does not work if the coordinates are the same one after another, and this removes
		# those points
		for i in range(min(len(x),len(y)))[1:]:
			if (abs(x[i] - x[i-1]) > 1e-4) or (abs(y[i] - y[i-1]) > 1e-4):
				newX.append(x[i])
				newY.append(y[i])

		newX = np.asarray(newX)
		newY = np.asarray(newY)

		#if all the points are not the same (i.e. calibration)....
		if len(newX) > 1:
			#Runs splprep with s as smoothing to generate the splines
			tck, u = splprep([newX, newY], k=5, s=10**6)
			#Creates the time points to that they go from 0 to 1 with a number of 
			# devisions equal to the number of frames
			newU = np.arange(0,1,t[1]/(t[-1]+t[1]))
			#Runs splev to generate new points from the spline from 0 to 1
			# with a number ofdivisions equal to the number of frames
			new_points = splev(newU, tck)
		else:
			print("Only one point per. Returning original points")
			new_points = [orig_x,orig_y]

	else:
		print("All points were masked. Returning original points")
		new_points = [orig_x,orig_y]
	  
	return(new_points)

#filename = "2020_6_29_24_TN_DN_F2_VDLC_resnet50_L8FVJul4shuffle1_50000_bx.h5"

set_calib = False
set_conf = False

#Not currently working as well as I would like
predict_data = False

p_cutoff = 0.001

new_under_conf = 0.001
new_over_conf = 0.01
finished_frames = 150

n_fish = 8
n_points = 6

filepath = "/Users/Ben/Desktop/Light_8_Fish_Ventral_DLC/L8FV-Ben-2020-07-04/videos/iteration-3/"

video_files = os.listdir(filepath)

for file in video_files:

	if file.endswith("_bx.h5"):

		print(file)

		if file.startswith("2020_6"):
			x_len = 1920
			y_len = 800
		else:
			x_len = 2368
			y_len = 1152

		rep_values = [x_len-x_len/10,y_len/2,0]

		hf = h5py.File(filepath+file, 'r+')
		table = hf['df_with_missing/table']

		n_table = np.array(table)
		new_table = n_table

		calib_points = n_table[0][1][n_fish*n_points*3:]


		for i in range(n_table.shape[0]):
			dlc_data = n_table[i][1]
			dlc_data = np.nan_to_num(dlc_data)

			if set_calib:
				new_table[i][1][n_fish*n_points*3:] = calib_points

			for j in range(len(dlc_data)):
				# j%3 gives 0 = x, 1 = y, 2 = prob
				pos = j%3

				#if there is no data then replace it with default
				if dlc_data[j] == 0:
					new_table[i][1][j] = rep_values[pos]

					if pos != 2:
						print("No Points. Set to {} in position {}.".format(rep_values[pos],pos))

				#if x is out of range replace with default
				if pos == 0 and (dlc_data[j] > x_len or dlc_data[j] < 0):
					new_table[i][1][j] = rep_values[pos]
					print("Out of X Bounds: Replaced {} with {} in position {}.".format(dlc_data[j],rep_values[pos],pos))

				#if y is out of range replace with default
				if pos == 1 and (dlc_data[j] > y_len or dlc_data[j] < 0):
					new_table[i][1][j] = rep_values[pos]
					print("Out of Y Bounds: Replaced {} with {} in position {}.".format(dlc_data[j],rep_values[pos],pos))

				#This is all to allow me to just correct the first n frames and only extract those

				#if we are changing the confidence values and the pos is conf
				if set_conf and pos == 2:
					print(dlc_data[j])
					#if we are looking at the first finished frames and conf > conf cutoff set it lower
					if i <= finished_frames and dlc_data[j] > new_under_conf:
						new_table[i][1][j] = new_under_conf
						print("Reduced Conf: Replaced {} with {} in frame {}.".format(dlc_data[j],new_under_conf,i))
					#if we are after the first finished frames and conf < conf cutoff
					elif i > finished_frames and dlc_data[j] < new_under_conf:
						new_table[i][1][j] = new_over_conf
						print("Increased Conf: Replaced {} with {} in frame {}.".format(dlc_data[j],new_over_conf,i))

		if predict_data:
			array_data = np.zeros((n_table.shape[0],n_table[0][1].shape[0]))

			print(n_table.shape)
			print(n_table[0][1].shape)
			print(array_data.shape)

			#Fill in array_data with the weird h5 data
			for i in range(n_table.shape[0]):
				array_data[i][:] = n_table[i][1]

			#predict new x and y points taking out the points with low prob
			#0 = x, 1 = y, 2 = p

			for j in range(0,array_data.shape[1],3):

				x = array_data[:,j]
				y = array_data[:,j+1]
				p = array_data[:,j+2]

				new_points = splprep_predict(x,y,p,len(x))

				# array_data[:,j] = new_points[0]
				# array_data[:,j+1] = new_points[1]

				#Only replace the bad points
				for i in range(array_data.shape[0]):
					if p[i] <= p_cutoff:
						array_data[i][j] = new_points[0][i]
						array_data[i][j] = new_points[1][i]

			#Now place those points in the h5
			for i in range(array_data.shape[0]):
				new_table[i][1] = array_data[i]


		table[...] = new_table
		hf.close()

#2368 × 1152 videos