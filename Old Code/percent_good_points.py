import os, sys
import numpy as np

cwd = os.getcwd()

new_csv_dir = cwd+"/Pruned_CSVs/"
video_data_dir = cwd+"/L8FV-Ben-2020-07-04/videos/iteration-3/"

#Create folder for new csvs
if not os.path.isdir(new_csv_dir):
	os.mkdir(new_csv_dir)

prob_cutoff = 0.9

files = []
good_percents = []

for file_name in os.listdir(video_data_dir):
	if file_name.endswith(".csv"):

		in_csv = open(video_data_dir+file_name,"r")

		#out_csv = open(new_csv_dir+file_name[:-4]+"_pruned.csv", "w")

		#Skip the first 4
		skip_lines = 0

		good_points = 0
		total_points = 0

		for line in in_csv:
			if skip_lines >= 4:
				lis = line.split(",")[:-24]

				#Off by 1 for the first row of frame counts
				probs = np.asarray(lis[3::3]).astype(np.float)

				#If there are no zeros in probs
				good_points += np.sum(probs > prob_cutoff)
				total_points += len(probs)

				#out_csv.write(','.join(lis)+"\n")

			else:
				skip_lines += 1

				lis = line.split(",")[:-24]
				#out_csv.write(','.join(lis)+"\n")

				# if skip_lines == 3:
				# 	print(len(lis))
				# 	print(','.join(lis))
				# 	sys.exit()

			

		#print("{} : {}% good lines".format(file_name[:13],round(good_lines/total_lines*100,2)))

		files.append(file_name)
		good_percents.append(round(good_points/total_points*100,2))

		in_csv.close()
		#out_csv.close()

files = np.asarray(files)
good_percents = np.asarray(good_percents)

sort_percent = np.flip(good_percents[np.argsort(good_percents)])
sorted_files = np.flip(files[np.argsort(good_percents)])

for i in range(len(sorted_files)):
	print("{} : {}% good points".format(sorted_files[i][:22],sort_percent[i]))



