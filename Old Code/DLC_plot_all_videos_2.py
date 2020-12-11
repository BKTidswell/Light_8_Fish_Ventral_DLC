import os
import subprocess
import deeplabcut
from deeplabcut.utils.make_labeled_video import create_video_from_pickled_tracks


video_path = '/Users/Ben/Desktop/Light_8_Fish_Ventral_DLC/L8FV-Ben-2020-07-04/videos/'
pickle_path = '/Users/Ben/Desktop/Light_8_Fish_Ventral_DLC/L8FV-Ben-2020-07-04/videos/{}DLC_resnet50_L8FVJul4shuffle1_100000_sk.pickle'
video_dir = '/Users/Ben/Desktop/Light_8_Fish_Ventral_DLC/L8FV-Ben-2020-07-04/videos/{}/{}'

# 2020_6_29_1_TN_DN_F0_V
# 2020_6_29_1_TN_DN_F0_VDLC_resnet50_L8FVJul4shuffle1_100000_sk.pickle

for file in os.listdir(video_path):
	if file.endswith(".mp4"):
		file_name = file[:-4]
		print(file_name)
		create_video_from_pickled_tracks(video_path+file,pickle_path.format(file_name))

# ffmpeg -r 10 -f image2 -s 1920x1080 -i *frame%03d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p *test.mp4
# ffmpeg -i VIDEO_NAMEDLClabeled.mp4 -pix_fmt yuv420p -crf 18 WORKING_MP4.mp4
for file in os.listdir(video_path):
	if file.endswith(".mp4"):
		file_name = file[:-4]
		print(file_name)
		subprocess.call(['ffmpeg', '-i', video_dir.format(file_name,file_name+"DLClabeled.mp4"), '-pix_fmt', 'yuv420p', '-crf', '18', video_dir.format(file_name,file_name+"DLClabeled_fixed.mp4")])
		subprocess.call(['ffmpeg', '-r', '10', '-f', 'image2', '-s', '1920x1080', '-i', video_dir.format(file_name,"""frame%03d.png"""), '-vcodec', 'libx264', '-crf', '25', '-pix_fmt', 'yuv420p', video_dir.format(file_name,"combined_png2.mp4")])
