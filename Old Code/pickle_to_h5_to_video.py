import deeplabcut
import os

config_path = '/Users/Ben/Desktop/Light_8_Fish_Ventral_DLC/L8FV-Ben-2020-07-04/config.yaml'
video_path = '/Users/Ben/Desktop/Light_8_Fish_Ventral_DLC/L8FV-Ben-2020-07-04/videos/iteration-3/'

video_files = os.listdir(video_path)

for file in video_files:
	if file.endswith("_bx.pickle"):
		h5_name = file.replace("pickle","h5")
		video_name = file.replace

		if h5_name in video_files:
			print("h5 for {} already exists!".format(file))
		else:
			print("Converting {} to h5!".format(file))
			deeplabcut.convert_raw_tracks_to_h5(config_path, video_path+file)


for file in video_files:
	if file.endswith(".mp4"):

		print(file)
		
		deeplabcut.filterpredictions(config_path,[video_path+file], videotype='mp4', shuffle=1, track_method = 'box')

		deeplabcut.create_labeled_video(config_path, [video_path+file],videotype='.mp4',shuffle=1,draw_skeleton=True,track_method='box',save_frames=False,filtered=True)