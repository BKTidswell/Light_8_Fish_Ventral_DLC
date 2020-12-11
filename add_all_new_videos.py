import deeplabcut
import os

cwd = os.getcwd()

config_path = '/Users/Ben/Desktop/Light_8_Fish_Ventral_DLC/L8FV-Ben-2020-07-04/config.yaml'

new_video_path = cwd+'/New_Videos'

for file in os.listdir(new_video_path):
	if file.endswith(".mp4"):
		full_file = new_video_path+'/'+file
		print(full_file)
		deeplabcut.add_new_videos(config_path, [full_file], copy_videos=True)
