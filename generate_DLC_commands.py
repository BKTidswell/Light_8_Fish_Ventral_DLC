import os

new_project = False

cwd = os.getcwd()

print("This will only work properly if:\n1. This is placed in the same directory where the DLC folder is.\n2. Your starting videos are placed in a folder labled 'Initial_Videos'")
print("The later commands generate videos from one of the outputs, so must be replaced manually")

if new_project:
	DLC_name = input('Enter the DLC folder name:')
	iter_num = 0
else:
	DLC_name = "L8FV-Ben-2020-07-04"
	iter_num = 1

cluster = "/cluster/tufts/tytelllab/btidsw01/Light_8_Fish_Ventral_DLC"

initial_videos = "{dir}/Initial_Videos".format(dir=cwd)
config_path = "{dir}/{DLC}/config.yaml".format(dir=cwd,DLC=DLC_name)
video_path = "{dir}/{DLC}/videos/iteration-{num}".format(dir=cwd,DLC=DLC_name,num=iter_num)

cluster_config_path = "{dir}/{DLC}/config.yaml".format(dir=cluster,DLC=DLC_name)
cluster_video_path = "{dir}/{DLC}/videos/iteration-{num}".format(dir=cluster,DLC=DLC_name,num=iter_num)

videofiles = [f for f in os.listdir(initial_videos) if os.path.isfile(os.path.join(initial_videos, f))]
videoStr = ""

for v in videofiles:
	videoStr += "'Initial_Videos/{vf}',".format(vf=v)

outStr = "You can find detailed instructions at: https://github.com/DeepLabCut/DeepLabCut/blob/master/docs/UseOverviewGuide.md \n\n\n"
outStr += "#### Restarting  Session ####\n\n"
outStr += "conda activate DLC-CPU\n\n"
outStr += "pythonw\n\n"
outStr += "import deeplabcut\n\n"
outStr += "config_path = '{cp}'\n\n".format(cp=config_path)
outStr += "#############################\n\n\n"
outStr += "###### Getting Started ######\n\n"
outStr += "conda activate DLC-CPU\n\n"
outStr += "pythonw\n\n"
outStr += "import deeplabcut\n\n"
outStr += "deeplabcut.create_new_project('{name}','Ben', [{vf}], videotype='.mp4', copy_videos=True, multianimal=True)\n\n".format(name = DLC_name, vf=initial_videos)
outStr += "config_path = '{cp}'\n\n".format(cp=config_path)
outStr += "deeplabcut.extract_frames(config_path, mode='automatic', algo='kmeans', crop=False)\n\n"
outStr += "#############################\n## CONFIGURE SKELETON HERE ##\n#############################\n\n"
outStr += "deeplabcut.label_frames(config_path)\n\n"
outStr += "deeplabcut.check_labels(config_path)\n\n"
outStr += "deeplabcut.cropimagesandlabels(config_path)\n\n"
outStr += "deeplabcut.create_multianimaltraining_dataset(config_path)\n\n"
outStr += "deeplabcut.train_network(config_path, shuffle=1, trainingsetindex=0, gputouse=None, max_snapshots_to_keep=5, autotune=False, displayiters=100, saveiters=100, maxiters=3000, allow_growth=True)\n\n"
outStr += "deeplabcut.evaluate_network(config_path, plotting=True)\n\n"
outStr += "deeplabcut.evaluate_multianimal_crossvalidate(config_path, Shuffles=[1], edgewisecondition=True, leastbpts=1, init_points=20, n_iter=50)\n\n"
outStr += "deeplabcut.extract_save_all_maps(config_path, shuffle=1, Indices=[0, 5])\n\n"
outStr += "deeplabcut.analyze_videos(config_path,['{vp}'], videotype='.mp4', save_as_csv = True)\n\n".format(vp=video_path)
outStr += "deeplabcut.convert_detections2tracklets(config_path, ['{vp}'], videotype='mp4', shuffle=1, trainingsetindex=0, track_method='box')\n\n".format(vp=video_path)
outStr += "deeplabcut.refine_tracklets(config_path, '{vp}/TRAINED_OUTPUT_bx.h5', '{vp}/VIDEO_NAME.mp4', trail_len=50)\n\n".format(vp=video_path)
outStr += "deeplabcut.filterpredictions(config_path,['{vp}/VIDEO_NAME.mp4'], videotype='mp4', shuffle=1, track_method = 'box')\n\n".format(vp=video_path)
outStr += "deeplabcut.plot_trajectories(config_path, ['{vp}/VIDEO_NAME.mp4'],track_method='box',filtered=True)\n\n".format(vp=video_path)
outStr += "deeplabcut.create_labeled_video(config_path, ['{vp}/VIDEO_NAME.mp4'],videotype='.mp4',shuffle=1,draw_skeleton=True,track_method='box',save_frames=False,filtered=True)\n\n\n".format(vp=video_path)
outStr += "### Optional: Learning ####\n\n"
outStr += "deeplabcut.extract_outlier_frames(config_path,['{vp}/VIDEO_NAME.mp4'],shuffle=1,videotype='.mp4',track_method ='box',outlieralgorithm='uncertain',p_bound = 0.5)\n\n""".format(vp=video_path)
outStr += "outlieralgorithm: 'fitting', 'jump', or 'uncertain'``\n\n\n"
outStr += "deeplabcut.refine_labels(config_path)\n\n"
outStr += "deeplabcut.merge_datasets(config_path)\n\n"
outStr += "deeplabcut.create_training_dataset(config_path)\n\n"
outStr += "deeplabcut.train_network(config_path)\n\n"

outStr += "#############################\n"
outStr += """CLUSTER COMMANDS \n
CLUSTER CONFIG PATH:
config_path = '{config}'

CLUSTER LOGIN
ssh -Y btidsw01@login.cluster.tufts.edu
srun -p gpu -n 4 --mem=32g --time=0-8:00:00 --x11=first --pty bash
module load anaconda
cd /cluster/tufts/tytelllab/deeplabcut
source activate_dlc

Check on Job:
squeue -u btidsw01

Start Job:
sbatch train_L8FV_Ben_2020_08_20

SED TO CONVERT FILE PATHS FOR CLUSTER

cd {cluster}{DLC}

sed -i 's|{dir}|{cluster}|g' config.yaml

In DLC Models:

sed -i 's|{dir}|{cluster}|g' pose_cfg.yaml

cd {cluster}/{DLC}/dlc-models/iteration-{num}/L8FVJul4-trainset95shuffle1/test

cd {cluster}/{DLC}/dlc-models/iteration-{num}/L8FVJul4-trainset95shuffle1/train

REPLACE USER MODEL WITH CLUSTER ONE

sed -i 's|/Users/Ben/anaconda2/envs/DLC-CPU/lib/python3.7/site-packages/deeplabcut/pose_estimation_tensorflow/models/pretrained/resnet_v1_50.ckpt|/cluster/tufts/tytelllab/deeplabcut/pipinstall/lib/python3.6/site-packages/deeplabcut/pose_estimation_tensorflow/models/pretrained/resnet_v1_50.ckpt|g' pose_cfg.yaml

REPLACE CLUSTER MODEL WITH PREVIOUS TRAINED ONE

sed -i 's|/cluster/tufts/tytelllab/deeplabcut/pipinstall/lib/python3.6/site-packages/deeplabcut/pose_estimation_tensorflow/models/pretrained/resnet_v1_50.ckpt|g' pose_cfg.yaml|{cluster}/{DLC}/dlc-models/iteration-{prev}/L8FVJul4-trainset95shuffle1/train/snapshot-100000|g' pose_cfg.yaml

""".format(config = cluster_config_path, cluster = cluster, num = iter_num, dir = cwd, DLC = DLC_name, prev = iter_num-1)

outStr += "#############################\n\n\n"

outStr += """Config.yaml Fish Parts: \n\n
individuals:
- individual1
- individual2
- individual3
- individual4
- individual5
- individual6
- individual7
- individual8
uniquebodyparts: [topbackleftP,topbackrightP,topfrontleftP,topfrontrightP,bottombackleftP,bottombackrightP,bottomfrontleftP,bottomfrontrightP]
multianimalbodyparts:
- head
- tailbase
- midline2
- midline1
- midline3
- tailtip
skeleton:
- - head
  - midline1
- - midline1
  - midline2
- - midline2
  - midline3
- - midline3
  - tailbase
- - tailbase
  - tailtip \n\n\n"""


f = open("DLC_Commands.txt", "w")
f.write(outStr)
print("Commands Written")
f.close()