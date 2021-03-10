import deeplabcut
import os, sys
import shutil
import h5py
import numpy as np

iteration = 3

inference_cfg_text = """
addlikelihoods: 0.15
averagescore: 0.1
boundingboxslack: {bbs}
detectionthresholdsquare: 0
distnormalization: 240.18
distnormalizationLOWER: 0
iou_threshold: {it}
lowerbound_factor: 0.5
max_age: 100
method: m1
min_hits: {mh}
minimalnumberofconnections: 1
pafthreshold: 0.07080893421472742
topktoretain: 9
upperbound_factor: 1.25
variant: 0
withid: false"""

inference_cfg_path = "/Users/Ben/Desktop/Light_8_Fish_Ventral_DLC/L8FV-Ben-2020-07-04/dlc-models/iteration-{}/L8FVJul4-trainset95shuffle1/test/inference_cfg.yaml".format(iteration)
config_path = '/Users/Ben/Desktop/Light_8_Fish_Ventral_DLC/L8FV-Ben-2020-07-04/config.yaml'

main_video_folder_path = '/Users/Ben/Desktop/Light_8_Fish_Ventral_DLC/L8FV-Ben-2020-07-04/videos/iteration-3/'
test_video_folder_path = '/Users/Ben/Desktop/Light_8_Fish_Ventral_DLC/L8FV-Ben-2020-07-04/videos/iteration-3/testing_cfg/{bbs}_{it}_{mh}_test/'

prob_cutoff = 0.9

og_video_files = ["2020_7_28_29_TN_DN_F2_V1.mp4","2020_6_29_20_TN_DN_F0_V.mp4","2020_6_29_1_TN_DN_F0_V.mp4","2020_6_29_14_TN_DN_F2_V.mp4","2020_7_28_3_TN_DN_F2_V1.mp4"]
og_pickle_files = ["2020_6_29_1_TN_DN_F0_VDLC_resnet50_L8FVJul4shuffle1_30000_full.pickle","2020_6_29_1_TN_DN_F0_VDLC_resnet50_L8FVJul4shuffle1_30000_meta.pickle",
                   "2020_6_29_20_TN_DN_F0_VDLC_resnet50_L8FVJul4shuffle1_30000_full.pickle","2020_6_29_20_TN_DN_F0_VDLC_resnet50_L8FVJul4shuffle1_30000_meta.pickle",
                   "2020_7_28_29_TN_DN_F2_V1DLC_resnet50_L8FVJul4shuffle1_30000_full.pickle","2020_7_28_29_TN_DN_F2_V1DLC_resnet50_L8FVJul4shuffle1_30000_meta.pickle",
                   "2020_6_29_14_TN_DN_F2_VDLC_resnet50_L8FVJul4shuffle1_30000_full.pickle","2020_6_29_14_TN_DN_F2_VDLC_resnet50_L8FVJul4shuffle1_30000_meta.pickle",
                   "2020_7_28_3_TN_DN_F2_V1DLC_resnet50_L8FVJul4shuffle1_30000_full.pickle","2020_7_28_3_TN_DN_F2_V1DLC_resnet50_L8FVJul4shuffle1_30000_meta.pickle"]


def Get_Percent_Good(coords):
    bbs,it,mh = coords[0],coords[1],coords[2]

    current_video_path = test_video_folder_path.format(bbs=bbs,it=it,mh=mh)

    #Writes a new file here for the future labeling and things to use
    i_cfg_file = open(inference_cfg_path,"w")
    i_cfg_file.write(inference_cfg_text.format(bbs=bbs,it=it,mh=mh))
    i_cfg_file.close()

    #Creates the folder and videos
    if not os.path.isdir(current_video_path):
        os.mkdir(current_video_path)

        for v in og_video_files:
            shutil.copy2(main_video_folder_path+v,current_video_path+v)

        for p in og_pickle_files:
            shutil.copy2(main_video_folder_path+p,current_video_path+p)

    #analyze video
    #No longer needed since they are the same each time
    #deeplabcut.analyze_videos(config_path,[current_video_path], videotype='.mp4', save_as_csv = True)

    #convert to tracklets
    deeplabcut.convert_detections2tracklets(config_path, [current_video_path], videotype='mp4', shuffle=1, trainingsetindex=0, track_method='box')

    video_files = os.listdir(current_video_path)

    #Convert to h5s
    for file in video_files:
        if file.endswith("_bx.pickle"):
            h5_name = file.replace("pickle","h5")
            video_name = file.replace

            if h5_name in video_files:
                print("h5 for {} already exists!".format(file))
            else:
                print("Converting {} to h5!".format(file))
                deeplabcut.convert_raw_tracks_to_h5(config_path, current_video_path+file)

    #Need to do this each time to update with new videos
    video_files = os.listdir(current_video_path)
    #fix the messy bits in the files so all points exist
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

            hf = h5py.File(current_video_path+file, 'r+')
            table = hf['df_with_missing/table']

            n_table = np.array(table)
            new_table = n_table

            for i in range(n_table.shape[0]):
                dlc_data = n_table[i][1]
                dlc_data = np.nan_to_num(dlc_data)

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

            table[...] = new_table
            hf.close()

    video_files = os.listdir(current_video_path)
    #Now create the filtered videos and CSVs
    for file in video_files:
        if file.endswith(".mp4"):

            print(file)
        
            deeplabcut.filterpredictions(config_path,[current_video_path+file], videotype='mp4', shuffle=1, track_method = 'box')
            deeplabcut.create_labeled_video(config_path, [current_video_path+file],videotype='.mp4',shuffle=1,draw_skeleton=True,track_method='box',save_frames=False,filtered=True)


    video_files = os.listdir(current_video_path)
    #Now finally get the percent of "good" points
    files = []
    good_percents = []

    for file_name in video_files:
        if file_name.endswith(".csv"):

            in_csv = open(current_video_path+file_name,"r")

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

                else:
                    skip_lines += 1

                    lis = line.split(",")[:-24]

            files.append(file_name)
            good_percents.append(round(good_points/total_points*100,2))

            in_csv.close()
            #out_csv.close()

    files = np.asarray(files)
    good_percents = np.asarray(good_percents)

    sort_percent = np.flip(good_percents[np.argsort(good_percents)])
    sorted_files = np.flip(files[np.argsort(good_percents)])

    out_file = open(current_video_path+"PGV_{bbs}_{it}_{mh}.txt".format(bbs=bbs,it=it,mh=mh),"w")

    for i in range(len(sorted_files)):
        out_file.write("{} : {}% good points\n".format(sorted_files[i][:22],sort_percent[i]))
        print("{} : {}% good points".format(sorted_files[i][:22],sort_percent[i]))

    out_file.close()

    score = np.sum((100-good_percents)**2)

    main_csv = open(main_video_folder_path+"PGV_all.csv".format(bbs=bbs,it=it,mh=mh),"a")
    main_csv.write("{bbs}_{it}_{mh},{bbs},{it},{mh},{v0},{v1},{v2},{v3},{v4},{m},{score}\n".format(bbs=bbs,it=it,mh=mh,v0=good_percents[0],v1=good_percents[1],v2=good_percents[2],v3=good_percents[3],v4=good_percents[4],m=np.mean(good_percents),score=score))
    main_csv.close()

    shutil.rmtree(current_video_path)

    return(score)

lr = [3,0.01,1]
multi = [-1,0,1]
min_diff = 1
runtime = 25
num_start_points = 10
num_nearby_points = 10

point_mins = [20,0,1]
points_max = [50,0.1,5]

# ~30s per video

def getNewPoints(cords):

    newPoints = []

    for i in range(3):
        for j in range(3):
            for k in range(3):
                dirs = [lr[0]*multi[i],lr[1]*multi[j],lr[2]*multi[k]]
                newPoint = cords+dirs

                if np.all(newPoint >= point_mins) and np.all(newPoint <= points_max):
                    newPoints.append(cords+dirs)

    return_points = np.asarray(newPoints)

    return return_points[np.random.choice(return_points.shape[0], min(num_nearby_points,return_points.shape[0]), replace=False), :]

start_points = np.random.rand(num_start_points,3)

#boundingboxslacks from 20 to 50
#iou_thresholds from 0 to 0.01
#min_hits from 1 to 4

start_points[:,0] = start_points[:,0]*30 + 20
start_points[:,1] = start_points[:,1]*0.1
start_points[:,2] = start_points[:,1]*4 + 1

global_mins = np.zeros(num_start_points)
global_points = np.zeros((num_start_points,3))

for g,sp in enumerate(start_points):

    print("##############################\n###           {g}            ###\n##############################".format(g=g))

    for i in range(runtime):
        start_val = Get_Percent_Good(sp)
        search_points = getNewPoints(sp)

        search_vals = []

        for search in search_points:
            search_vals.append(Get_Percent_Good(search))

        search_vals = np.asarray(search_vals)

        if np.min(search_vals) < start_val and abs(start_val - np.min(search_vals)) > min_diff:
            sp = search_points[np.argmin(search_vals)]
        else:
            global_mins[g] = start_val
            global_points[g] = sp
            break

        if i == runtime-1:
            global_mins[g] = start_val
            global_points[g] = sp
            break

global_min = np.min(global_mins)
global_min_point = global_points[np.argmin(global_mins)]

print(global_mins)
print(global_min)
print(global_min_point)
