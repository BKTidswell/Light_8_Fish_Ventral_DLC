import os

f = open("all_vid_out.csv", "w")

csv_str = "{vid_name},{points},{score}\n"

file_format = "PGV_all_vids_{p}P.txt"

points = [3,4,6]

f.write(csv_str.format(vid_name="name",points="points",score="score"))

for p in points:
    read_file = open("PGV_all_vids_{p}P.txt".format(p=p),"r")
    lines = read_file.readlines()

    for line in lines:
        if line[0] != "P":
            split = line.replace(" :",",").replace("%",",").split(",")
            f.write(csv_str.format(vid_name=split[0],points=p,score=float(split[1])))
