
library(dplyr)
library(ggplot2)

cfg_data <- read.csv("PGV_all.csv")

ggplot(cfg_data, aes(BBS, Mean))+
  geom_point()

ggplot(cfg_data, aes(log(IOUT), Mean))+
  geom_point()

unique(cfg_data$IOUT)
#Filter out IOUT >= 5e-02

ggplot(cfg_data, aes(MINHITS, Mean))+
  geom_point()

cfg_no_mh <- cfg_data %>% group_by(BBS,IOUT) %>% summarize(MMean = mean(Mean))

ggplot(cfg_no_mh, aes(BBS, IOUT, z = MMean))+
  geom_contour_filled()

cfg_data[which.max(cfg_data$Mean),]
#Time to explore higher BBS and lower IOUT

