library(tidyverse)
library(ggplot2)

data <- read.csv("all_vid_out.csv")

model <- aov(score ~ factor(points), data = data)
summary(model)
TukeyHSD(model)

ggplot(data, aes(x = factor(points), y = score, color = factor(points)))+
  geom_boxplot()
