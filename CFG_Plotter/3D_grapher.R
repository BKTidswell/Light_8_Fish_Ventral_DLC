library(dplyr)
library(ggplot2)
library(plotly)

cfg_data_3D <- read.csv("PGV_all_2P.csv")

min(cfg_data_3D$Score)
max(cfg_data_3D$Score)

cfg_data_3D <- cfg_data_3D %>% filter(Score < 1500)

fig <- plot_ly(cfg_data_3D, x = ~BBS, y = ~IOUT, z = ~MINHITS,
               type = 'scatter3d', mode = 'markers',
               marker = list(color = ~Score, colorscale = "Portland", showscale = TRUE))
fig <- fig %>% layout(scene = list(xaxis = list(title = 'BBS'),
                                   yaxis = list(title = 'IOUT'),
                                   zaxis = list(title = 'MINHITS')))
fig

