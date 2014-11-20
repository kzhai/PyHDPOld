#!/usr/bin/env Rscript

# ==========
# ==========
# ==========

# load ggplot2 library
library(ggplot2)
library(grid)
library(scales)

#file_name="140806-165121-log-likelihood"
file_name="140912-100218-log-likelihood"

#project="/windows/d/Workspace/OnlineLDA/"
project_home="/Users/student/Workspace/PyHDP/"

input_directory=paste(project_home, "result/", sep="");
output_directory=paste(project_home, "figure/", sep="");

input_file=paste(input_directory, file_name, ".csv", sep="");
output_file=paste(output_directory, file_name, ".pdf", sep="");

pdf(width=6, height=2.8)

# load in csv data
input_data <- read.csv(input_file)

true_input_data <- subset(input_data, input_data$inference=='null' & input_data$iteration=100000)$topic;

#input_data <- subset(input_data, input_data$iteration>=1 & input_data$iteration<=1000)
input_data <- subset(input_data, input_data$inference!='null')
input_data <- subset(input_data, input_data$iteration %% 100==0)

plot_pic <- ggplot() +
	#geom_line(data=input_data, aes(x=iteration, y=likelihood, color=inference, group=inference), alpha=0.75, size=1) +
	geom_smooth(data=input_data, aes(x=iteration, y=topic, color=inference, group=inference), alpha=0.75, size=1) +
	geom_hline(yintercept=true_input_data) +
	labs(x="Iteration", y="Log-Likelihood") +
	
	#guides(colour = guide_legend(nrow = 3)) +
	
	theme(legend.margin = unit(0, "line")) +
	theme(legend.key.height=unit(1,"line")) +
	theme(legend.position="bottom") +
	theme(legend.title = element_text(size = 0, angle = 90), legend.text = element_text(size = 13)) +
	
	#coord_cartesian(ylim=c(-2.4,-2.25)) +
    #coord_cartesian(xlim=c(100,1000)) +
	#scale_y_continuous(breaks = round(seq(-6000, -4000, by=1000), 1)) +
	#scale_x_continuous(breaks = round(seq(200, 1000, by=200), 1)) + 
	
	#theme(legend.direction='vertical', legend.box='vertical', legend.position = c(1, 0)) +
	#theme(legend.direction='vertical', legend.box='vertical', legend.position = c(0, 0), legend.justification = c(0, 1)) +
	theme(axis.text.x = element_text(size = 15, colour = "black")) +
	theme(axis.text.y = element_text(size = 15, colour = "black")) +
	theme(axis.title.x = element_text(size = 15), axis.title.y = element_text(size = 15, angle = 90)) +
	theme(strip.text.x = element_text(size = 15), strip.text.y = element_text(size = 15, angle = 45))
					
ggsave(plot_pic,filename=output_file);
