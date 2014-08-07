#!/usr/bin/env Rscript

# ==========
# ==========
# ==========

# load ggplot2 library
library(ggplot2)
library(grid)
library(scales)

#file_name="140730-224558-cluster-counts"
file_name="140806-165121-cluster-counts"

project_home="/windows/d/Workspace/PyHDP/"
#project_home="/Users/kezhai/Workspace/PyHDP/"

input_directory=paste(project_home, "result/", sep="");
output_directory=paste(project_home, "figure/", sep="");

input_file=paste(input_directory, file_name, ".csv", sep="");
output_file=paste(output_directory, file_name, ".pdf", sep="");

pdf(width=10, height=10)

# load in csv data
input_data <- read.csv(input_file)

plot_pic <- qplot(vocabulary, factor(topic), data=input_data, size=count) +
	facet_grid(inference ~ .) +
	labs(size="Points\nCount") +
	labs(x="Vocabulary", y="Topic") +

	#scale_x_log10() +

	theme(legend.margin = unit(0, "line")) +
	theme(legend.key.height=unit(1.5,"line")) +
	theme(legend.position="bottom") +
	theme(legend.title = element_text(size = 12, angle = 0), legend.text = element_text(size = 15)) +
	guides(colour = guide_legend(nrow = 1)) +
		
	#coord_cartesian(xlim=c(0, 10, 100)) +
	#coord_cartesian(ylim=c(0, 13, 2)) +
	#scale_y_continuous(breaks = round(seq(-6000, -4000, by=1000), 1)) +
	#scale_x_continuous(breaks = round(seq(1, 500, by=100), 1)) + 
	
	#theme(legend.direction='vertical', legend.box='vertical', legend.position = c(1, 0)) +
	#theme(legend.direction='vertical', legend.box='vertical', legend.position = c(0, 0), legend.justification = c(0, 1)) +
	theme(axis.text.x = element_text(size = 15, colour = "black")) +
	theme(axis.text.y = element_text(size = 15, colour = "black")) +
	theme(axis.title.x = element_text(size = 12), axis.title.y = element_text(size = 15, angle = 90)) +
	theme(strip.text.x = element_text(size = 15), strip.text.y = element_text(size = 15, angle = 0))
					
ggsave(plot_pic,filename=output_file);
