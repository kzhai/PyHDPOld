import os
import sys
import re

#cn='ap-50-2K-10-700-100'
#cn='denews-50-4K-10-3200-30'
#cn='nips-150-15K-50-1500-500'
#cn='pnas-100-10K-50-1K-20'
cn='denews-25-set1'
#cn='ap-50-set2'
#cn='ap-50-set3'

model_settings_pattern = re.compile('\d+-\d+-hdp-I(?P<iteration>\d+)-S(?P<snapshot>\d+)-aa(?P<alpha>[\d\.]+)-ag(?P<gamma>[\d\.]+)-ae(?P<eta>[\d\.]+)((?P<postfix>.+))?');

def generate_train_script():
    output_directory = "/fs/clip-scratch/zhaike/PyHDP/output";
    
    project_directory = os.path.abspath(sys.argv[1]);
    count = int(sys.argv[2]);
    
    snapshot_index = int(sys.argv[3]);
    training_iteration = int(sys.argv[4]);

    output_dataset_directory = os.path.join(output_directory, cn);
    for model_settings in os.listdir(output_dataset_directory):
        model_directory = os.path.join(output_dataset_directory, model_settings);
        if os.path.isfile(model_directory):
            continue;

        matches = re.match(model_settings_pattern, model_settings);
        alpha_alpha = float(matches.group('alpha'));
        alpha_gamma = float(matches.group('gamma'));
        alpha_eta = float(matches.group('eta'));
        model_postfix = matches.group('postfix');
        if model_postfix==None:
            model_postfix = "";
        
        count += 1;
        
        parameterStrings="%s-%d" % (cn, count)
        
        input_stream = open(os.path.join(project_directory, 'src', 'qjob', 'qjob_resume.sh'), 'r');
        output_stream = open(os.path.join(project_directory, 'qjob', cn+"-resume-"+str(count)+".sh"), 'w');
        
        for line in input_stream:
            line = line.rstrip();
            if line.startswith("SET_PARAMETER"):
                output_stream.write("CorpusName=" + cn + "\n");
                output_stream.write("ModelSetting=" + model_settings + "\n");
                output_stream.write("TrainingIteration=" + str(training_iteration) + "\n");
                output_stream.write("SnapshotIndex=" + str(snapshot_index) + "\n")

                output_stream.write("AlphaAlpha=" + str(alpha_alpha) + "\n");
                output_stream.write("AlphaGamma=" + str(alpha_gamma) + "\n");
                output_stream.write("AlphaEta=" + str(alpha_eta) + "\n");
                
                output_stream.write("PostFix=" + model_postfix + "\n");

                continue;
                                
            if line.startswith("#PBS -N"):
                line += " " + parameterStrings;
                
            output_stream.write(line + "\n");

if __name__ == '__main__':
    generate_train_script();
