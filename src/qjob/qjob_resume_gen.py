import os
import sys

#cn='ap-50-2K-10-700-100'
#cn='denews-50-4K-10-3200-30'
#cn='nips-150-15K-50-1500-500'
cn='pnas-100-10K-50-1K-20'

def generate_train_script():
    output_directory = "/fs/clip-scratch/zhaike/PyHDP/output/"

    project_directory = os.path.abspath(sys.argv[1]);
    count = int(sys.argv[2]);

    output_dataset_directory = os.path.join(output_directory, cn);
    for model_settings in os.listdir(output_dataset_directory):
        model_directory = os.path.join(output_dataset_directory, model_settings);
        if os.path.isfile(model_directory):
            continue;

        count += 1;
                    
        parameterStrings="%s-%d" % (cn, count)
    
        input_stream = open(os.path.join(project_directory, 'src', 'qjob', 'qjob_resume.sh'), 'r');
        output_stream = open(os.path.join(project_directory, 'qjob', cn+"-resume-"+str(count)+".sh"), 'w');
    
        for line in input_stream:
            line = line.rstrip();
            if line.startswith("SET_PARAMETER"):
                output_stream.write("CorpusName=" + cn + "\n");
                output_stream.write("ModelSetting=" + str(alpha_gamma) + "\n");
                output_stream.write("TrainingIteration=" + str(train_iter) + "\n");

#                output_stream.write("AlphaAlpha=" + str(alpha_alpha) + "\n");

#                output_stream.write("SMH=" + str(smh) + "\n");
#                output_stream.write("SP=" + str(sp) + "\n");
#                output_stream.write("MP=" + str(mp) + "\n");

#                if smh<0:
#                    output_stream.write("PostFix=\n");
#                elif smh==0:
#                    output_stream.write("PostFix=-smh%d\n" % (smh));
#                else:
#                    output_stream.write("PostFix=-smh%d-sp%d-mp%d\n" % (smh, sp, mp));
    
                continue;
                                
            if line.startswith("#PBS -N"):
                line += " " + parameterStrings;
                
            output_stream.write(line + "\n");

if __name__ == '__main__':
    generate_train_script();
