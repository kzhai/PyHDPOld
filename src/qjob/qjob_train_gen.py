import os
import sys

cn='ap-50-2K-10-700-100'
#cn='denews-50-4K-10-3200-30'
#cn='nips-150-15K-50-1500-500'
#cn='pnas-100-10K-50-1K-20'

def generate_train_script():
    project_directory = os.path.abspath(sys.argv[1]);
    count = int(sys.argv[2]);

    for train_iter in [1000]:
        for (alpha_alpha, alpha_gamma, alpha_eta) in [(0.001, 0.001, 0), (0.00001, 0.00001, 0), (0.001, 0.001, 0.001), (0.00001, 0.00001, 0.001)]:
            for (smh, sp, mp) in [(-1,0,0), (0,0,0), (1,0,0), (1,1,1), (1,2,0)]:
                count += 1;
                
                parameterStrings="%s-train-%d" % (cn, count)
    
                input_stream = open(os.path.join(project_directory, 'src', 'qjob', 'qjob_train.sh'), 'r');
                output_stream = open(os.path.join(project_directory, 'qjob', cn+"-train-"+str(count)+".sh"), 'w');
    
                for line in input_stream:
                    line = line.rstrip();
                    if line.startswith("SET_PARAMETER"):
                        output_stream.write("CorpusName=" + cn + "\n");

                        output_stream.write("TrainingIteration=" + str(train_iter) + "\n");
                        output_stream.write("AlphaGamma=" + str(alpha_gamma) + "\n");
                        output_stream.write("AlphaAlpha=" + str(alpha_alpha) + "\n");
                        output_stream.write("AlphaEta=" + str(alpha_eta) + "\n");

                        output_stream.write("SMH=" + str(smh) + "\n");
                        output_stream.write("SP=" + str(sp) + "\n");
                        output_stream.write("MP=" + str(mp) + "\n");

                        if smh<0:
                            output_stream.write("PostFix=\n");
                        elif smh==0:
                            output_stream.write("PostFix=-smh%d\n" % (smh));
                        else:
                            output_stream.write("PostFix=-smh%d-sp%d-mp%d\n" % (smh, sp, mp));
                            
                        continue;
                                
                    if line.startswith("#PBS -N"):
                        line += " " + parameterStrings;
    
                    output_stream.write(line + "\n");

if __name__ == '__main__':
    generate_train_script();
