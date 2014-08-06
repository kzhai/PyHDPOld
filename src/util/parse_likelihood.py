"""
@author: Ke Zhai (zhaike@cs.umd.edu)
"""
import numpy
import optparse;
import os
import re;
import sys
import matplotlib
import matplotlib.pyplot

split_merge_proposal_pattern = re.compile('nohup.smh(?P<smh>[\d]+).sp(?P<sp>[\d]+).mp(?P<mp>[\d]+).out');
split_merge_heuristics_pattern = re.compile('nohup.smh(?P<smh>[\d]+).out');
no_split_merge_pattern = re.compile('nohup.out');

def parse_args():
    parser = optparse.OptionParser()
    parser.set_defaults(# parameter set 1
                        input_directory=None,
                        # dataset_name=None,
                        
                        output_file=None,
                        )
    # parameter set 1
    parser.add_option("--input_directory", type="string", dest="input_directory",
                      help="input directory [None]");
    # parser.add_option("--dataset_name", type="string", dest="dataset_name",
                      # help="the corpus name [None]");
                      
    parser.add_option("--output_file", type="string", dest="output_file",
                      help="output file [None]");                      
    #parser.add_option("--snapshot_index", type="int", dest="snapshot_index",
                      #help="snapshot index [-1]");
                      
    (options, args) = parser.parse_args();
    return options;

def main():
    options = parse_args();
    
    # parameter set 1
    # assert(options.dataset_name!=None);
    assert(options.input_directory != None);
    assert(options.output_file != None);
    
    # dataset_name = options.dataset_name;
    # input_directory = options.input_directory;
    # input_directory = os.path.join(input_directory, dataset_name);
    
    input_directory = options.input_directory;
    input_directory = input_directory.rstrip("/");
    dataset_name = os.path.basename(input_directory);
    
    # extract information from nohup.out files
    output_file = options.output_file;
    output_file_stream = open(output_file, "w");
    output_file_stream.write("%s,%s,%s,%s,%s\n" % ("inference", "iteration", "duration", "topic", "likelihood"));
    
    for file_name in os.listdir(input_directory):
        input_file_path = os.path.join(input_directory, file_name);
        if os.path.isdir(input_file_path):
            continue;
        
        inference = None;
        if re.match(split_merge_proposal_pattern, file_name):
            m = re.match(split_merge_proposal_pattern, file_name);
            if int(m.group("sp")) == 0 and int(m.group("mp")) == 0:
                inference = "random";
            elif int(m.group("sp")) == 1 and int(m.group("mp")) == 1:
                inference = "res_gibbs";
            elif int(m.group("sp")) == 2 and int(m.group("mp")) == 0:
                inference = "seq_alloc";
            else:
                print "unrecognized model patterns %s" % file_name
                continue;
        elif re.match(split_merge_heuristics_pattern, file_name):
            m = re.match(split_merge_heuristics_pattern, file_name);
            if int(m.group("smh")) == 0:
                inference = "com_resam"
            elif int(m.group("smh")) == 1:
                inference = "random"
            else:
                print "unrecognized model patterns %s" % file_name
                continue;
        elif re.match(no_split_merge_pattern, file_name):
            inference = "null";
        else:
            print "unrecognized model patterns %s" % file_name
            continue;
        
        input_file_stream = open(input_file_path, 'r');
        for line in input_file_stream:
            if not line.startswith("training iteration"):
                continue;
            
            # training iteration [d+] finished in [d.]+ seconds: number-of-topics = [d]+, log-likelihood = [d]+
            line = line.strip();
            tokens = line.split();
            tokens[9] = tokens[9].strip(",")
            
            iteration_count = int(tokens[2]);
            iteration_time = float(tokens[5]);
            number_of_topics = int(tokens[9]);
            log_likelihood = float(tokens[12]);
            
            output_file_stream.write("%s,%d,%g,%d,%g\n" % (inference, iteration_count, iteration_time, number_of_topics, log_likelihood));
        
        print "successfully parsed output %s..." % (file_name);
        
if __name__ == '__main__':
    main()
