"""
@author: Ke Zhai (zhaike@cs.umd.edu)
"""
import numpy
import nltk;
import optparse;
import os
import re
import sys
import matplotlib
import matplotlib.pyplot

split_merge_proposal_pattern = re.compile('\d+-\d+-hdp-I(?P<iteration>\d+)-S(?P<snapshot>\d+)-aa(?P<alpha>[\d.]+)-ag(?P<gamma>[\d.]+)-ae(?P<eta>[\d.]+)-smh(?P<smh>[\d]+)-sp(?P<sp>[\d]+)-mp(?P<mp>[\d]+)');
split_merge_heuristics_pattern = re.compile('\d+-\d+-hdp-I(?P<iteration>\d+)-S(?P<snapshot>\d+)-aa(?P<alpha>[\d.]+)-ag(?P<gamma>[\d.]+)-ae(?P<eta>[\d.]+)-smh(?P<smh>[\d]+)');
no_split_merge_pattern = re.compile('\d+-\d+-hdp-I(?P<iteration>\d+)-S(?P<snapshot>\d+)-aa(?P<alpha>[\d.]+)-ag(?P<gamma>[\d.]+)-ae(?P<eta>[\d.]+)');

def parse_args():
    parser = optparse.OptionParser()
    parser.set_defaults(# parameter set 1
                        input_directory=None,
                        output_directory=None,
                        output_file=None,
                        snapshot_index=500
                        )
    # parameter set 1
    parser.add_option("--input_directory", type="string", dest="input_directory",
                      help="input directory [None]");
    parser.add_option("--output_directory", type="string", dest="output_directory",
                      help="the output directory [None]");
                      
    parser.add_option("--output_file", type="string", dest="output_file",
                      help="output directory [None]");                      
    parser.add_option("--snapshot_index", type="int", dest="snapshot_index",
                      help="snapshot index [-1]");
                      
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
    
    input_voc_file = os.path.join(input_directory, "voc.dat");
    input_voc_file_stream = open(input_voc_file, 'r');
    topic_indices = set();
    vocab_mapping = {};
    for line in input_voc_file_stream:
        line = line.strip();
        vocab_mapping[line] = len(vocab_mapping);
        line = line.strip("k");
        tokens = line.split('v');
        topic_index = int(tokens[0]);
        topic_indices.add(topic_index);
    
    input_doc_file = os.path.join(input_directory, "doc.dat");
    input_doc_file_stream = open(input_doc_file, 'r');
    n_kv = numpy.zeros((len(topic_indices), len(vocab_mapping)));
    for line in input_doc_file_stream:
        line = line.strip();
        tokens = line.split();
        for token in tokens:
            vocab_index = vocab_mapping[token];
            topic_index = int(token.strip("k").split("v")[0]);
            n_kv[topic_index, vocab_index] += 1;

    # extract information from nohup.out files
    output_file = options.output_file;
    output_file_stream = open(output_file, "w");
    
    (number_of_rows, number_of_columns) = n_kv.shape;
    for topic_index in xrange(number_of_rows):
        for vocab_index in xrange(number_of_columns):
            output_file_stream.write("%s,%d,%d,%d\n" % ("true", topic_index + 1, vocab_index + 1, n_kv[topic_index, vocab_index]));    
    
    output_directory = options.output_directory;
    snapshot_index = options.snapshot_index;
    for model_name in os.listdir(output_directory):
        output_model_path = os.path.join(output_directory, model_name);
        if os.path.isfile(output_model_path):
            continue;
        
        inference = None;
        if re.match(split_merge_proposal_pattern, model_name):
            m = re.match(split_merge_proposal_pattern, model_name);
            if int(m.group("sp")) == 0 and int(m.group("mp")) == 0:
                inference = "random";
            elif int(m.group("sp")) == 1 and int(m.group("mp")) == 1:
                inference = "res_gibbs";
            elif int(m.group("sp")) == 2 and int(m.group("mp")) == 0:
                inference = "seq_alloc";
            else:
                print "unrecognized model patterns %s" % model_name
                continue;
        elif re.match(split_merge_heuristics_pattern, model_name):
            m = re.match(split_merge_heuristics_pattern, model_name);
            if int(m.group("smh")) == 0:
                inference = "com_resam"
            elif int(m.group("smh")) == 1:
                inference = "random"
            else:
                print "unrecognized model patterns %s" % model_name
                continue;
        elif re.match(no_split_merge_pattern, model_name):
            inference = "null";
        else:
            print "unrecognized model patterns %s" % model_name
            continue;

        snapshot_file = os.path.join(output_model_path, "n_kv-%d" % (snapshot_index));
        if not os.path.exists(snapshot_file):
            continue;
        
        n_kv = numpy.loadtxt(snapshot_file);
        (number_of_rows, number_of_columns) = n_kv.shape;
        for topic_index in xrange(number_of_rows):
            for vocab_index in xrange(number_of_columns):
                output_file_stream.write("%s,%d,%d,%d\n" % (inference, topic_index + 1, vocab_index + 1, n_kv[topic_index, vocab_index]));    
        
        print "successfully parsed output %s..." % (model_name);
        
if __name__ == '__main__':
    main()
