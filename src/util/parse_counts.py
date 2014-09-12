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

split_merge_proposal_pattern = re.compile('\d+-\d+-hdp-I(?P<iteration>\d+)-S(?P<snapshot>\d+)-aa(?P<alpha>[\d\.]+)-ag(?P<gamma>[\d\.]+)-ae(?P<eta>[\d\.]+)-smh(?P<smh>[\d]+)-sp(?P<sp>[\d]+)-mp(?P<mp>[\d]+)');
split_merge_heuristics_pattern = re.compile('\d+-\d+-hdp-I(?P<iteration>\d+)-S(?P<snapshot>\d+)-aa(?P<alpha>[\d\.]+)-ag(?P<gamma>[\d\.]+)-ae(?P<eta>[\d\.]+)-smh(?P<smh>[\d]+)');
no_split_merge_pattern = re.compile('\d+-\d+-hdp-I(?P<iteration>\d+)-S(?P<snapshot>\d+)-aa(?P<alpha>[\d\.]+)-ag(?P<gamma>[\d\.]+)-ae(?P<eta>[\d\.]+)');

def parse_args():
    parser = optparse.OptionParser()
    parser.set_defaults(# parameter set 1
                        input_directory=None,
                        output_directory=None,
                        output_file=None,
                        snapshot_index=1000
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

    '''
    input_voc_file = os.path.join(input_directory, "voc.dat");
    input_voc_file_stream = open(input_voc_file, 'r');
    vocab_set = set();
    for line in input_voc_file_stream:
        line = line.strip();
        vocab_set.add(int(line.strip()));
    '''
        
    n_kv = parse_beta_file(os.path.join(input_directory, "topic.dat"));
    
    # extract information from nohup.out files
    output_file = options.output_file;
    output_file_stream = open(output_file, "w");
    output_file_stream.write("%s,%s,%s,%s\n" % ("inference", "topic", "vocabulary", "count"));
    
    (number_of_rows, number_of_columns) = n_kv.shape;
    for topic_index in xrange(number_of_rows):
        for vocab_index in xrange(number_of_columns):
            if n_kv[topic_index, vocab_index]==0:
                continue;
            output_file_stream.write("%s,%d,%d,%g\n" % ("true", topic_index + 1, vocab_index + 1, n_kv[topic_index, vocab_index]));    
    
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
                inference = "rand";
            elif int(m.group("sp")) == 1 and int(m.group("mp")) == 1:
                inference = "rgs";
            elif int(m.group("sp")) == 2 and int(m.group("mp")) == 0:
                inference = "sa";
            else:
                print "unrecognized model patterns %s" % model_name
                continue;
        elif re.match(split_merge_heuristics_pattern, model_name):
            m = re.match(split_merge_heuristics_pattern, model_name);
            if int(m.group("smh")) == 0:
                inference = "crs"
            elif int(m.group("smh")) == 1:
                inference = "rand"
            else:
                print "unrecognized model patterns %s" % model_name
                continue;
        elif re.match(no_split_merge_pattern, model_name):
            inference = "null";
        else:
            print "unrecognized model patterns %s" % model_name
            continue;
        
        n_kv = parse_beta_file(os.path.join(output_model_path, "exp_beta-%d" % (snapshot_index)));
        (number_of_rows, number_of_columns) = n_kv.shape;
        
        for topic_index in xrange(number_of_rows):
            for vocab_index in xrange(number_of_columns):
                if n_kv[topic_index, vocab_index]==0:
                    continue;
                output_file_stream.write("%s,%d,%d,%g\n" % (inference, topic_index + 1, vocab_index + 1, n_kv[topic_index, vocab_index]));    
        
        print "successfully parsed output %s..." % (model_name);

def parse_beta_file(beta_file_path):
    n_kv = numpy.zeros((0, 0));
    
    topic_index = -1;
    input_stream = open(beta_file_path, 'r');
    for line in input_stream:
        line = line.strip();
        tokens = line.split();
        if len(tokens)==3:
            assert tokens[0]=="==========";
            assert tokens[2]=="==========";
            topic_index = int(tokens[1]);
            n_kv = numpy.vstack((n_kv, numpy.zeros((1, n_kv.shape[1]))))
        else:
            vocabulary_index = int(tokens[0]);
            if vocabulary_index>=n_kv.shape[1]:
                n_kv = numpy.hstack((n_kv, numpy.zeros((n_kv.shape[0], vocabulary_index-n_kv.shape[1]+1))));
            n_kv[topic_index, vocabulary_index] = float(tokens[1]);

    (number_of_topics, number_of_vocabularies) = n_kv.shape;
    topic_weights = numpy.argmax(n_kv, axis=1);
    '''
    vocab_weights = numpy.arange(number_of_vocabularies)[numpy.newaxis, :]
    #vocab_weights = numpy.cumsum(vocab_weights, axis=1);
    vocab_weights = vocab_weights.T;
    topic_weights = numpy.dot(n_kv, vocab_weights)[:, 0];
    '''
    print topic_weights;
    topic_ranking = numpy.argsort(topic_weights);
    print topic_ranking
    
    n_kv = n_kv[topic_ranking, :];
    
    return n_kv;
    
if __name__ == '__main__':
    main()