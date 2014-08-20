import numpy;
import scipy;
import random;
import itertools;
import nltk;
import time;
import os;
import sys;

def filter():
    input_directory = sys.argv[1];
    input_directory = input_directory.rstrip("/");
    
    min_term_frequency_cutoff = int(sys.argv[2]);
    max_term_frequency_cutoff = int(sys.argv[3]);
    
    min_doc_frequency_cutoff = int(sys.argv[4]);
    max_doc_frequency_cutoff = int(sys.argv[5]);

    min_doc_length = int(sys.argv[6])
    
    min_tf_idf = 1.5;
    
    output_directory = input_directory;
    output_directory += "-min_tf%d-max_tf%d" % (min_term_frequency_cutoff, max_term_frequency_cutoff);
    output_directory += "-min_df%d-max_df%d" % (min_doc_frequency_cutoff, max_doc_frequency_cutoff);
    output_directory += "-min_dl%d" % (min_doc_length);
    if not os.path.exists(output_directory):
        os.mkdir(output_directory);
    
    term_freq = nltk.probability.FreqDist();
    doc_freq = nltk.probability.FreqDist();
    input_doc_file = os.path.join(input_directory, "doc.dat");
    input_doc_stream = open(input_doc_file, 'r');
    for line in input_doc_stream:
        line = line.strip();
        tokens = line.split();
        for token in tokens:
            term_freq.inc(token);
            
        for token in set(tokens):
            doc_freq.inc(token);
    input_doc_stream.close();
            
    valid_vocab = [word for word in term_freq.keys() \
                    if term_freq[word]>=min_term_frequency_cutoff \
                    and term_freq[word]<=max_term_frequency_cutoff \
                    and doc_freq[word]>=min_doc_frequency_cutoff \
                    and doc_freq[word]<=max_doc_frequency_cutoff \
                    and term_freq[word]*1.0/doc_freq[word]>min_tf_idf];
    output_voc_stream = open(os.path.join(output_directory, "voc.dat"), "w");
    for word in valid_vocab:
        output_voc_stream.write("%s\t%d\t%d\n" % (word, term_freq[word], doc_freq[word]));
    output_voc_stream.close();
    
    valid_vocab = set(valid_vocab);
    input_doc_stream = open(input_doc_file, 'r');
    output_doc_stream = open(os.path.join(output_directory, "doc.dat"), "w");
    for line in input_doc_stream:
        line = line.strip();
        tokens = [token for token in line.split() if token in valid_vocab];
        if len(tokens)<min_doc_length:
            continue;
        output_doc_stream.write("%s\n" % (" ".join(tokens)));
        
if __name__ == '__main__':
    filter();