import numpy;
import scipy;
import random;
import itertools;
import nltk;
import time;
import os;
import sys;

def sampler():
    input_directory = sys.argv[1];
    percentage_of_data = float(sys.argv[2]);
    
    input_directory = input_directory.rstrip("/");
    
    output_directory = input_directory;
    output_directory += "-%d" % (percentage_of_data);
    if not os.path.exists(output_directory):
        os.mkdir(output_directory);
    
    term_freq = nltk.probability.FreqDist();
    doc_freq = nltk.probability.FreqDist();
    input_doc_file = os.path.join(input_directory, "doc.dat");
    input_doc_stream = open(input_doc_file, 'r');
    output_doc_file = os.path.join(output_directory, "doc.dat")
    output_doc_stream = open(output_doc_file, "w");
    for line in input_doc_stream:
        line = line.strip();
        if numpy.random.random()>percentage_of_data/100:
            continue;

        output_doc_stream.write("%s\n" % line)
        tokens = line.split();
        for token in tokens:
            term_freq.inc(token);
        for token in set(tokens):
            doc_freq.inc(token);
    input_doc_stream.close();

    output_voc_stream = open(os.path.join(output_directory, "voc.dat"), "w");
    for word in term_freq.keys():
        output_voc_stream.write("%s\t%d\t%d\n" % (word, term_freq[word], doc_freq[word]));
    output_voc_stream.close();
    
if __name__ == '__main__':
    sampler();