import numpy;
import scipy;
import random;
import itertools;
import nltk;
import time;
import os;
import sys;

def main():
    input_directory = sys.argv[1];
    input_directory = input_directory.rstrip("/");
    
    #output_directory = sys.argv[2];
    vocab_cutoff = int(sys.argv[2]);
    doc_cutoff = int(sys.argv[3]);
    
    output_directory = input_directory + "-v%d-d%d" % (vocab_cutoff, doc_cutoff);
    os.mkdir(output_directory);
    
    word_freq = nltk.probability.FreqDist();
    input_doc_file = os.path.join(input_directory, "doc.dat");
    input_doc_stream = open(input_doc_file, 'r');
    for line in input_doc_stream:
        line = line.strip();
        tokens = line.split();
        for token in tokens:
            word_freq.inc(token);
    input_doc_stream.close();
            
    valid_vocab = word_freq.keys()[:vocab_cutoff];
    output_voc_stream = open(os.path.join(output_directory, "voc.dat"), "w");
    for word in valid_vocab:
        output_voc_stream.write("%s\t%d\n" % (word, word_freq[word]));
    output_voc_stream.close();
    
    valid_vocab = set(valid_vocab);
    input_doc_stream = open(input_doc_file, 'r');
    output_doc_stream = open(os.path.join(output_directory, "doc.dat"), "w");
    for line in input_doc_stream:
        line = line.strip();
        tokens = [token for token in line.split() if token in valid_vocab];
        if len(tokens)<doc_cutoff:
            continue;
        output_doc_stream.write("%s\n" % ("\t".join(tokens)));
        
if __name__ == '__main__':
    main();