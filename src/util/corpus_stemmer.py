import numpy;
import scipy;
import random;
import itertools;
import nltk;
import time;
import os;
import sys;

import nltk.corpus;
import nltk.stem.porter;

from nltk.corpus import stopwords
stop_words = stopwords.words("english");

from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer();

from nltk.tokenize.punkt import PunktWordTokenizer 
punkt_tokenizer = PunktWordTokenizer()

def stemmer():
    input_directory = sys.argv[1];
    output_directory = sys.argv[2]
    
    min_doc_length = 10;
    
    term_frequency = nltk.probability.FreqDist();
    document_frequency = nltk.probability.FreqDist();
    
    output_doc_file = os.path.join(output_directory, "doc.dat");
    output_doc_stream = open(output_doc_file, 'w');
    
    input_doc_file = os.path.join(input_directory, "doc.dat");
    input_doc_stream = open(input_doc_file, 'r');
    for line in input_doc_stream:
        line = line.strip();
        line = line.lower();
        
        #words = [porter_stemmer.stem(x) for x in punkt_tokenizer.tokenize(line) if (x not in stop_words)];
        words = [porter_stemmer.stem(x) for x in punkt_tokenizer.tokenize(line) if (x not in stop_words) and (x!="nbsp")];
        
        if len(words)>=min_doc_length:
            output_doc_stream.write(" ".join(words) + "\n");
        
        for word in words:
            term_frequency.inc(word);
            
        for word in set(words):
            document_frequency.inc(word);
            
    output_voc_file = os.path.join(output_directory, "voc.dat");
    output_voc_stream = open(output_voc_file, 'w');
    for word in term_frequency.keys():
        output_voc_stream.write("%s\t%d\t%d\n" % (word, term_frequency[word], document_frequency[word]));
        
    return;

if __name__ == '__main__':
    stemmer();