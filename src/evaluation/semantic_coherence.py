from nltk.corpus import stopwords;
from nltk.probability import FreqDist;

import sys, os;

import nltk;
import numpy;

class SemanticCoherence():
    def __init__(self, epsilon=1):
        print "Gentle Reminder: are you using consistent stop-word list, word stemmer and string tokenizer?"

        #self._language = language;
        #from nltk.corpus import stopwords
        #self._stop = stopwords.words(language);
        
        #from nltk.stem.porter import PorterStemmer
        #self._stemmer = PorterStemmer();
        
        #from nltk.tokenize.punkt import PunktWordTokenizer 
        #self._tokenizer = PunktWordTokenizer()
        
        self._unitype_freqdist = nltk.probability.FreqDist();
        self._bitype_freqdist = nltk.probability.FreqDist();
        self._epsilon = epsilon;

    def collect_statistics(self, data_input_file):
        self._unitype_freqdist.clear();
        self._bitype_freqdist.clear();
        
        testing_docs = open(data_input_file, 'r');
        
        self._unitype_freqdist = nltk.probability.FreqDist();
        self._bitype_freqdist = nltk.probability.FreqDist();
        for line in testing_docs:
            line = line.strip();
            content = line.split();
            content = list(set(content));
            for index1 in xrange(len(content)):
                self._unitype_freqdist[content[index1]]+=1;
                for index2 in xrange(index1+1, len(content)):
                    if content[index1] < content[index2]:
                        self._bitype_freqdist[(content[index1], content[index2])]+=1;
                    elif content[index2] < content[index1]:
                        self._bitype_freqdist[(content[index2], content[index1])]+=1;

    def evaluate(self, topic_file, top_words=10, topic_title_indicator='=========='):
        #unitype_probdist = nltk.probability.WittenBellProbDist(self._unitype_freqdist, self._unitype_freqdist.B()+1);
        #bitype_probdist = nltk.probability.WittenBellProbDist(self._bitype_freqdist, self._bitype_freqdist.B()+1);
        
        input = open(topic_file, 'r');
        top_word_list = [];
        coherence_score = numpy.zeros((1, 1));
        topic_id = 0;
        topic_word_probability = float('+inf');
        for line in input:
            line = line.strip();
            content = line.split();
            
            if line.startswith(topic_title_indicator):
                top_word_list = [];
                topic_id = int(content[1]);
                topic_word_probability = float('+inf');
                continue;
            
            if len(top_word_list)>=top_words:
                continue;
            
            assert(topic_word_probability>=float(content[1]));
            top_word_list.append(content[0]);
            topic_word_probability = float(content[1]);
            
            if len(top_word_list)==top_words:
                if coherence_score.shape[1]<=topic_id:
                    coherence_score = numpy.append(coherence_score, numpy.zeros((1, topic_id-coherence_score.shape[1]+1)), 1);
                    
                for i in xrange(1, len(top_word_list)):
                    for j in xrange(i):
                        coherence_score[0, topic_id] += numpy.log(self._bitype_freqdist[(top_word_list[i], top_word_list[j])] + self._epsilon);
                        coherence_score[0, topic_id] -= numpy.log(self._unitype_freqdist[top_word_list[j]]);
    
        return coherence_score

def main():

    directory = sys.argv[1].strip();
    #directory = os.path.abspath('../../output/de-news-filtered/12Aug02-13:56:55-D9800-K20-T50-P10-S10-B70-O280-t64-k0.6-at0.05-ab1');
    assert(not directory.endswith('/'));
    
    test_document = int(sys.argv[2]);
    maximum_snapshot = int(sys.argv[3]);
    top_words = int(sys.argv[4]);
    
    option_file = os.path.join(directory, 'option.txt');
    input = open(option_file, 'r');
    for line in input:
        line = line.strip();
        content = line.split('=');
        if line.startswith('input_directory'):
            input_directory = content[1];
        elif line.startswith('corpus_name'):
            corpus_name = content[1];
        elif line.startswith('dictionary_file'):
            dict_file = content[1];
        elif line.startswith('number_of_documents'):
            number_of_documents = int(content[1]);
        elif line.startswith('number_of_topics'):
            number_of_topics = int(content[1]);
        elif line.startswith('desired_truncation_level'):
            desired_truncation_level = int(content[1]);
        elif line.startswith('vocab_prune_interval'):
            vocab_prune_interval = int(content[1]);
        elif line.startswith('snapshot_interval'):
            snapshot_interval = int(content[1]);
        elif line.startswith('batch_size'):
            batch_size = int(content[1]);
        elif line.startswith('online_iterations'):
            online_iterations = int(content[1]);
        elif line.startswith('tau'):
            tau = float(content[1]);
        elif line.startswith('kappa'):
            kappa = float(content[1]);
        elif line.startswith('alpha_theta'):
            alpha_theta = float(content[1]);
        elif line.startswith('alpha_beta'):
            alpha_beta = float(content[1]);
        else:
            print "warning: unparsed options...";
            
    testing_docs = open(os.path.join(input_directory, "train.dat"), 'r');
    
    unitoken_freqdist = nltk.probability.FreqDist();
    bitoken_freqdist = nltk.probability.FreqDist();
    index = 0;
    for line in testing_docs:
        index += 1;
        if index <= test_document:
            continue;
        line = line.strip();
        content = line.split();
        content = list(set(content));
        for index1 in xrange(len(content)):
            unitoken_freqdist[content[index1]]+=1;
            for index2 in xrange(len(content)):
                bitoken_freqdist[(content[index1], content[index2])]+=0.5;
                bitoken_freqdist[(content[index2], content[index1])]+=0.5;
                
                '''
                if index1==index2:
                    continue;
                
                if content[index2] < content[index1]:
                    bitoken_freqdist.inc((content[index2], content[index1]));
                else:
                    bitoken_freqdist.inc((content[index1], content[index2]));
                '''
    
    #total_vocab = 5000;
    #unitoken_probdist = nltk.probability.WittenBellProbDist(unitoken_freqdist, total_vocab);
    #bitoken_probdist = nltk.probability.WittenBellProbDist(bitoken_freqdist, total_vocab**2);

    (parent_directory, file_name) = os.path.split(os.path.abspath(directory));
    for sub_directory in os.listdir(parent_directory):
        if not sub_directory.startswith(file_name):
            continue;
        
        if os.path.isfile(os.path.join(parent_directory, sub_directory)):
            continue;
        
        content = sub_directory.split('olda');
        print content
        if len(content)==1:
            for index in xrange(0, maximum_snapshot+1, 10):
                #coherence_score = evaluate_coherence(unitoken_probdist, bitoken_probdist, top_words, number_of_topics, os.path.join(parent_directory, sub_directory, 'exp_beta-'+str(index)));
                coherence_score = evaluate_coherence(unitoken_freqdist, bitoken_freqdist, top_words, number_of_topics, os.path.join(parent_directory, sub_directory, 'exp_beta-'+str(index)));
                print numpy.mean(coherence_score);
        else:
            for index in xrange(0, maximum_snapshot+1, 10):
                #coherence_score = evaluate_coherence(unitoken_probdist, bitoken_probdist, top_words, number_of_topics, os.path.join(parent_directory, sub_directory, 'exp_beta-'+str(index)));
                coherence_score = evaluate_coherence(unitoken_freqdist, bitoken_freqdist, top_words, number_of_topics, os.path.join(parent_directory, sub_directory, 'exp_beta-'+str(index)));
                print numpy.mean(coherence_score);
                
if __name__ == '__main__':
    main()