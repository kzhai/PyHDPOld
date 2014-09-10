from nltk.corpus import stopwords;
from nltk.probability import FreqDist;

import sys, os;
import collections;
import nltk;
import numpy;

class TopicUniqueness():
    def evaluate(self, topic_file, top_words=10, topic_title_indicator='=========='):
        input_stream = open(topic_file, 'r');
        top_word_set = collections.defaultdict(set);
        top_word_count = 0;
        
        for line in input_stream:
            line = line.strip();
            content = line.split();
            
            if line.startswith(topic_title_indicator):
                topic_id = int(content[1]);
                top_word_set[topic_id] = set();
                top_word_count = 0;
                continue;
            
            if top_word_count >= top_words:
                continue;
            
            top_word_set[topic_id].add(content[0]);
            top_word_count += 1;
        
        inner_topic_distance = numpy.zeros((1, len(top_word_set)));
        for topic_index_1 in xrange(len(top_word_set)):
            for word in top_word_set[topic_index_1]:
                for topic_index_2 in xrange(len(top_word_set)):
                    if topic_index_2==topic_index_1:
                        continue;
                    if word not in top_word_set[topic_index_2]:
                        inner_topic_distance[0, topic_index_1] += 1;
                
        return inner_topic_distance
    
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
            unitoken_freqdist.inc(content[index1]);
            for index2 in xrange(len(content)):
                bitoken_freqdist.inc((content[index1], content[index2]), 0.5);
                bitoken_freqdist.inc((content[index2], content[index1]), 0.5);
                
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