import nltk;
import numpy;
import sys;

#from nltk.tokenize import wordpunct_tokenize
#from nltk.tokenize import PunktWordTokenizer
#from PMI_statistics import get_tfidf
#from glob import glob
#from math import log
#from topicmod.util import flags
#from topicmod.ling.snowball_wrapper import Snowball

"""
"""
class PointwiseMutualInformation():
    def __init__(self, vocab_file=None, window_size=-1, language='english'):
        print "Gentle Reminder: are you using consistent stop-word list, word stemmer and string tokenizer?"

        self._vocab_file = vocab_file;
        self._vocab = [];
        if self._vocab_file!=None:
            input = open(self._vocab_file, 'r');
            for line in input:
                self._vocab.append(line.strip());
        print "vocab size:", len(self._vocab);
        
        self._window_size = window_size

        self._language = language;
        from nltk.corpus import stopwords
        self._stop = stopwords.words(language);
        
        from nltk.stem.porter import PorterStemmer
        self._stemmer = PorterStemmer();
        
        from nltk.tokenize.punkt import PunktWordTokenizer 
        self._tokenizer = PunktWordTokenizer()
        
        self._unitype_freqdist = nltk.probability.FreqDist();
        self._bitype_freqdist = nltk.probability.FreqDist();

    def collect_statistics(self, input_file='../../data/wikipedia/ascii-en-wiki.txt'):
        input = open(input_file, 'r');
        i=0
        for line in input:
            i += 1;
            line = line.strip();
            line = line.lower();
            #words = [self._stemmer.stem(x) for x in self._tokenizer.tokenize(line) if (x not in self._stop) and (min(y in ascii_lowercase for y in x))];
            words = [self._stemmer.stem(x) for x in self._tokenizer.tokenize(line) if (x not in self._stop)];
            
            if len(self._vocab)>0:
                words = [word for word in words if word in self._vocab];
                
            for index1 in xrange(len(words)):
                word1 = words[index1];
                self._unitype_freqdist.inc(word1);
                
                if self._window_size == -1:
                    index_end = len(words);
                else:
                    index_end = min(len(words), index1 + self._window_size + 1);
                    
                for index2 in range(index1 + 1, index_end):
                    word2 = words[index2]
                    
                    if word1 < word2:
                        self._bitype_freqdist.inc((word1, word2));
                    elif word1 > word2:
                        self._bitype_freqdist.inc((word2, word1));
            
            if i%1000 == 0:
                print "collected pmi statistics from %d documents" % (i);
                        
    def export_pmi_statistics(self, output_file):
        output = open(output_file, 'w');
        #output.write("==========" + "unitype" + "==========");
        for unitype in self._unitype_freqdist.keys():
            output.write(unitype + "\t" + self._unitype_freqdist[unitype]);
            
        #output.write("==========\t" + "unitype" + "\t==========");
        for bitype in self._bitype_freqdist.keys():
            output.write(' '.join(bitype) + "\t" + self._bitype_freqdist[bitype]);
            
    def import_pmi_statistics(self, input_file, target_vocab=None):
        if self._vocab_file!=None:
            print "warning: you have seeded a vocab file for evaluation model!"
            return;
        
        if target_vocab!=None:
            '''
            input = open(topic_file, 'r');
            top_word_list = [];
            top_word_counter = 0;
            pmi_score = numpy.zeros((1, 1));
            topic_id = 0;
            for line in input:
                line = line.strip();
                content = line.split();
                if line.startswith(topic_title_indicator):
                    top_word_counter = 0;
                    continue;
                if top_word_counter>=top_words:
                    continue;
                top_word_counter += 1;
                top_word_list.append(content[0])
            '''
            
            line_counter = 0;
            dummy_type = '#';
            input = open(input_file, 'r');
            self._unitype_freqdist = nltk.probability.FreqDist();
            self._bitype_freqdist = nltk.probability.FreqDist();
            for line in input:
                line_counter += 1;
                line = line.strip();
                content = line.split();
                if len(content)==2:
                    if content[0] in target_vocab:
                        self._unitype_freqdist.inc(target_vocab[content[0]], float(content[1]));
                    else:
                        self._unitype_freqdist.inc(content[0], float(content[1]));
                    #if content[0] in target_vocab:
                        #_unitype_freqdist.inc(content[0], float(content[1]));
                    #else:
                        #_unitype_freqdist.inc(dummy_type[0], float(content[1]));
                    #self._vocab.append(content[0]);
                elif len(content)==3:
                    assert(content[0]<content[1]);
                    
                    if content[0] in target_vocab and content[1] in target_vocab:
                        self._bitype_freqdist.inc((target_vocab[content[0]], target_vocab[content[1]]), float(content[2]));
                    else:
                        self._bitype_freqdist.inc((dummy_type, dummy_type), float(content[2]));
                        
                if line_counter % 10000000==0:
                    print "loaded %d lines in pmi-statistics..." % line_counter
        else:
            input = open(input_file, 'r');
            self._unitype_freqdist = nltk.probability.FreqDist();
            self._bitype_freqdist = nltk.probability.FreqDist();
            line_counter = 0;
            for line in input:
                line_counter += 1;
                line = line.strip();
                content = line.split();
                if len(content)==2:
                    self._unitype_freqdist.inc(content[0], float(content[1]));
                elif len(content)==3:
                    assert(content[0]<content[1]);
                    self._bitype_freqdist.inc((content[0], content[1]), float(content[2]));
                    
                if line_counter % 10000000==0:
                    print "loaded %d lines in pmi-statistics..." % line_counter
                    
        #self._vocab = self._unitype_freqdist.keys();
        #return self._vocab, self._unitype_freqdist, self._bitype_freqdist
    
        return self._unitype_freqdist, self._bitype_freqdist

    def evaluate(self, topic_file, top_words=10, target_vocab=None, topic_title_indicator='=========='):
        #unitype_probdist = nltk.probability.MLEProbDist(self._unitype_freqdist);
        #bitype_probdist = nltk.probability.MLEProbDist(self._bitype_freqdist);
        unitype_probdist = nltk.probability.LidstoneProbDist(self._unitype_freqdist, 0.001, self._unitype_freqdist.B() + 1);
        bitype_probdist = nltk.probability.LidstoneProbDist(self._bitype_freqdist, 0.001, self._unitype_freqdist.B()**2);
        #unitype_probdist = nltk.probability.WittenBellProbDist(self._unitype_freqdist, self._unitype_freqdist.B()+1);
        #bitype_probdist = nltk.probability.WittenBellProbDist(self._bitype_freqdist, self._bitype_freqdist.B()+1);
        #bitype_probdist = nltk.probability.WittenBellProbDist(self._bitype_freqdist, self._unitype_freqdist.B()**2);

        input = open(topic_file, 'r');
        top_word_list = [];
        pmi_score = numpy.zeros((1, 1));
        topic_id = 0;
        for line in input:
            line = line.strip();
            content = line.split();
            if line.startswith(topic_title_indicator):
                top_word_list = [];
                topic_id = int(content[1]);
                continue;
            
            if len(top_word_list)>=top_words:
                continue;

            top_word_list.append(content[0]);
            if len(top_word_list)==top_words:
                if pmi_score.shape[1]<=topic_id:
                    pmi_score = numpy.append(pmi_score, numpy.zeros((1, topic_id-pmi_score.shape[1]+1)), 1);
                for i in xrange(len(top_word_list)-1):
                    for j in xrange(i+1, len(top_word_list)):
                        if target_vocab!=None:
                            if top_word_list[i]<=top_word_list[j]:
                                pmi_score[0, topic_id] += numpy.log(bitype_probdist.prob((target_vocab[top_word_list[i]], target_vocab[top_word_list[j]])));
                            else:
                                pmi_score[0, topic_id] += numpy.log(bitype_probdist.prob((target_vocab[top_word_list[j]], target_vocab[top_word_list[i]])));
                            pmi_score[0, topic_id] -= numpy.log(unitype_probdist.prob(target_vocab[top_word_list[i]]));
                            pmi_score[0, topic_id] -= numpy.log(unitype_probdist.prob(target_vocab[top_word_list[j]]));
                        else:
                            if top_word_list[i]<=top_word_list[j]:
                                pmi_score[0, topic_id] += numpy.log(bitype_probdist.prob((top_word_list[i], top_word_list[j])));
                            else:
                                pmi_score[0, topic_id] += numpy.log(bitype_probdist.prob((top_word_list[j], top_word_list[i])));
                            pmi_score[0, topic_id] -= numpy.log(unitype_probdist.prob(top_word_list[i]));
                            pmi_score[0, topic_id] -= numpy.log(unitype_probdist.prob(top_word_list[j]));
 
        return pmi_score

    def collect_statistics_mapper(self):
        for line in sys.stdin:
            line = line.strip();
            line = line.lower();
            #words = [self._stemmer.stem(x) for x in self._tokenizer.tokenize(line) if (x not in self._stop) and (min(y in ascii_lowercase for y in x))];
            words = [self._stemmer.stem(x) for x in self._tokenizer.tokenize(line) if (x not in self._stop)];
            
            if len(self._vocab)>0:
                words = [word for word in words if word in self._vocab];
                
            for index1 in xrange(len(words)):
                word1 = words[index1];
                #self._unitype_freqdist.inc(word1);
                print "%s\t%d" % (word1, 1);
                
                if self._window_size == -1:
                    index_end = len(words);
                else:
                    index_end = min(len(words), index1 + self._window_size + 1);
                    
                for index2 in range(index1 + 1, index_end):
                    word2 = words[index2]
                    
                    if word1 < word2:
                        print "%s\t%s\t%d" % (word1, word2, 1);
                        #self._bitype_freqdist.inc((word1, word2));
                    elif word1 > word2:
                        print "%s\t%s\t%d" % (word2, word1, 1);
                        #self._bitype_freqdist.inc((word2, word1));
                        
    def collect_statistics_reducer(self):
        self._unitype_freqdist = nltk.probability.FreqDist();
        self._bitype_freqdist = nltk.probability.FreqDist();
     
        for line in sys.stdin:
            line = line.strip();
            line = line.lower();
            
            content = line.split();
            if len(content)==2:
                self._unitype_freqdist.inc(content[0], int(content[1]));
            elif len(content)==3:
                self._bitype_freqdist.inc((content[0], content[1]), int(content[2]));
            else:
                print "warning: unexpected key-value sequence..."
            
        for key in self._unitype_freqdist.keys():
            print "%s\t%d" % (key, self._unitype_freqdist[key]);
        for key in self._bitype_freqdist.keys():
            print "%s\t%d" % (' '.join(key), self._bitype_freqdist[key]);
            
if __name__ == "__main__":
    #pmi_statistics_collector = PointwiseMutualInformation('../data/de-news-filtered/voc.dat');
    pmi_statistics_collector = PointwiseMutualInformation();
    pmi_statistics_collector.collect_statistics('../data/wikipedia/ascii-en-wiki.txt');
    #pmi_statistics_collector.export_pmi_statistics('../data/de-news-filtered/pmi_statistics_wiki_en.txt');
    pmi_statistics_collector.export_pmi_statistics('../data/pmi_statistics_wiki_en.txt');
