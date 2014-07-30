"""
@author: Ke Zhai (zhaike@cs.umd.edu)

This code was modified from the code originally written by Chong Wang (chongw@cs.princeton.edu).
Implements uncollapsed Gibbs sampling for the hierarchical Dirichlet process (HDP).

References:
[1] Chong Wang and David Blei, A Split-Merge MCMC Algorithm for the Hierarchical Dirichlet Process, available online www.cs.princeton.edu/~chongw/papers/sm-hdp.pdf.
"""

import numpy
import scipy
import scipy.special
import scipy.stats;
import os
import re
import sys
import string
import math
import time
import nltk;
import random

negative_infinity = -1e500;

# We will be taking log(0) = -Inf, so turn off this warning
numpy.seterr(divide='ignore')

class MonteCarlo(object):
    """
    @param truncation_level: the maximum number of clusters, used for speeding up the computation
    @param snapshot_interval: the interval for exporting a snapshot of the model
    """
    def __init__(self,
                 resample_topics=False,
                 hash_oov_words=False
                 ):
        self._resample_topics = resample_topics;
        self._hash_oov_words = hash_oov_words;
        
    """
    @param data: a N-by-D numpy array object, defines N points of D dimension
    @param K: number of topics, number of broke sticks
    @param alpha: the probability of f_{k_{\mathsf{new}}}^{-x_{dv}}(x_{dv}), the prior probability density for x_{dv}
    @param gamma: the smoothing value for a table to be assigned to a new topic
    @param eta: the smoothing value for a word to be assigned to a new topic
    """
    def _initialize(self,
                    corpus,
                    vocab,
                    alpha_alpha,
                    alpha_gamma,
                    alpha_eta=0
                    ):
        
        self._word_to_index = {};
        self._index_to_word = {};
        for word in set(vocab):
            self._index_to_word[len(self._index_to_word)] = word;
            self._word_to_index[word] = len(self._word_to_index);
            
        self._vocab = self._word_to_index.keys();
        self._vocabulary_size = len(self._vocab);
        
        # top level smoothing
        self._alpha_alpha = alpha_alpha
        # bottom level smoothing
        self._alpha_gamma = alpha_gamma
        # vocabulary smoothing
        if alpha_eta<=0:
            self._alpha_eta = 1.0/self._vocabulary_size;
        else:
            self._alpha_eta = alpha_eta;
            
        self._counter = 0;
        
        self._K = 1;
        
        # initialize the documents, key by the document path, value by a list of non-stop and tokenized words, with duplication.
        self._corpus = self.parse_doc_list(corpus);
        
        # initialize the size of the collection, i.e., total number of documents.
        self._D = len(self._corpus)

        # initialize the word count matrix indexed by topic id and word id, i.e., n_{\cdot \cdot k}^v
        self._n_kv = numpy.zeros((self._K, self._vocabulary_size));
        
        # initialize the table count matrix indexed by topic id, i.e., m_{\cdot k}
        self._m_k = numpy.zeros(self._K);
        
        # initialize the word count matrix indexed by topic id and document id, i.e., n_{j \cdot k}
        self._n_dk = numpy.zeros((self._D, self._K));
        
        # random initialize all documents
        
        # initialize the table information vectors indexed by document id and word id, i.e., t{j i}
        self._t_dv = {};
        # initialize the topic information vectors indexed by document id and table id, i.e., k_{j t}
        self._k_dt = {};
        # initialize the word count vectors indexed by document id and table id, i.e., n_{j t \cdot}
        self._n_dt = {};
        
        # we assume all words in a document belong to one table which was assigned to topic 0 
        for d in xrange(self._D):
            # initialize the table information vector indexed by document and records down which table a word belongs to 
            self._t_dv[d] = numpy.zeros(len(self._corpus[d]), dtype=numpy.int);
            
            # self._k_dt records down which topic a table was assigned to
            self._k_dt[d] = numpy.zeros(1, dtype=numpy.int);
            assert(len(self._k_dt[d]) == len(numpy.unique(self._t_dv[d])));
            
            # word_count_table records down the number of words sit on every table
            self._n_dt[d] = numpy.zeros(1, dtype=numpy.int) + len(self._corpus[d]);
            assert(len(self._n_dt[d]) == len(numpy.unique(self._t_dv[d])));
            assert(numpy.sum(self._n_dt[d]) == len(self._corpus[d]));
            
            for v in self._corpus[d]:
                self._n_kv[0, v] += 1;
            self._n_dk[d, 0] = len(self._corpus[d])
            
            self._m_k[0] += len(self._k_dt[d]);

    def parse_doc_list(self, docs):
        if (type(docs).__name__ == 'str'):
            temp = list()
            temp.append(docs)
            docs = temp
            
        D = len(docs)
        wordids = list()

        for d in xrange(D):
            docs[d] = docs[d].lower()
            words = docs[d].split()
            wordid = list(); 
            for word in words:
                if (word in self._vocab):
                    wordtoken = self._word_to_index[word]
                    wordid.append(wordtoken);
                else:
                    if self._hash_oov_words:
                        wordtoken = hash(word) % len(self._vocab);
                        wordid.append(wordtoken);
                        
            wordids.append(wordid)
            
        return wordids

    def learning(self):
        self._counter += 1;
        
        self.sample_cgs();
    
        # compact all the parameters, including removing unused topics and unused tables
        self.compact_params();
        
        print "accumulated number of tables:", self._m_k;
        print "accumulated number of tokens:", numpy.sum(self._n_kv, axis=1)[:, numpy.newaxis].T;
        
        return 0;

    """
    sample the data to train the parameters
    """
    def sample_cgs(self):
        #sample the total data
        for document_index in numpy.random.permutation(xrange(self._D)):
            # sample word assignment, see which table it should belong to
            for word_index in numpy.random.permutation(xrange(len(self._corpus[document_index]))):
                #self.update_params(document_index, word_index, -1);
                
                # retrieve the table_id of the current word of current document
                table_id = self._t_dv[document_index][word_index];
                # retrieve the topic_id of the table that current word of current document sit on
                topic_id = self._k_dt[document_index][table_id];
                # get the word_id of at the word_index of the document_index
                word_id = self._corpus[document_index][word_index];
        
                self._n_dt[document_index][table_id] -= 1;
                assert(numpy.all(self._n_dt[document_index] >= 0));
                self._n_kv[topic_id, word_id] -= 1;
                assert(numpy.all(self._n_kv >= 0));
                self._n_dk[document_index, topic_id] -= 1;
                assert(numpy.all(self._n_dk >= 0));
                
                # if current table in current document becomes empty 
                if self._n_dt[document_index][table_id] == 0:
                    # adjust the table counts
                    self._m_k[topic_id] -= 1;
                    
                assert(numpy.all(self._m_k >= 0));
                assert(numpy.all(self._k_dt[document_index] >= 0));
                
                # get the word at the index position
                word_id = self._corpus[document_index][word_index];

                n_k = numpy.sum(self._n_kv, axis=1);
                assert(len(n_k) == self._K);
                f = numpy.zeros(self._K);
                f_new = self._alpha_alpha / self._vocabulary_size;
                for k in xrange(self._K):
                    f[k] = (self._n_kv[k, word_id] + self._alpha_eta) / (n_k[k] + self._vocabulary_size * self._alpha_eta);
                    f_new += self._m_k[k] * f[k];
                f_new /= (numpy.sum(self._m_k) + self._alpha_alpha);
                
                # compute the probability of this word sitting at every table 
                table_probablity = numpy.zeros(len(self._k_dt[document_index]) + 1);
                for t in xrange(len(self._k_dt[document_index])):
                    if self._n_dt[document_index][t] > 0:
                        # if there are some words sitting on this table, the probability will be proportional to the population
                        assigned_topic = self._k_dt[document_index][t];
                        assert(assigned_topic >= 0 or assigned_topic < self._K);
                        table_probablity[t] = f[assigned_topic] * self._n_dt[document_index][t];
                    else:
                        # if there are no words sitting on this table
                        # note that it is an old table, hence the prior probability is 0, not self._alpha_gamma
                        table_probablity[t] = 0.;
                # compute the probability of current word sitting on a new table, the prior probability is self._alpha_gamma
                table_probablity[len(self._k_dt[document_index])] = self._alpha_gamma * f_new;

                # sample a new table this word should sit in
                table_probablity /= numpy.sum(table_probablity);
                cdf = numpy.cumsum(table_probablity);
                new_table = numpy.uint8(numpy.nonzero(cdf >= numpy.random.random())[0][0]);

                # assign current word to new table
                self._t_dv[document_index][word_index] = new_table;
                
                # if current word sits on a new table, we need to get the topic of that table
                if new_table == len(self._k_dt[document_index]):
                    # expand the vectors to fit in new table
                    self._n_dt[document_index] = numpy.hstack((self._n_dt[document_index], numpy.zeros(1)));
                    self._k_dt[document_index] = numpy.hstack((self._k_dt[document_index], numpy.zeros(1)));
                    
                    assert(len(self._n_dt) == self._D and numpy.all(self._n_dt[document_index] >= 0));
                    assert(len(self._k_dt) == self._D and numpy.all(self._k_dt[document_index] >= 0));
                    assert(len(self._n_dt[document_index]) == len(self._k_dt[document_index]));

                    # compute the probability of this table having every topic
                    topic_probability = numpy.zeros(self._K + 1);
                    for k in xrange(self._K):
                        topic_probability[k] = self._m_k[k] * f[k];
                    topic_probability[self._K] = self._alpha_alpha / self._vocabulary_size;

                    # sample a new topic this table should be assigned
                    topic_probability /= numpy.sum(topic_probability);
                    cdf = numpy.cumsum(topic_probability);
                    k_new = numpy.uint8(numpy.nonzero(cdf >= numpy.random.random())[0][0]);
                    
                    # if current table requires a new topic
                    if k_new == self._K:
                        # expand the matrices to fit in new topic
                        self._K += 1;
                        self._n_kv = numpy.vstack((self._n_kv, numpy.zeros((1, self._vocabulary_size))));
                        assert(self._n_kv.shape == (self._K, self._vocabulary_size));
                        #self._n_dk = numpy.vstack((self._n_dk, numpy.zeros((1, self._D))));
                        #assert(self._n_dk.shape == (self._K, self._D));
                        self._n_dk = numpy.hstack((self._n_dk, numpy.zeros((self._D, 1))));
                        assert(self._n_dk.shape == (self._D, self._K));
                        self._m_k = numpy.hstack((self._m_k, numpy.zeros(1)));
                        assert(len(self._m_k) == self._K);
                        
                #self.update_params(document_index, word_index, +1);
                
                # retrieve the table_id of the current word of current document
                table_id = self._t_dv[document_index][word_index];
                # retrieve the topic_id of the table that current word of current document sit on
                topic_id = self._k_dt[document_index][table_id];
                # get the word_id of at the word_index of the document_index
                word_id = self._corpus[document_index][word_index];
        
                self._n_dt[document_index][table_id] += 1;
                assert(numpy.all(self._n_dt[document_index] >= 0));
                self._n_kv[topic_id, word_id] += 1;
                assert(numpy.all(self._n_kv >= 0));
                self._n_dk[document_index, topic_id] += 1;
                assert(numpy.all(self._n_dk >= 0));
                
                # if a new table is created in current document
                if self._n_dt[document_index][table_id] == 1:
                    # adjust the table counts
                    self._m_k[topic_id] += 1;
                    
                assert(numpy.all(self._m_k >= 0));
                assert(numpy.all(self._k_dt[document_index] >= 0));
            
            self.sample_tables();
            
            '''
            # sample table assignment, see which topic it should belong to
            for table_index in numpy.random.permutation(xrange(len(self._k_dt[document_index]))):
                # if this table is not empty, sample the topic assignment of this table
                if self._n_dt[document_index][table_index] > 0:
                    old_topic = self._k_dt[document_index][table_index];

                    # find the index of the words sitting on the current table
                    selected_word_index = numpy.nonzero(self._t_dv[document_index] == table_index)[0];
                    # find the frequency distribution of the words sitting on the current table
                    selected_word_freq_dist = FreqDist([self._corpus[document_index][term] for term in list(selected_word_index)]);

                    # compute the probability of assigning current table every topic
                    topic_probability = numpy.zeros(self._K + 1);
                    topic_probability[self._K] = scipy.special.gammaln(self._vocabulary_size * self._alpha_eta) - scipy.special.gammaln(self._n_dt[document_index][table_index] + self._vocabulary_size * self._alpha_eta);
                    for word_id in selected_word_freq_dist.keys():
                        topic_probability[self._K] += scipy.special.gammaln(selected_word_freq_dist[word_id] + self._alpha_eta) - scipy.special.gammaln(self._alpha_eta);
                    topic_probability[self._K] += numpy.log(self._alpha_alpha);
                    
                    n_k = numpy.sum(self._n_kv, axis=1);
                    assert(len(n_k) == (self._K))
                    for topic_index in xrange(self._K):
                        if topic_index == old_topic:
                            if self._m_k[topic_index] <= 1:
                                # if current table is the only table assigned to current topic,
                                # it means this topic is probably less useful or less generalizable to other documents,
                                # it makes more sense to collapse this topic and hence assign this table to other topic.
                                topic_probability[topic_index] = negative_infinity;
                            else:
                                # if there are other tables assigned to current topic
                                topic_probability[topic_index] = scipy.special.gammaln(self._vocabulary_size * self._alpha_eta + n_k[topic_index] - self._n_dt[document_index][table_index]) - scipy.special.gammaln(self._vocabulary_size * self._alpha_eta + n_k[topic_index]);
                                for word_id in selected_word_freq_dist.keys():
                                    topic_probability[topic_index] += scipy.special.gammaln(self._n_kv[topic_index, word_id] + self._alpha_eta) - scipy.special.gammaln(self._n_kv[topic_index, word_id] + self._alpha_eta - selected_word_freq_dist[word_id]);
                                # compute the prior if we move this table from this topic
                                topic_probability[topic_index] += numpy.log(self._m_k[topic_index] - 1);
                        else:
                            topic_probability[topic_index] = scipy.special.gammaln(self._vocabulary_size * self._alpha_eta + n_k[topic_index]) - scipy.special.gammaln(self._vocabulary_size * self._alpha_eta + n_k[topic_index] + self._n_dt[document_index][table_index]);
                            for word_id in selected_word_freq_dist.keys():
                                topic_probability[topic_index] += scipy.special.gammaln(self._n_kv[topic_index, word_id] + self._alpha_eta + selected_word_freq_dist[word_id]) - scipy.special.gammaln(self._n_kv[topic_index, word_id] + self._alpha_eta);
                            topic_probability[topic_index] += numpy.log(self._m_k[topic_index]);

                    # normalize the distribution and sample new topic assignment for this topic
                    #topic_probability = numpy.exp(topic_probability);
                    #topic_probability = topic_probability/numpy.sum(topic_probability);
                    #topic_probability = numpy.exp(log_normalize(topic_probability));
                    topic_probability -= scipy.misc.logsumexp(topic_probability);
                    topic_probability = numpy.exp(topic_probability);
                    
                    cdf = numpy.cumsum(topic_probability);
                    new_topic = numpy.uint8(numpy.nonzero(cdf >= numpy.random.random())[0][0]);
                    
                    # if the table is assigned to a new topic
                    if new_topic != old_topic:
                        # assign this table to new topic
                        self._k_dt[document_index][table_index] = new_topic;
                        
                        # if this table starts a new topic, expand all matrix
                        if new_topic == self._K:
                            self._K += 1;
                            #self._n_dk = numpy.vstack((self._n_dk, numpy.zeros((1, self._D))));
                            #assert(self._n_dk.shape == (self._K, self._D));
                            self._n_dk = numpy.hstack((self._n_dk, numpy.zeros((self._D, 1))));
                            assert(self._n_dk.shape == (self._D, self._K));
                            
                            self._n_kv = numpy.vstack((self._n_kv, numpy.zeros((1, self._vocabulary_size))));
                            assert(self._n_kv.shape == (self._K, self._vocabulary_size));
                            self._m_k = numpy.hstack((self._m_k, numpy.zeros(1)));
                            assert(len(self._m_k) == self._K);
                            
                        # adjust the statistics of all model parameter
                        self._m_k[old_topic] -= 1;
                        self._m_k[new_topic] += 1;
                        self._n_dk[document_index, old_topic] -= self._n_dt[document_index][table_index];
                        self._n_dk[document_index, new_topic] += self._n_dt[document_index][table_index];
                        for word_id in selected_word_freq_dist.keys():
                            self._n_kv[old_topic, word_id] -= selected_word_freq_dist[word_id];
                            assert(self._n_kv[old_topic, word_id] >= 0)
                            self._n_kv[new_topic, word_id] += selected_word_freq_dist[word_id];
            '''
            
        if self._resample_topics:
            self.sample_topics();
            
    def sample_tables(self, document_index):
        from nltk.probability import FreqDist;
        
        # sample table assignment, see which topic it should belong to
        for table_index in numpy.random.permutation(xrange(len(self._k_dt[document_index]))):
            # if this table is not empty, sample the topic assignment of this table
            if self._n_dt[document_index][table_index] > 0:
                old_topic = self._k_dt[document_index][table_index];

                # find the index of the words sitting on the current table
                selected_word_index = numpy.nonzero(self._t_dv[document_index] == table_index)[0];
                # find the frequency distribution of the words sitting on the current table
                selected_word_freq_dist = FreqDist([self._corpus[document_index][term] for term in list(selected_word_index)]);

                # compute the probability of assigning current table every topic
                topic_probability = numpy.zeros(self._K + 1);
                topic_probability[self._K] = scipy.special.gammaln(self._vocabulary_size * self._alpha_eta) - scipy.special.gammaln(self._n_dt[document_index][table_index] + self._vocabulary_size * self._alpha_eta);
                for word_id in selected_word_freq_dist.keys():
                    topic_probability[self._K] += scipy.special.gammaln(selected_word_freq_dist[word_id] + self._alpha_eta) - scipy.special.gammaln(self._alpha_eta);
                topic_probability[self._K] += numpy.log(self._alpha_alpha);
                
                n_k = numpy.sum(self._n_kv, axis=1);
                assert(len(n_k) == (self._K))
                for topic_index in xrange(self._K):
                    if topic_index == old_topic:
                        if self._m_k[topic_index] <= 1:
                            # if current table is the only table assigned to current topic,
                            # it means this topic is probably less useful or less generalizable to other documents,
                            # it makes more sense to collapse this topic and hence assign this table to other topic.
                            topic_probability[topic_index] = negative_infinity;
                        else:
                            # if there are other tables assigned to current topic
                            topic_probability[topic_index] = scipy.special.gammaln(self._vocabulary_size * self._alpha_eta + n_k[topic_index] - self._n_dt[document_index][table_index]) - scipy.special.gammaln(self._vocabulary_size * self._alpha_eta + n_k[topic_index]);
                            for word_id in selected_word_freq_dist.keys():
                                topic_probability[topic_index] += scipy.special.gammaln(self._n_kv[topic_index, word_id] + self._alpha_eta) - scipy.special.gammaln(self._n_kv[topic_index, word_id] + self._alpha_eta - selected_word_freq_dist[word_id]);
                            # compute the prior if we move this table from this topic
                            topic_probability[topic_index] += numpy.log(self._m_k[topic_index] - 1);
                    else:
                        topic_probability[topic_index] = scipy.special.gammaln(self._vocabulary_size * self._alpha_eta + n_k[topic_index]) - scipy.special.gammaln(self._vocabulary_size * self._alpha_eta + n_k[topic_index] + self._n_dt[document_index][table_index]);
                        for word_id in selected_word_freq_dist.keys():
                            topic_probability[topic_index] += scipy.special.gammaln(self._n_kv[topic_index, word_id] + self._alpha_eta + selected_word_freq_dist[word_id]) - scipy.special.gammaln(self._n_kv[topic_index, word_id] + self._alpha_eta);
                        topic_probability[topic_index] += numpy.log(self._m_k[topic_index]);

                # normalize the distribution and sample new topic assignment for this topic
                #topic_probability = numpy.exp(topic_probability);
                #topic_probability = topic_probability/numpy.sum(topic_probability);
                #topic_probability = numpy.exp(log_normalize(topic_probability));
                topic_probability -= scipy.misc.logsumexp(topic_probability);
                topic_probability = numpy.exp(topic_probability);
                
                cdf = numpy.cumsum(topic_probability);
                new_topic = numpy.uint8(numpy.nonzero(cdf >= numpy.random.random())[0][0]);
                
                # if the table is assigned to a new topic
                if new_topic != old_topic:
                    # assign this table to new topic
                    self._k_dt[document_index][table_index] = new_topic;
                    
                    # if this table starts a new topic, expand all matrix
                    if new_topic == self._K:
                        self._K += 1;
                        #self._n_dk = numpy.vstack((self._n_dk, numpy.zeros((1, self._D))));
                        #assert(self._n_dk.shape == (self._K, self._D));
                        self._n_dk = numpy.hstack((self._n_dk, numpy.zeros((self._D, 1))));
                        assert(self._n_dk.shape == (self._D, self._K));
                        
                        self._n_kv = numpy.vstack((self._n_kv, numpy.zeros((1, self._vocabulary_size))));
                        assert(self._n_kv.shape == (self._K, self._vocabulary_size));
                        self._m_k = numpy.hstack((self._m_k, numpy.zeros(1)));
                        assert(len(self._m_k) == self._K);
                        
                    # adjust the statistics of all model parameter
                    self._m_k[old_topic] -= 1;
                    self._m_k[new_topic] += 1;
                    self._n_dk[document_index, old_topic] -= self._n_dt[document_index][table_index];
                    self._n_dk[document_index, new_topic] += self._n_dt[document_index][table_index];
                    for word_id in selected_word_freq_dist.keys():
                        self._n_kv[old_topic, word_id] -= selected_word_freq_dist[word_id];
                        assert(self._n_kv[old_topic, word_id] >= 0)
                        self._n_kv[new_topic, word_id] += selected_word_freq_dist[word_id];
                                        
    def sample_topics(self):
        # sample topic assignment, see which topic it should belong to
        #for topic_id in numpy.random.permutation(xrange(self._K)):
        
        for word_id in self._index_to_word:
            # find the topics that this word currently sits on
            word_topic_index = numpy.nonzero(self._n_kv[:, word_id]>0)[0];
            # find the frequency distribution of the words sitting on the current table
            selected_word_freq_dist = self._n_kv[topic_id, selected_word_index];
            
            
        for topic_id in numpy.argsort(self._m_k):
            # if this table is empty, no need to resample the topic assignment 
            if self._m_k[topic_id] <= 0:
                continue;
            
            # find the index of the words sitting on the current table
            selected_word_index = numpy.nonzero(self._n_kv[topic_id, :]>0)[0];
            # find the frequency distribution of the words sitting on the current table
            selected_word_freq_dist = self._n_kv[topic_id, selected_word_index];
            
            
            
            # compute the probability of assigning current table every topic
            log_topic_probability = numpy.zeros(self._K) + negative_infinity;
            
            n_k = numpy.sum(self._n_kv, axis=1);
            assert(len(n_k) == (self._K))
            #test_log_probability = numpy.zeros(self._K) + negative_infinity;
            
            for topic_index in xrange(self._K):
                # if current table is the only table assigned to current topic,
                # it means this topic is probably less useful or less generalizable to other documents,
                # it makes more sense to collapse this topic and hence assign this table to other topic.
                if self._m_k[topic_index]==0:
                    log_topic_probability[topic_index] = negative_infinity;
                    continue;
                
                if topic_index==topic_id:
                    log_topic_probability[topic_index] = numpy.log(self._alpha_gamma) * self._m_k[topic_id];
                    
                    log_topic_probability[topic_index] += scipy.special.gammaln(self._vocabulary_size * self._alpha_eta) - scipy.special.gammaln(self._vocabulary_size * self._alpha_eta + n_k[topic_id]);
                    for word_pos in xrange(len(selected_word_index)):
                        word_id = selected_word_index[word_pos];
                        log_topic_probability[topic_index] += scipy.special.gammaln(self._alpha_eta + selected_word_freq_dist[word_pos]) - scipy.special.gammaln(self._alpha_eta);
                        
                    #test_log_probability[topic_index] = numpy.log(self._alpha_gamma) * self._m_k[topic_id];
                else:
                    log_topic_probability[topic_index] = numpy.log(self._m_k[topic_index]) * self._m_k[topic_id];
                    
                    log_topic_probability[topic_index] += scipy.special.gammaln(self._vocabulary_size * self._alpha_eta + n_k[topic_index]) - scipy.special.gammaln(self._vocabulary_size * self._alpha_eta + n_k[topic_index] + n_k[topic_id]);
                    for word_pos in xrange(len(selected_word_index)):
                        word_id = selected_word_index[word_pos];
                        log_topic_probability[topic_index] += scipy.special.gammaln(self._n_kv[topic_index, word_id] + self._alpha_eta + selected_word_freq_dist[word_pos]) - scipy.special.gammaln(self._n_kv[topic_index, word_id] + self._alpha_eta);
                    
                    #test_log_probability[topic_index] = numpy.log(self._m_k[topic_index]) * self._m_k[topic_id];    
            # normalize the distribution and sample new topic assignment for this topic
            #log_topic_probability = numpy.exp(log_topic_probability);
            #log_topic_probability = log_topic_probability/numpy.sum(log_topic_probability);
            #log_topic_probability = numpy.exp(log_normalize(log_topic_probability));
            log_topic_probability -= scipy.misc.logsumexp(log_topic_probability);
            topic_probability = numpy.exp(log_topic_probability);
            print "topic merging probability of topic %d: %s" % (topic_id, topic_probability);
            cdf = numpy.cumsum(topic_probability);
            new_topic = numpy.uint8(numpy.nonzero(cdf >= numpy.random.random())[0][0]);
            
            #test_log_probability -= scipy.misc.logsumexp(test_log_probability);
            #print "test_log_probability:", numpy.exp(test_log_probability);
            
            # if the entire topic is assigned to a new topic
            if new_topic != topic_id:
                print "merge topic %d to topic %d..." % (topic_id, new_topic);
                print self._n_kv[topic_id, :];
                print self._n_kv[new_topic, :];
                
                self._m_k[new_topic] += self._m_k[topic_id];
                self._m_k[topic_id] = 0;
                self._n_kv[new_topic, :] += self._n_kv[topic_id, :];
                self._n_kv[topic_id, :] = 0;
                self._n_dk[:, new_topic] += self._n_dk[:, topic_id];
                self._n_dk[:, topic_id] = 0;
            
                assert self._n_kv.shape == (self._K, self._vocabulary_size);
                assert len(self._m_k) == self._K;
                assert self._n_dk.shape == (self._D, self._K);
                
                for d in xrange(self._D):
                    self._k_dt[d][numpy.nonzero(self._k_dt[d]==topic_id)[0]] = new_topic;

        return;

    def resample_topics_backup(self):
        # sample topic assignment, see which topic it should belong to
        #for topic_id in numpy.random.permutation(xrange(self._K)):
        for topic_id in numpy.argsort(self._m_k):
            # if this table is empty, no need to resample the topic assignment 
            if self._m_k[topic_id] <= 0:
                continue;
            
            # find the index of the words sitting on the current table
            selected_word_index = numpy.nonzero(self._n_kv[topic_id, :]>0)[0];
            # find the frequency distribution of the words sitting on the current table
            selected_word_freq_dist = self._n_kv[topic_id, selected_word_index];
            
            # compute the probability of assigning current table every topic
            log_topic_probability = numpy.zeros(self._K) + negative_infinity;
            
            n_k = numpy.sum(self._n_kv, axis=1);
            assert(len(n_k) == (self._K))
            #test_log_probability = numpy.zeros(self._K) + negative_infinity;
            
            for topic_index in xrange(self._K):
                # if current table is the only table assigned to current topic,
                # it means this topic is probably less useful or less generalizable to other documents,
                # it makes more sense to collapse this topic and hence assign this table to other topic.
                if self._m_k[topic_index]==0:
                    log_topic_probability[topic_index] = negative_infinity;
                    continue;
                
                if topic_index==topic_id:
                    log_topic_probability[topic_index] = numpy.log(self._alpha_gamma) * self._m_k[topic_id];
                    
                    log_topic_probability[topic_index] += scipy.special.gammaln(self._vocabulary_size * self._alpha_eta) - scipy.special.gammaln(self._vocabulary_size * self._alpha_eta + n_k[topic_id]);
                    for word_pos in xrange(len(selected_word_index)):
                        word_id = selected_word_index[word_pos];
                        log_topic_probability[topic_index] += scipy.special.gammaln(self._alpha_eta + selected_word_freq_dist[word_pos]) - scipy.special.gammaln(self._alpha_eta);
                        
                    #test_log_probability[topic_index] = numpy.log(self._alpha_gamma) * self._m_k[topic_id];
                else:
                    log_topic_probability[topic_index] = numpy.log(self._m_k[topic_index]) * self._m_k[topic_id];
                    
                    log_topic_probability[topic_index] += scipy.special.gammaln(self._vocabulary_size * self._alpha_eta + n_k[topic_index]) - scipy.special.gammaln(self._vocabulary_size * self._alpha_eta + n_k[topic_index] + n_k[topic_id]);
                    for word_pos in xrange(len(selected_word_index)):
                        word_id = selected_word_index[word_pos];
                        log_topic_probability[topic_index] += scipy.special.gammaln(self._n_kv[topic_index, word_id] + self._alpha_eta + selected_word_freq_dist[word_pos]) - scipy.special.gammaln(self._n_kv[topic_index, word_id] + self._alpha_eta);
                    
                    #test_log_probability[topic_index] = numpy.log(self._m_k[topic_index]) * self._m_k[topic_id];    
            # normalize the distribution and sample new topic assignment for this topic
            #log_topic_probability = numpy.exp(log_topic_probability);
            #log_topic_probability = log_topic_probability/numpy.sum(log_topic_probability);
            #log_topic_probability = numpy.exp(log_normalize(log_topic_probability));
            log_topic_probability -= scipy.misc.logsumexp(log_topic_probability);
            topic_probability = numpy.exp(log_topic_probability);
            print "topic merging probability of topic %d: %s" % (topic_id, topic_probability);
            cdf = numpy.cumsum(topic_probability);
            new_topic = numpy.uint8(numpy.nonzero(cdf >= numpy.random.random())[0][0]);
            
            #test_log_probability -= scipy.misc.logsumexp(test_log_probability);
            #print "test_log_probability:", numpy.exp(test_log_probability);
            
            # if the entire topic is assigned to a new topic
            if new_topic != topic_id:
                print "merge topic %d to topic %d..." % (topic_id, new_topic);
                print self._n_kv[topic_id, :];
                print self._n_kv[new_topic, :];
                
                self._m_k[new_topic] += self._m_k[topic_id];
                self._m_k[topic_id] = 0;
                self._n_kv[new_topic, :] += self._n_kv[topic_id, :];
                self._n_kv[topic_id, :] = 0;
                self._n_dk[:, new_topic] += self._n_dk[:, topic_id];
                self._n_dk[:, topic_id] = 0;
            
                assert self._n_kv.shape == (self._K, self._vocabulary_size);
                assert len(self._m_k) == self._K;
                assert self._n_dk.shape == (self._D, self._K);
                
                for d in xrange(self._D):
                    self._k_dt[d][numpy.nonzero(self._k_dt[d]==topic_id)[0]] = new_topic;

        return;

    """
    """
    def compact_params(self):
        # find unused and used topics
        unused_topics = numpy.nonzero(self._m_k == 0)[0];
        used_topics = numpy.nonzero(self._m_k != 0)[0];
        
        self._K -= len(unused_topics);
        assert(self._K >= 1 and self._K == len(used_topics));
        
        self._n_dk = numpy.delete(self._n_dk, unused_topics, axis=1);
        assert(self._n_dk.shape == (self._D, self._K));
        self._n_kv = numpy.delete(self._n_kv, unused_topics, axis=0);
        assert(self._n_kv.shape == (self._K, self._vocabulary_size));
        self._m_k = numpy.delete(self._m_k, unused_topics);
        assert(len(self._m_k) == self._K);
        
        for d in xrange(self._D):
            # find the unused and used tables
            unused_tables = numpy.nonzero(self._n_dt[d] == 0)[0];
            used_tables = numpy.nonzero(self._n_dt[d] != 0)[0];

            self._n_dt[d] = numpy.delete(self._n_dt[d], unused_tables);
            self._k_dt[d] = numpy.delete(self._k_dt[d], unused_tables);
            
            # shift down all the table indices of all words in current document
            # @attention: shift the used tables in ascending order only.
            for t in xrange(len(self._n_dt[d])):
                self._t_dv[d][numpy.nonzero(self._t_dv[d] == used_tables[t])[0]] = t;
            
            # shrink down all the topics indices of all tables in current document
            # @attention: shrink the used topics in ascending order only.
            for k in xrange(self._K):
                self._k_dt[d][numpy.nonzero(self._k_dt[d] == used_topics[k])[0]] = k;

    """
    compute the log likelihood of the model
    """
    def log_likelihood(self):
        log_likelihood = 0.;
        # compute the document level log likelihood
        log_likelihood += self.table_log_likelihood();
        # compute the table level log likelihood
        log_likelihood += self.topic_log_likelihood();
        # compute the word level log likelihood
        log_likelihood += self.word_log_likelihood();
        
        #todo: add in the likelihood for hyper-parameter
        
        return log_likelihood
        
    """
    compute the table level prior in log scale \prod_{d=1}^D (p(t_{d})), where p(t_d) = \frac{ \alpha^m_d \prod_{t=1}^{m_d}(n_di-1)! }{ \prod_{v=1}^{n_d}(v+\alpha-1) }
    """
    def table_log_likelihood(self):
        log_likelihood = 0.;
        for document_index in xrange(self._D):
            log_likelihood += len(self._k_dt[document_index]) * numpy.log(self._alpha_gamma) - log_factorial(len(self._t_dv[document_index]), self._alpha_gamma);
            for table_index in xrange(len(self._k_dt[document_index])):
                log_likelihood += scipy.special.gammaln(self._n_dt[document_index][table_index]);
            
        return log_likelihood
    
    """
    compute the topic level prior in log scale p(k) = \frac{ \gamma^K \prod_{k=1}^{K}(m_k-1)! }{ \prod_{s=1}^{m}(s+\gamma-1) }
    """
    def topic_log_likelihood(self):
        log_likelihood = self._K * numpy.log(self._alpha_alpha) - log_factorial(numpy.sum(self._m_k), self._alpha_alpha);
        for topic_index in xrange(self._K):
            log_likelihood += scipy.special.gammaln(self._m_k[topic_index]);
        
        return log_likelihood
    
    """
    compute the word level log likelihood p(x | t, k) = \prod_{k=1}^K f(x_{ij} | z_{ij}=k), where f(x_{ij} | z_{ij}=k) = \frac{\Gamma(V \eta)}{\Gamma(n_k + V \eta)} \frac{\prod_{v} \Gamma(n_{k}^{v} + \eta)}{\Gamma^V(\eta)}
    """
    def word_log_likelihood(self):
        n_k = numpy.sum(self._n_dk, axis=0);
        assert(len(n_k) == self._K);
        
        log_likelihood = self._K * scipy.special.gammaln(self._vocabulary_size * self._alpha_eta);
        for topic_index in xrange(self._K):
            log_likelihood -= scipy.special.gammaln(self._vocabulary_size * self._alpha_eta + n_k[topic_index]);
            for word_index in xrange(self._vocabulary_size):
                if self._n_kv[topic_index, word_index] > 0:
                    log_likelihood += scipy.special.gammaln(self._n_kv[topic_index, word_index] + self._alpha_eta) + scipy.special.gammaln(self._alpha_eta);
                    
        return log_likelihood
        
    """
    """
    def export_beta(self, exp_beta_path, top_display=-1):
        n_k_sum_over_v = numpy.sum(self._n_kv, axis=1)[:, numpy.newaxis];
        assert(n_k_sum_over_v.shape == (self._K, 1));
        beta_probability = (self._n_kv + self._alpha_eta) / (n_k_sum_over_v + self._vocabulary_size * self._alpha_eta);
        
        output = open(exp_beta_path, 'w');
        for topic_index in xrange(self._K):
            output.write("==========\t%d\t==========\n" % (topic_index));
            
            i=0;
            for word_index in reversed(numpy.argsort(beta_probability[topic_index, :])):
                i += 1;
                output.write(self._index_to_word[word_index] + "\t" + str(beta_probability[topic_index, word_index]) + "\n");
                if top_display>0 and i>=top_display:
                    break

        output.close();

























    """
    @param document_index: the document index to update
    @param word_index: the word index to update
    @param update: the update amount for this document and this word
    @attention: the update table index and topic index is retrieved from self._t_dv and self._k_dt, so make sure these values were set properly before invoking this function
    """
    def update_params(self, document_index, word_index, update):
        # retrieve the table_id of the current word of current document
        table_id = self._t_dv[document_index][word_index];
        # retrieve the topic_id of the table that current word of current document sit on
        topic_id = self._k_dt[document_index][table_id];
        # get the word_id of at the word_index of the document_index
        word_id = self._corpus[document_index][word_index];

        self._n_dt[document_index][table_id] += update;
        assert(numpy.all(self._n_dt[document_index] >= 0));
        self._n_kv[topic_id, word_id] += update;
        assert(numpy.all(self._n_kv >= 0));
        self._n_dk[document_index, topic_id] += update;
        assert(numpy.all(self._n_dk >= 0));
        
        # if current table in current document becomes empty 
        if update == -1 and self._n_dt[document_index][table_id] == 0:
            # adjust the table counts
            self._m_k[topic_id] -= 1;
            
        # if a new table is created in current document
        if update == 1 and self._n_dt[document_index][table_id] == 1:
            # adjust the table counts
            self._m_k[topic_id] += 1;
            
        assert(numpy.all(self._m_k >= 0));
        assert(numpy.all(self._k_dt[document_index] >= 0));        
        
"""
@param n: an integer data type
@param a: 
@attention: n must be an integer
this function is to compute the log(n!), since n!=Gamma(n+1), which means log(n!)=lngamma(n+1)
"""
def log_factorial(n, a):
    if n == 0:
        return 0.;
    return scipy.special.gammaln(n + a) - scipy.special.gammaln(a);

"""
"""
def print_topics(n_kv, term_mapping, top_words=10):
    input = open(term_mapping);
    vocab = {};
    i = 0;
    for line in input:
        vocab[i] = line.strip();
        i += 1;

    (K, V) = n_kv.shape;
    assert(V == len(vocab));

    if top_words >= V:
        sorted_counts = numpy.zeros((1, K)) - numpy.log(V);
    else:
        sorted_counts = numpy.sort(n_kv, axis=1);
        sorted_counts = sorted_counts[:, -top_words][:, numpy.newaxis];
    
    assert(sorted_counts.shape==(K, 1));

    for k in xrange(K):
        display = (n_kv[[k], :] >= sorted_counts[k, :]);
        assert(display.shape == (1, V));
        output_str = str(k) + ": ";
        for v in xrange(self._vocabulary_size):
            if display[:, v]:
                output_str += vocab[v] + "\t";
        print output_str

"""
run HDP on a synthetic corpus.
"""
if __name__ == '__main__':
    a = numpy.random.random((2, 3));
    print a;
    print a/numpy.sum(a, axis=1)[:, numpy.newaxis];