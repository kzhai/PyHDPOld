"""
@author: Ke Zhai (zhaike@cs.umd.edu)

This code was modified from the code originally written by Chong Wang (chongw@cs.princeton.edu).
Implements uncollapsed Gibbs sampling for the hierarchical Dirichlet process (HDP).

References:
[1] Chong Wang and David Blei, A Split-Merge MCMC Algorithm for the Hierarchical Dirichlet Process, available online www.cs.princeton.edu/~chongw/papers/sm-hdp.pdf.
"""

import copy
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
                 split_merge_heuristics=0,
                 split_proposal=0,
                 merge_proposal=0,
                 split_merge_iteration=1,
                 restrict_gibbs_sampling_iteration=10,
                 hash_oov_words=False
                 ):
        self._split_merge_heuristics = split_merge_heuristics;
        self._split_proposal = split_proposal;
        self._merge_proposal = merge_proposal;
        
        self._split_merge_iteration = split_merge_iteration;
        self._restrict_gibbs_sampling_iteration = restrict_gibbs_sampling_iteration;

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
            
        self._vocabulary = self._word_to_index.keys();
        self._vocabulary_size = len(self._vocabulary);
        
        # top level smoothing
        self._alpha_alpha = alpha_alpha
        # bottom level smoothing
        self._alpha_gamma = alpha_gamma
        # vocabulary smoothing
        if alpha_eta <= 0:
            self._alpha_eta = 1.0 / self._vocabulary_size;
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
            assert (len(self._k_dt[d]) == len(numpy.unique(self._t_dv[d]))), (len(self._k_dt[d]), self._t_dv[d], len(numpy.unique(self._t_dv[d])));
            
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
            words = docs[d].split()
            wordid = list();
            for word in words:
                if (word in self._vocabulary):
                    wordtoken = self._word_to_index[word]
                    wordid.append(wordtoken);
                else:
                    if self._hash_oov_words:
                        wordtoken = hash(word) % len(self._vocabulary);
                        wordid.append(wordtoken);
                        
            wordids.append(wordid)
            
        return wordids

    def learning(self):
        self._counter += 1;
        
        self.sample_cgs();
        
        if self._split_merge_heuristics > 0:
            self.sample_topics();
    
        print "accumulated number of tables:", self._m_k;
        print "accumulated number of tokens:", numpy.sum(self._n_kv, axis=1)[:, numpy.newaxis].T;
        
        return self.log_posterior();

    """
    sample the data to train the parameters
    """
    def sample_cgs(self):
        # sample the total data
        for document_index in numpy.random.permutation(xrange(self._D)):
            # sample word assignment, see which table it should belong to
            for word_index in numpy.random.permutation(xrange(len(self._corpus[document_index]))):
                # get the word_id of at the word_index of the document_index
                word_id = self._corpus[document_index][word_index];
                
                # retrieve the old_table_id of the current word of current document
                old_table_id = self._t_dv[document_index][word_index];
                # retrieve the old_topic_id of the table that current word of current document sit on
                old_topic_id = self._k_dt[document_index][old_table_id];
        
                self._n_dt[document_index][old_table_id] -= 1;
                assert(numpy.all(self._n_dt[document_index] >= 0));
                self._n_kv[old_topic_id, word_id] -= 1;
                assert(numpy.all(self._n_kv >= 0));
                self._n_dk[document_index, old_topic_id] -= 1;
                assert(numpy.all(self._n_dk >= 0));
                
                # if current table in current document becomes empty 
                if self._n_dt[document_index][old_table_id] == 0:
                    # adjust the table counts
                    self._m_k[old_topic_id] -= 1;
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
                new_table_id = numpy.uint8(numpy.nonzero(cdf >= numpy.random.random())[0][0]);
                
                # if current word sits on a new table, we need to get the topic of that table
                if new_table_id == len(self._k_dt[document_index]):
                    if self._n_dt[document_index][old_table_id] == 0:
                        # if the old table is empty, reuse it
                        new_table_id = old_table_id;
                    else:
                        # else expand the vectors to fit in new table
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
                    new_topic_id = numpy.uint8(numpy.nonzero(cdf >= numpy.random.random())[0][0]);
                    
                    # if current table requires a new topic
                    if new_topic_id == self._K:
                        if self._m_k[old_topic_id] == 0:
                            # if the old topic is empty, reuse it
                            new_topic_id = old_topic_id;
                        else:
                            # else expand matrices to fit in new topic
                            self._K += 1;
                            self._n_kv = numpy.vstack((self._n_kv, numpy.zeros((1, self._vocabulary_size))));
                            assert(self._n_kv.shape == (self._K, self._vocabulary_size));
                            self._n_dk = numpy.hstack((self._n_dk, numpy.zeros((self._D, 1))));
                            assert(self._n_dk.shape == (self._D, self._K));
                            self._m_k = numpy.hstack((self._m_k, numpy.zeros(1)));
                            assert(len(self._m_k) == self._K);
                    
                    # assign current table to new topic
                    self._k_dt[document_index][new_table_id] = new_topic_id;
                    
                # assign current word to new table
                self._t_dv[document_index][word_index] = new_table_id;
                
                # retrieve the new_table_id of the current word of current document
                new_table_id = self._t_dv[document_index][word_index];
                # retrieve the new_topic_id of the table that current word of current document sit on
                new_topic_id = self._k_dt[document_index][new_table_id];
        
                self._n_dt[document_index][new_table_id] += 1;
                assert(numpy.all(self._n_dt[document_index] >= 0));
                self._n_kv[new_topic_id, word_id] += 1;
                assert(numpy.all(self._n_kv >= 0));
                self._n_dk[document_index, new_topic_id] += 1;
                assert(numpy.all(self._n_dk >= 0));
                
                # if a new table is created in current document
                if self._n_dt[document_index][new_table_id] == 1:
                    # adjust the table counts
                    self._m_k[new_topic_id] += 1;
                    
                assert(numpy.all(self._m_k >= 0));
                assert(numpy.all(self._k_dt[document_index] >= 0));
            
            # sample table assignment, see which topic it should belong to
            for table_index in numpy.random.permutation(xrange(len(self._k_dt[document_index]))):
                self.sample_tables(document_index, table_index);
            
        # compact all the parameters, including removing unused topics and unused tables
        self.compact_params();

    def sample_tables(self, document_index, table_index):
        # if this table is empty, skip the sampling directly
        if self._n_dt[document_index][table_index] <= 0:
            continue;
            
        old_topic_id = self._k_dt[document_index][table_index];

        # find the index of the words sitting on the current table
        selected_word_index = numpy.nonzero(self._t_dv[document_index] == table_index)[0];
        # find the frequency distribution of the words sitting on the current table
        selected_word_freq_dist = nltk.probability.FreqDist([self._corpus[document_index][term] for term in list(selected_word_index)]);

        # compute the probability of assigning current table every topic
        topic_log_probability = numpy.zeros(self._K + 1);
        
        topic_log_probability[self._K] = scipy.special.gammaln(self._vocabulary_size * self._alpha_eta) - scipy.special.gammaln(self._n_dt[document_index][table_index] + self._vocabulary_size * self._alpha_eta);
        for word_id in selected_word_freq_dist.keys():
            topic_log_probability[self._K] += scipy.special.gammaln(selected_word_freq_dist[word_id] + self._alpha_eta) - scipy.special.gammaln(self._alpha_eta);
        topic_log_probability[self._K] += numpy.log(self._alpha_alpha);
        
        n_k = numpy.sum(self._n_kv, axis=1);
        assert(len(n_k) == (self._K))
        for topic_index in xrange(self._K):
            # if current table is the only table assigned to current topic,
            if self._m_k[topic_index] <= 1:
                # it means this topic is probably less useful or less generalizable to other documents,
                # it makes more sense to collapse this topic and hence assign this table to other topic.
                topic_log_probability[topic_index] = negative_infinity;
                continue;

            # if there are other tables assigned to current topic
            if topic_index == old_topic_id:
                topic_log_probability[topic_index] = scipy.special.gammaln(self._vocabulary_size * self._alpha_eta + n_k[topic_index] - self._n_dt[document_index][table_index]) - scipy.special.gammaln(self._vocabulary_size * self._alpha_eta + n_k[topic_index]);
                for word_id in selected_word_freq_dist.keys():
                    topic_log_probability[topic_index] += scipy.special.gammaln(self._n_kv[topic_index, word_id] + self._alpha_eta) - scipy.special.gammaln(self._n_kv[topic_index, word_id] + self._alpha_eta - selected_word_freq_dist[word_id]);
                # compute the prior if we move this table from this topic
                topic_log_probability[topic_index] += numpy.log(self._m_k[topic_index] - 1);
            else:
                topic_log_probability[topic_index] = scipy.special.gammaln(self._vocabulary_size * self._alpha_eta + n_k[topic_index]) - scipy.special.gammaln(self._vocabulary_size * self._alpha_eta + n_k[topic_index] + self._n_dt[document_index][table_index]);
                for word_id in selected_word_freq_dist.keys():
                    topic_log_probability[topic_index] += scipy.special.gammaln(self._n_kv[topic_index, word_id] + self._alpha_eta + selected_word_freq_dist[word_id]) - scipy.special.gammaln(self._n_kv[topic_index, word_id] + self._alpha_eta);
                # compute the prior if we move this table from this topic
                topic_log_probability[topic_index] += numpy.log(self._m_k[topic_index]);

        # normalize the distribution and sample new topic assignment for this topic
        # topic_log_probability = numpy.exp(topic_log_probability);
        # topic_log_probability = topic_log_probability/numpy.sum(topic_log_probability);
        # topic_log_probability = numpy.exp(log_normalize(topic_log_probability));
        topic_log_probability -= scipy.misc.logsumexp(topic_log_probability);
        topic_log_probability = numpy.exp(topic_log_probability);
        
        cdf = numpy.cumsum(topic_log_probability);
        new_topic_id = numpy.uint8(numpy.nonzero(cdf >= numpy.random.random())[0][0]);
        
        if new_topic_id == old_topic_id:
            continue;
    
        # if the table is assigned to a new topic
        if new_topic_id == self._K:
            if self._m_k[old_topic_id] <= 1:
                # if old topic is empty, reuse it
                new_topic_id = old_topic_id
            else:
                # else expand all matrices to fit new topics
                self._K += 1;
                self._n_dk = numpy.hstack((self._n_dk, numpy.zeros((self._D, 1))));
                assert(self._n_dk.shape == (self._D, self._K));
                
                self._n_kv = numpy.vstack((self._n_kv, numpy.zeros((1, self._vocabulary_size))));
                assert(self._n_kv.shape == (self._K, self._vocabulary_size));
                self._m_k = numpy.hstack((self._m_k, numpy.zeros(1)));
                assert(len(self._m_k) == self._K);
            
        # assign this table to new topic
        self._k_dt[document_index][table_index] = new_topic_id;                
        
        # adjust the statistics of all model parameter
        self._m_k[old_topic_id] -= 1;
        self._m_k[new_topic_id] += 1;
        self._n_dk[document_index, old_topic_id] -= self._n_dt[document_index][table_index];
        self._n_dk[document_index, new_topic_id] += self._n_dt[document_index][table_index];
        for word_id in selected_word_freq_dist.keys():
            self._n_kv[old_topic_id, word_id] -= selected_word_freq_dist[word_id];
            assert(self._n_kv[old_topic_id, word_id] >= 0)
            self._n_kv[new_topic_id, word_id] += selected_word_freq_dist[word_id];

    def split_merge(self):
        for iteration in xrange(self._split_merge_iteration):
            if self._split_merge_heuristics == 1:
                temp_cluster_probability = numpy.random.multinomial(1, self._m_k)[numpy.newaxis, :];
                random_label_1 = numpy.nonzero(temp_cluster_probability == 1)[1][0];
                temp_cluster_probability = numpy.random.multinomial(1, self._m_k)[numpy.newaxis, :];
                random_label_2 = numpy.nonzero(temp_cluster_probability == 1)[1][0];
            elif self._split_merge_heuristics == 2:
                temp_cluster_probability = numpy.random.multinomial(1, self._m_k)[numpy.newaxis, :];
                random_label_1 = numpy.nonzero(temp_cluster_probability == 1)[1][0];
                random_label_2 = numpy.random.randint(self._K);
            elif self._split_merge_heuristics == 3:
                random_label_1 = numpy.random.randint(self._K);
                random_label_2 = numpy.random.randint(self._K);
            else:
                sys.stderr.write("error: unrecognized split-merge heuristics %d...\n" % (self._split_merge_heuristics));
                return;
        
            if random_label_1 == random_label_2:
                self.split_metropolis_hastings(random_label_1);
            else:
                self.merge_metropolis_hastings(random_label_1, random_label_2);

    def split_metropolis_hastings(self, cluster_label):
        # record down the old cluster assignment
        old_log_posterior = self.log_posterior();

        proposed_K = self._K;
        
        proposed_n_kv = numpy.copy(self._n_kv);
        proposed_m_k = numpy.copy(self._m_k);
        proposed_n_dk = numpy.copy(self._n_dk);
        
        proposed_n_dt = copy.deepcopy(self._n_dt);
        
        proposed_t_dv = copy.deepcopy(self._t_dv);
        proposed_k_dt = copy.deepcopy(self._k_dt);
        
        model_parameter = (proposed_K, proposed_n_kv, proposed_m_k, proposed_n_dk, proposed_n_dt, proposed_t_dv, proposed_k_dt);

        if self._split_proposal == 0:
            # perform random split for split proposal
            model_parameter = self.random_split(cluster_label, model_parameter);
            if model_parameter == None:
                return;
        
            (proposed_K, proposed_n_kv, proposed_m_k, proposed_n_dk, proposed_n_dt, proposed_t_dv, proposed_k_dt) = model_parameter;
            log_proposal_probability = (proposed_m_k[cluster_label] + proposed_m_k[proposed_K - 1] - 2) * numpy.log(2);
        elif self._split_proposal == 1:
            # perform restricted gibbs sampling for split proposal
            model_parameter = self.random_split(cluster_label, model_parameter);
            # split a singleton cluster
            if model_parameter == None:
                return;
            
            model_parameter, transition_log_probability = self.restrict_gibbs_sampling(cluster_label, proposed_K - 1, model_parameter, self._restrict_gibbs_sampling_iteration + 1);
            
            (proposed_K, proposed_n_kv, proposed_m_k, proposed_n_dk, proposed_n_dt, proposed_t_dv, proposed_k_dt) = model_parameter;
            
            log_proposal_probability = transition_log_probability;
        elif self._split_proposal == 2:
            pass;
        else:
            pass        
        
        new_log_posterior = self.log_posterior(model_parameter);
        
        acceptance_log_probability = log_proposal_probability + new_log_posterior - old_log_posterior;
        acceptance_probability = numpy.exp(acceptance_log_probability);
        
        (proposed_K, proposed_n_kv, proposed_m_k, proposed_n_dk, proposed_n_dt, proposed_t_dv, proposed_k_dt) = model_parameter;
        
        if numpy.random.random() < acceptance_probability:
            print "split operation granted from %s to %s with acceptance probability %s" % (self._m_k, proposed_m_k, acceptance_probability);
            
            self._K = proposed_K;
        
            self._n_kv = proposed_n_kv;
            self._m_k = proposed_m_k;
            self._n_dk = proposed_n_dk;
            
            self._n_dt = proposed_n_dt;
            
            self._t_dv = proposed_t_dv;
            self._k_dt = proposed_k_dt;
        
        #
        #
        #
        #
        #
        
        '''
        if self._split_proposal == 1:
            # data_point_indices = numpy.nonzero(proposed_label==cluster_label)[0];
            
            # perform restricted gibbs sampling for split proposal
            model_parameter = self.random_split(cluster_label, model_parameter);
            # split a singleton cluster
            if model_parameter == None:
                return;
            
            (proposed_label, proposed_K, proposed_count, proposed_mu, proposed_sum, proposed_log_sigma_det, proposed_sigma_inv) = model_parameter;
            model_parameter = self.restrict_gibbs_sampling(cluster_label, proposed_K - 1, model_parameter, self._restrict_gibbs_sampling_iteration);
            (proposed_label, proposed_K, proposed_count, proposed_mu, proposed_sum, proposed_log_sigma_det, proposed_sigma_inv) = model_parameter;
            
            if proposed_count[cluster_label] == 0 or proposed_count[proposed_K - 1] == 0:
                return;
            
            self._K = proposed_K
            self._label = proposed_label;
            
            self._count = proposed_count;
            self._sum = proposed_sum;
            
            self._mu = proposed_mu;
            self._sigma_inv = proposed_sigma_inv;
            self._log_sigma_det = proposed_log_sigma_det;
            
            assert numpy.all(self._count > 0)
            old_log_posterior = self.log_posterior();
            
            model_parameter = self.restrict_gibbs_sampling(cluster_label, proposed_K - 1, model_parameter);
            
            if proposed_count[cluster_label] == 0 or proposed_count[proposed_K - 1] == 0:
                return;
        elif self._split_proposal == 2:
            # perform sequential allocation gibbs sampling for split proposal
            model_parameter = self.sequential_allocation(cluster_label, model_parameter);
        else:
            sys.stderr.write("error: unrecognized split proposal strategy %d...\n" % (self._split_proposal));
        '''

    def random_split(self, component_index, model_parameter):
        # sample the data points set
        (proposed_K, proposed_n_kv, proposed_m_k, proposed_n_dk, proposed_n_dt, proposed_t_dv, proposed_k_dt) = model_parameter;

        if proposed_m_k[component_index] <= 1:
            return None;
        
        number_of_unvisited_target_tables = proposed_m_k[component_index];
        
        proposed_K += 1;
        
        proposed_n_dk = numpy.hstack((proposed_n_dk, numpy.zeros((self._D, 1))));
        assert(proposed_n_dk.shape == (self._D, proposed_K));
                    
        proposed_n_kv = numpy.vstack((proposed_n_kv, numpy.zeros((1, self._vocabulary_size))));
        assert(proposed_n_kv.shape == (proposed_K, self._vocabulary_size));

        proposed_m_k = numpy.hstack((proposed_m_k, numpy.zeros(1)));
        assert(len(proposed_m_k) == proposed_K);

        for document_index in numpy.random.permutation(xrange(self._D)):
            for table_index in numpy.random.permutation(xrange(len(proposed_k_dt[document_index]))):
                if proposed_k_dt[document_index][table_index] != component_index:
                    continue;
                
                if numpy.random.random() < 0.5:
                    proposed_k_dt[document_index][table_index] = proposed_K - 1;
                    
                    proposed_m_k[component_index] -= 1;
                    proposed_m_k[proposed_K - 1] += 1;
                    
                    proposed_n_dk[document_index, component_index] -= proposed_n_dt[table_index];
                    proposed_n_dk[document_index, proposed_K - 1] += proposed_n_dt[table_index];
                    
                    selected_word_index = numpy.nonzero(proposed_t_dv[document_index] == table_index)[0];
                    for word_index in selected_word_index:
                        proposed_n_kv[component_index, word_index] -= 1;
                        proposed_n_kv[proposed_K - 1, word_index] += 1;

                number_of_unvisited_target_tables -= 1;
                
                if number_of_unvisited_target_tables == 0:
                    break;
                
            if number_of_unvisited_target_tables == 0:
                    break;
                
        if proposed_m_k[component_index] == 0 or proposed_m_k[proposed_K - 1] == 0:
            return None;
        else:
            model_parameter = (proposed_K, proposed_n_kv, proposed_m_k, proposed_n_dk, proposed_n_dt, proposed_t_dv, proposed_k_dt);
            return model_parameter

    def restrict_gibbs_sampling(self, cluster_index_1, cluster_index_2, model_parameter, restricted_gibbs_sampling_iteration=1):
        (proposed_K, proposed_n_kv, proposed_m_k, proposed_n_dk, proposed_n_dt, proposed_t_dv, proposed_k_dt) = model_parameter;

        document_table_indices = [];
        for document_index in xrange(self._D):
            for table_index in xrange(len(proposed_k_dt[document_index])):
                if proposed_k_dt[document_index][table_index] == cluster_index_1 or proposed_k_dt[document_index][table_index] == cluster_index_2:
                    document_table_indices.append((document_index, table_index));
                    
                if len(document_table_indices) == proposed_m_k[cluster_index_1] + proposed_m_k[cluster_index_2]:
                    break;
            if len(document_table_indices) == proposed_m_k[cluster_index_1] + proposed_m_k[cluster_index_2]:
                    break;
                
        assert len(document_table_indices) == proposed_m_k[cluster_index_1] + proposed_m_k[cluster_index_2];
        
        for restricted_gibbs_sampling_iteration_index in xrange(restricted_gibbs_sampling_iteration):
            transition_log_probability = 0;
            for (document_index, table_index) in numpy.random.permutation(document_table_indices):
                current_topic_id = proposed_k_dt[document_index][table_index];
                
                if current_topic_id == cluster_index_1:
                    other_topic_id = cluster_index_2;
                elif current_topic_id == cluster_index_2:
                    other_topic_id = cluster_index_1;
                else:
                    sys.stderr.write("error: table does not belong to proposed split clusters...\n");
        
                # find the index of the words sitting on the current table
                selected_word_index = numpy.nonzero(proposed_t_dv[document_index] == table_index)[0];
                # find the frequency distribution of the words sitting on the current table
                selected_word_freq_dist = nltk.probability.FreqDist([self._corpus[document_index][term] for term in list(selected_word_index)]);
        
                n_k = numpy.sum(proposed_n_kv, axis=1);
                
                if proposed_m_k[current_topic_id] <= 1:
                    # if current table is the only table assigned to current topic,
                    current_topic_probability = scipy.special.gammaln(self._vocabulary_size * self._alpha_eta);
                    current_topic_probability -= scipy.special.gammaln(self._vocabulary_size * self._alpha_eta + proposed_n_dt[document_index][table_index]);
                    for word_id in selected_word_freq_dist.keys():
                        current_topic_probability += scipy.special.gammaln(selected_word_freq_dist[word_id] + self._alpha_eta)
                        current_topic_probability -= scipy.special.gammaln(self._alpha_eta);
                    current_topic_probability += numpy.log(self._alpha_alpha);
                else:
                    # if there are other tables assigned to current topic
                    current_topic_probability = scipy.special.gammaln(self._vocabulary_size * self._alpha_eta + n_k[current_topic_id] - proposed_n_dt[document_index][table_index])
                    current_topic_probability -= scipy.special.gammaln(self._vocabulary_size * self._alpha_eta + n_k[current_topic_id]);
                    for word_id in selected_word_freq_dist.keys():
                        current_topic_probability += scipy.special.gammaln(proposed_n_kv[current_topic_id, word_id] + self._alpha_eta); 
                        current_topic_probability -= scipy.special.gammaln(proposed_n_kv[current_topic_id, word_id] + self._alpha_eta - selected_word_freq_dist[word_id]);
                    # compute the prior if we move this table from this topic
                    current_topic_probability += numpy.log(proposed_m_k[current_topic_id] - 1);
                
                if proposed_m_k[other_topic_id] <= 0:
                    # if current table is the only table assigned to current topic,
                    other_topic_probability = scipy.special.gammaln(self._vocabulary_size * self._alpha_eta);
                    other_topic_probability -= scipy.special.gammaln(self._vocabulary_size * self._alpha_eta + proposed_n_dt[document_index][table_index]);
                    for word_id in selected_word_freq_dist.keys():
                        other_topic_probability += scipy.special.gammaln(selected_word_freq_dist[word_id] + self._alpha_eta)
                        other_topic_probability -= scipy.special.gammaln(self._alpha_eta);
                    other_topic_probability += numpy.log(self._alpha_alpha);
                else:
                    other_topic_probability = scipy.special.gammaln(self._vocabulary_size * self._alpha_eta + n_k[other_topic_id]);
                    other_topic_probability -= scipy.special.gammaln(self._vocabulary_size * self._alpha_eta + n_k[other_topic_id] + proposed_n_dt[document_index][table_index]);
                    for word_id in selected_word_freq_dist.keys():
                        other_topic_probability += scipy.special.gammaln(proposed_n_kv[other_topic_id, word_id] + self._alpha_eta + selected_word_freq_dist[word_id]);
                        other_topic_probability -= scipy.special.gammaln(proposed_n_kv[other_topic_id, word_id] + self._alpha_eta);
                    other_topic_probability += numpy.log(proposed_m_k[other_topic_id]);
                    
                # sample a new cluster label for current point
                ratio_current_over_other = numpy.exp(current_topic_probability - other_topic_probability);
                probability_other_topic = 1. / (1. + ratio_current_over_other);
                
                # if this table does not change topic assignment
                if numpy.random.random() > probability_other_topic:
                    transition_log_probability += numpy.log(1 - probability_other_topic);
                    continue;
                
                transition_log_probability += numpy.log(probability_other_topic);
                
                # assign this table to new topic
                proposed_k_dt[document_index][table_index] = other_topic_id;
                
                # adjust the statistics of all model parameter
                proposed_m_k[current_topic_id] -= 1;
                proposed_m_k[other_topic_id] += 1;
                proposed_n_dk[document_index, current_topic_id] -= proposed_n_dt[document_index][table_index];
                proposed_n_dk[document_index, other_topic_id] += proposed_n_dt[document_index][table_index];
                for word_id in selected_word_freq_dist.keys():
                    proposed_n_kv[current_topic_id, word_id] -= selected_word_freq_dist[word_id];
                    assert(proposed_n_kv[current_topic_id, word_id] >= 0)
                    proposed_n_kv[other_topic_id, word_id] += selected_word_freq_dist[word_id];
                    
        model_parameter = (proposed_K, proposed_n_kv, proposed_m_k, proposed_n_dk, proposed_n_dt, proposed_t_dv, proposed_k_dt);
        return model_parameter, transition_log_probability;
        
    def merge_metropolis_hastings(self, component_index_1, component_index_2):
        old_log_posterior = self.log_posterior();
        
        # this is to switch the label, make sure we always 
        if component_index_1 > component_index_2:
            temp_random_label = component_index_1;
            component_index_1 = component_index_2;
            component_index_2 = temp_random_label;

        proposed_K = self._K;
        
        proposed_n_kv = numpy.copy(self._n_kv);
        proposed_m_k = numpy.copy(self._m_k);
        proposed_n_dk = numpy.copy(self._n_dk);
        
        proposed_n_dt = copy.deepcopy(self._n_dt);
        
        proposed_t_dv = copy.deepcopy(self._t_dv);
        proposed_k_dt = copy.deepcopy(self._k_dt);
        
        model_parameter = (proposed_K, proposed_n_kv, proposed_m_k, proposed_n_dk, proposed_n_dt, proposed_t_dv, proposed_k_dt);

        if self._merge_proposal == 0:            
            # perform random merge for merge proposal
            model_parameter = self.random_merge(component_index_1, component_index_2, model_parameter);
            
            (proposed_K, proposed_n_kv, proposed_m_k, proposed_n_dk, proposed_n_dt, proposed_t_dv, proposed_k_dt) = model_parameter;
            
            log_proposal_probability = -(proposed_m_k[component_index_1] - 2) * numpy.log(2);
        elif self._merge_proposal == 1:
            # perform restricted gibbs sampling for merge proposal
            model_parameter, transition_log_probability = self.restrict_gibbs_sampling(component_index_1, component_index_2, model_parameter, self._restrict_gibbs_sampling_iteration + 1);
            
            (proposed_K, proposed_n_kv, proposed_m_k, proposed_n_dk, proposed_n_dt, proposed_t_dv, proposed_k_dt) = model_parameter;
            
            if proposed_m_k[component_index_1] == 0 or proposed_m_k[component_index_2] == 0:
                print "merge cluster %d and %d during restricted gibbs sampling step..." % (component_index_1, component_index_2);
                
                if proposed_m_k[component_index_1] == 0:
                    collapsed_cluster = component_index_1;
                elif proposed_m_k[component_index_2] == 0:
                    collapsed_cluster = component_index_2;

                # since one cluster is empty now, switch it with the last one
                proposed_n_kv[collapsed_cluster, :] = proposed_n_kv[proposed_K - 1, :];
                proposed_m_k[collapsed_cluster] = proposed_m_k[proposed_K - 1];
                proposed_n_dk[:, collapsed_cluster] = proposed_n_dk[:, proposed_K - 1];
                
                for document_index in xrange(self._D):
                    proposed_k_dt[document_index][numpy.nonzero(proposed_k_dt[document_index] == (proposed_K - 1))] = collapsed_cluster;
                
                # remove the very last empty cluster, to remain compact cluster
                proposed_n_kv = numpy.delete(proposed_n_kv, [proposed_K - 1], axis=0);
                proposed_m_k = numpy.delete(proposed_m_k, [proposed_K - 1], axis=0);
                proposed_n_dk = numpy.delete(proposed_n_dk, [proposed_K - 1], axis=1);
                
                proposed_count = numpy.delete(proposed_count, [proposed_K - 1], axis=0);
                proposed_sum = numpy.delete(proposed_sum, [proposed_K - 1], axis=0);
                proposed_mu = numpy.delete(proposed_mu, [proposed_K - 1], axis=0);
                proposed_sigma_inv = numpy.delete(proposed_sigma_inv, [proposed_K - 1], axis=0);
                proposed_log_sigma_det = numpy.delete(proposed_log_sigma_det, [proposed_K - 1], axis=0);
                proposed_K -= 1;
                
                self._K = proposed_K
                self._label = proposed_label;
                
                self._count = proposed_count;
                self._sum = proposed_sum;
                
                self._mu = proposed_mu;
                self._sigma_inv = proposed_sigma_inv;
                self._log_sigma_det = proposed_log_sigma_det;
                
                assert numpy.all(self._count > 0)
                    
                return;
            
            log_proposal_probability = transition_log_probability;            
        elif self._merge_proposal == 2:
            pass
        else:
            pass
            
        new_log_posterior = self.log_posterior(model_parameter);
        
        acceptance_log_probability = log_proposal_probability + new_log_posterior - old_log_posterior;
        acceptance_probability = numpy.exp(acceptance_log_probability);
        
        (proposed_K, proposed_n_kv, proposed_m_k, proposed_n_dk, proposed_n_dt, proposed_t_dv, proposed_k_dt) = model_parameter;
        
        if numpy.random.random() < acceptance_probability:
            print "merge operation granted from %s to %s with acceptance probability %s" % (self._m_k, proposed_m_k, acceptance_probability);
            
            self._K = proposed_K;
        
            self._n_kv = proposed_n_kv;
            self._m_k = proposed_m_k;
            self._n_dk = proposed_n_dk;
            
            self._n_dt = proposed_n_dt;
            
            self._t_dv = proposed_t_dv;
            self._k_dt = proposed_k_dt;
            
            
        #
        #
        #
        #
        #
        '''
        if self._merge_proposal == 1:
            # perform restricted gibbs sampling for merge proposal
            # (proposed_label, proposed_K, proposed_count, proposed_mu, proposed_sum, proposed_log_sigma_det, proposed_sigma_inv) = model_parameter;
            model_parameter = self.restrict_gibbs_sampling(component_index_1, component_index_2, model_parameter, self._restrict_gibbs_sampling_iteration);
            (proposed_label, proposed_K, proposed_count, proposed_mu, proposed_sum, proposed_log_sigma_det, proposed_sigma_inv) = model_parameter;

            self._K = proposed_K
            self._label = proposed_label;
            
            self._count = proposed_count;
            self._sum = proposed_sum;
            
            self._mu = proposed_mu;
            self._sigma_inv = proposed_sigma_inv;
            self._log_sigma_det = proposed_log_sigma_det;
            
            collapsed_cluster = -1;
            if proposed_count[component_index_1] == 0:
                collapsed_cluster = component_index_1;
            elif proposed_count[component_index_2] == 0:
                collapsed_cluster = component_index_2;
            
            if collapsed_cluster != -1:
                print "merge cluster %d and %d during restricted gibbs sampling step..." % (component_index_1, component_index_2);
                
                # remove the empty cluster, to remain compact cluster
                self._count = numpy.delete(self._count, [collapsed_cluster], axis=0);
                self._sum = numpy.delete(self._sum, [collapsed_cluster], axis=0);
                self._mu = numpy.delete(self._mu, [collapsed_cluster], axis=0);
                self._sigma_inv = numpy.delete(self._sigma_inv, [collapsed_cluster], axis=0);
                self._log_sigma_det = numpy.delete(self._log_sigma_det, [collapsed_cluster], axis=0);
                self._K -= 1;
            
                assert numpy.all(self._count > 0)
                    
                return;
            else:
                old_log_posterior = self.log_posterior();
        elif self._merge_proposal == 2:
            # perform gibbs sampling for merge proposal
            cluster_log_probability = numpy.log(proposed_count);
            cluster_log_probability = numpy.sum(cluster_log_probability) - cluster_log_probability;
            cluster_log_probability -= scipy.misc.logsumexp(cluster_log_probability);
    
            cluster_probability = numpy.exp(cluster_log_probability);
            temp_cluster_probability = numpy.random.multinomial(1, cluster_probability)[numpy.newaxis, :];
            cluster_label = numpy.nonzero(temp_cluster_probability == 1)[1][0];
            
            model_parameter = self.merge_gibbs_sampling(cluster_label, model_parameter);
            
            if model_parameter == None:
                return;
        else:
            sys.stderr.write("error: unrecognized merge proposal strategy %d...\n" % (self._merge_proposal));
        
        (proposed_label, proposed_K, proposed_count, proposed_mu, proposed_sum, proposed_log_sigma_det, proposed_sigma_inv) = model_parameter;
        assert numpy.all(proposed_count > 0)
        
        
        
        assert proposed_K == len(proposed_log_sigma_det);
        assert numpy.max(proposed_label) < len(proposed_log_sigma_det);
        assert numpy.sum(proposed_count) == self._N;
        assert proposed_mu.shape == (proposed_K, self._D);

        # model_parameter = (proposed_label, proposed_K, proposed_count, proposed_mu, proposed_sum, proposed_log_sigma_det, proposed_sigma_inv);
        new_log_posterior = self.log_posterior(model_parameter);
        
        acceptance_log_probability = log_proposal_probability + new_log_posterior - old_log_posterior;
        acceptance_probability = numpy.exp(acceptance_log_probability);
        
        if numpy.random.random() < acceptance_probability:
            print "merge operation granted from %s to %s with acceptance probability %s" % (self._count, proposed_count, acceptance_probability);
            
            self._K = proposed_K
            self._label = proposed_label;
            
            self._count = proposed_count;
            self._sum = proposed_sum;
            
            self._mu = proposed_mu;
            self._sigma_inv = proposed_sigma_inv;
            self._log_sigma_det = proposed_log_sigma_det;
            
        assert self._count.shape == (self._K,), (self._count.shape, self._K)
        assert self._sum.shape == (self._K, self._D);
        assert self._mu.shape == (self._K, self._D);
        assert self._sigma_inv.shape == (self._K, self._D, self._D);
        assert self._log_sigma_det.shape == (self._K,);
        '''

    def random_merge(self, component_index_1, component_index_2, model_parameter):
        assert component_index_2 > component_index_1;
        
        # sample the data points set
        (proposed_K, proposed_n_kv, proposed_m_k, proposed_n_dk, proposed_n_dt, proposed_t_dv, proposed_k_dt) = model_parameter;
        
        if component_index_2 == proposed_K - 1:
            number_of_unvisited_target_tables = proposed_m_k[component_index_2] + proposed_m_k[proposed_K - 1];
        else:
            number_of_unvisited_target_tables = proposed_m_k[component_index_2];
        
        for document_index in numpy.random.permutation(xrange(self._D)):
            for table_index in numpy.random.permutation(xrange(len(proposed_k_dt[document_index]))):
                if proposed_k_dt[document_index][table_index] == component_index_2:
                    # merge component_index_2 with component_index_1
                    proposed_k_dt[document_index][table_index] = component_index_1;
                    number_of_unvisited_target_tables -= 1;
                
                if proposed_k_dt[document_index][table_index] == proposed_K - 1:
                    # shift the very last component to component_index_2
                    proposed_k_dt[document_index][table_index] = component_index_2;
                    number_of_unvisited_target_tables -= 1;
                
                if number_of_unvisited_target_tables == 0:
                    break;
                
            if number_of_unvisited_target_tables == 0:
                    break;

        # merge component_index_2 with component_index_1
        proposed_n_kv[component_index_1, :] += proposed_n_kv[component_index_2, :];
        proposed_m_k[component_index_1] += proposed_m_k[component_index_2];
        proposed_n_dk[:, component_index_1] += proposed_n_dt[:, component_index_2];
        
        # shift the very last component to component_index_2
        proposed_n_kv[component_index_2, :] = proposed_n_kv[proposed_K - 1, :];
        proposed_m_k[component_index_2] = proposed_m_k[proposed_K - 1];
        proposed_n_dk[:, component_index_2] += proposed_n_dt[:, proposed_K - 1];
        
        # remove the very last component
        proposed_n_kv = numpy.delete(proposed_n_kv, [proposed_K - 1], axis=0);
        proposed_m_k = numpy.delete(proposed_m_k, [proposed_K - 1], axis=0);
        proposed_n_dk = numpy.delete(proposed_n_dk, [proposed_K - 1], axis=1);
        proposed_K -= 1;
        
        model_parameter = (proposed_K, proposed_n_kv, proposed_m_k, proposed_n_dk, proposed_n_dt, proposed_t_dv, proposed_k_dt);
        
        return model_parameter;


    
    '''
    def sample_topics(self):
        # sample topic assignment, see which topic it should belong to
        # for topic_id in numpy.random.permutation(xrange(self._K)):
        
        for word_id in self._index_to_word:
            # find the topics that this word currently sits on
            word_topic_index = numpy.nonzero(self._n_kv[:, word_id] > 0)[0];
            # find the frequency distribution of the words sitting on the current table
            selected_word_freq_dist = self._n_kv[topic_id, selected_word_index];
            
            
        for topic_id in numpy.argsort(self._m_k):
            # if this table is empty, no need to resample the topic assignment 
            if self._m_k[topic_id] <= 0:
                continue;
            
            # find the index of the words sitting on the current table
            selected_word_index = numpy.nonzero(self._n_kv[topic_id, :] > 0)[0];
            # find the frequency distribution of the words sitting on the current table
            selected_word_freq_dist = self._n_kv[topic_id, selected_word_index];
            
            
            
            # compute the probability of assigning current table every topic
            log_topic_probability = numpy.zeros(self._K) + negative_infinity;
            
            n_k = numpy.sum(self._n_kv, axis=1);
            assert(len(n_k) == (self._K))
            # test_log_probability = numpy.zeros(self._K) + negative_infinity;
            
            for topic_index in xrange(self._K):
                # if current table is the only table assigned to current topic,
                # it means this topic is probably less useful or less generalizable to other documents,
                # it makes more sense to collapse this topic and hence assign this table to other topic.
                if self._m_k[topic_index] == 0:
                    log_topic_probability[topic_index] = negative_infinity;
                    continue;
                
                if topic_index == topic_id:
                    log_topic_probability[topic_index] = numpy.log(self._alpha_gamma) * self._m_k[topic_id];
                    
                    log_topic_probability[topic_index] += scipy.special.gammaln(self._vocabulary_size * self._alpha_eta) - scipy.special.gammaln(self._vocabulary_size * self._alpha_eta + n_k[topic_id]);
                    for word_pos in xrange(len(selected_word_index)):
                        word_id = selected_word_index[word_pos];
                        log_topic_probability[topic_index] += scipy.special.gammaln(self._alpha_eta + selected_word_freq_dist[word_pos]) - scipy.special.gammaln(self._alpha_eta);
                        
                    # test_log_probability[topic_index] = numpy.log(self._alpha_gamma) * self._m_k[topic_id];
                else:
                    log_topic_probability[topic_index] = numpy.log(self._m_k[topic_index]) * self._m_k[topic_id];
                    
                    log_topic_probability[topic_index] += scipy.special.gammaln(self._vocabulary_size * self._alpha_eta + n_k[topic_index]) - scipy.special.gammaln(self._vocabulary_size * self._alpha_eta + n_k[topic_index] + n_k[topic_id]);
                    for word_pos in xrange(len(selected_word_index)):
                        word_id = selected_word_index[word_pos];
                        log_topic_probability[topic_index] += scipy.special.gammaln(self._n_kv[topic_index, word_id] + self._alpha_eta + selected_word_freq_dist[word_pos]) - scipy.special.gammaln(self._n_kv[topic_index, word_id] + self._alpha_eta);
                    
                    # test_log_probability[topic_index] = numpy.log(self._m_k[topic_index]) * self._m_k[topic_id];    
            # normalize the distribution and sample new topic assignment for this topic
            # log_topic_probability = numpy.exp(log_topic_probability);
            # log_topic_probability = log_topic_probability/numpy.sum(log_topic_probability);
            # log_topic_probability = numpy.exp(log_normalize(log_topic_probability));
            log_topic_probability -= scipy.misc.logsumexp(log_topic_probability);
            topic_probability = numpy.exp(log_topic_probability);
            print "topic merging probability of topic %d: %s" % (topic_id, topic_probability);
            cdf = numpy.cumsum(topic_probability);
            new_topic = numpy.uint8(numpy.nonzero(cdf >= numpy.random.random())[0][0]);
            
            # test_log_probability -= scipy.misc.logsumexp(test_log_probability);
            # print "test_log_probability:", numpy.exp(test_log_probability);
            
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
                    self._k_dt[d][numpy.nonzero(self._k_dt[d] == topic_id)[0]] = new_topic;

        return;
    '''
    
    def resample_topics_backup(self):
        # sample topic assignment, see which topic it should belong to
        # for topic_id in numpy.random.permutation(xrange(self._K)):
        for topic_id in numpy.argsort(self._m_k):
            # if this table is empty, no need to resample the topic assignment 
            if self._m_k[topic_id] <= 0:
                continue;
            
            # find the index of the words sitting on the current table
            selected_word_index = numpy.nonzero(self._n_kv[topic_id, :] > 0)[0];
            # find the frequency distribution of the words sitting on the current table
            selected_word_freq_dist = self._n_kv[topic_id, selected_word_index];
            
            # compute the probability of assigning current table every topic
            log_topic_probability = numpy.zeros(self._K) + negative_infinity;
            
            n_k = numpy.sum(self._n_kv, axis=1);
            assert(len(n_k) == (self._K))
            # test_log_probability = numpy.zeros(self._K) + negative_infinity;
            
            for topic_index in xrange(self._K):
                # if current table is the only table assigned to current topic,
                # it means this topic is probably less useful or less generalizable to other documents,
                # it makes more sense to collapse this topic and hence assign this table to other topic.
                if self._m_k[topic_index] == 0:
                    log_topic_probability[topic_index] = negative_infinity;
                    continue;
                
                if topic_index == topic_id:
                    log_topic_probability[topic_index] = numpy.log(self._alpha_gamma) * self._m_k[topic_id];
                    
                    log_topic_probability[topic_index] += scipy.special.gammaln(self._vocabulary_size * self._alpha_eta) - scipy.special.gammaln(self._vocabulary_size * self._alpha_eta + n_k[topic_id]);
                    for word_pos in xrange(len(selected_word_index)):
                        word_id = selected_word_index[word_pos];
                        log_topic_probability[topic_index] += scipy.special.gammaln(self._alpha_eta + selected_word_freq_dist[word_pos]) - scipy.special.gammaln(self._alpha_eta);
                        
                    # test_log_probability[topic_index] = numpy.log(self._alpha_gamma) * self._m_k[topic_id];
                else:
                    log_topic_probability[topic_index] = numpy.log(self._m_k[topic_index]) * self._m_k[topic_id];
                    
                    log_topic_probability[topic_index] += scipy.special.gammaln(self._vocabulary_size * self._alpha_eta + n_k[topic_index]) - scipy.special.gammaln(self._vocabulary_size * self._alpha_eta + n_k[topic_index] + n_k[topic_id]);
                    for word_pos in xrange(len(selected_word_index)):
                        word_id = selected_word_index[word_pos];
                        log_topic_probability[topic_index] += scipy.special.gammaln(self._n_kv[topic_index, word_id] + self._alpha_eta + selected_word_freq_dist[word_pos]) - scipy.special.gammaln(self._n_kv[topic_index, word_id] + self._alpha_eta);
                    
                    # test_log_probability[topic_index] = numpy.log(self._m_k[topic_index]) * self._m_k[topic_id];    
            # normalize the distribution and sample new topic assignment for this topic
            # log_topic_probability = numpy.exp(log_topic_probability);
            # log_topic_probability = log_topic_probability/numpy.sum(log_topic_probability);
            # log_topic_probability = numpy.exp(log_normalize(log_topic_probability));
            log_topic_probability -= scipy.misc.logsumexp(log_topic_probability);
            topic_probability = numpy.exp(log_topic_probability);
            print "topic merging probability of topic %d: %s" % (topic_id, topic_probability);
            cdf = numpy.cumsum(topic_probability);
            new_topic = numpy.uint8(numpy.nonzero(cdf >= numpy.random.random())[0][0]);
            
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
                    self._k_dt[d][numpy.nonzero(self._k_dt[d] == topic_id)[0]] = new_topic;

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

    def log_posterior(self, model_parameter=None):
        log_posterior = 0.;
        
        # compute the document level log likelihood
        log_posterior += self.table_log_likelihood(model_parameter);
        # compute the table level log likelihood
        log_posterior += self.topic_log_likelihood(model_parameter);
        # compute the word level log likelihood
        log_posterior += self.word_log_likelihood(model_parameter);
        
        return log_posterior;

    """
    compute the word level log likelihood p(x | t, k) = \prod_{k=1}^K f(x_{ij} | z_{ij}=k), where f(x_{ij} | z_{ij}=k) = \frac{\Gamma(V \eta)}{\Gamma(n_k + V \eta)} \frac{\prod_{v} \Gamma(n_{k}^{v} + \eta)}{\Gamma^V(\eta)}
    """
    def word_log_likelihood(self, model_parameter=None):
        if model_parameter == None:
            K = self._K;
            n_kv = self._n_kv;
            m_k = self._m_k;
            n_dk = self._n_dk;
            n_dt = self._n_dt;
            t_dv = self._t_dv;
            k_dt = self._k_dt;
        else:
            (K, n_kv, m_k, n_dk, n_dt, t_dv, k_dt) = model_parameter;
        
        n_k = numpy.sum(n_kv, axis=1);
        assert(len(n_k) == K);
        
        log_likelihood = 0;
        
        log_likelihood += K * scipy.special.gammaln(self._vocabulary_size * self._alpha_eta);
        log_likelihood -= numpy.sum(scipy.special.gammaln(self._vocabulary_size * self._alpha_eta + n_k));
        
        log_likelihood += numpy.sum(scipy.special.gammaln(self._alpha_eta + n_kv));
        log_likelihood -= K * self._vocabulary_size * scipy.special.gammaln(self._alpha_eta);
        
        return log_likelihood
        
    """
    compute the table level prior in log scale \prod_{d=1}^D (p(t_{d})), where p(t_d) = \frac{ \alpha^m_d \prod_{t=1}^{m_d}(n_di-1)! }{ \prod_{v=1}^{n_d}(v+\alpha-1) }
    """
    def table_log_likelihood(self, model_parameter=None):
        if model_parameter == None:
            K = self._K;
            n_kv = self._n_kv;
            m_k = self._m_k;
            n_dk = self._n_dk;
            n_dt = self._n_dt;
            t_dv = self._t_dv;
            k_dt = self._k_dt;
        else:
            (K, n_kv, m_k, n_dk, n_dt, t_dv, k_dt) = model_parameter;
        
        log_likelihood = 0.;
        for document_index in xrange(self._D):
            log_likelihood += len(k_dt[document_index]) * numpy.log(self._alpha_gamma);
            log_likelihood += numpy.sum(scipy.special.gammaln(n_dt[document_index]));
            log_likelihood -= scipy.special.gammaln(len(t_dv[document_index]) + self._alpha_gamma)
            log_likelihood += scipy.special.gammaln(self._alpha_gamma);
            
        return log_likelihood
    
    """
    compute the topic level prior in log scale p(k) = \frac{ \gamma^K \prod_{k=1}^{K}(m_k-1)! }{ \prod_{s=1}^{m}(s+\gamma-1) }
    """
    def topic_log_likelihood(self, model_parameter=None):
        if model_parameter == None:
            K = self._K;
            n_kv = self._n_kv;
            m_k = self._m_k;
            n_dk = self._n_dk;
            n_dt = self._n_dt;
            t_dv = self._t_dv;
            k_dt = self._k_dt;
        else:
            (K, n_kv, m_k, n_dk, n_dt, t_dv, k_dt) = model_parameter;
        
        log_likelihood = 0;
        log_likelihood += K * numpy.log(self._alpha_alpha)
        log_likelihood += numpy.sum(scipy.special.gammaln(m_k));
        log_likelihood -= scipy.special.gammaln(numpy.sum(m_k) + self._alpha_alpha);
        log_likelihood += scipy.special.gammaln(self._alpha_alpha);
        
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
            
            i = 0;
            for word_index in reversed(numpy.argsort(beta_probability[topic_index, :])):
                i += 1;
                output.write(self._index_to_word[word_index] + "\t" + str(beta_probability[topic_index, word_index]) + "\n");
                if top_display > 0 and i >= top_display:
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
    
    assert(sorted_counts.shape == (K, 1));

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
    print a / numpy.sum(a, axis=1)[:, numpy.newaxis];
