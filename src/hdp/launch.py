import cPickle;
import string, numpy, getopt, sys, random, time, re, pprint
import datetime, os;

import nltk;
import numpy;

def main():
    import option_parser
    options = option_parser.parse_args();
    
    # parameter set 1
    # assert(options.corpus_name!=None);
    assert(options.input_directory != None);
    assert(options.output_directory != None);
    
    input_directory = options.input_directory;
    input_directory = input_directory.rstrip("/");
    corpus_name = os.path.basename(input_directory);
    
    output_directory = options.output_directory;
    if not os.path.exists(output_directory):
        os.mkdir(output_directory);
    output_directory = os.path.join(output_directory, corpus_name);
    if not os.path.exists(output_directory):
        os.mkdir(output_directory);
    
    # Document
    train_docs = [];
    input_doc_stream = open(os.path.join(input_directory, 'doc.dat'), 'r');
    for line in input_doc_stream:
        train_docs.append(line.strip().lower());
    print "successfully load all training documents..."
    
    # Vocabulary
    dictionary_file = os.path.join(input_directory, 'voc.dat');
    input_voc_stream = open(dictionary_file, 'r');
    vocab = [];
    for line in input_voc_stream:
        vocab.append(line.strip().lower().split()[0]);
    vocab = list(set(vocab));
    print "successfully load all the words from %s..." % (dictionary_file);
    
    # parameter set 2
    alpha_eta = 1.0 / len(vocab);
    if options.alpha_eta > 0:
        alpha_eta = options.alpha_eta;
    assert(options.alpha_alpha > 0);
    alpha_alpha = options.alpha_alpha;
    assert(options.alpha_gamma > 0);
    alpha_gamma = options.alpha_gamma;
    
    # parameter set 3
    if options.training_iterations > 0:
        training_iterations = options.training_iterations;
    if options.snapshot_interval > 0:
        snapshot_interval = options.snapshot_interval;
        
    # resample_topics = options.resample_topics;
    # hash_oov_words = options.hash_oov_words;
            
    # parameter set 4
    split_merge_heuristics = options.split_merge_heuristics;
    split_proposal = options.split_proposal;
    merge_proposal = options.merge_proposal;

    # create output directory
    now = datetime.datetime.now();
    suffix = now.strftime("%y%m%d-%H%M%S") + "";
    suffix += "-%s" % ("hdp");
    suffix += "-I%d" % (training_iterations);
    suffix += "-S%d" % (snapshot_interval);
    suffix += "-aa%g" % (alpha_alpha);
    suffix += "-ag%g" % (alpha_gamma);
    suffix += "-ae%g" % (alpha_eta);
    # suffix += "-%s" % (resample_topics);
    # suffix += "-%s" % (hash_oov_words);
    if split_merge_heuristics >= 0:
        suffix += "-smh%d" % (split_merge_heuristics);
    if split_merge_heuristics >= 1:
        suffix += "-sp%d" % (split_proposal);
        suffix += "-mp%d" % (merge_proposal);
    suffix += "/";

    output_directory = os.path.join(output_directory, suffix);
    os.mkdir(os.path.abspath(output_directory));
    
    # store all the options to a input_doc_stream
    options_output_file = open(output_directory + "option.txt", 'w');
    # parameter set 1
    options_output_file.write("input_directory=" + input_directory + "\n");
    options_output_file.write("corpus_name=" + corpus_name + "\n");
    options_output_file.write("dictionary_file=" + str(dictionary_file) + "\n");
    # parameter set 2
    options_output_file.write("alpha_eta=" + str(alpha_eta) + "\n");
    options_output_file.write("alpha_alpha=" + str(alpha_alpha) + "\n");
    options_output_file.write("alpha_gamma=" + str(alpha_gamma) + "\n");
    # parameter set 3
    options_output_file.write("training_iteration=%d\n" % training_iterations);
    options_output_file.write("snapshot_interval=%d\n" % snapshot_interval);
    # options_output_file.write("resample_topics=%s\n" % resample_topics);
    # options_output_file.write("hash_oov_words=%s\n" % hash_oov_words);
    # parameter set 4
    if split_merge_heuristics >= 0:
        options_output_file.write("split_merge_heuristics=%d\n" % split_merge_heuristics);
    if split_merge_heuristics >= 1:
        options_output_file.write("split_proposal=%d\n" % split_proposal);
        options_output_file.write("merge_proposal=%d\n" % merge_proposal);
    options_output_file.close()
    
    print "========== ========== ========== ========== =========="
    # parameter set 1
    print "output_directory=" + output_directory
    print "input_directory=" + input_directory
    print "corpus_name=" + corpus_name
    print "dictionary_file=" + str(dictionary_file)
    # parameter set 2
    print "alpha_eta=" + str(alpha_eta)
    print "alpha_alpha=" + str(alpha_alpha)
    print "alpha_gamma=" + str(alpha_gamma)
    # parameter set 3
    print "training_iteration=%d" % (training_iterations);
    print "snapshot_interval=%d" % (snapshot_interval);
    # print "resample_topics=%s" % (resample_topics)
    # print "hash_oov_words=%s" % (hash_oov_words)
    # parameter set 4
    if split_merge_heuristics >= 0:
        print "split_merge_heuristics=%d" % (split_merge_heuristics)
    if split_merge_heuristics >= 1:
        print "split_proposal=%d" % split_proposal;
        print "merge_proposal=%d" % merge_proposal;
    print "========== ========== ========== ========== =========="
    
    import monte_carlo;
    hdp = monte_carlo.MonteCarlo(split_merge_heuristics, split_proposal, merge_proposal);
    hdp._initialize(train_docs, vocab, alpha_alpha, alpha_gamma, alpha_eta)
    
    hdp.export_beta(os.path.join(output_directory, 'exp_beta-' + str(hdp._iteration_counter)), 50);
    numpy.savetxt(os.path.join(output_directory, 'n_kv-' + str(hdp._iteration_counter)), hdp._n_kv, fmt="%d");
    
    for iteration in xrange(training_iterations):
        clock = time.time();
        log_likelihood = hdp.learning();
        clock = time.time() - clock;
        print 'training iteration %d finished in %f seconds: number-of-topics = %d, log-likelihood = %f' % (hdp._iteration_counter, clock, hdp._K, log_likelihood);

        # Save lambda, the parameters to the variational distributions over topics, and batch_gamma, the parameters to the variational distributions over topic weights for the articles analyzed in the last iteration.
        if (hdp._iteration_counter % snapshot_interval == 0):
            hdp.export_beta(os.path.join(output_directory, 'exp_beta-' + str(hdp._iteration_counter)), 50);
            #numpy.savetxt(os.path.join(output_directory, 'n_kv-' + str(hdp._iteration_counter)), hdp._n_kv, fmt="%d");
            
    
    # gamma_path = os.path.join(output_directory, 'gamma.txt');
    # numpy.savetxt(gamma_path, hdp._document_topic_distribution);
    
    # topic_inactive_counts_path = os.path.join(output_directory, "topic_inactive_counts.txt");
    # numpy.savetxt(topic_inactive_counts_path, hdp._topic_inactive_counts);

if __name__ == '__main__':
    main()
