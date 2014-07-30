import cPickle, string, numpy, getopt, sys, random, time, re, pprint
import datetime, os;

import nltk;
import numpy;

def main():
    import option_parser
    options = option_parser.parse_args();
    
    # parameter set 1
    #assert(options.corpus_name!=None);
    assert(options.input_directory!=None);
    assert(options.output_directory!=None);
    
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
    file = open(os.path.join(input_directory, 'doc.dat'), 'r');
    for line in file:
        train_docs.append(line.strip());
    print "successfully load all training documents..."
    
    # Vocabulary
    dictionary_file = os.path.join(input_directory, 'voc.dat');
    input_file = open(dictionary_file, 'r');
    vocab = [];
    for line in input_file:
        vocab.append(line.strip().split()[0]);
    vocab = list(set(vocab));
    print "successfully load all the words from %s..." % (dictionary_file);
    
    # parameter set 2
    alpha_eta = 1.0/len(vocab);
    if options.alpha_eta>0:
        alpha_eta=options.alpha_eta;
    assert(options.alpha_alpha>0);
    alpha_alpha = options.alpha_alpha;
    assert(options.alpha_gamma>0);
    alpha_gamma = options.alpha_gamma;
    
    # parameter set 3
    if options.training_iterations>0:
        training_iterations=options.training_iterations;
    if options.snapshot_interval>0:
        snapshot_interval=options.snapshot_interval;
    resample_topics = options.resample_topics;
    hash_oov_words = options.hash_oov_words;
    
    # create output directory
    now = datetime.datetime.now();
    suffix = now.strftime("%y%b%d-%H%M%S")+"";
    suffix += "-%s" % ("hdp");
    suffix += "-I%d" % (training_iterations);
    suffix += "-S%d" % (snapshot_interval);
    suffix += "-aa%g" % (alpha_alpha);
    suffix += "-ag%g" % (alpha_gamma);
    suffix += "-ae%g" % (alpha_eta);
    suffix += "-%s" % (resample_topics);
    suffix += "-%s" % (hash_oov_words);
    suffix += "/";

    output_directory = os.path.join(output_directory, suffix);
    os.mkdir(os.path.abspath(output_directory));
    
    # store all the options to a file
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
    options_output_file.write("resample_topics=%s\n" % resample_topics);
    options_output_file.write("hash_oov_words=%s\n" % hash_oov_words);
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
    print "resample_topics=%s" % (resample_topics)
    print "hash_oov_words=%s" % (hash_oov_words)
    print "========== ========== ========== ========== =========="
    
    import monte_carlo;
    hdp = monte_carlo.MonteCarlo(resample_topics, hash_oov_words);
    hdp._initialize(train_docs, vocab, alpha_alpha, alpha_gamma, alpha_eta)
    
    for iteration in xrange(training_iterations):
        clock = time.time();
        log_likelihood = hdp.learning();
        clock = time.time()-clock;
        print 'training iteration %d finished in %f seconds: number-of-topics = %d, log-likelihood = %f' % (hdp._counter, clock, hdp._K, log_likelihood);

        # Save lambda, the parameters to the variational distributions over topics, and batch_gamma, the parameters to the variational distributions over topic weights for the articles analyzed in the last iteration.
        #if ((hdp._counter+1) % snapshot_interval == 0):
            #hdp.export_beta(output_directory + 'exp_beta-' + str(hdp._counter+1));
        if (hdp._counter % snapshot_interval == 0):
            hdp.export_beta(os.path.join(output_directory, 'exp_beta-' + str(hdp._counter)), 50);
            numpy.savetxt(os.path.join(output_directory, 'n_kv-' + str(hdp._counter)), hdp._n_kv, fmt="%d");
            #numpy.savetxt(os.path.join(output_directory, 'beta-' + str(hdp._counter)), hdp._E_log_beta);
            #numpy.savetxt(os.path.join(output_directory, 'lambda-' + str(hdp._counter)), hdp._lambda);
    
    #gamma_path = os.path.join(output_directory, 'gamma.txt');
    #numpy.savetxt(gamma_path, hdp._document_topic_distribution);
    
    #topic_inactive_counts_path = os.path.join(output_directory, "topic_inactive_counts.txt");
    #numpy.savetxt(topic_inactive_counts_path, hdp._topic_inactive_counts);

if __name__ == '__main__':
    main()