from nltk.corpus import stopwords;
from nltk.probability import FreqDist;

import os, sys
import nltk;
import numpy;
import matplotlib;
import time;

def main():
    model_result = {};

    pmi_statistics_file = sys.argv[1];
    input_file_path = sys.argv[2];
    
    model_directory = sys.argv[3];
    model_directory = model_directory.rstrip("/");
    corpus_name = os.path.basename(model_directory);
    
    output_file_path = sys.argv[4];
    
    top_words = int(sys.argv[5]);
    snapshot_index = int(sys.argv[6]);

    clock = time.time();

    target_word_list = {};
    invert_target_word_list = {};
    for model_name in os.listdir(model_directory):
        sub_directory = os.path.join(model_directory, model_name);
        if os.path.isfile(sub_directory):
            continue;
        
        exp_beta_file = os.path.join(sub_directory, "exp_beta-%d" % (snapshot_index));
        input = open(exp_beta_file, 'r');
        top_word_counter = 0;
        for line in input:
            line = line.strip();
            if line.startswith('=========='):
                top_word_counter = 0;
                continue;
            if top_word_counter >= top_words + 1:
                continue;
            model_settings = line.split();
            top_word_counter += 1;
            if model_settings[0] in target_word_list:
                continue;
            else:
                target_word_list[model_settings[0]] = len(target_word_list);
                invert_target_word_list[len(invert_target_word_list)] = model_settings[0];
                
    # target_word_list = target_word_list.keys();
    print "total number of words in target %d" % len(target_word_list);

    import pointwise_mutual_information;
    pmi_scores = pointwise_mutual_information.PointwiseMutualInformation();
    pmi_scores.import_pmi_statistics(pmi_statistics_file, target_word_list);
    clock = time.time() - clock;
    print 'time to load point-mutual-information statistics: %d seconds' % (clock);
    
    import semantic_coherence;
    clock = time.time();
    sc_scores = semantic_coherence.SemanticCoherence(0.001);
    sc_scores.collect_statistics(os.path.join(input_file_path));
    clock = time.time() - clock;
    print 'time to load semantic coherence statistics: %d seconds' % (clock);
    
    import inter_topic_distance;
    itd_scores = inter_topic_distance.InterTopicDistance();
    
    output_file = open(output_file_path, 'w');
    #output_file.write(", ".join(['topic', 'pmi', 'sc', 'its', 'alpha', 'gamma', 'eta', 'inference','dataset']) + "\n");
    output_file.write(", ".join(['topic', 'metric', 'value', 'alpha', 'gamma', 'eta', 'inference','dataset']) + "\n");
    for model_name in os.listdir(model_directory):
        sub_directory = os.path.join(model_directory, model_name);
        if os.path.isfile(sub_directory):
            continue;
        
        model_settings = model_name.split("-");
        alpha = float(model_settings[5].strip("aa"));
        gamma = float(model_settings[6].strip("ag"));
        eta = float(model_settings[7].strip("ae"));
        inference = "-".join(model_settings[8:]);
        if inference=="":
            inference = "null";
        elif inference=="smh0":
            inference = "crs";
        elif inference=="smh1-sp0-mp0":
            inference = "rand";
        elif inference=="smh1-sp1-mp1":
            inference = "rgs";
        elif inference=="smh1-sp2-mp0":
            inference = "sa";
        else:
            sys.stderr.write("warning: unrecognized inference...\n");

        snapshot_file = os.path.join(sub_directory, "exp_beta-%d" % (snapshot_index));
        pmi_score = pmi_scores.evaluate(snapshot_file, top_words, target_word_list);
        sc_score = sc_scores.evaluate(snapshot_file, top_words)
        itd_score = itd_scores.evaluate(snapshot_file, top_words);
        itd_score /= (itd_score.shape[1]-1.0);

        for topic_index in xrange(pmi_score.shape[1]):
            #output_file.write("%d,%g,%g,%g,%g,%g,%g,%s,%s\n" % (topic_index, pmi_score[0, topic_index], sc_score[0, topic_index], itd_score[0, topic_index], alpha, gamma, eta, inference, corpus_name));
            output_file.write("%d,%s,%g,%g,%g,%g,%s,%s\n" % (topic_index, "pmi", pmi_score[0, topic_index], alpha, gamma, eta, inference, corpus_name));
            output_file.write("%d,%s,%g,%g,%g,%g,%s,%s\n" % (topic_index, "coh", sc_score[0, topic_index], alpha, gamma, eta, inference, corpus_name));
            output_file.write("%d,%s,%g,%g,%g,%g,%s,%s\n" % (topic_index, "itd", itd_score[0, topic_index], alpha, gamma, eta, inference, corpus_name));
        
        #print "%d\t%g\t%g\t%g\t%g\t%g\t%s" % (pmi_score.shape[1], numpy.mean(pmi_score), numpy.mean(sc_score), alpha, gamma, eta, inference);
        print "%d\t%g\t%g\t%g\t%g\t%g\t%g\t%s" % (pmi_score.shape[1], numpy.mean(pmi_score), numpy.mean(sc_score), numpy.mean(itd_score), alpha, gamma, eta, inference);
        
if __name__ == '__main__':
    main()
