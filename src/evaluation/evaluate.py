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
    model_directory = sys.argv[2];
    output_file_path = sys.argv[3];
    
    top_words = int(sys.argv[4]);
    snapshot_index = int(sys.argv[5]);

    clock = time.time();
    if False:
        target_word_list = None;
    else:
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
    
    import inner_topic_similarity;
    its_scores = inner_topic_similarity.InnerTopicSimilarity();
    
    output_file = open(output_file_path, 'w');
    output_file.write(", ".join(['topic', 'pmi', 'its', 'alpha', 'gamma', 'eta', 'inference']) + "\n");
    for model_name in os.listdir(model_directory):
        sub_directory = os.path.join(model_directory, model_name);
        if os.path.isfile(sub_directory):
            continue;
        
        model_settings = model_name.split("-");
        alpha = float(model_settings[5].strip("aa"));
        gamma = float(model_settings[6].strip("ag"));
        eta = float(model_settings[7].strip("ae"));
        inference = "-".join(model_settings[8:]);

        snapshot_file = os.path.join(sub_directory, "exp_beta-%d" % (snapshot_index));
        pmi_score = pmi_scores.evaluate(snapshot_file, top_words, target_word_list);
        its_score = its_scores.evaluate(snapshot_file, top_words);

        output_file.write("%d,%g,%d,%g,%g,%g,%s\n" % (pmi_score.shape[1], numpy.mean(pmi_score), its_score, alpha, gamma, eta, inference));
        
        print "%d\t%g\t%g\t%g\t%g\t%s" % (pmi_score.shape[1], numpy.mean(pmi_score), alpha, gamma, eta, inference);
    
if __name__ == '__main__':
    main()
