import optparse;

def parse_args():
    parser = optparse.OptionParser()
    parser.set_defaults(# parameter set 1
                        input_directory=None,
                        output_directory=None,
                        #corpus_name=None,
                        
                        # parameter set 2
                        alpha_eta=-1,
                        alpha_alpha=0.1,
                        alpha_gamma=0.1,
                        
                        # parameter set 3
                        training_iterations=1000,
                        snapshot_interval=100,
                        #resample_topics=False,
                        #hash_oov_words=False,
                        
                        # parameter set 4
                        split_proposal=0,
                        merge_proposal=0,
                        split_merge_heuristics=-1,
                        )
    # parameter set 1
    parser.add_option("--input_directory", type="string", dest="input_directory",
                      help="input directory [None]");
    parser.add_option("--output_directory", type="string", dest="output_directory",
                      help="output directory [None]");
    #parser.add_option("--corpus_name", type="string", dest="corpus_name",
                      #help="the corpus name [None]");

    # parameter set 2
    parser.add_option("--alpha_eta", type="float", dest="alpha_eta",
                      help="hyper-parameter for Dirichlet distribution of vocabulary [1.0/number_of_types]")
    parser.add_option("--alpha_alpha", type="float", dest="alpha_alpha",
                      help="hyper-parameter for top level Dirichlet process of distribution over topics [0.1]")
    parser.add_option("--alpha_gamma", type="float", dest="alpha_gamma",
                      help="hyper-parameter for bottom level Dirichlet process of distribution over topics [0.1]")
    
    # parameter set 3
    parser.add_option("--training_iterations", type="int", dest="training_iterations",
                      help="number of training iterations [1000]");
    parser.add_option("--snapshot_interval", type="int", dest="snapshot_interval",
                      help="snapshot interval [100]");
    #parser.add_option("--resample_topics", action="store_true", dest="resample_topics",
                      #help="resample topics [False]")
    #parser.add_option("--hash_oov_words", action="store_true", dest="hash_oov_words",
                      #help="hash out-of-vocabulary words to run this model in pseudo infinite vocabulary mode [False]")
    
    # parameter set 4
    parser.add_option("--merge_proposal", type="int", dest="merge_proposal",
                      help="propose merge operation via [ " + 
                            "0 (default): metropolis-hastings, " + 
                            "1: restricted gibbs sampler and metropolis-hastings, " + 
                            "2: gibbs sampler and metropolis-hastings " + 
                            "]")
    parser.add_option("--split_proposal", type="int", dest="split_proposal",
                      help="propose split operation via [ " + 
                            "0 (default): metropolis-hastings, " + 
                            "1: restricted gibbs sampler and metropolis-hastings, " + 
                            "2: sequential allocation and metropolis-hastings " + 
                            "]")    
    parser.add_option("--split_merge_heuristics", type="int", dest="split_merge_heuristics",
                      help="split-merge heuristics [ " + 
                            "-1 (default): no split-merge operation, " +
                            "0: component resampling, " + 
                            "1: random choose candidate clusters by points, " + 
                            "2: random choose candidate clusters by point-cluster, " + 
                            "3: random choose candidate clusters by clusters " +
                            "]")
    
    (options, args) = parser.parse_args();
    return options;