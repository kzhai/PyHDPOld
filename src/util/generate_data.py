import numpy;
import nltk;
import scipy;
import optparse;
import datetime;
import os;

#uninformed_count = 1.;
#informed_count =  100.;

#vocab = ['python', 'java', 'umiacs', 'clip', 'doctor', 'master'];
#total_vocab = xrange(6);
#total_topic = 3;
#topic_vocab = 2;

#topic = {};
#topic[0] = numpy.array([informed_count, informed_count, uninformed_count, uninformed_count, uninformed_count, uninformed_count]);
#topic[1] = numpy.array([uninformed_count, uninformed_count, informed_count, informed_count, uninformed_count, uninformed_count]);
#topic[2] = numpy.array([uninformed_count, uninformed_count, uninformed_count, uninformed_count, informed_count, informed_count]);

#for k in xrange(total_topic):
#    topic[k] = FreqDist();
#    for v in total_vocab:
#        topic[k].inc(v, uninformed_count);
#    topic[k].inc(k*topic_vocab, informed_count); 
#    topic[k].inc(k*topic_vocab+1, informed_count);
    
def parse_args():
    parser = optparse.OptionParser()
    parser.set_defaults(output_directory=None,
                        number_of_topics=10,
                        number_of_documents=1000,
                        asymmetric_alpha_prior=False,
                        
                        number_of_types_per_topic=10,
                        number_of_tokens_per_document=30,
                        )
    parser.add_option("--output_directory", type="string", dest="output_directory",
                      help="output directory [None]");
    parser.add_option("--number_of_topics", type="int", dest="number_of_topics",
                      help="number of clusters [10]");
    parser.add_option("--number_of_documents", type="int", dest="number_of_documents",
                      help="number of points [1000]");
    parser.add_option("--asymmetric_alpha_prior", action='store_true', default=False, dest="asymmetric_alpha_prior",
                      help="asymmetric alpha prior [False]")
    
    parser.add_option("--number_of_types_per_topic", type="int", dest="number_of_types_per_topic",
                      help="number of types per topic [10]");
    parser.add_option("--number_of_tokens_per_documnet", type="int", dest="number_of_tokens_per_document",
                      help="number of tokens per document [30]");
                      
    (options, args) = parser.parse_args();
    return options;
    
def main():
    options = parse_args();
    
    assert(options.output_directory!=None);
    output_directory = options.output_directory;
    number_of_topics = options.number_of_topics;
    number_of_documents = options.number_of_documents;
    asymmetric_alpha_prior = options.asymmetric_alpha_prior;
    
    number_of_types_per_topic = options.number_of_types_per_topic;
    number_of_tokens_per_document = options.number_of_tokens_per_document;
    
    now = datetime.datetime.now();
    suffix = now.strftime("%y%m%d-%H%M%S")+"";
    suffix += "-k%d" % (number_of_topics);
    suffix += "-d%d" % (number_of_documents);
    suffix += "-vpk%d" % (number_of_types_per_topic);
    suffix += "-wpd%d" % (number_of_tokens_per_document);
    suffix += "-%s" % (asymmetric_alpha_prior);
    suffix += "/";
    output_directory = os.path.join(output_directory, suffix);
    os.mkdir(os.path.abspath(output_directory));
    
    if asymmetric_alpha_prior:
        alpha = numpy.random.random(number_of_topics);
        alpha /= numpy.sum(alpha);
    else:
        alpha = numpy.zeros(number_of_topics) + 1.0/number_of_topics;

    output_file_stream = open(os.path.join(output_directory, "doc.dat"), 'w');
    for doc_id in xrange(number_of_documents):
        doc = [];
        gamma = numpy.random.mtrand.dirichlet(alpha);
        for token_id in xrange(numpy.random.poisson(number_of_tokens_per_document)):
            topic_id = numpy.nonzero(numpy.random.multinomial(1, gamma))[0][0];
            type_id = numpy.random.randint(0, number_of_types_per_topic);
            doc.append("K%dV%d" % (topic_id, type_id));
        
        output_file_stream.write("%s\n" % (" ".join(doc)));
    output_file_stream.close();

    output_file_stream = open(os.path.join(output_directory, "voc.dat"), 'w');
    for topic_id in xrange(number_of_topics):
        for type_id in xrange(number_of_types_per_topic):
            output_file_stream.write("K%dV%d\n" % (topic_id, type_id));
    output_file_stream.close();
        
if __name__ == '__main__':
    main();