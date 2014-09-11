import numpy;
import nltk;
import scipy;
import optparse;
import datetime;
import os;
import matplotlib
import matplotlib.pyplot
import matplotlib.pylab
    
def parse_args():
    parser = optparse.OptionParser()
    parser.set_defaults(number_of_topics=10,
                        number_of_vocabularies=100,
                        number_of_documents=1000,                        
                        output_directory=None,
                        asymmetric_alpha_prior=False,
                        number_of_tokens_per_document=30,
                            )
    parser.add_option("--number_of_topics", type="int", dest="number_of_topics",
                      help="number of topics [10]");
    parser.add_option("--number_of_vocabularies", type="int", dest="number_of_vocabularies",
                      help="number of types per topic [10]");
    parser.add_option("--number_of_documents", type="int", dest="number_of_documents",
                      help="number of documents [1000]");
                       
    parser.add_option("--output_directory", type="string", dest="output_directory",
                      help="output directory [None]");                     
    parser.add_option("--asymmetric_alpha_prior", action='store_true', default=False, dest="asymmetric_alpha_prior",
                      help="asymmetric alpha prior [False]")
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
    
    number_of_vocabularies = options.number_of_vocabularies;
    number_of_tokens_per_document = options.number_of_tokens_per_document;
    
    now = datetime.datetime.now();
    suffix = now.strftime("%y%m%d-%H%M%S")+"";
    suffix += "-k%d" % (number_of_topics);
    suffix += "-d%d" % (number_of_documents);
    suffix += "-v%d" % (number_of_vocabularies);
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

    output_file_stream = open(os.path.join(output_directory, "voc.dat"), 'w');
    for type_id in xrange(number_of_vocabularies):
        output_file_stream.write("%d\n" % (type_id));
    output_file_stream.close();
    
    beta = numpy.zeros(number_of_vocabularies) + 2.0/number_of_vocabularies;
    topic_vocabulary_probability = numpy.zeros((number_of_topics, number_of_vocabularies));
    
    output_file_stream = open(os.path.join(output_directory, "topic.dat"), 'w');
    for topic_index in xrange(number_of_topics):
        topic_vocabulary_probability[topic_index, :] = numpy.random.dirichlet(beta);
        
        output_file_stream.write("==========\t%d\t==========\n" % (topic_index));
        for type_index in xrange(number_of_vocabularies):
            output_file_stream.write("%d\t%f\n" % (type_index, topic_vocabulary_probability[topic_index, type_index]));

    output_file_stream = open(os.path.join(output_directory, "doc.dat"), 'w');
    for doc_id in xrange(number_of_documents):
        doc = [];
        topic_distribution = numpy.random.dirichlet(alpha);
        for token_id in xrange(numpy.random.poisson(number_of_tokens_per_document)):
            topic_id = numpy.nonzero(numpy.random.multinomial(1, topic_distribution))[0][0];
            type_id = numpy.nonzero(numpy.random.multinomial(1, topic_vocabulary_probability[topic_id, :]))[0][0];
            doc.append("%d" % (type_id));
        
        output_file_stream.write("%s\n" % (" ".join(doc)));
    output_file_stream.close();
    
    # plot the vocabulary distribution per topic
    x_ticks = numpy.arange(number_of_vocabularies);
    width = 0.5;
    
    axis = matplotlib.pylab.subplot(number_of_topics, 1, 0)
    axis.bar(x_ticks-width, topic_vocabulary_probability[0, :], 1, color='g')
    axis.set_xlim(x_ticks[0]-width, x_ticks[-1]+width);
    for topic_index in xrange(1, number_of_topics):
        ax1 = matplotlib.pylab.subplot(number_of_topics, 1, topic_index, sharex=axis, sharey=axis);
        ax1.bar(x_ticks-width, topic_vocabulary_probability[topic_index, :], 1, color='g')
        ax1.set_xlim(x_ticks[0]-width, x_ticks[-1]+width);
        matplotlib.pylab.setp(ax1.get_xticklabels(), fontsize=2.5, visible=False)
    axis.set_xticks(x_ticks);
    
    figure_path = os.path.join(output_directory, "topic.pdf");
    matplotlib.pyplot.savefig(figure_path);
    
if __name__ == '__main__':
    main();