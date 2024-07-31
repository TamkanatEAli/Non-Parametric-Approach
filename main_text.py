import argparse
from data import *
from compressors import *
from experiments import *
from utils import *
from functools import partial
from pathos.multiprocessing import ProcessingPool as Pool
import time
from torchtext.datasets import IMDB, AG_NEWS, SogouNews, DBpedia, YelpReviewPolarity, YahooAnswers, AmazonReviewPolarity



def record_distance(compressor_name, test_data, test_portion_name, train_data, agg_func, dis_func, out_dir, para=True):
    print("compressor={}".format(compressor_name))
    numpy_dir = os.path.join(out_dir, compressor_name)
    if not os.path.exists(numpy_dir):
        os.makedirs(numpy_dir)
    out_fn = os.path.join(numpy_dir, test_portion_name)
    cp = DefaultCompressor(compressor_name)
    knn_exp = KnnExpText(agg_func, cp, dis_func)
    start = time.time()
    if para:
        with Pool(6) as p:
            distance_for_selected_test = p.map(partial(knn_exp.calc_dis_single_multi, train_data), test_data)
        np.save(out_fn, np.array(distance_for_selected_test))
        del distance_for_selected_test
    else:
        knn_exp.calc_dis(test_data, train_data=train_data)
        knn_exp.dis_matrix = 0.5 * (np.array(knn_exp.dis_matrix) + np.array(knn_exp.dis_matrix).T)
        symmetric_distance_matrix = np.array(knn_exp.dis_matrix)
        np.save(out_fn, np.array(symmetric_distance_matrix))
    print("spent: {}".format(time.time() - start))

record_distance('gzip', Seq_input, test_portion_name , train_data, agg_by_concat_space, NCD, output_dir, para)


