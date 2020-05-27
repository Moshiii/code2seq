from argparse import ArgumentParser
import numpy as np
import tensorflow as tf

from config import Config
from interactive_predict import InteractivePredictor
from model import Model

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("-l", "--load", dest="load_path",
                        help="path to saved file", metavar="FILE", required=False)
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', type=int, default=239)
    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    if args.debug:
        config = Config.get_debug_config(args)
    else:
        config = Config.get_default_config(args)

    model = Model(config)
    print('Created model')
    if args.predict:
        predictor = InteractivePredictor(config, model)
        predictor.predict()

    model.close_session()
