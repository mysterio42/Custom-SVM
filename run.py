import argparse

import numpy as np
import pandas as pd

from utils.config import params_grid
from utils.data import CustomData
from utils.model import train_model, load_model
from utils.plot import plot_data, plot_boundary, plot_prediction


def parse_args():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected')

    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str2bool, default=True,
                        help='True: Load trained model  False: Train model default: True')

    parser.add_argument('--gen', type=str2bool, default=False,
                        help='True: Generate Fake data False: Use already generated data default: False')

    parser.add_argument('--gs', type=str2bool, default=False,
                        help='True: Run GridSearchCV default: False')

    subp = parser.add_subparsers(help='Choose kernel for SVM model', dest='kernel')

    parser_lin = subp.add_parser('linear', help='linear kernel')
    parser_lin.add_argument('--C', choices=params_grid['C'], type=float, help='C params')

    parser_rbf = subp.add_parser('rbf', help='rbf kernel')
    parser_rbf.add_argument('--gamma', choices=params_grid['gamma'], type=float, help='Gamma params')
    parser_rbf.add_argument('--C', choices=params_grid['C'], type=float, help='C params')

    parser_poly = subp.add_parser('poly', help='poly kernel')
    parser_poly.add_argument('--gamma', choices=params_grid['gamma'], type=float, help='Gamma params')
    parser_poly.add_argument('--C', choices=params_grid['C'], type=float, help='C params')
    parser_poly.add_argument('--degree', choices=params_grid['degree'], type=int, help='degree params')

    parser.print_help()

    return parser.parse_args()


if __name__ == '__main__':

    np.random.seed(2)
    args = parse_args()
    input('---')

    if args.gen:
        data = CustomData(path='data/', nums=100)
        data.generate_cluster(x_range=(3, 7), y_range=(3, 6), cluster_key=0)
        data.generate_cluster(x_range=(13, 16), y_range=(11, 15), cluster_key=1)
        data.generate_cluster(x_range=(1, 5), y_range=(10, 16), cluster_key=2)
        data.generate_cluster(x_range=(11, 14), y_range=(4, 7), cluster_key=3)

        data.to_csv(name='clusters')

    df = pd.read_csv('data/clusters.csv')

    dot_colors = CustomData.generate_colors(df)

    if args.load:
        model = load_model()
        plot_data(df, dot_colors)

        plot_prediction(model, df, (2.3, 9.1), dot_colors)
        plot_prediction(model, df, (9.7, 8.1), dot_colors)
        plot_prediction(model, df, (9.3, 10.2), dot_colors)

        param = {}
        data = CustomData.load_data(df)

        param['mode'] = 'train'
        plot_boundary(model, data['train'], dot_colors, param)

        param['mode'] = 'test'
        plot_boundary(model, data['test'], dot_colors, param)

        plot_prediction(model, df, (2.3, 9.1), dot_colors)
        plot_prediction(model, df, (9.7, 8.1), dot_colors)
        plot_prediction(model, df, (9.3, 10.2), dot_colors)
    else:

        plot_data(df, dot_colors)

        data = CustomData.load_data(df)

        if args.gs:
            model, param = train_model(data, params_grid, gs=True)
        else:
            model, param = train_model(data, args)

        param['mode'] = 'train'
        plot_boundary(model, data['train'], dot_colors, param)

        param['mode'] = 'test'
        plot_boundary(model, data['test'], dot_colors, param)

        plot_prediction(model, df, (2.3, 9.1), dot_colors)
        plot_prediction(model, df, (9.7, 8.1), dot_colors)
        plot_prediction(model, df, (9.3, 10.2), dot_colors)
