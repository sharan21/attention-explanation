import argparse

parser = argparse.ArgumentParser(description='Run experiments on a dataset')
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--output_dir", type=str)
parser.add_argument('--encoder', type=str, choices=['cnn', 'lstm', 'average', 'all', 'vanilla_lstm'], required=True)
parser.add_argument('--attention', type=str, choices=['tanh', 'dot', 'all', 'none'], required=True)

args, extras = parser.parse_known_args()
args.extras = extras

from Trainers.DatasetBC import *
from ExperimentsBC import *

dataset = datasets[args.dataset](args)

if args.output_dir is not None:
	dataset.output_dir = args.output_dir

encoders = ['cnn', 'lstm', 'average'] if args.encoder == 'all' else [args.encoder]

if args.attention == "none": # encoder, config = ['vanilla_lstm']
	train_dataset_on_encoders(dataset, encoders)
	generate_graphs_on_encoders(dataset, encoders)





# if args.attention in ['tanh', 'all']:
# 	train_dataset_on_encoders(dataset, encoders)
# 	generate_graphs_on_encoders(dataset, encoders)
#
# if args.attention in ['dot', 'all']:
# 	encoders = [e + '_dot' for e in encoders]
# 	train_dataset_on_encoders(dataset, encoders)
# 	generate_graphs_on_encoders(dataset, encoders)
#








