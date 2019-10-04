from Transparency.common_code.common import *
from Transparency.Trainers.PlottingBC import generate_graphs, plot_adversarial_examples, plot_logodds_examples
from Transparency.configurations import configurations
from Transparency.Trainers.TrainerBC import Trainer, Evaluator
from Transparency.model.LR import LR
from datetime import datetime
from itertools import zip_longest
from lime.lime_text import LimeTextExplainer


def generate_graphs_on_latest_model_old(dataset, config='lstm'):
	config = configurations[config](dataset)
	latest_model = get_latest_model(os.path.join(config['training']['basepath'], config['training']['exp_dirname']))
	evaluator = Evaluator(dataset, latest_model, _type=dataset.trainer_type)
	_ = evaluator.evaluate(dataset.test_data, save_results=False)
	generate_graphs(dataset, config['training']['exp_dirname'], evaluator.model, test_data=dataset.test_data)


def generate_graphs_on_latest_model(dataset, config='lstm'):
	dataset_name = dataset.name

	config = configurations[config](dataset)
	latest_model = get_latest_model(os.path.join(config['training']['basepath'], config['training']['exp_dirname']))

	"""Get evaluator"""
	print("getting evaluator")
	evaluator = Evaluator(dataset, latest_model, _type=dataset.trainer_type)

	print("gettings predictions and updating testdata.yhat and testdata.attn")
	_ = evaluator.evaluate(dataset.test_data, save_results=False)

	"""Get trained nn.embedding weights"""
	embd_dict = np.array(evaluator.model.encoder.embedding.weight.data)

	"""get pre trained vectorizer"""
	print("getting vectorizer")
	try:
		file = open('./pickles/{}_vectorizer.pickle'.format(dataset_name), 'rb')
	except:
		print("need to store vectorizer first from ./preprocess/preprocess_data*.py")
	vectorizer = pickle.load(file)

	"""get idx2word and reverse dict"""
	idx2word = vectorizer.idx2word
	word2idx = vectorizer.word2idx

	"""get testdata in back to english"""
	testdata_eng = get_sentence_from_testdata(vectorizer, dataset.test_data.X)
	print("getting testdata back in english")

	"""get complete testdata as embeddings"""
	# test_data_embd_full = []
	# for e in dataset.test_data.X:
	# 	test_data_embd_full.append(get_embeddings_for_testdata(e, embd_dict))

	""" Compute Normal Grads"""
	print("getting NG for testdata")
	normal_grads = evaluator.get_grads_from_custom_td(dataset.test_data.X)

	# normal_grads_norm = normalise_grads(normal_grads['H'])

	################################# LIME STARTS HERE #################################

	# def lime_raw_string_preprocessor(word2idx,
	#                                  testdata_raw):  # customized for lime input collection which perturbs inputs by randomly masking words
	#
	# 	default = "<SOS> <UNK> <EOS>"  # all blank sentences must be corrected to this format
	#
	# 	unknowns = ['<SOS>', '<EOS>', '<PAD>', '<UNK>']
	# 	indexs = [2, 3, 0, 1]
	# 	mapped = dict(zip(unknowns, indexs))
	#
	# 	testdata_tokens = []
	#
	# 	for j in range(len(testdata_raw)):
	# 		t = testdata_raw[j]
	#
	# 		""" Check if t has any words"""
	# 		if (len(t.split()) == t.split().count('')):
	# 			t = default
	#
	# 		words = t.split()
	#
	# 		if (words[0] != '<SOS>'):
	# 			words.insert(0, '<SOS>')
	# 		if (words[-1] != '<EOS>'):
	# 			words.insert(len(words), '<EOS>')
	#
	# 		if (len(words) == 2):
	# 			words.insert(1, '<UNK>')
	#
	# 		token_list = []
	#
	# 		for i in range(len(words)):
	#
	# 			if words[i] in unknowns:  # because lime considers <,SOS and > as 3 separate words we remove them
	# 				token_list.append(mapped[words[i]])
	# 				continue
	#
	# 			token_list.append(word2idx[words[i]])
	#
	# 		testdata_tokens.append(token_list)
	# 	return testdata_tokens
	#
	# def model_pipeline(raw_string_ip, word2idx=word2idx):  # always load idx2word dict as default
	# 	# To be passed to lime explanation evaluator
	# 	# input: list of d input strings
	# 	# output: (d,k) ndarray where k is the number of classes
	#
	# 	raw_string_ip_tokens = lime_raw_string_preprocessor(word2idx, raw_string_ip)
	# 	raw_string_ip_preds = evaluator.evaluate_outputs_from_custom_td(raw_string_ip_tokens)
	# 	inv = np.ones_like(raw_string_ip_preds) - raw_string_ip_preds
	#
	# 	return np.concatenate((inv, raw_string_ip_preds), axis=-1)
	#
	# def custom_regex(string):  # limes regex doesnt recognise < and > to be a part of a word
	#
	# 	words = string.split(" ")
	#
	# 	return words
	#
	# """Test lime word attri for one instance"""
	#
	# sample = 1
	# categories = ['Bad', 'Good']
	# instance_of_interest = testdata_eng[sample]
	#
	# explainer = LimeTextExplainer(class_names=categories, verbose=True, split_expression=custom_regex)
	# exp = explainer.explain_instance(instance_of_interest, model_pipeline, num_features=6)
	# exp_for_instance = exp.as_list()
	#
	# print(exp_for_instance)
	#
	# exit(0)

	################################ LIME ENDS HERE #################################

	################################# IG STARTS HERE #################################

	"""Get Testdata_embd_collection of shape [testdata_count, Steps, Wordcount, Hiddensize] """
	print("converting dataset.testdata.X.embeddings to dataset.testdata.X.embedding.collection")
	test_data_embd_col = get_complete_testdata_embed_col(dataset, embd_dict, testdata_count=300, steps=50)

	"""Get preds for testdata from raw input"""
	print("getting preds and attn for dataset.test_data")
	preds_from_raw_input, attn_from_raw_input = evaluator.evaluate(dataset.test_data, save_results=False)

	"""get int_grads"""

	print("Getting {} instances of int_grads for test_data".format(len(test_data_embd_col)))

	int_grads = []

	for i in tqdm(range(len(test_data_embd_col))):
		sample = i
		one_sample = test_data_embd_col[sample]
		grads = evaluator.get_grads_from_custom_td(one_sample)
		int_grads.append(integrated_gradients(grads, one_sample, grads_wrt='XxE[X]'))

	"""Saving new IG as pickle"""
	# print("saving IG")
	# with open("./pickles/int_grads_x_500.pickle", "wb") as file:
	# 	pickle.dump(obj=int_grads, file=file)
	# print("saved")

	"""Loading IG from pickle"""
	#
	# print("getting IG from pickle")
	# file = open('./pickles/int_grads_x_1000.pickle', 'rb')
	# int_grads = pickle.load(file)

	################################# IG ENDS HERE #################################

	generate_graphs(evaluator, dataset, config['training']['exp_dirname'], evaluator.model, test_data=dataset.test_data,
	                int_grads=int_grads, norm_grads=normal_grads, for_only=len(test_data_embd_col))


def train_dataset(dataset, config='lstm', n_iter=1):
	try:
		# building the config and initializing the BC model with it through trainer wrapper class
		config = configurations[config](dataset)
		trainer = Trainer(dataset, config=config, _type=dataset.trainer_type)  # will create new model

		trainer.train(dataset.train_data, dataset.dev_data, n_iters=n_iter, save_on_metric=dataset.save_on_metric)
		evaluator = Evaluator(dataset, trainer.model.dirname, _type=dataset.trainer_type)
		_ = evaluator.evaluate(dataset.test_data, save_results=True)
		return trainer, evaluator
	except:
		return


def train_dataset_on_encoders(dataset, encoders):
	for e in encoders:
		train_dataset(dataset, e)
		run_experiments_on_latest_model(dataset, e)


def generate_graphs_on_encoders(dataset, encoders):
	for e in encoders:
		generate_graphs_on_latest_model(dataset, e)


def train_lr_on_dataset(dataset):
	config = {
		'vocab': dataset.vec,
		'stop_words': True,
		'type': dataset.trainer_type,
		'exp_name': dataset.name
	}

	dataset.train_data.y = np.array(dataset.train_data.y)
	dataset.test_data.y = np.array(dataset.test_data.y)
	if len(dataset.train_data.y.shape) == 1:
		dataset.train_data.y = dataset.train_data.y[:, None]
		dataset.test_data.y = dataset.test_data.y[:, None]
	lr = LR(config)
	lr.train(dataset.train_data)
	lr.evaluate(dataset.test_data, save_results=True)
	lr.save_estimator_logodds()
	return lr


def run_evaluator_on_latest_model(dataset, config='lstm'):
	config = configurations[config](dataset)
	latest_model = get_latest_model(os.path.join(config['training']['basepath'], config['training']['exp_dirname']))
	evaluator = Evaluator(dataset, latest_model, _type=dataset.trainer_type)
	_ = evaluator.evaluate(dataset.test_data, save_results=True)
	return evaluator


def run_experiments_on_latest_model(dataset, config='lstm', force_run=True):
	try:
		evaluator = run_evaluator_on_latest_model(dataset, config)
		test_data = dataset.test_data
		evaluator.gradient_experiment(test_data, force_run=force_run)
		evaluator.permutation_experiment(test_data, force_run=force_run)
		evaluator.adversarial_experiment(test_data, force_run=force_run)
	#        evaluator.remove_and_run_experiment(test_data, force_run=force_run)
	except Exception as e:
		print(e)
		return


################################################################################################ MODIFICATIONS START HERE


def integrated_gradients(grads, testdata, grads_wrt='H'):
	grads_list = grads[grads_wrt]
	input = np.array(testdata).sum(-1)

	x_dash = input[0]
	x = input[-1]
	diff = x - x_dash

	grads_list = np.add(grads_list[:-1], grads_list[1:])

	integral = np.average(np.array(grads_list), axis=0)

	int_grads = np.multiply(integral, diff)

	return int_grads


def normalise_grads(grads_list):
	cleaned = []

	for g in grads_list:
		sum = np.sum(g)
		c = [e / sum * 100 for e in g]
		cleaned.append(c)

	return cleaned


def make_single_attri_dict(txt, int_grads, norm_grads_unpruned):
	words = [e for e in txt.split(" ")]

	int_grads_dict = {}
	norm_grads_dict = {}
	norm_grads_pruned = (norm_grads_unpruned[0])[:len(int_grads[0])]

	assert len(int_grads[0]) == len(norm_grads_pruned)

	for i in range(len(words)):
		int_grads_dict[words[i]] = int_grads[0][i]
		norm_grads_dict[words[i]] = norm_grads_unpruned[0][i]

	return (int_grads_dict, norm_grads_dict)


def write_ig_to_file(int_grads, normal_grads_norm, preds, testdata_eng):
	print("Writing IG vs SG results to file")

	with open("./analysis/ig_vs_norm.txt", "a") as f:
		now = datetime.now()
		current_time = now.strftime("%H:%M:%S")
		f.write("\n\nCurrent Time = {}".format(current_time))

		for i in range(len(testdata_eng)):
			f.write("\nSentence:\n")
			f.write("prediction is: {}\n".format(preds[i]))
			f.write(testdata_eng[i] + "\n")
			i, n = make_single_attri_dict(testdata_eng[i], int_grads[i], normal_grads_norm[i])
			f.write("IG Says:\n")
			f.write(str(i) + "\n")
			f.write("Normal grad says\n")
			f.write(str(n))
			f.write("\n")


def get_sentence_from_testdata(vec, testdata):
	# testdata.X is a list of ndarrays
	reverse_dict = vec.idx2word

	txt = []

	for t in testdata:
		try:
			sent = []
			for ele in t:
				sent.append(reverse_dict[ele])
			sent = " ".join(sent)
			txt.append(sent)
		except:
			pass

	return (txt)


def load_int_grads(file='./pickles/int_grads.pickle'):
	print("loading int_grads from pickle")
	# load int_grads from pickle, wont affect because dataset random seed is fixed
	with open(file, 'rb') as handle:
		int_grads = pickle.load(handle)
	return int_grads


def integrated_grads_for_instance(grads, steps=50):
	grads_list = grads['H']
	int_grads_of_sample = []

	sum = np.zeros_like(grads_list[0])

	for sent in grads_list:
		sum = np.add(sum, sent)

	avg_grads = np.divide(sum, steps)

	return avg_grads


def swap_axis(test):
	# swap 0 and 1 axis of 3d list
	return [[i for i in element if i is not None] for element in list(zip_longest(*test))]


def get_collection_from_embeddings(embd_sent, steps=50):
	# takes test sentence embedding list [wc, 300] and converts into collection [steps, wc, 300]
	# embd_sent is a list of ndarrays

	embed_collection = []

	for e in embd_sent:  # word wise

		zero_vector = np.zeros_like(e)
		diff = e - zero_vector
		inc = np.divide(diff, steps)

		buffer = []
		buffer.append(list(zero_vector))

		for i in range(steps - 2):
			zero_vector = np.add(zero_vector, inc)
			buffer.append(list(zero_vector))

		buffer.append(list(e))
		embed_collection.append(buffer)

	return embed_collection


def get_complete_testdata_embed_col(dataset, embd_dict, testdata_count=1, steps=50):
	# returns tesdata of shape [No.of.instances, Steps, WC, hidden_size] for IG
	# testdata_count => how many sentences to convert, max = 4356 for imdb

	test_data_embeds = []

	for i in tqdm(range(testdata_count)):
		embds = get_embeddings_for_testdata(dataset.test_data.X[i], embd_dict)
		embds_col = get_collection_from_embeddings(embds, steps=steps)

		# swap axis 0 and 1 to ensure evaluator.evaluate is fed properly
		# assert embds_col[i][-1][5] == embds[i][5]  # check that the last embd in col == embd of testdata instance
		embds_col_swapped = swap_axis(embds_col)
		test_data_embeds.append(embds_col_swapped)

	print("done")

	return test_data_embeds


def get_embeddings_for_testdata(test_data, embd_dict):
	# takes one instance of testdata of shape 1xWC and returns embds of instance of shape 1xWCx300
	# returns list of ndarrays
	embd_sentence = []

	for t in test_data:  # token wise
		embd_sentence.append(list(embd_dict[t]))

	return embd_sentence


def get_embeddings_for_testdata_full(test_data_full, embd_dict, testdata_count=50):
	# does the same thing as the above function but returns the entire collection of test_data

	embed_col = []

	for i in range(testdata_count):
		sent = test_data_full[i]
		buffer = []
		for word in sent:
			buffer.append(list(embd_dict[word]))

		embed_col.append(buffer)

	return embed_col


###################################################################################################################   MODIFICATIONS END HERE


def generate_adversarial_examples(dataset, config='lstm'):
	evaluator = run_evaluator_on_latest_model(dataset, config)
	config = configurations[config](dataset)
	plot_adversarial_examples(dataset, config['training']['exp_dirname'], evaluator.model, test_data=dataset.test_data)


def generate_logodds_examples(dataset, config='lstm'):
	evaluator = run_evaluator_on_latest_model(dataset, config)
	config = configurations[config](dataset)
	plot_logodds_examples(dataset, config['training']['exp_dirname'], evaluator.model, test_data=dataset.test_data)


def run_logodds_experiment(dataset, config='lstm'):
	model = get_latest_model(os.path.join('outputs', dataset.name, 'LR+TFIDF'))
	print(model)
	logodds = pickle.load(open(os.path.join(model, 'logodds.p'), 'rb'))
	evaluator = run_evaluator_on_latest_model(dataset, config)
	evaluator.logodds_attention_experiment(dataset.test_data, logodds, save_results=True)


def run_logodds_substitution_experiment(dataset):
	model = get_latest_model(os.path.join('outputs', dataset.name, 'LR+TFIDF'))
	print(model)
	logodds = pickle.load(open(os.path.join(model, 'logodds.p'), 'rb'))
	evaluator = run_evaluator_on_latest_model(dataset)
	evaluator.logodds_substitution_experiment(dataset.test_data, logodds, save_results=True)


def get_top_words(dataset, config='lstm'):
	evaluator = run_evaluator_on_latest_model(dataset, config)
	test_data = dataset.test_data
	test_data.top_words_attn = find_top_words_in_all(dataset, test_data.X, test_data.attn_hat)


def get_results(path):
	latest_model = get_latest_model(path)
	if latest_model is not None:
		evaluations = json.load(open(os.path.join(latest_model, 'evaluate.json'), 'r'))
		return evaluations
	else:
		raise LookupError("No Latest Model ... ")


names = {
	'vanilla_lstm': 'LSTM',
	'lstm': 'LSTM + Additive Attention',
	'logodds_lstm': 'LSTM + Log Odds Attention',
	'lr': 'LR + BoW',
	'logodds_lstm_post': 'LSTM + Additive Attention (Log Odds at Test)'
}


def push_all_models(dataset, keys):
	model_evals = {}
	for e in ['vanilla_lstm', 'lstm', 'logodds_lstm']:
		config = configurations[e](dataset)
		path = os.path.join(config['training']['basepath'], config['training']['exp_dirname'])
		evals = get_results(path)
		model_evals[names[e]] = {keys[k]: evals[k] for k in keys}

	path = os.path.join('outputs', dataset.name, 'LR+TFIDF')
	evals = get_results(path)
	model_evals[names['lr']] = {keys[k]: evals[k] for k in keys}

	path = os.path.join('outputs', dataset.name, 'lstm+tanh+logodds(posthoc)')
	evals = get_results(path)
	model_evals[names['logodds_lstm_post']] = {keys[k]: evals[k] for k in keys}

	df = pd.DataFrame(model_evals).transpose()
	df['Model'] = df.index
	df = df.loc[[names[e] for e in ['lr', 'vanilla_lstm', 'lstm', 'logodds_lstm_post', 'logodds_lstm']]]

	os.makedirs(os.path.join('graph_outputs', 'evals'), exist_ok=True)
	df.to_csv(os.path.join('graph_outputs', 'evals', dataset.name + '+lstm+tanh.csv'), index=False)
	return df
