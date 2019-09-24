from Transparency.common_code.common import *
from Transparency.Trainers.PlottingBC import generate_graphs, plot_adversarial_examples, plot_logodds_examples
from Transparency.configurations import configurations
from Transparency.Trainers.TrainerBC import Trainer, Evaluator
from Transparency.model.LR import LR
from datetime import datetime
from itertools import zip_longest


def train_dataset(dataset, config='lstm', n_iter=2):
	try:
		# building the config and initializing the BC model with it through trainer wrapper class
		config = configurations[config](dataset)
		trainer = Trainer(dataset, config=config, _type=dataset.trainer_type)
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
	diff = x-x_dash

	d_alpha = np.divide(diff, len(testdata))

	integral = np.zeros_like(x)

	for g in grads_list:
		integral_body = np.multiply(g, d_alpha)
		integral = np.add(integral, integral_body)


	int_grads = np.multiply(integral,diff)


	return int_grads

def normalise_grads(grads_list):

	cleaned = []

	for g in grads_list:
		sum = np.sum(g)
		c = [e/sum*100 for e in g]
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

	return(int_grads_dict, norm_grads_dict)


def write_ig_to_file(int_grads, normal_grads_norm , preds, testdata_eng):

	print("Writing IG vs SG results to file")

	with open("./analysis/ig_vs_norm.txt", "a") as f:

		now = datetime.now()
		current_time = now.strftime("%H:%M:%S")
		f.write("\n\nCurrent Time = {}".format(current_time))

		for i in range(len(testdata_eng)):

			f.write("\nSentence:\n")
			f.write("prediction is: {}\n".format(preds[i]))
			f.write(testdata_eng[i]+"\n")
			i, n = make_single_attri_dict(testdata_eng[i], int_grads[i], normal_grads_norm[i])
			f.write("IG Says:\n")
			f.write(str(i)+"\n")
			f.write("Normal grad says\n")
			f.write(str(n))
			f.write("\n")


def get_sentence_from_testdata(vec, testdata):
	# testdata.X is a list of ndarrays
	reverse_dict = vec.idx2word

	txt = []

	for t in testdata:
		sent = []
		for ele in t:
			sent.append(reverse_dict[ele])
		sent = " ".join(sent)
		txt.append(sent)

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


def get_complete_testdata_embed_col(dataset, imdb_embd_dict, testdata_count=1):
	# returns tesdata of shape [No.of.instances, Steps, WC, hidden_size] for IG
	# testdata_count => how many sentences to convert, max = 4356 for imdb


	test_data_embeds = []

	for i in tqdm(range(testdata_count)):
		embds = get_embeddings_for_testdata(dataset.test_data.X[i], imdb_embd_dict)
		embds_col = get_collection_from_embeddings(embds, steps=50)

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
		embd_sentence.append(embd_dict[t])

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



def generate_graphs_on_latest_model_sst(dataset, config='lstm') :

	config = configurations[config](dataset)
	latest_model = get_latest_model(os.path.join(config['training']['basepath'], config['training']['exp_dirname']))
	evaluator = Evaluator(dataset, latest_model, _type=dataset.trainer_type)
	_ = evaluator.evaluate(dataset.test_data, save_results=False)

	normal_grads = evaluator.get_grads_from_custom_td(dataset.test_data.X)

	generate_graphs(evaluator, dataset, config['training']['exp_dirname'], evaluator.model, test_data=dataset.test_data, norm_grads=normal_grads)




def generate_graphs_on_latest_model(dataset, config='lstm'):

	print("INSIDE GENERATE GRAPHS FOR IMDB FUNCTION")
	config = configurations[config](dataset)
	latest_model = get_latest_model(os.path.join(config['training']['basepath'], config['training']['exp_dirname']))

	"""Get evaluator"""
	print("getting evaluator")
	evaluator = Evaluator(dataset, latest_model,
						  _type=dataset.trainer_type)  # 'evaluator' is wrapper for your loaded model

	"""get imdb vectorizer"""
	print("getting imdb vectorizer")
	file = open('./pickles/imdb_vectorizer.pickle', 'rb')
	imdb_vectorizer = pickle.load(file)

	"""get idx2word and reverse dict"""
	idx2word = imdb_vectorizer.idx2word
	word2idx = imdb_vectorizer.word2idx

	"""Get embed dictionary from imdb vectorizer"""
	imdb_embd_dict = dataset.vec.embeddings

	"""get testdata in back to english"""
	print("getting testdata back in english")
	testdata_eng = get_sentence_from_testdata(imdb_vectorizer, dataset.test_data.X)

	"""Get Testdata_embd_collection of shape [testdata_count, Steps, Wordcount, Hiddensize] """
	print("converting dataset.testdata.X.embeddings to dataset.testdata.X.embedding.collection")
	test_data_embd_col = get_complete_testdata_embed_col(dataset, imdb_embd_dict, testdata_count=51)

	"""Get preds for testdata from raw input"""
	print("gettings predictions and updating testdata.yhat and testdata.attn")
	preds_from_raw_input, attn_from_raw_input = evaluator.evaluate(dataset.test_data, save_results=False)

	""" Compute Normal Grads"""
	print("getting NG for testdata")
	normal_grads = evaluator.get_grads_from_custom_td(dataset.test_data.X)
	normal_grads_norm = normalise_grads(normal_grads['H'])


	"""Testing error in pred calc with direct embds"""

	# diff = []
	# for i in range(20):
	# 	sample = i
	# 	one_sample = test_data_embd_col[sample]
	# 	preds, attn = preds_from_raw_input[sample], attn_from_raw_input[sample]
	# 	preds_from_embd, attn_from_embd = evaluator.evaluate_outputs_from_embeds(one_sample)
	# 	preds2, attn2 = preds_from_embd[-1], attn_from_embd[-1]
	# 	diff.append(abs(preds-preds2))


	# """Sanity check"""
	# sample = 3
	# embds = get_embeddings_for_testdata(dataset.test_data.X[sample], embd_dict=imdb_embd_dict)
	# embds_zero = np.zeros_like(embds)
	#
	# pred_for_x_dash = evaluator.evaluate_outputs_from_embeds(np.expand_dims(embds_zero, axis=0))
	# pred_for_x = evaluator.evaluate_outputs_from_embeds()

	"""get int_grads for custom embeds"""

	int_grads = []

	for i in tqdm(range(50)): # only for 50 sentences

		sample = i
		one_sample = test_data_embd_col[sample]

		preds_for_embd, attn_for_embd = evaluator.evaluate_outputs_from_embeds(one_sample)
		# preds, attn = preds_from_raw_input[sample], attn_from_raw_input[sample]

		grads = evaluator.get_grads_from_custom_td(one_sample)

		int_grads.append(integrated_gradients(grads, one_sample))



	"""Saving new IG as pickle"""
	# print("saving")
	# with open("int_grads_new.pickle" , "wb") as file:
	# 	pickle.dump(obj=int_grads, file=file)
	#
	# print("saved")


	"""Normalize int grads"""
	# int_grads_norm = normalise_grads(int_grads)
	# attn_mod = attn[0:len(int_grads_norm)]*100


	"""Validate and write IG and NG results to file"""
	# write_ig_to_file(int_grads_norm, normal_grads_norm[0:50], preds_from_raw_input[0:50], testdata_eng[0:50])



	"""Set grads to None for normal repo functioning"""
	# normal_grads = None
	# int_grads = None


	"""Get predictions for entire testdata with raw input"""
	# preds, attn = evaluator.evaluate(dataset.test_data, save_results=False)


	"""Generate graphs for normal grads and int grads"""
	generate_graphs(evaluator, dataset, config['training']['exp_dirname'], evaluator.model,
					test_data=dataset.test_data, int_grads=int_grads, norm_grads=normal_grads)


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


