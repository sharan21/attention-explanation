from Transparency.common_code.common import *
from Transparency.Trainers.PlottingBC import generate_graphs, plot_adversarial_examples, plot_logodds_examples
from Transparency.configurations import configurations
from Transparency.Trainers.TrainerBC import Trainer, Evaluator
from Transparency.model.LR import LR


def train_dataset(dataset, config='lstm', n_iter=2):
	try:
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

def integrated_gradients(grads, testdata):

	# grads is a dict of 3 gradients, H , XxE, and XxE[X]
	grads_list = grads['H']

	x_dash = np.array(testdata[0])
	x = np.array(testdata[-1])
	diff = x-x_dash

	d_alpha = np.divide(diff, len(testdata))

	integral = np.zeros_like(x)

	for g in grads_list:
		integral_body = np.multiply(np.abs(g), d_alpha)
		integral = np.add(integral, integral_body)


	int_grads = np.multiply(integral,diff)

	# perturb int_grads to see effect on corr plot
	# int_grads = np.abs(np.random.randn(*int_grads.shape))
	# int_grads = np.zeros_like(int_grads)


	int_grads = np.divide(int_grads, np.sum(int_grads))*100

	return int_grads



def normalise_grads(grads_list):
	# takes grads['H'/'XxE'/'XxE[X]'

	cleaned = []

	for g in grads_list:
		sum = np.sum(g)
		c = [e/sum*100 for e in g]
		cleaned.append(c)

	return cleaned

def generate_input_collection_from_sample(dataset, steps = 10, sample=0):
	"""Returns list of ndarrays"""
	collection = []
	final_vector = dataset.test_data.X[sample]
	#5497 => horrible

	zero_vector = [5497]*len(final_vector)

	diff = list(np.abs(np.array(final_vector) - np.array(zero_vector)))
	inc = np.array([int(e/steps) for e in diff])

	collection.append(zero_vector)

	add = np.array(zero_vector)

	for i in range(steps-2): #-2 because we append zero and final vector
		add = add + inc
		collection.append(list(add))

	collection.append(final_vector)

	return(collection)


def generate_integrated_grads(evaluator, dataset):

	print("\nGenerating IG for {} input vectors from test_data...\n".format(len(dataset.test_data.X)))

	int_grads = []

	for i in tqdm(range(len(dataset.test_data.X))): # number of testing examples
		collection = generate_input_collection_from_sample(dataset, sample=i)
		# _ = evaluator.evaluate(collection, save_results=False)
		grads = evaluator.get_grads_from_custom_td(collection)
		int_grads_of_sample = integrated_gradients(grads, collection)
		# int_grads_of_sample = process_int_grads(int_grads_of_sample)


		int_grads.append(int_grads_of_sample)



	return(int_grads)

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

	return(txt)

def load_int_grads(file = 'int_grads_20_steps.pickle'):
	print("loading int_grads from pickle")
	# load int_grads from pickle, wont affect because dataset random seed is fixed
	with open(file, 'rb') as handle:
		int_grads = pickle.load(handle)
	return int_grads

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


def write_ig_to_file(int_grads, normal_grads_norm , preds, testdata_eng, iter=10):

	print("Writing IG vs SG results to file")

	with open("ig_vs_norm.txt", "w") as f:
		for i in range(iter):

			f.write("\nSentence:\n")
			f.write("prediction is: {}\n".format(preds[i]))
			f.write(testdata_eng[i]+"\n")
			i, n = make_single_attri_dict(testdata_eng[i], int_grads[i], normal_grads_norm[i])
			f.write("IG Says:\n")
			f.write(str(i)+"\n")
			f.write("Normal grad says\n")
			f.write(str(n))
			f.write("\n")



def generate_graphs_on_latest_model(dataset, config='lstm'):
	config = configurations[config](dataset)
	latest_model = get_latest_model(os.path.join(config['training']['basepath'], config['training']['exp_dirname']))

	# 'evaluator' is wrapper for your loaded model
	print("getting evaluator")
	evaluator = Evaluator(dataset, latest_model, _type=dataset.trainer_type)

	#get imdb vectorizer
	print("getting imdb vectorizer")
	file = open('imdb_vectorizer.pickle', 'rb')
	imdb_vectorizer = pickle.load(file)




	# get testdata in back to english
	print("getting testdata back in english")
	testdata_eng = get_sentence_from_testdata(imdb_vectorizer, dataset.test_data.X)

	# load int_grads to save time
	# int_grads = load_int_grads(file='int_grads_10_steps_bad.pickle')

	# compute integrated grads for whole dataset.testdata.X
	int_grads = generate_integrated_grads(evaluator, dataset) # get integrated gradients, [4356*Word_count]

	# compute normal grads for whole dataset.testdata.X
	normal_grads = evaluator.get_grads_from_custom_td(dataset.test_data.X)
	normal_grads_norm = normalise_grads(normal_grads['H'])

	# pickle dump int_grads for future use
	print("saving int_grads")
	with open("int_grads_10_steps_bad.pickle", 'wb') as handle:
		pickle.dump(int_grads, handle)


	# this updates test_data.X_hat and test_data_attn, needed for corr plot
	preds, attn = evaluator.evaluate(dataset.test_data, save_results=False)

	# Validating IG amd SG
	write_ig_to_file(int_grads, normal_grads_norm, preds, testdata_eng)

	#exit(0)

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

