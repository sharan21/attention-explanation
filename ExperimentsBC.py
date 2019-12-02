from common_code.common import *
from Trainers.PlottingBC import generate_graphs, plot_adversarial_examples, plot_logodds_examples
from configurations import configurations
from Trainers.TrainerBC import Trainer, Evaluator
from model.LR import LR
from helpers import *


def generate_graphs_on_encoders(dataset, encoders):
    for e in encoders:
        generate_graphs_on_latest_model(dataset, e)


def generate_graphs_on_latest_model(dataset, config='lstm'):

    config = configurations[config](dataset)
    latest_model = get_latest_model(os.path.join(config['training']['basepath'], config['training']['exp_dirname']))
    print("latest model is {}".format(latest_model))
    evaluator = Evaluator(dataset, latest_model, _type=dataset.trainer_type)
    _ = evaluator.evaluate(dataset.test_data, save_results=False)

    """Get Attributions"""
    # lrp_attri = evaluator.model.lrp_mem(dataset.test_data.X, no_of_instances=10)
    # int_grads = evaluator.model.integrated_gradient_mem(dataset, no_of_instances=10)
    # normal_grads = evaluator.get_grads_from_custom_td(dataset.test_data.X):w
    # lime_attri = evaluator.model.lime_attribution_mem(dataset, no_of_instances=10)
    # dl_attri = evaluator.model.deeplift_mem(dataset, no_of_instances=10)


    generate_graphs(dataset, config['training']['exp_dirname'], evaluator.model, test_data=dataset.test_data,
                    int_grads=None, norm_grads=None, lime=None, lrp=None, dl=None, for_only=-1)


def train_dataset(dataset, config='lstm', n_iter=2):
    try:
        # building the config and initializing the BC model with it through trainer wrapper class
        config = configurations[config](dataset)
        # trainer = Trainer(dataset, config=config, _type=dataset.trainer_type)  # will create new model

        # trainer.train(dataset.train_data, dataset.dev_data, n_iters=n_iter, save_on_metric=dataset.save_on_metric)
        evaluator = Evaluator(dataset, trainer.model.dirname, _type=dataset.trainer_type)
        _ = evaluator.evaluate(dataset.test_data, save_results=True)
        return trainer, evaluator
    except Exception as e:

        print(e)
        exit()
        return

def train_dataset_on_encoders(dataset, encoders):
    for e in encoders:
        # train_dataset(dataset, e, n_iter=1)
        run_experiments_on_latest_model(dataset, e)


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

    evaluator = run_evaluator_on_latest_model(dataset, config)
    test_data = dataset.test_data
    # evaluator.gradient_experiment(test_data, force_run=force_run)
    #evaluator.permutation_experiment(test_data, force_run=force_run)
    # evaluator.adversarial_experiment(test_data, force_run=force_run)
    # evaluator.integrated_gradient_experiment(dataset, force_run=force_run)
    # evaluator.lime_attribution_experiment(dataset, force_run=force_run)
    evaluator.lrp_attribution_experiment(dataset, force_run=force_run)


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


######## OLD CODE #######

def generate_graphs_on_latest_model_debug(dataset, config='lstm'):
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

    # print("getting NG for testdata")
    normal_grads = evaluator.get_grads_from_custom_td(dataset.test_data.X)
    # normal_grads_norm = normalise_grads(normal_grads['H'])

    ################################ LIME STARTS HERE #################################

    def custom_regex(string):  # limes regex doesnt recognise < and > to be a part of a word

        words = string.split(" ")
        return words

    def lime_raw_string_preprocessor(word2idx, testdata_raw):
        # customized for lime input collection which perturbs inputs by randomly masking words

        default = "<SOS> <UNK> <EOS>"  # all blank sentences must be corrected to this format
        unknowns = ['<SOS>', '<EOS>', '<PAD>', '<UNK>']
        indexs = [2, 3, 0, 1]
        mapped = dict(zip(unknowns, indexs))

        testdata_tokens = []

        for j in range(len(testdata_raw)):
            t = testdata_raw[j]

            """ Check if t has any words"""
            if (len(t.split()) == t.split().count('')):
                t = default

            words = t.split()

            if (words[0] != '<SOS>'):
                words.insert(0, '<SOS>')
            if (words[-1] != '<EOS>'):
                words.insert(len(words), '<EOS>')

            if (len(words) == 2):
                words.insert(1, '<UNK>')

            token_list = []
            for i in range(len(words)):

                if words[i] in unknowns:  # because lime considers <,SOS and > as 3 separate words we remove them
                    token_list.append(mapped[words[i]])
                    continue

                token_list.append(word2idx[words[i]])

            testdata_tokens.append(token_list)
        return testdata_tokens

    def model_pipeline(raw_string_ip, word2idx=word2idx):  # always load idx2word dict as default
        # To be passed to lime explanation evaluator
        # input: list of d input strings
        # output: (d,k) ndarray where k is the number of classes

        raw_string_ip_tokens = lime_raw_string_preprocessor(word2idx, raw_string_ip)
        raw_string_ip_preds = evaluator.evaluate_outputs_from_custom_td(raw_string_ip_tokens, use_tqdm=False)
        inv = np.ones_like(raw_string_ip_preds) - raw_string_ip_preds

        return np.concatenate((inv, raw_string_ip_preds), axis=-1)

    def unshuffle(explanations, sample):
        # input is list of keyword tuples
        # output is list of float attributions`

        words, weights = zip(*explanations)
        words = list(words)
        weights = list(weights)
        sample = sample.split(" ")
        attri = []

        for s in sample:
            try:
                if (s == "<SOS>" or s == "<EOS>"):
                    attri.append(0.0)
                    continue
                if(s in words):
                    index = words.index(s)
                    attri.append(abs(weights[index]))
                else:
                    attri.append(0.0)
            except Exception as e:
                print(e)

        return attri


    """ Get lime attributions"""

    testdata_instances = 10
    print("find lime attributions for {} instances".format(testdata_instances))

    lime_attri = []
    categories = ['Bad', 'Good']

    for i in tqdm(range(10)):

        sample = i
        instance_of_interest = testdata_eng[sample]
        explainer = LimeTextExplainer(class_names=categories, verbose=True, split_expression=custom_regex)
        exp = explainer.explain_instance(instance_of_interest, model_pipeline, num_features=6)
        exp_for_instance = exp.as_list()
        attri = unshuffle(exp_for_instance, instance_of_interest)
        lime_attri.append(attri)


    """Get int_grads"""
    int_grads = evaluator.model.integrated_gradient_mem(dataset, no_of_instances=testdata_instances)

    """get deep lift attri"""
    """Get trained nn.embedding weights"""
    embd_dict = np.array(evaluator.model.encoder.embedding.weight.data)

    """get complete testdata as embeddings"""
    test_data_embds_full = []
    baseline_embds_full = []

    for e in dataset.test_data.X:
        test_data_embds_full.append(get_embeddings_for_testdata(e, embd_dict))
        baseline_embds_full.append(get_baseline_embeddings_for_testdata(e, embd_dict))

    hs_bs, attn_bs, ctx_bs, u_outs_bs, outs_bs = evaluator.model.evaluate_and_buffer(baseline_embds_full,
                                                                                     no_of_instances=50)
    hs, attn, ctx, u_outs, outs = evaluator.model.evaluate_and_buffer(test_data_embds_full, no_of_instances=50)

    delta_x = dict()

    delta_x['d_o'] = np.subtract(outs, outs_bs)
    delta_x['d_uo'] = np.subtract(u_outs, u_outs_bs)
    delta_x['d_ctx'] = np.subtract(ctx, ctx_bs)
    delta_x['d_attn'] = np.subtract(attn, attn_bs)
    delta_x['d_hs'] = np.subtract(hs, hs_bs)

    rel_ctx, rel_attn = evaluator.model.get_deeplift(delta_x)



    print(np.array(int_grads).shape)
    print(np.array(lime_attri).shape)


    generate_graphs(dataset, config['training']['exp_dirname'], evaluator.model, test_data=dataset.test_data,
                    int_grads=int_grads, norm_grads=normal_grads, lime=lime_attri, for_only=testdata_instances)




# for i in range(len(testdata_eng)):
# 	print(testdata_eng[i])
# 	print(len(testdata_eng[i].split(" ")))
# 	print(rel_attn[i])
# 	print(len(rel_attn[i]))
# 	print(" sum is: {}".format(rel_attn[i].sum(0)))

# forward_weight_names = evaluator.model.encoder.rnn._all_weights[0]
# reverse_weight_names = evaluator.model.encoder.rnn._all_weights[1]
#
# f_w = []
# for f in forward_weight_names:
# 	f_w.append(evaluator.model.encoder.rnn._parameters[f])
#
# r_w = []
# for r in reverse_weight_names:
# 	f_w.append(evaluator.model.encoder.rnn._parameters[f])
#
# print(evaluator.model.encoder_params)
