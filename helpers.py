import numpy as np
from datetime import datetime
from tqdm import tqdm
import pickle
from itertools import zip_longest
from lime.lime_text import LimeTextExplainer


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

def model_pipeline(raw_string_ip, word2idx, evaluator):  # always load idx2word dict as default
    # To be passed to lime explanation evaluator
    # input: list of d input strings
    # output: (d,k) ndarray where k is the number of classes

    raw_string_ip_tokens = lime_raw_string_preprocessor(word2idx, raw_string_ip)
    raw_string_ip_preds = evaluator.evaluate_outputs_from_custom_td(raw_string_ip_tokens)
    inv = np.ones_like(raw_string_ip_preds) - raw_string_ip_preds

    return np.concatenate((inv, raw_string_ip_preds), axis=-1)

def custom_regex(string):  # limes regex doesnt recognise < and > to be a part of a word

    words = string.split(" ")
    return words
