import json
import os
import shutil
from copy import deepcopy
from typing import Dict

import torch
import torch.nn as nn
from allennlp.common import Params
from sklearn.utils import shuffle

from model.modules.Decoder import AttnDecoder
from model.modules.Encoder import Encoder

from .modelUtils import BatchHolder, get_sorting_index_with_noise_from_lengths
from .modelUtils import jsd as js_divergence

from helpers import *
from model.modules import lrplstm

file_name = os.path.abspath(__file__)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class AdversaryMulti(nn.Module) :
    def __init__(self, decoder=None) :
        super().__init__()
        self.decoder = decoder
        self.K = 5

    def forward(self, data) :
        data.hidden_volatile = data.hidden.detach()

        new_attn = torch.log(data.generate_uniform_attn()).unsqueeze(1).repeat(1, self.K, 1) #(B, 10, L)
        new_attn = new_attn + torch.randn(new_attn.size()).to(device)*3

        new_attn.requires_grad = True

        data.log_attn_volatile = new_attn
        optim = torch.optim.Adam([data.log_attn_volatile], lr=0.01, amsgrad=True)

        for _ in range(500) :
            log_attn = data.log_attn_volatile + 1 - 1
            log_attn.masked_fill_(data.masks.unsqueeze(1), -float('inf'))
            data.attn_volatile = nn.Softmax(dim=-1)(log_attn) #(B, 10, L)
            self.decoder.get_output(data)
            predict_new = data.predict_volatile #(B, 10, O)

            y_diff = torch.sigmoid(predict_new) - torch.sigmoid(data.predict.detach()).unsqueeze(1) #(B, 10, O)
            diff = nn.ReLU()(torch.abs(y_diff).sum(-1, keepdim=True) - 1e-2) #(B, 10, 1)

            jsd = js_divergence(data.attn_volatile, data.attn.detach().unsqueeze(1)) #(B, 10, 1)
            cross_jsd = js_divergence(data.attn_volatile.unsqueeze(1), data.attn_volatile.unsqueeze(2))

            loss =  -(jsd**1) + 500 * diff
            loss = loss.sum() - cross_jsd.sum(0).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()

        log_attn = data.log_attn_volatile + 1 - 1
        log_attn.masked_fill_(data.masks.unsqueeze(1), -float('inf'))
        data.attn_volatile = nn.Softmax(dim=-1)(log_attn)
        self.decoder.get_output(data)
        data.predict_volatile = torch.sigmoid(data.predict_volatile)

class Model() :

    def __init__(self, configuration, pre_embed=None) : #pre_embed is dataset.vec.embeddings, [12k, 300d]


        configuration = deepcopy(configuration)
        self.configuration = deepcopy(configuration)

        # saving pre_embeds in encoder config
        configuration['model']['encoder']['pre_embed'] = pre_embed
        # initializing encoder with pre_embeddings
        self.encoder = Encoder.from_params(Params(configuration['model']['encoder'])).to(device)

        configuration['model']['decoder']['hidden_size'] = self.encoder.output_size
        self.decoder = AttnDecoder.from_params(Params(configuration['model']['decoder'])).to(device)

        self.encoder_params = list(self.encoder.parameters())
        self.attn_params = list([v for k, v in self.decoder.named_parameters() if 'attention' in k])
        self.decoder_params = list([v for k, v in self.decoder.named_parameters() if 'attention' not in k])

        self.bsize = configuration['training']['bsize']

        weight_decay = configuration['training'].get('weight_decay', 1e-5)
        self.encoder_optim = torch.optim.Adam(self.encoder_params, lr=0.001, weight_decay=weight_decay, amsgrad=True)
        self.attn_optim = torch.optim.Adam(self.attn_params, lr=0.001, weight_decay=0, amsgrad=True)
        self.decoder_optim = torch.optim.Adam(self.decoder_params, lr=0.001, weight_decay=weight_decay, amsgrad=True)
        self.adversarymulti = AdversaryMulti(decoder=self.decoder)

        pos_weight = configuration['training'].get('pos_weight', [1.0]*self.decoder.output_size)
        self.pos_weight = torch.Tensor(pos_weight).to(device)
        self.criterion = nn.BCEWithLogitsLoss(reduction='none').to(device)

        import time
        dirname = configuration['training']['exp_dirname']
        basepath = configuration['training'].get('basepath', 'outputs')
        self.time_str = time.ctime().replace(' ', '_')
        self.dirname = os.path.join(basepath, dirname, self.time_str)

    @classmethod
    def init_from_config(cls, dirname, **kwargs) :
        config = json.load(open(dirname + '/config.json', 'r'))
        config.update(kwargs)
        obj = cls(config)
        obj.load_values(dirname)
        return obj

    def train(self, data_in, target_in, train=True) :
        sorting_idx = get_sorting_index_with_noise_from_lengths([len(x) for x in data_in], noise_frac=0.1)
        data = [data_in[i] for i in sorting_idx]
        target = [target_in[i] for i in sorting_idx]

        self.encoder.train()
        self.decoder.train()
        bsize = self.bsize
        N = len(data)
        loss_total = 0

        batches = list(range(0, N, bsize))
        batches = shuffle(batches)

        for n in tqdm(batches) :
            torch.cuda.empty_cache()
            batch_doc = data[n:n+bsize]
            batch_data = BatchHolder(batch_doc)

            self.encoder(batch_data)
            self.decoder(batch_data)

            batch_target = target[n:n+bsize]
            batch_target = torch.Tensor(batch_target).to(device)

            if len(batch_target.shape) == 1 : #(B, )
                batch_target = batch_target.unsqueeze(-1) #(B, 1)

            bce_loss = self.criterion(batch_data.predict, batch_target)
            weight = batch_target * self.pos_weight + (1 - batch_target)
            bce_loss = (bce_loss * weight).mean(1).sum()

            loss = bce_loss

            if hasattr(batch_data, 'reg_loss') :
                loss += batch_data.reg_loss

            if train :
                self.encoder_optim.zero_grad()
                self.decoder_optim.zero_grad()
                self.attn_optim.zero_grad()
                loss.backward()
                self.encoder_optim.step()
                self.decoder_optim.step()
                self.attn_optim.step()

            loss_total += float(loss.data.cpu().item())
        return loss_total*bsize/N

    def evaluate(self, data, use_tqdm=True) :

        if(len(np.array(data).shape) == 3): #fails when leading length is very big, shape will be 2 dimentional
            is_embed = True
        else:
            is_embed = False

        print(is_embed)



        self.encoder.train()
        self.decoder.train()

        bsize = self.bsize

        N = len(data)

        outputs = []
        attns = []

        for n in range(0, N, bsize):


            torch.cuda.empty_cache()

            batch_doc = data[n:n+bsize]
            # from batch_doc => batch_data, type gets converted from float64 => torch.int64 instead of torch.float64

            batch_data = BatchHolder(batch_doc, is_embed=is_embed)

            self.encoder(batch_data)
            self.decoder(batch_data)

            # return self.decoder.get_context(batch_data)




            batch_data.predict = torch.sigmoid(batch_data.predict)
            if self.decoder.use_attention :

                attn = batch_data.attn.cpu().data.numpy()
                attns.append(attn)

            predict = batch_data.predict.cpu().data.numpy()
            outputs.append(predict)

        outputs = [x for y in outputs for x in y]
        if self.decoder.use_attention :
            attns = [x for y in attns for x in y]

        return outputs, attns


    def evaluate_and_buffer(self, data_in, no_of_instances):
        # Get the outputs of each layer to feed into deeplift
        # raw_inp -> embeddings -> hidden states -> attention weights -> context vector -> output

        data = data_in[0:no_of_instances]

        # fails when leading length is very big, shape will be 2 dimentional
        if(len(np.array(data).shape) == 3):
            is_embed = True
        else:
            is_embed = False

        print("in evaluate_and_buffer")
        print(is_embed)
        print(np.array(data[0]).shape)
        bsize = self.bsize

        N = len(data)

        outputs = []
        unactivated_outputs = []
        attns = []
        context = []
        hidden_states = []

        for n in range(0, N, bsize):
            torch.cuda.empty_cache()
            batch_doc = data[n:n+bsize]
            batch_data = BatchHolder(batch_doc, is_embed=is_embed)
            self.encoder(batch_data)
            hs = batch_data.hidden.cpu().data.numpy()
            hidden_states.append(hs)
            self.decoder(batch_data)
            unactivated_outputs.append(batch_data.predict)


            batch_data.predict = torch.sigmoid(batch_data.predict)

            if self.decoder.use_attention :
                attn = batch_data.attn.cpu().data.numpy()
                attns.append(attn)
                cxt = self.decoder.get_context(batch_data).cpu().data.numpy()
                context.append(cxt)
            predict = batch_data.predict.cpu().data.numpy()
            outputs.append(predict)

        #Unpacking batches->instances
        context = [x for y in context for x in y]
        u_outputs = [x for y in outputs for x in y]
        unactivated_outputs = [x for y in unactivated_outputs for x in y]
        hidden_states = [x for y in hidden_states for x in y]
        if self.decoder.use_attention :
            attns = [x for y in attns for x in y]

        return hidden_states, attns, context, u_outputs, outputs

    def lrp_mem(self, data_in, no_of_instances = 100) :
        # returns LRP Decomposition wrt attention layer (B, L) and wrt Decoder context input (B, H)

        data_in = data_in[0:no_of_instances]

        sorting_idx = get_sorting_index_with_noise_from_lengths([len(x) for x in data_in], noise_frac=0.1)
        data = [data_in[i] for i in sorting_idx]
        # target = [target_in[i] for i in sorting_idx]


        bsize = self.bsize
        N = len(data)
        # loss_total = 0

        batches = list(range(0, N, bsize))
        batches = shuffle(batches)

        lrp_attri = []

        # output_buffer = []

        for n in tqdm(batches) :
            torch.cuda.empty_cache()
            batch_doc = data[n:n+bsize]
            batch_data = BatchHolder(batch_doc)

            self.encoder(batch_data)
            lrp_attn, _ = self.decoder.lrp(batch_data)

            lrp_attri.extend(lrp_attn)

        return lrp_attri

    def get_attention(self, data, no_of_instances=10):
        #Used for human eval scripts
        #returns list of ndarrays

        data = data[0:no_of_instances]

        # sorting_idx = get_sorting_index_with_noise_from_lengths([len(x) for x in data], noise_frac=0.1)
        # data = [data[i] for i in sorting_idx]

        bsize = self.bsize
        N = len(data)

        batches = list(range(0, N, bsize))
        batches = shuffle(batches)

        attn = []

        for n in tqdm(batches):

            torch.cuda.empty_cache()
            batch_doc = data[n:n + bsize]
            batch_data = BatchHolder(batch_doc)

            self.encoder(batch_data)

            batch_attn = self.decoder.get_attention(batch_data)

            attn.extend(np.array(batch_attn.data))

        return attn





    def deeplift_mem(self, dataset, no_of_instances=100):
        # returns deeplift relevances scores wrt attention weights and context vector (only Decoder level)
        # does not propagate to encoder level yet because im not sure how to propagate through bahdanu attention layer

        embd_dict = np.array(self.encoder.embedding.weight.data)

        test_data_embds_full = []
        baseline_embds_full = []

        for e in dataset.test_data.X:
            test_data_embds_full.append(get_embeddings_for_testdata(e, embd_dict))
            baseline_embds_full.append(get_baseline_embeddings_for_testdata(e, embd_dict))

        hs_bs, attn_bs, ctx_bs, u_outs_bs, outs_bs = self.evaluate_and_buffer(baseline_embds_full, no_of_instances=no_of_instances)

        hs, attn, ctx, u_outs, outs = self.evaluate_and_buffer(test_data_embds_full, no_of_instances=no_of_instances)

        delta_x = dict() #holds difference in outputs for each layer

        delta_x['d_o'] = np.subtract(outs, outs_bs)
        delta_x['d_uo'] = np.subtract(u_outs, u_outs_bs)
        delta_x['d_ctx'] = np.subtract(ctx, ctx_bs)
        delta_x['d_attn'] = np.subtract(attn, attn_bs)
        delta_x['d_hs'] = np.subtract(hs, hs_bs)

        rel_attn, rel_ctx = self.decoder.deeplift(delta_x)

        return rel_attn, rel_ctx



    def gradient_mem(self, data) :

        if (len(np.array(data).shape) == 3):
            is_embed = True
        else:
            is_embed = False


        self.encoder.train()
        self.decoder.train()
        bsize = self.bsize
        N = len(data)

        grads = {'XxE' : [], 'XxE[X]' : [], 'H' : []}


        for n in range(0, N, bsize) :
            torch.cuda.empty_cache()
            batch_doc = data[n:n+bsize]

            grads_xxe = []
            grads_xxex = []
            grads_H = []

            for i in range(self.decoder.output_size) :
                batch_data = BatchHolder(batch_doc, is_embed=is_embed)
                batch_data.keep_grads = True
                batch_data.detach = True

                # running encoder updates batch_data with embedding, embedding.grad_fn() hook and hidden, hidden.grad_fn()
                self.encoder(batch_data)
                # running decoder calculates batch_data.embeddings.grad and batch_data.hidden.grad
                self.decoder(batch_data)

                torch.sigmoid(batch_data.predict[:, i]).sum().backward()


                """get XxE[X]"""

                g = batch_data.embedding.grad
                em = batch_data.embedding
                g1 = (g * em).sum(-1)


                grads_xxex.append(g1.cpu().data.numpy())

                """get XxE"""

                g1 = (g * self.encoder.embedding.weight.sum(0)).sum(-1)
                grads_xxe.append(g1.cpu().data.numpy())


                """get H"""

                g1 = batch_data.hidden.grad.sum(-1)
                grads_H.append(g1.cpu().data.numpy())


            grads_xxe = np.array(grads_xxe).swapaxes(0, 1)
            grads['XxE'].append(grads_xxe)

            grads_xxex = np.array(grads_xxex).swapaxes(0, 1)
            grads['XxE[X]'].append(grads_xxex)

            grads_H = np.array(grads_H).swapaxes(0, 1)
            grads['H'].append(grads_H)


        for k in grads :
            grads[k] = [x for y in grads[k] for x in y]

        return grads

    def integrated_gradient_mem(self, data, grads_wrt='XxE[X]', no_of_instances=100, steps=50):

        #NOTE: Integrated gradients by default will only calculate IG for 100 instances and wrt grads['XxE[X]'] to reduce computation time
        #Change 'grads_wrt' and 'no_of_instances' accordingly to match correlation plot of normal gradients
        #Unlike gradients_mem, IG should be invoked after model has been trained, so that self.encoder.embeddings.weight.data is accurate

        embd_dict = np.array(self.encoder.embedding.weight.data)
        print("getting testdata embed col")
        test_data_embd_col = get_complete_testdata_embed_col(data, embd_dict, testdata_count=no_of_instances, steps=steps)
        int_grads = []

        print("calculating IG")
        for i in tqdm(range(len(test_data_embd_col))):

            sample = i
            one_sample = test_data_embd_col[sample]
            grads = self.get_grads_from_custom_td(one_sample)
            int_grads.append(integrated_gradients(grads, one_sample, grads_wrt=grads_wrt))

        return int_grads

    def lime_attribution_mem(self, dataset, no_of_instances=10):
        #NOTE: Lime attributions by default calculate only 10 instances since to reduce computation time

        path = 'preprocess/{}/vec_{}.p'.format(dataset.name, dataset.name)
        try:
            file = open(path, 'rb')
        except:
            print("preprocess/{}/vec_{}.py not found".format(dataset.name, dataset.name))

        vectorizer = pickle.load(file)
        testdata_eng = get_sentence_from_testdata(vectorizer, dataset.test_data.X)


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

        def model_pipeline(raw_string_ip, word2idx=vectorizer.word2idx):
            # To be passed to lime explanation evaluator
            # input: list of d input strings
            # output: (d,k) ndarray where k is the number of classes

            raw_string_ip_tokens = lime_raw_string_preprocessor(word2idx, raw_string_ip)
            raw_string_ip_preds = self.evaluate_outputs_from_custom_td(raw_string_ip_tokens)
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
                    if (s in words):
                        index = words.index(s)
                        attri.append(abs(weights[index]))
                    else:
                        attri.append(0.0)
                except Exception as e:
                    print(e)

            return attri

        print("find lime attributions for {} instances".format(no_of_instances))

        lime_attri = []
        categories = ['Bad', 'Good']

        for i in tqdm(range(no_of_instances)):
            sample = i
            instance_of_interest = testdata_eng[sample]
            explainer = LimeTextExplainer(class_names=categories, verbose=True, split_expression=custom_regex)
            exp = explainer.explain_instance(instance_of_interest, model_pipeline, num_features=6)
            exp_for_instance = exp.as_list()
            attri = unshuffle(exp_for_instance, instance_of_interest)
            lime_attri.append(attri)

        return lime_attri


    def get_grads_from_custom_td(self, test_data):
        grads = self.gradient_mem(test_data)
        return grads


    def evaluate_outputs_from_custom_td(self, testdata):
        predictions, _ = self.evaluate(testdata)
        return predictions


    def remove_and_run(self, data) :
        self.encoder.train()
        self.decoder.train()
        bsize = self.bsize
        N = len(data)

        outputs = []

        for n in range(0, N, bsize) :
            batch_doc = data[n:n+bsize]
            batch_data = BatchHolder(batch_doc)
            po = np.zeros((batch_data.B, batch_data.maxlen, self.decoder.output_size))

            for i in range(1, batch_data.maxlen - 1) :
                batch_data = BatchHolder(batch_doc)

                batch_data.seq = torch.cat([batch_data.seq[:, :i], batch_data.seq[:, i+1:]], dim=-1)
                batch_data.lengths = batch_data.lengths - 1
                batch_data.masks = torch.cat([batch_data.masks[:, :i], batch_data.masks[:, i+1:]], dim=-1)

                self.encoder(batch_data)
                self.decoder(batch_data)

                po[:, i] = torch.sigmoid(batch_data.predict).cpu().data.numpy()

            outputs.append(po)

        outputs = [x for y in outputs for x in y]

        return outputs

    def permute_attn(self, data, num_perm=100) :
        self.encoder.train()
        self.decoder.train()
        bsize = self.bsize
        N = len(data)

        permutations = []

        for n in tqdm(range(0, N, bsize)) :
            torch.cuda.empty_cache()
            batch_doc = data[n:n+bsize]
            batch_data = BatchHolder(batch_doc)

            batch_perms = np.zeros((batch_data.B, num_perm, self.decoder.output_size))

            self.encoder(batch_data)
            self.decoder(batch_data)

            for i in range(num_perm) :
                batch_data.permute = True
                self.decoder(batch_data)
                output = torch.sigmoid(batch_data.predict)
                batch_perms[:, i] = output.cpu().data.numpy()

            permutations.append(batch_perms)

        permutations = [x for y in permutations for x in y]

        return permutations

    def save_values(self, use_dirname=None, save_model=True) :
        if use_dirname is not None :
            dirname = use_dirname
        else :
            dirname = self.dirname
        os.makedirs(dirname, exist_ok=True)
        shutil.copy2(file_name, dirname + '/')
        json.dump(self.configuration, open(dirname + '/config.json', 'w'))

        if save_model :
            torch.save(self.encoder.state_dict(), dirname + '/enc.th')
            torch.save(self.decoder.state_dict(), dirname + '/dec.th')

        return dirname

    def load_values(self, dirname) :
        self.encoder.load_state_dict(torch.load(dirname + '/enc.th', map_location={'cuda:1': 'cuda:0'}))
        self.decoder.load_state_dict(torch.load(dirname + '/dec.th', map_location={'cuda:1': 'cuda:0'}))

    def adversarial_multi(self, data) :
        self.encoder.eval()
        self.decoder.eval()

        for p in self.encoder.parameters() :
            p.requires_grad = False

        for p in self.decoder.parameters() :
            p.requires_grad = False

        bsize = self.bsize
        N = len(data)

        adverse_attn = []
        adverse_output = []

        for n in tqdm(range(0, N, bsize)) :
            torch.cuda.empty_cache()
            batch_doc = data[n:n+bsize]
            batch_data = BatchHolder(batch_doc)

            self.encoder(batch_data)
            self.decoder(batch_data)

            self.adversarymulti(batch_data)

            attn_volatile = batch_data.attn_volatile.cpu().data.numpy() #(B, 10, L)
            predict_volatile = batch_data.predict_volatile.cpu().data.numpy() #(B, 10, O)

            adverse_attn.append(attn_volatile)
            adverse_output.append(predict_volatile)

        adverse_output = [x for y in adverse_output for x in y]
        adverse_attn = [x for y in adverse_attn for x in y]

        return adverse_output, adverse_attn

    def logodds_attention(self, data, logodds_map:Dict) :
        self.encoder.eval()
        self.decoder.eval()

        bsize = self.bsize
        N = len(data)

        adverse_attn = []
        adverse_output = []

        logodds = np.zeros((self.encoder.vocab_size, ))
        for k, v in logodds_map.items() :
            if v is not None :
                logodds[k] = abs(v)
            else :
                logodds[k] = float('-inf')
        logodds = torch.Tensor(logodds).to(device)

        for n in tqdm(range(0, N, bsize)) :
            torch.cuda.empty_cache()
            batch_doc = data[n:n+bsize]
            batch_data = BatchHolder(batch_doc)

            self.encoder(batch_data)
            self.decoder(batch_data)

            attn = batch_data.attn #(B, L)
            batch_data.attn_logodds = logodds[batch_data.seq]
            self.decoder.get_output_from_logodds(batch_data)

            attn_volatile = batch_data.attn_volatile.cpu().data.numpy() #(B, L)
            predict_volatile = torch.sigmoid(batch_data.predict_volatile).cpu().data.numpy() #(B, O)

            adverse_attn.append(attn_volatile)
            adverse_output.append(predict_volatile)

        adverse_output = [x for y in adverse_output for x in y]
        adverse_attn = [x for y in adverse_attn for x in y]

        return adverse_output, adverse_attn

    def logodds_substitution(self, data, top_logodds_words:Dict) :
        self.encoder.eval()
        self.decoder.eval()

        bsize = self.bsize
        N = len(data)

        adverse_X = []
        adverse_attn = []
        adverse_output = []

        words_neg = torch.Tensor(top_logodds_words[0][0]).long().cuda().unsqueeze(0)
        words_pos = torch.Tensor(top_logodds_words[0][1]).long().cuda().unsqueeze(0)

        words_to_select = torch.cat([words_neg, words_pos], dim=0) #(2, 5)

        for n in tqdm(range(0, N, bsize)) :
            torch.cuda.empty_cache()
            batch_doc = data[n:n+bsize]
            batch_data = BatchHolder(batch_doc)

            self.encoder(batch_data)
            self.decoder(batch_data)
            predict_class = (torch.sigmoid(batch_data.predict).squeeze(-1) > 0.5)*1 #(B,)

            attn = batch_data.attn #(B, L)
            top_val, top_idx = torch.topk(attn, 5, dim=-1)
            subs_words = words_to_select[1 - predict_class.long()] #(B, 5)

            batch_data.seq.scatter_(1, top_idx, subs_words)

            self.encoder(batch_data)
            self.decoder(batch_data)

            attn_volatile = batch_data.attn.cpu().data.numpy() #(B, L)
            predict_volatile = torch.sigmoid(batch_data.predict).cpu().data.numpy() #(B, O)
            X_volatile = batch_data.seq.cpu().data.numpy()

            adverse_X.append(X_volatile)
            adverse_attn.append(attn_volatile)
            adverse_output.append(predict_volatile)

        adverse_X = [x for y in adverse_X for x in y]
        adverse_output = [x for y in adverse_output for x in y]
        adverse_attn = [x for y in adverse_attn for x in y]

        return adverse_output, adverse_attn, adverse_X
