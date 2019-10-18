import json
import os
import shutil
from copy import deepcopy
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli
from allennlp.common import Params
from sklearn.utils import shuffle
from tqdm import tqdm
import sys
from helpers import *

from scipy.special import softmax
from scipy.special import expit as sigmoid
from tensorboardX import SummaryWriter

from Transparency.model.modules.Decoder import AttnDecoder
from Transparency.model.modules.Encoder import Encoder
from Transparency.model.modules.Attention import masked_softmax
from Transparency.model.modules.Rationale_Generator import RGenerator
from Transparency.model.modules.contextual_decomposition import CD
from Transparency.common_code.metrics import calc_metrics_classification, calc_metrics_multilabel
from Transparency.common_code.common import pload1
from Transparency.Trainers.PlottingBC import process_grads, process_int_grads

from .modelUtils import BatchHolder, get_sorting_index_with_noise_from_lengths
from .modelUtils import jsd as js_divergence
import pathlib
import nltk
from multiprocessing import Pool

file_name = os.path.abspath(__file__)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

metrics_type = {
    'Single_Label' : calc_metrics_classification,
    'Multi_Label' : calc_metrics_multilabel
}

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
            log_attn.masked_fill_(data.masks.unsqueeze(1).bool(), -float('inf'))
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
        log_attn.masked_fill_(data.masks.unsqueeze(1).bool(), -float('inf'))
        data.attn_volatile = nn.Softmax(dim=-1)(log_attn)
        self.decoder.get_output(data)
        data.predict_volatile = torch.sigmoid(data.predict_volatile)

class Model() :
    def __init__(self, configuration, pre_embed=None) :

        torch.manual_seed(0)

        configuration = deepcopy(configuration)
        self.configuration = deepcopy(configuration)

        configuration['model']['encoder']['pre_embed'] = pre_embed
        print ("encoder params",configuration['model']['encoder'])
        sys.stdout.flush()
        self.encoder = Encoder.from_params(Params(configuration['model']['encoder'])).to(device)

        configuration['model']['decoder']['hidden_size'] = self.encoder.output_size
        self.decoder = AttnDecoder.from_params(Params(configuration['model']['decoder'])).to(device)

        self.encoder_params = list(self.encoder.parameters())
        self.attn_params = list([v for k, v in self.decoder.named_parameters() if 'attention' in k])
        self.decoder_params = list([v for k, v in self.decoder.named_parameters() if 'attention' not in k])

        print ('configuration', configuration, self.configuration)

        self.generator = RGenerator(vocab_size=self.configuration['model']['encoder']['vocab_size'], embed_size=self.configuration['model']['encoder']['embed_size'],
                                                                 hidden_size=self.configuration['model']['generator']['hidden_size'], pre_embed=pre_embed).to(device)
        self.generator_params = list(self.generator.parameters())

        self.bsize = configuration['training']['bsize']

        print ('config ',configuration)

        self.diversity_weight = self.configuration['training'].get('diversity_weight',0)
        self.entropy_weight = self.configuration['training'].get('entropy_weight',0)

        weight_decay = configuration['training'].get('weight_decay', 1e-5)
        self.encoder_optim = torch.optim.Adam(self.encoder_params, lr=0.001, weight_decay=weight_decay, amsgrad=True)
        self.attn_optim = torch.optim.Adam(self.attn_params, lr=0.001, weight_decay=0, amsgrad=True)
        self.decoder_optim = torch.optim.Adam(self.decoder_params, lr=0.001, weight_decay=weight_decay, amsgrad=True)
        self.generator_optim = torch.optim.Adam(self.generator_params, lr=0.001, weight_decay=weight_decay, amsgrad=True)

        self.adversarymulti = AdversaryMulti(decoder=self.decoder)

        pos_weight = configuration['training'].get('pos_weight', [1.0]*self.decoder.output_size)
        self.pos_weight = torch.Tensor(pos_weight).to(device)
        self.criterion = nn.BCEWithLogitsLoss(reduction='none').to(device)

        import time
        dirname = configuration['training']['exp_dirname']
        basepath = configuration['training'].get('basepath', 'outputs')
        self.time_str = time.ctime().replace(' ', '_')
        self.dirname = os.path.join(basepath, dirname, self.time_str)
        self.tb_dir = os.path.join(self.dirname,"tensorboard")

        # pathlib.Path(self.tb_dir).mkdir(parents=True, exist_ok=True)
        # self.writer = SummaryWriter(self.tb_dir)

    @classmethod
    def init_from_config(cls, dirname, config_update=None, load_gen=False) :
        config = json.load(open(dirname + '/config.json', 'r'))
        print ('old config',config)
        if config_update is not None:
            config.update(config_update)
        print ('new config',config)
        obj = cls(config)
        obj.load_values(dirname)
        if load_gen:
            obj.load_values_generator(dirname)
        return obj

    def train(self, data_in, target_in, train=True,epoch=0) :
        sorting_idx = get_sorting_index_with_noise_from_lengths([len(x) for x in data_in], noise_frac=0.1)
        data = [data_in[i] for i in sorting_idx]
        target = [target_in[i] for i in sorting_idx]

        self.encoder.train()
        self.decoder.train()
        bsize = self.bsize
        N = len(data)
        loss_total = 0

        batches = list(range(0, N, bsize))
        total_iter = len(batches)
        batches = shuffle(batches)

        for idx,n in enumerate(batches):
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
            bce_loss = (bce_loss * weight).mean(1).mean()

            diverity_loss = self.conicity(batch_data).mean()
            entropy_loss = self.entropy_normalized(batch_data).mean()

            loss = bce_loss + self.diversity_weight*diverity_loss + self.entropy_weight*entropy_loss

            if hasattr(batch_data, 'reg_loss') :
                loss += batch_data.reg_loss

            if train:
                self.encoder_optim.zero_grad()
                self.decoder_optim.zero_grad()
                self.attn_optim.zero_grad()
                loss.backward()
                self.encoder_optim.step()
                self.decoder_optim.step()
                self.attn_optim.step()
                print ("Epoch: {} Step: {} Total Loss: {}, BCE loss: {}, Diversity Loss: {} (Diversity_weight = {}) Entropy Loss: {} (Entropy_weight = {})".format(epoch,idx,loss,bce_loss.cpu().data, diverity_loss, self.diversity_weight,entropy_loss,self.entropy_weight))
                n_iters = total_iter*epoch + idx
                # self.writer.add_scalar("loss",loss,n_iters)
                sys.stdout.flush()

            loss_total += float(loss.data.cpu().item())
        return loss_total*bsize/N

    def train_generator(self, data_in, target_in, train=True,epoch=0) :

        sorting_idx = get_sorting_index_with_noise_from_lengths([len(x) for x in data_in], noise_frac=0.1)
        data = [data_in[i] for i in sorting_idx]
        target = [target_in[i] for i in sorting_idx]

        self.encoder.train()
        self.decoder.train()
        self.generator.train()
        bsize = self.bsize
        N = len(data)
        loss_total = 0

        batches = list(range(0, N, bsize))
        total_iter = len(batches)
        batches = shuffle(batches)

        for idx,n in enumerate(batches):
            torch.cuda.empty_cache()
            batch_doc = data[n:n+bsize]
            batch_data = BatchHolder(batch_doc)

            probs = self.generator(batch_data)
            m = Bernoulli(probs=probs)
            rationale = m.sample().squeeze(-1)
            batch_data.seq = batch_data.seq * rationale.long()  #(B,L)
            masks = batch_data.masks.float()

            with torch.no_grad():
                self.encoder(batch_data)
                self.decoder(batch_data)

                batch_target = target[n:n+bsize]
                batch_target = torch.Tensor(batch_target).to(device)

                if len(batch_target.shape) == 1 : #(B, )
                    batch_target = batch_target.unsqueeze(-1) #(B, 1)

                bce_loss = self.criterion(batch_data.predict, batch_target)
                weight = batch_target * self.pos_weight + (1 - batch_target)
                bce_loss = (bce_loss * weight).mean(1)

            lengths = (batch_data.lengths-2)  #excl <s> and <eos>
            temp = (1-rationale)*(1-masks)
            sparsity_reward = temp.sum(1)/ (lengths.float())


            # sparsity_reward = torch.sum((1-rationale)*(1-masks),axis=1)/(lengths.float())

            total_reward = -1*bce_loss +  self.configuration['model']['generator']['sparsity_lambda']*sparsity_reward

            log_probs = m.log_prob(rationale.unsqueeze(-1)).squeeze(-1)
            # print ('log_probs, total_reward',log_probs.shape, total_reward.shape)
            loss = -log_probs * total_reward.unsqueeze(-1)
            loss = loss.sum(1).mean(0)


            if train:
                self.generator_optim.zero_grad()
                loss.backward()
                self.generator_optim.step()
                print ("Step: {} Loss {}, Total Reward: {}, BCE loss: {} Sparsity Reward: {} (sparsity_lambda = {})".format(idx, loss, total_reward.mean(),
                                                                                     bce_loss.mean(), sparsity_reward.mean(), self.configuration['model']['generator']['sparsity_lambda']))
                n_iters = total_iter*epoch + idx
                # self.writer.add_scalar("loss",loss,n_iters)
                sys.stdout.flush()
            loss_total += float(loss.data.cpu().item())
        return loss_total*bsize/N

    def eval_generator(self, dataset, data, target, epoch, name="") :

        self.encoder.train()
        self.decoder.train()
        self.generator.eval()
        bsize = self.bsize
        N = len(data)
        loss_total = 0

        batches = list(range(0, N, bsize))
        total_iter = len(batches)
        batches = shuffle(batches)
        overall_reward = 0
        # f = open(self.dirname + '/rationale_' + name + "_" + str(epoch) + '.txt', 'w')

        for idx,n in enumerate(batches):
            torch.cuda.empty_cache()
            batch_doc = data[n:n+bsize]
            batch_data = BatchHolder(batch_doc)

            with torch.no_grad():

                probs = self.generator(batch_data)
                m = Bernoulli(probs=probs)
                rationale = m.sample().squeeze(-1)

                input_seq = batch_data.seq.cpu().data.numpy()
                output_seq = input_seq* rationale.long().cpu().data.numpy() #(B,L)
                masks = batch_data.masks.float()

                batch_data.seq = batch_data.seq*rationale.long()
                self.encoder(batch_data)
                self.decoder(batch_data)

                batch_target = target[n:n+bsize]
                batch_target = torch.Tensor(batch_target).to(device)

                predict = torch.sigmoid(batch_data.predict).cpu().data.numpy()

                if len(batch_target.shape) == 1 : #(B, )
                    batch_target = batch_target.unsqueeze(-1) #(B, 1)

                bce_loss = self.criterion(batch_data.predict, batch_target)
                weight = batch_target * self.pos_weight + (1 - batch_target)
                bce_loss = (bce_loss * weight).mean(1).cpu().data.numpy()

                lengths = (batch_data.lengths-2)  #excl <s> and <eos>

                sum = ((1 - rationale) * (1 - masks)).sum(1)

                sparsity_reward = (sum.float()/(lengths.float())).cpu().data.numpy() #TODO check again

                total_reward = -1*bce_loss +  self.configuration['model']['generator']['sparsity_lambda']*sparsity_reward

                label = batch_target.cpu().data.numpy()

                for i in range((batch_data.seq.shape[0])):
                    output_dict = {}
                    output_dict['input_sentence'] = dataset.vec.map2words(input_seq[i] )
                    output_dict['generated_rationale'] = dataset.vec.map2words(output_seq[i])
                    output_dict['sparsity_reward'] = sparsity_reward[i]
                    output_dict['bce_loss'] = bce_loss[i]
                    output_dict['total_reward'] = total_reward[i]
                    output_dict['predict'] = predict[i][0]
                    output_dict['label'] = label[i][0]
                    # f.write(str(output_dict) + '\n')

                    overall_reward +=  total_reward[i]

        overall_reward = overall_reward/N
        # f.close()
        return overall_reward

    def rationale_attn(self, dataset, data, target, name="") :
                
        self.encoder.train()
        self.decoder.train()
        self.generator.eval()
        bsize = self.bsize
        N = len(data)
        loss_total = 0

        batches = list(range(0, N, bsize))
        total_iter = len(batches)
        batches = shuffle(batches)
        overall_reward = 0
        f = open(self.dirname + '/rationale_' + name + '.txt', 'w')

        fracs = []
        sum_attns = []
        losses = []
        rationales = []
        predictions = []

        for idx,n in enumerate(batches):
            torch.cuda.empty_cache()
            batch_doc = data[n:n+bsize]
            batch_data = BatchHolder(batch_doc)

            with torch.no_grad():

                masks = batch_data.masks.float()

                probs = self.generator(batch_data)
                m = Bernoulli(probs=probs)
                rationale = m.sample().squeeze(-1)

                input_seq = batch_data.seq
                output_seq = input_seq* rationale.long() #(B,L)
                batch_data.seq = output_seq

                self.encoder(batch_data)
                self.decoder(batch_data)

                ######### Attn Comparision #############
                attn = batch_data.attn.cpu().data.numpy()  #(B,L)
                sum_attn = np.sum(attn*rationale.cpu().data.numpy(),axis=1).tolist()
                sum_attns.extend(sum_attn)

                ########  BCE Loss  ##################
                batch_target = target[n:n+bsize]
                batch_target = torch.Tensor(batch_target).to(device)
                predict = torch.sigmoid(batch_data.predict).cpu().data.numpy()

                predictions.append(predict)

                if len(batch_target.shape) == 1 : #(B, )
                    batch_target = batch_target.unsqueeze(-1) #(B, 1)

                bce_loss = self.criterion(batch_data.predict, batch_target)
                weight = batch_target * self.pos_weight + (1 - batch_target)
                bce_loss = (bce_loss * weight).mean(1).cpu().data.numpy()

                ######### Sparsity  #####################
                lengths = (batch_data.lengths-2)  #excl <s> and <eos>
                sparsity_reward = (torch.sum((1-rationale)*(1-masks),axis=1).float()/(lengths.float())).cpu().data.numpy()

                rationale_frac = torch.sum(rationale,axis=1).float()/lengths.float()
                rationale_frac = rationale_frac.cpu().data.numpy().tolist()
                loss = bce_loss.tolist()

                fracs.extend(rationale_frac)
                losses.extend(loss)

                total_reward = -1*bce_loss +  self.configuration['model']['generator']['sparsity_lambda']*sparsity_reward

                label = batch_target.cpu().data.numpy()

                for i in range((batch_data.seq.shape[0])):
                    output_dict = {}
                    output_dict['input_sentence'] = dataset.vec.map2words(input_seq[i].cpu().data.numpy() )
                    output_dict['generated_rationale'] = dataset.vec.map2words(output_seq[i].cpu().data.numpy())
                    output_dict['sparsity_reward'] = sparsity_reward[i]
                    output_dict['bce_loss'] = bce_loss[i]
                    output_dict['total_reward'] = total_reward[i]
                    output_dict['predict'] = predict[i][0]
                    output_dict['label'] = label[i][0]
                    f.write(str(output_dict) + '\n')

                    overall_reward +=  total_reward[i]

        overall_reward = overall_reward/N
        f.close()

        fracs = np.array(fracs)
        sum_attns = np.array(sum_attns)
        losses = np.array(losses)

        predictions = [x for y in predictions for x in y]

        result_summary = {'Fraction Length Average':np.mean(fracs),'Fraction Length STD':np.std(fracs),'Attn Sum Average':np.mean(sum_attns),'Attn Sum STD':np.std(sum_attns),'loss':np.mean(losses)}
        print ("Summary on Test Dataset",result_summary)
        results = {'fraction_lengths':fracs,'Sum_Attentions':sum_attns, 'losses':losses}
        return results, predictions

    def cd_batch(self, data):

        attention_weights = self.decoder.attention.state_dict()
        lstm_weights = self.encoder.rnn.state_dict()

        cd = CD()
        cd.W_ii, cd.W_if, cd.W_ig, cd.W_io = np.split(lstm_weights['cell_0.weight_ih'].cpu().data.numpy(), 4, 0)
        cd.W_hi, cd.W_hf, cd.W_hg, cd.W_ho = np.split(lstm_weights['cell_0.weight_hh'].cpu().data.numpy(), 4, 0)
        cd.b_i, cd.b_f, cd.b_g, cd.b_o = np.split(lstm_weights['cell_0.bias_ih'].cpu().data.numpy() + lstm_weights['cell_0.bias_hh'].cpu().data.numpy() , 4)

        # for tanh attention weights
        cd.W_attn1 = attention_weights['attn1.weight'].squeeze(0).cpu().data.numpy()
        # cd.b_attn1 = attention_weights['attn1.bias'].squeeze(0).cpu().data.numpy()
        # cd.W_attn2 = attention_weights['attn2.weight'].squeeze(0).cpu().data.numpy()
        #cd.W_attn = attention_weights['attn1.weight'].squeeze(0).cpu().data.numpy()

        cd.hidden_dim = 2*self.encoder.hidden_size
        batch_size = data.embedding.size(0)

        cd.embeddings = data.embedding.cpu().data.numpy()
        cd.attn_logits = data.attn_logit.cpu().data.numpy()
        cd.attns = data.attn.cpu().data.numpy()

        cd.lengths = data.lengths.cpu().data.numpy()
        cd.masks = data.masks.cpu().data.numpy()

        indices = range(batch_size)
        p = Pool(4)
        outputs = (p.map(cd.cd_single_new, indices))
        return zip(*outputs)

    def cd(self,data):

        self.encoder.train()
        self.decoder.train()
        bsize = self.bsize
        data = data[:100]
        N = len(data)

        outputs = []
        attns = []
        new_attns = []
        cd_matrices = []

        for idx,n in enumerate(tqdm(range(0, N, bsize))) :

            with torch.no_grad():

                torch.cuda.empty_cache()
                batch_doc = data[n:n+bsize]
                batch_data = BatchHolder(batch_doc)

                self.encoder(batch_data)
                self.decoder(batch_data)

                new_attn_score, attn_check ,_, cd_matrix = self.cd_batch(batch_data)
                batch_data.predict = torch.sigmoid(batch_data.predict)
                attn = batch_data.attn.cpu().data.numpy()

                attns.append(attn)
                new_attns.append(new_attn_score)
                cd_matrices.append(cd_matrix)

                predict = batch_data.predict.cpu().data.numpy()
                outputs.append(predict)
                

        outputs = [x for y in outputs for x in y]
        attns = [x for y in attns for x in y]
        new_attns = [x for y in new_attns for x in y]
        cd_matrices = [x for y in cd_matrices for x in y]
        return new_attns,cd_matrices

    def evaluate(self, data) :

        # is_embed check fails when B is very large, embed inputs is not (B,L,E) ndarray
        if (len(np.array(data).shape) == 3):
            is_embed = True
        else:
            is_embed = False


        self.encoder.train()
        self.decoder.train()
        bsize = self.bsize
        N = len(data)

        outputs = []
        attns = []
        conicity_values = []

        for n in (range(0, N, bsize)) :
            torch.cuda.empty_cache()
            batch_doc = data[n:n+bsize]
            batch_data = BatchHolder(batch_doc, is_embed=is_embed)

            self.encoder(batch_data)
            self.decoder(batch_data)

            batch_data.predict = torch.sigmoid(batch_data.predict)
            if self.decoder.use_attention :
                attn = batch_data.attn.cpu().data.numpy()
                attns.append(attn)

            conicity_values.append(self.conicity(batch_data).cpu().data.numpy())

            predict = batch_data.predict.cpu().data.numpy()
            outputs.append(predict)

        outputs = [x for y in outputs for x in y]
        if self.decoder.use_attention :
            attns = [x for y in attns for x in y]

        conicity_values = np.concatenate(conicity_values,axis=0)

        # third variable is a placeholder for new_attns values
        return outputs, attns, conicity_values

    def embeddings_analysis(self,data):

        self.encoder.train()
        self.decoder.train()
        bsize = self.bsize
        N = len(data)

        outputs = []
        attns = []
        cosine_sim_values = []

        for n in (range(0, N, bsize)) :
            torch.cuda.empty_cache()
            batch_doc = data[n:n+bsize]
            batch_data = BatchHolder(batch_doc)

            self.encoder(batch_data)
            self.decoder(batch_data)
                
            hidden_states = batch_data.hidden    # (B,L,H)
            cosine_sim = torch.abs(torch.nn.functional.cosine_similarity(hidden_states.unsqueeze(2), hidden_states.unsqueeze(1), dim=3, eps=1e-6))  #cosine sim between (B,L,1,H) and (B,1,L,H) --> (B,L,L)
            cosine_sim_values.append(cosine_sim.cpu().data.numpy())
        
        cosine_sim_values = [x for y in cosine_sim_values for x in y]

        return cosine_sim_values
    
    def quantitative_analysis(self, data, target, dataset) :

        self.encoder.train()
        self.decoder.train()
        bsize = self.bsize
        N = len(data)

        outputs = []
        attns = []
        pos_tag_dict = {}
        
        word_attn_positive = {}
        word_attn_negative = {}

        for key in dataset.vec.word2idx.keys():
            word_attn_positive[key] = [0,0]
            word_attn_negative[key] = [0,0]

        for n in (range(0, N, bsize)) :
            torch.cuda.empty_cache()
            batch_doc = data[n:n+bsize]
            batch_data = BatchHolder(batch_doc)
            batch_target = target[n:n+bsize]

            self.encoder(batch_data)
            self.decoder(batch_data)

            batch_data.predict = torch.sigmoid(batch_data.predict)
            attn = batch_data.attn.cpu().data.numpy()
            attns.append(attn)
            predict = batch_data.predict.cpu().data.numpy()

            for idx in range(len(batch_doc)):

                L = batch_data.lengths[idx].cpu().data
                label = batch_target[idx]
                seq = batch_data.seq[idx][1:L-1].cpu().data.numpy()
                attention = attn[idx][1:L-1]

                words = dataset.vec.map2words(seq)
                words = [word if word != "" else "'<UNK>'" for word in words if word]
                tags = nltk.tag.pos_tag(words)
                tags = [(word, nltk.tag.map_tag('en-ptb', 'universal', tag)) for word, tag in tags]

                for i, (word,tag) in enumerate(tags):

                    if tag not in pos_tag_dict.keys():
                        pos_tag_dict[tag] = []
                        pos_tag_dict[tag].extend([1, attention[i]])

                    else:
                        pos_tag_dict[tag][0] += 1
                        pos_tag_dict[tag][1] += attention[i] 
                    
                    if label == 0:
                        word_attn_negative[word][0] += 1
                        word_attn_negative[word][1] += attention[i]
                        # print ("word, word_attn_negative",word, word_attn_negative[word])

                    else:
                        word_attn_positive[word][0] += 1
                        word_attn_positive[word][1] += attention[i]
                        # print ("word, word_attn_positive",word, word_attn_positive[word])
        
        for keys,values in pos_tag_dict.items():
            
            if values[0] == 0:
                pos_tag_dict[keys].append(0)
            else:    
                pos_tag_dict[keys].append(values[1]/values[0])
        
        for keys,values in word_attn_positive.items():

            if values[0] == 0:
                word_attn_positive[keys].append(0)
            else:    
                word_attn_positive[keys].append(values[1]/values[0])
        
        for keys,values in word_attn_negative.items():
            
            if values[0] == 0:
                word_attn_negative[keys].append(0)
            else:
                word_attn_negative[keys].append(values[1]/values[0])
                    

        pos_tag_sorted = sorted(pos_tag_dict.items(), key=lambda kv: kv[1][1],reverse=True)
        word_attn_positive = sorted(word_attn_positive.items(), key=lambda kv: kv[1][1],reverse=True)
        word_attn_negative = sorted(word_attn_negative.items(), key=lambda kv: kv[1][1],reverse=True)

        outputs = {'pos_tags':pos_tag_sorted,'word_attn_positive':word_attn_positive,'word_attn_negative':word_attn_negative}
        print ("Pos_attn")
        print(pos_tag_sorted)
        print ("word_attn_positive")
        print(word_attn_positive[:20])
        print("word_attn_negative")
        print(word_attn_negative[:20])
        return outputs

    def conicity(self,data):

        hidden_states = data.hidden    # (B,L,H)
        b,l,h = hidden_states.size()
        masks = data.masks.float() #(B,L)
        lengths = (data.lengths.float() - 2) ## (B)

        hidden_states = hidden_states* (1-masks.unsqueeze(2))
        # hidden_states.masked_fill_(masks.unsqueeze(2).bool(), 0.0)

        mean_state = hidden_states.sum(1) / lengths.unsqueeze(1)

        mean_state = mean_state.unsqueeze(1) #.repeat(1,l,1) #(B,L,H)
        #print (mean_state.size(), hidden_states.size())
        sys.stdout.flush()
        cosine_sim = torch.abs(torch.nn.functional.cosine_similarity(hidden_states, mean_state, dim=2, eps=1e-6))  #(B,L)
        # cosine_sim.masked_fill_(masks.bool(), 0.0)
        cosine_sim = cosine_sim*(1-masks)

        conicity = cosine_sim.sum(1) / lengths  # (B)
        return conicity

    
    def entropy_normalized(self,data):
        
        attention = data.attn  #(B,L)
        lengths = (data.lengths.float() - 2) ## (B)
        masks = data.masks.float() #(B,L)

        entropy = -attention*torch.log(attention + 1e-6) *(1-masks)
        entropy = torch.sum(entropy,dim=1)/lengths
        # max_entropy = torch.log(lengths)
        # normalized_entropy = entropy/max_entropy
        return entropy

    def gradient_mem(self, data) :

        if (len(np.array(data).shape) == 3):
            is_embed = True
        else:
            is_embed = False

        self.encoder.train()
        self.decoder.train()
        bsize = self.bsize
        N = len(data)


        grads = {'XxE' : [], 'XxE[X]' : [], 'H' : [], 'X':[]}
        output_arr = []
        conicity_list = []

        for n in (range(0, N, bsize)):
            torch.cuda.empty_cache()
            batch_doc = data[n:n+bsize]

            grads_xxe = []
            grads_xxex = []
            grads_H = []
            grads_x = []
            outputs = []


            for i in range(self.decoder.output_size) :

                batch_data = BatchHolder(batch_doc, is_embed=is_embed)
                batch_data.keep_grads = True
                batch_data.detach = True

                self.encoder(batch_data)
                self.decoder(batch_data)

                if i==0:
                    conicity = self.conicity(batch_data).cpu().data.numpy()

                torch.sigmoid(batch_data.predict[:, i]).sum().backward()
                g = batch_data.embedding.grad
                em = batch_data.embedding
                g1 = (g * em).sum(-1)

                grads_x.append(g.cpu().data.numpy()) 

                grads_xxex.append(g1.cpu().data.numpy())

                g1 = (g * self.encoder.embedding.weight.sum(0)).sum(-1)
                grads_xxe.append(g1.cpu().data.numpy())

                outputs.append(torch.sigmoid(batch_data.predict[:, i]).cpu().data.numpy())

                g1 = (batch_data.hidden.grad * batch_data.hidden).sum(-1)
                grads_H.append(g1.cpu().data.numpy())

            grads_xxe = np.array(grads_xxe).swapaxes(0, 1)
        

            conicity_list.append(conicity)

            grads_xxex = np.array(grads_xxex).swapaxes(0, 1)  #(batch_size, 1 , L)
            grads_H = np.array(grads_H).swapaxes(0, 1)
            grads_x = np.array(grads_x).swapaxes(0,1).squeeze(1)  #(batch_size, L, hidden_size)
            
            outputs = np.array(outputs).swapaxes(0,1).squeeze(1)  #(batch_size, 1)

            grads['XxE'].append(grads_xxe)
            grads['XxE[X]'].append(grads_xxex)
            grads['H'].append(grads_H)
            grads['X'].append(grads_x)
            output_arr.append(outputs)

        for k in grads :
            grads[k] = [x for y in grads[k] for x in y] #(N * 1 * L)

        outputs = [x for y in output_arr for x in y]

        conicity_array = np.concatenate(conicity_list,axis=0)
        grads['conicity'] = conicity_array
        # print (grads['conicity'])
        return grads, outputs


    def integrated_gradient_mem(self, data, grads_wrt='X', no_of_instances=100, steps=50):

        #NOTE: Integrated gradients by default will only calculate IG for 100 instances and wrt grads['XxE[X]'] to reduce computation time
        #Change 'grads_wrt' and 'no_of_instances' accordingly to match correlation plot of normal gradients
        #Unlike gradients_mem, IG should be invoked after model has been trained, so that self.encoder.embeddings.weight.data


        no_of_instances = len(data.test_data.X)

        embd_dict = np.array(self.encoder.embedding.weight.cpu().data)
        test_data_embd_col = get_complete_testdata_embed_col(data, embd_dict, testdata_count=no_of_instances, steps=steps)
        int_grads = []

        print("calculating IG")
        for i in tqdm(range(len(test_data_embd_col))):

            # try:

                sample = i
                one_sample = test_data_embd_col[sample]
                grads,outputs = self.get_grads_from_custom_td(one_sample)
                int_grads.append(integrated_gradients(grads, outputs, one_sample, grads_wrt='X'))

            # except:
                # print("exception in integrated_gradein")


        print("int_grads are {}".format(int_grads))

        return int_grads

    def lime_attribution_mem(self, dataset, no_of_instances=10):
        # NOTE: Lime attributions by default calculate only 10 instances since to reduce computation time

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
        grads,outputs = self.gradient_mem(test_data)
        return grads,outputs

    def evaluate_outputs_from_embeds(self, embds):
        predictions, attentions = self.evaluate(embds)
        return predictions, attentions

    def evaluate_outputs_from_custom_td(self, testdata):
        predictions, _, _ = self.evaluate(testdata)
        return predictions
    def remove_and_run(self, data) :
        self.encoder.train()
        self.decoder.train()
        bsize = self.bsize
        N = len(data)

        outputs = []

        for n in (range(0, N, bsize)) :
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

        for n in (range(0, N, bsize)) :
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

    def minimum_length(self,batch_data, ranking):

        length = batch_data.lengths[0]
        logit = batch_data.logit[0]

        flip = 0
        for i in range(1,(length-2)+1):  #excluding <s> and <eos>

            batch_data.erase_attn = ranking[:,:i]
            
            # print ('length here, ranking',length,ranking[:,:i])

            self.decoder(batch_data)
            new_logit = batch_data.predict[0]
            # print ('new_attn',batch_data.attn)
            new_attn = batch_data.attn.cpu().data.numpy()

            # print ('i new_logit new_attn',i, new_logit, new_attn)

            if ((new_logit*logit) < 0):
                flip = 1
                fraction_length = (float(i)/float(length))
                return fraction_length

        return 1.0    

    def importance_ranking(self,data):

        self.encoder.train()
        self.decoder.train()
        bsize = 1
        N = len(data)

        erase_max = []
        erase_random = []

        attention_lengths = []
        random_lengths = []
        grads_x_lengths = []
        grads_h_lengths = []
        int_grads_lengths = []
        lengths = {}

        grads = pload1(self.dirname, 'gradients')
        process_grads(grads,data)
        grads_x = grads['XxE[X]']
        grads_h = grads['H']

        # int_grads = pload1(self.dirname, 'integrated_gradients')
        # int_grads = process_int_grads(int_grads)

        for n in (range(0, N, bsize)):

            torch.cuda.empty_cache()
            batch_doc = data[n:n+bsize]
            batch_data = BatchHolder(batch_doc)

            self.encoder(batch_data)
            self.decoder(batch_data)

            batch_data.logit = batch_data.predict[0]
            attn = batch_data.attn
            mask = batch_data.masks.float()
            batch_data.erase_given = True

            # print ('logit attn', batch_data.logit, attn)

            attention_ranking = attn.sort(dim=1,descending=True)[1]
            length = self.minimum_length(batch_data, attention_ranking)
            attention_lengths.append(length)

            random_ranking = (1+torch.randperm(batch_data.lengths[0]-2)).view(1,-1)  #excluding <start> and <eos>
            length = self.minimum_length(batch_data, random_ranking)
            random_lengths.append(length)

            l = batch_data.lengths[0]
            
            grad_x = np.array(grads_x[n:n+bsize])[:,:l]
            grads_x_batch = torch.FloatTensor(grad_x).to(device)  
            grads_x_ranking = grads_x_batch.sort(dim=1,descending=True)[1]
            length = self.minimum_length(batch_data, grads_x_ranking)
            grads_x_lengths.append(length)

            grad_h = np.array(grads_h[n:n+bsize])[:,:l]
            grads_h_batch = torch.FloatTensor(grad_h).to(device)
            grads_h_ranking = grads_h_batch.sort(dim=1,descending=True)[1]
            length = self.minimum_length(batch_data, grads_h_ranking)
            grads_h_lengths.append(length)

            """
            int_grad = np.array(int_grads[n:n+bsize])[:,:l]
            int_grads_batch = torch.FloatTensor(int_grad).to(device)
            int_grads_ranking = int_grads_batch.sort(dim=1,descending=True)[1]
            length = self.minimum_length(batch_data, int_grads_ranking)
            int_grads_lengths.append(length)
            """
        lengths = {'attention':attention_lengths,'random':random_lengths,'grad_x':grads_x_lengths,'grad_h':grads_h_lengths}# , 'int_grad':int_grads_lengths}
        return lengths

    def ersae_attn(self, data, num_perm=100):
        self.encoder.train()
        self.decoder.train()
        bsize = self.bsize
        N = len(data)

        erase_max = []
        erase_random = []

        for n in (range(0, N, bsize)) :
            torch.cuda.empty_cache()
            batch_doc = data[n:n+bsize]
            batch_data = BatchHolder(batch_doc)

            batch_perms = np.zeros((batch_data.B, num_perm, self.decoder.output_size))

            ######
            self.encoder(batch_data)
            # self.decoder(batch_data)

            batch_data.erase_max = True
            self.decoder(batch_data)
            output_max = torch.sigmoid(batch_data.predict).cpu().data.numpy()
            erase_max.append(output_max)

            batch_data.erase_max = False
            batch_data.erase_random = True

            self.decoder(batch_data)
            output_random = torch.sigmoid(batch_data.predict).cpu().data.numpy()
            erase_random.append(output_random)

        erase_random = [x for y in erase_random for x in y]
        erase_max = [x for y in erase_max for x in y]

        return (erase_max, erase_random)

    def save_values(self, use_dirname=None, save_model=True) :

        print ('saved config ',self.configuration)

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


    def save_values_generator(self, use_dirname=None, save_model=True) :

        if use_dirname is not None :
            dirname = use_dirname
        else :
            dirname = self.dirname
        os.makedirs(dirname, exist_ok=True)
        shutil.copy2(file_name, dirname + '/')
        json.dump(self.configuration, open(dirname + '/config.json', 'w'))

        if save_model :
            torch.save(self.generator.state_dict(), dirname + '/gen.th')
        return dirname

    def load_values_generator(self, dirname) :
        self.generator.load_state_dict(torch.load(dirname + '/gen.th', map_location={'cuda:1': 'cuda:0'}))

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

        for n in (range(0, N, bsize)) :
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

        for n in (range(0, N, bsize)) :
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

        for n in (range(0, N, bsize)) :
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
