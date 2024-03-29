from allennlp.common.from_params import FromParams
import torch
import torch.nn as nn
from typing import Dict
from allennlp.common import Params
import numpy as np

from model.modules.Attention import Attention, masked_softmax
from model.modelUtils import isTrue, BatchHolder, BatchMultiHolder

from common_code.common import pload, pdump

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class AttnDecoder(nn.Module, FromParams) :
    def __init__(self, hidden_size:int,
                       attention:Dict,
                       output_size:int = 1,
                       use_attention:bool = True,
                       regularizer_attention:Dict = None) :
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.linear_1 = nn.Linear(hidden_size, output_size)

        attention['hidden_size'] = self.hidden_size
        self.attention = Attention.from_params(Params(attention))


        self.use_regulariser_attention = False
        if regularizer_attention is not None :
            regularizer_attention['hidden_size'] = self.hidden_size
            self.regularizer_attention = Attention.from_params(Params(regularizer_attention))
            self.use_regulariser_attention = True

        self.use_attention = use_attention

    def decode(self, predict) :
        predict = self.linear_1(predict)
        return predict

    def lrp(self, data:BatchHolder):
        # sanity check: rel_<input>[0].sum(0) = 1

        """ Output to Context vEctor relevance propagation """
        weights = np.array(self.linear_1.weight.data) #(1, H)
        bias = np.array(self.linear_1.bias.data) #not used for naive lrp
        X = self.get_context(data).data #(B, H)
        wX = np.abs(np.multiply(weights, X))
        sum = np.abs(wX.sum(-1).unsqueeze(-1))
        rel_context = torch.div(wX, sum) #normalizing


        """ Context vector to Attention weights relevance propagation """
        attn = np.array(self.get_attention(data).data.unsqueeze(-1)).transpose(2,0,1) # (L, 1)
        hidden = np.array(data.hidden.data) # (L, H)
        rel = rel_context.transpose(1, 0)  # (H, 1) => output

        accu = np.zeros_like(attn)

        for i in range(self.hidden_size):
            aH = np.multiply(attn, hidden[:, :, i]) #(L, 1)

            den = aH.sum(-1)
            norm = np.divide(aH, np.expand_dims(den, axis=-1))
            rel_here = np.multiply(norm, rel[i].unsqueeze(0).unsqueeze(-1)) #(L, 1)
            accu = np.add(accu, rel_here)

        rel_attn = np.squeeze(accu.data)

        # print(rel_attn[0].sum(0))
        # exit('hello')
        rel_attn= np.multiply(rel_attn, 100)

        return np.array(rel_attn.data), np.array(rel_context.data)



    def deeplift(self, delta_x: dict, data = None ):
        #delta_x is a dict of ndarrays with shape (B,L,E)
        #returns relevances scores for attention weights and context batch wise

        d_X = delta_x['d_ctx']
        d_Y = delta_x['d_uo']
        d_attn = delta_x['d_attn']
        d_Yhat = delta_x['d_o']

        weights = np.array(self.linear_1.weight.data)

        # Rescale Rule for sigmoid
        sigm_mul = d_Yhat/d_Y

        # Unactivated output -> context propagation
        wX = np.multiply(weights, d_X)
        rel_context = wX

        """SANITY CHECK: rel_context[0].sum(-1) == d_Y[0]"""

        # Context -> Attention weights propagation
        attn = np.expand_dims(d_attn, axis=-1).transpose(2, 0, 1)
        hidden = delta_x['d_hs']
        accu = np.zeros_like(attn)

        for i in range(self.hidden_size):
            aH = np.multiply(attn, hidden[:, :, i])
            accu = np.add(accu, aH)

        rel_attn = np.squeeze(accu.data)

        """SANITY CHECK: Del_context = del_(A*H)"""

        print(rel_attn[0].sum(-1))
        print(d_X[0].sum())

        return rel_attn, rel_context


    def get_context(self, data: BatchHolder):
        output = data.hidden
        mask = data.masks
        attn = self.attention(data.seq, output, mask)
        if self.use_regulariser_attention:
            data.reg_loss = 5 * self.regularizer_attention.regularise(data.seq, output, mask, attn)

        if isTrue(data, 'detach'):
            attn = attn.detach()

        if isTrue(data, 'permute'):
            permutation = data.generate_permutation()
            attn = torch.gather(attn, -1, torch.LongTensor(permutation).to(device))
        return(attn.unsqueeze(-1) * output).sum(1)

    def get_attention(self, data):
        output = data.hidden
        mask = data.masks
        attn = self.attention(data.seq, output, mask)

        return attn


    def forward(self, data:BatchHolder) :
        if self.use_attention :
            output = data.hidden
            mask = data.masks
            attn = self.attention(data.seq, output, mask)

            if self.use_regulariser_attention :
                data.reg_loss = 5 * self.regularizer_attention.regularise(data.seq, output, mask, attn)

            if isTrue(data, 'detach') :
                attn = attn.detach()

            if isTrue(data, 'permute') :
                permutation = data.generate_permutation()
                attn = torch.gather(attn, -1, torch.LongTensor(permutation).to(device))

            context = (attn.unsqueeze(-1) * output).sum(1) #B, H

            data.attn = attn

        else :
            context = data.last_hidden

        predict = self.decode(context)
        data.predict = predict


    def get_output(self, data:BatchHolder) :
        output = data.hidden_volatile #(B, L, H)
        attn = data.attn_volatile #(B, *, L)

        if len(attn.shape) == 3 :
            context = (attn.unsqueeze(-1) * output.unsqueeze(1)).sum(2) #(B, *, H)
            predict = self.decode(context)
        else :
            context = (attn.unsqueeze(-1) * output).sum(1)
            predict = self.decode(context)

        data.predict_volatile = predict

    def get_output_from_logodds(self, data:BatchHolder) :
        attn_logodds = data.attn_logodds #(B, L)
        attn = masked_softmax(attn_logodds, data.masks)

        data.attn_volatile = attn
        data.hidden_volatile = data.hidden

        self.get_output(data)

class AttnDecoderQA(nn.Module, FromParams) :
    def __init__(self, hidden_size:int,
                       attention:Dict,
                       output_size:int = 1,
                       use_attention:bool = True,
                       regularizer_attention:Dict = None) :
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.linear_1q = nn.Linear(hidden_size, hidden_size // 2)
        self.linear_1p = nn.Linear(hidden_size, hidden_size // 2)

        self.linear_2 = nn.Linear(hidden_size // 2, output_size)

        attention['hidden_size'] = self.hidden_size
        self.attention = Attention.from_params(Params(attention))

        self.use_regulariser_attention = False
        if regularizer_attention is not None :
            regularizer_attention['hidden_size'] = self.hidden_size
            self.regularizer_attention = Attention.from_params(Params(regularizer_attention))
            self.use_regulariser_attention = True

        self.use_attention = use_attention

    def decode(self, Poutput, Qoutput, entity_mask) :
        predict = self.linear_2(nn.Tanh()(self.linear_1p(Poutput) + self.linear_1q(Qoutput))) #(B, O)
        predict.masked_fill_(1 - entity_mask, -float('inf'))

        return predict

    def forward(self, data: BatchMultiHolder) :
        if self.use_attention :
            Poutput = data.P.hidden #(B, H, L)
            Qoutput = data.Q.last_hidden #(B, H)
            mask = data.P.masks

            attn = self.attention(data.P.seq, Poutput, Qoutput, mask) #(B, L)

            if isTrue(data, 'detach') :
                attn = attn.detach()

            if isTrue(data, 'permute') :
                permutation = data.P.generate_permutation()
                attn = torch.gather(attn, -1, torch.LongTensor(permutation).to(device))

            context = (attn.unsqueeze(-1) * Poutput).sum(1) #(B, H)
            data.attn = attn


        else :
            context = data.P.last_hidden

        predict = self.decode(context, Qoutput, data.entity_mask)
        data.predict = predict

    def get_attention(self, data:BatchMultiHolder) :
        output = data.P.hidden_volatile
        mask = data.P.masks
        attn = self.attention(data.P.seq, output, mask)
        data.attn_volatile = attn

    def get_output(self, data: BatchMultiHolder) :
        Poutput = data.P.hidden_volatile #(B, L, H)
        Qoutput = data.Q.last_hidden_volatile #(B, H)
        attn = data.attn_volatile #(B, K, L)

        if len(attn.shape) == 3 :
            predict = (attn.unsqueeze(-1) * Poutput.unsqueeze(1)).sum(2) #(B, K, H)
            predict = self.decode(predict, Qoutput.unsqueeze(1), data.entity_mask.unsqueeze(1))
        else :
            predict = (attn.unsqueeze(-1) * Poutput).sum(1) #(B, H)
            predict = self.decode(predict, Qoutput, data.entity_mask)

        data.predict_volatile = predict
