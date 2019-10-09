import json
import os
import time
import shutil
from copy import deepcopy
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from allennlp.common import Params
from sklearn.utils import shuffle
from tqdm import tqdm

from model.modules.Decoder import AttnDecoder
from model.modules.Encoder import Encoder

from .modelUtils import BatchHolder, get_sorting_index_with_noise_from_lengths
from .modelUtils import jsd as js_divergence

file_name = os.path.abspath(__file__)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.optim as optim

from typing import *
from pathlib import Path

from enum import IntEnum
class Dim(IntEnum):
	batch = 0
	seq = 1
	feature = 2


class OptimizedLSTM(nn.Module):
	def __init__(self, input_sz: int, hidden_sz: int):
		super().__init__()
		self.input_sz = input_sz
		self.hidden_size = hidden_sz
		self.weight_ih = Parameter(torch.Tensor(input_sz, hidden_sz * 4))
		self.weight_hh = Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
		self.bias = Parameter(torch.Tensor(hidden_sz * 4))
		self.init_weights()

	def init_weights(self):
		for p in self.parameters():
			if p.data.ndimension() >= 2:
				nn.init.xavier_uniform_(p.data)
			else:
				nn.init.zeros_(p.data)

	def forward(self, x: torch.Tensor,
				init_states: Optional[Tuple[torch.Tensor]] = None
				) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

		"""Assumes x is of shape (batch, sequence, feature)"""
		bs, seq_sz, _ = x.size()
		hidden_seq = []

		if init_states is None:
			h_t, c_t = (torch.zeros(self.hidden_size).to(x.device),
						torch.zeros(self.hidden_size).to(x.device))
		else:
			h_t, c_t = init_states

		HS = self.hidden_size
		for t in range(seq_sz):
			x_t = x[:, t, :]
			# batch the computations into a single matrix multiplication
			gates = x_t @ self.weight_ih + h_t @ self.weight_hh + self.bias
			i_t, f_t, g_t, o_t = (
				torch.sigmoid(gates[:, :HS]),  # input
				torch.sigmoid(gates[:, HS:HS * 2]),  # forget
				torch.tanh(gates[:, HS * 2:HS * 3]),
				torch.sigmoid(gates[:, HS * 3:]),  # output
			)
			c_t = f_t * c_t + i_t * g_t
			h_t = o_t * torch.tanh(c_t)
			hidden_seq.append(h_t.unsqueeze(Dim.batch))
		hidden_seq = torch.cat(hidden_seq, dim=Dim.batch)
		# reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
		hidden_seq = hidden_seq.transpose(Dim.batch, Dim.seq).contiguous()
		return hidden_seq, (h_t, c_t)