import os
import torch

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


class NaiveLSTM(nn.Module):
	def __init__(self, input_sz: int, hidden_sz: int):
		super().__init__()
		self.input_size = input_sz
		self.hidden_size = hidden_sz
		# input gate
		self.W_ii = Parameter(torch.Tensor(input_sz, hidden_sz))
		self.W_hi = Parameter(torch.Tensor(hidden_sz, hidden_sz))
		self.b_i = Parameter(torch.Tensor(hidden_sz))
		# forget gate
		self.W_if = Parameter(torch.Tensor(input_sz, hidden_sz))
		self.W_hf = Parameter(torch.Tensor(hidden_sz, hidden_sz))
		self.b_f = Parameter(torch.Tensor(hidden_sz))
		# ???
		self.W_ig = Parameter(torch.Tensor(input_sz, hidden_sz))
		self.W_hg = Parameter(torch.Tensor(hidden_sz, hidden_sz))
		self.b_g = Parameter(torch.Tensor(hidden_sz))
		# output gate
		self.W_io = Parameter(torch.Tensor(input_sz, hidden_sz))
		self.W_ho = Parameter(torch.Tensor(hidden_sz, hidden_sz))
		self.b_o = Parameter(torch.Tensor(hidden_sz))

		self.init_weights()

	def init_weights(self):
		for p in self.parameters():
			if p.data.ndimension() >= 2:
				nn.init.xavier_uniform_(p.data)
			else:
				nn.init.zeros_(p.data)

	def forward(self, x: torch.Tensor,
	            init_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
	            ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
		"""Assumes x is of shape (batch, sequence, feature)"""
		bs, seq_sz, _ = x.size()
		hidden_seq = []
		if init_states is None:
			h_t, c_t = torch.zeros(self.hidden_size).to(x.device), torch.zeros(self.hidden_size).to(x.device)
		else:
			h_t, c_t = init_states
		for t in range(seq_sz):  # iterate over the time steps
			x_t = x[:, t, :]
			i_t = torch.sigmoid(x_t @ self.W_ii + h_t @ self.W_hi + self.b_i)
			f_t = torch.sigmoid(x_t @ self.W_if + h_t @ self.W_hf + self.b_f)
			g_t = torch.tanh(x_t @ self.W_ig + h_t @ self.W_hg + self.b_g)
			o_t = torch.sigmoid(x_t @ self.W_io + h_t @ self.W_ho + self.b_o)
			c_t = f_t * c_t + i_t * g_t
			h_t = o_t * torch.tanh(c_t)
			hidden_seq.append(h_t.unsqueeze(Dim.batch))
		hidden_seq = torch.cat(hidden_seq, dim=Dim.batch)
		# reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
		hidden_seq = hidden_seq.transpose(Dim.batch, Dim.seq).contiguous()
		return hidden_seq, (h_t, c_t)


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



if __name__ == "__main__":

	lstm = OptimizedLSTM(100, 32)
	a = torch.arange(5 * 10 * 100).view((5, 10, 100))
	hs, _ = lstm(a.float())
	print(hs.shape)
