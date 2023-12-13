


import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.functional as F
import torch.nn.functional as F_nn
import torch.optim as optim

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNN(nn.Module):

	def __init__(self, embedding_dim, hidden_dim, output_size):

		super(RNN, self).__init__()

		#self.embedding_dim = embedding_dim
		self.hidden_dim = hidden_dim
		#self.vocab_size = vocab_size

		#self.embedding = nn.Embedding(vocab_size, embedding_dim)
		self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=2, batch_first=True, nonlinearity='relu')

		self.hidden1 = nn.Linear(hidden_dim, output_size)
		#self.hidden2 = nn.Linear(output_size*4, output_size)




	def init_hidden(self, batch_size):
		return(autograd.Variable(torch.randn(2, batch_size, self.hidden_dim)).to('cuda'),
						autograd.Variable(torch.randn(2, batch_size, self.hidden_dim)).to('cuda'))


	def forward(self, input):
		#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		input = input.float()
		h = torch.zeros(2, input.size(0), self.hidden_dim).requires_grad_().to('cuda').float()

		#embeds = self.embedding(batch)
		#packed_input = pack_padded_sequence(input, lengths)
		outputs, ht = self.rnn(input, h.detach())
		#print(outputs.shape)
		#print(outputs[:, -1, :].shape)
		outputs = self.hidden1(outputs[:, -1, :]) 


		return outputs
