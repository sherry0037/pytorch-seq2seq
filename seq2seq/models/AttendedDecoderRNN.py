import random

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from .ScoredAttention import ScoredAttention
from .baseRNN import BaseRNN

if torch.cuda.is_available():
    import torch.cuda as device
else:
    import torch as device


class AttendedDecoderRNN(BaseRNN):
    r"""
    Provides functionality for decoding in a seq2seq framework, with an option for attention.

    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): a maximum allowed length for the sequence to be processed
        hidden_size (int): the number of features in the hidden state `h`
        sos_id (int): index of the start of sentence symbol
        eos_id (int): index of the end of sentence symbol
        n_layers (int, optional): number of recurrent layers (default: 1)
        rnn_cell (str, optional): type of RNN cell (default: gru)
        input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
        dropout_p (float, optional): dropout probability for the output sequence (default: 0)
        use_attention(bool, optional): flag indication whether to use attention mechanism or not (default: false)

    Attributes:
        KEY_ATTN_SCORE (str): key used to indicate attention weights in `ret_dict`
        KEY_LENGTH (str): key used to indicate a list representing lengths of output sequences in `ret_dict`
        KEY_SEQUENCE (str): key used to indicate a list of sequences in `ret_dict`

    Inputs: inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio
        - **inputs** (batch, seq_len, input_size): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs.  It is used for teacher forcing when provided. (default `None`)
        - **encoder_hidden** (num_layers * num_directions, batch_size, hidden_size): tensor containing the features in the
          hidden state `h` of encoder. Used as the initial hidden state of the decoder. (default `None`)
        - **encoder_outputs** (batch, seq_len, hidden_size): tensor with containing the outputs of the encoder.
          Used for attention mechanism (default is `None`).
        - **function** (torch.nn.Module): A function used to generate symbols from RNN hidden state
          (default is `torch.nn.functional.log_softmax`).
        - **teacher_forcing_ratio** (float): The probability that teacher forcing will be used. A random number is
          drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0).

    Outputs: decoder_outputs, decoder_hidden, ret_dict
        - **decoder_outputs** (seq_len, batch, vocab_size): list of tensors with size (batch_size, vocab_size) containing
          the outputs of the decoding function.
        - **decoder_hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the last hidden
          state of the decoder.
        - **ret_dict**: dictionary containing additional information as follows {*KEY_LENGTH* : list of integers
          representing lengths of output sequences, *KEY_SEQUENCE* : list of sequences, where each sequence is a list of
          predicted token IDs }.
    """

    KEY_ATTN_SCORE = 'attention_score'
    KEY_LENGTH = 'length'
    KEY_SEQUENCE = 'sequence'

    def __init__(self, vocab_size, max_len, hidden_size,
            sos_id, eos_id,
            n_layers=1, rnn_cell='gru',
            input_dropout_p=0, dropout_p=0, attention_method="general"):
        super(AttendedDecoderRNN, self).__init__(vocab_size, max_len, hidden_size,
                input_dropout_p, dropout_p,
                n_layers, rnn_cell)

        self.output_size = vocab_size
        self.max_length = max_len
        self.attention_method= attention_method
        self.eos_id = eos_id
        self.sos_id = sos_id

        self.init_input = None

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        if self.attention_method:
            self.attention = ScoredAttention(self.hidden_size, self.attention_method)

        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward_step(self, input_var, hidden, encoder_outputs, function):
        batch_size = input_var.size(0)
        output_size = input_var.size(1)
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)

        output, hidden = self.rnn(embedded, hidden)

        attn = None
        if self.attention_method:
            output, attn = self.attention(output, encoder_outputs)

        predicted_softmax = function(self.out(output.view(-1, self.hidden_size))).view(batch_size, output_size, -1)
        return predicted_softmax, hidden, attn

    def forward(self, inputs=None, encoder_hidden=None, function=F.log_softmax,
                    encoder_outputs=None, teacher_forcing_ratio=0):
        ret_dict = dict()
        if self.attention_method:
            if encoder_outputs is None:
                raise ValueError("Argument encoder_outputs cannot be None when attention is used.")
            ret_dict[AttendedDecoderRNN.KEY_ATTN_SCORE] = list()
        if inputs is None:
            if teacher_forcing_ratio > 0:
                raise ValueError("Teacher forcing has to be disabled (set 0) when no inputs is provided.")
        if inputs is None and encoder_hidden is None:
            batch_size = 1
        else:
            if inputs is not None:
                batch_size = inputs.size(0)
            else:
                if self.rnn_cell is nn.LSTM:
                    batch_size = encoder_hidden[0].size(1)
                elif self.rnn_cell is nn.GRU:
                    batch_size = encoder_hidden.size(1)

        if inputs is None:
            decoder_input = Variable(torch.LongTensor([self.sos_id]),
                                    volatile=True).view(batch_size, -1)
        else:
            decoder_input = inputs[:, 0].unsqueeze(1)
            inputs = None if inputs.size(1) == 1 else inputs[:, 1:]
        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        decoder_outputs = []
        sequence_symbols = []
        lengths = np.array([self.max_length] * batch_size)

        def decode(step, step_output, step_attn):
            decoder_outputs.append(step_output)
            if self.attention_method:
                ret_dict[AttendedDecoderRNN.KEY_ATTN_SCORE].append(step_attn)
            symbols = decoder_outputs[-1].topk(1)[1]
            sequence_symbols.append(symbols)

            eos_batches = symbols.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > di) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)
            return symbols

        def get_local_contexts(position, encoder_outputs, window=2): 
            indices = np.arange(position-window, position+window+1)
            indices = indices[(indices >= 0) & (indices < encoder_outputs.size(1))]
            indices = torch.from_numpy(indices)
            if torch.cuda.is_available():
                indices = indices.cuda()
            local_hidden_states = torch.index_select(encoder_outputs, 1, Variable(indices))
            return local_hidden_states

        fill_contexts = Variable(torch.zeros(encoder_outputs.size()))
        if torch.cuda.is_available():
	    fill_contexts.cuda()
 
        for di in range(self.max_length):
            local_contexts = get_local_contexts(di, encoder_outputs) if di <= encoder_outputs.size(1) else fill_contexts
            decoder_output, decoder_hidden, step_attn = self.forward_step(decoder_input, decoder_hidden, local_contexts,
                                                                     function=function)
            step_output = decoder_output.squeeze(1)
            symbols = decode(di, step_output, step_attn)
            if use_teacher_forcing:
                if di>=inputs.size(1):
                    break
                else:
                    symbols = inputs[:, di]
            decoder_input = symbols

        ret_dict[AttendedDecoderRNN.KEY_SEQUENCE] = sequence_symbols
        ret_dict[AttendedDecoderRNN.KEY_LENGTH] = lengths.tolist()

        return decoder_outputs, decoder_hidden, ret_dict