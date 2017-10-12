import sys
sys.path.append('../')
sys.path.append('../seq2seq/')

import os
import argparse
import logging

import torch
import torchtext
import ConfigParser
#from torch.optim.lr_scheduler import StepLR

import seq2seq
from seq2seq.trainer import SupervisedTrainer
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from seq2seq.models.AttendedDecoderRNN import AttendedDecoderRNN
from seq2seq.loss import Perplexity
from seq2seq.optim import Optimizer
from seq2seq.dataset import SourceField, TargetField
from seq2seq.evaluator import Predictor
from seq2seq.util.checkpoint import Checkpoint

try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3



parser = argparse.ArgumentParser()
parser.add_argument('--config', action='store', dest='config_path', default="config.ini",
                    help='Path to the configuration file')
parser.add_argument('--model', action='store', dest='model', default="0",
                    help='Name of the model')
parser.add_argument('--decoder', action='store', dest='decoder_type', default="simple",
                    help='Choose decoder type from: simple, error. Default: simple')
parser.add_argument('--attention', action='store', dest='attention_type', default="global",
                    help='Choose attention type from: global, local. Default: global')
parser.add_argument('--n_epoch', action='store', type=int, dest='n_epoch', default=30,
                    help='Number of epoch to train. Default: 30')
parser.add_argument('--save', action='store', dest='checkpoint_every', default="better",
                    help='Save after certain number of training data (int), or after each epoch ("epoch"), or if performance is better ("better"). Default: better')
parser.add_argument('--resume', action='store_true', dest='resume',
                    default=False,
                    help='Indicates if training has to be resumed from the latest checkpoint')
parser.add_argument('--log-level', dest='log_level',
                    default='info',
                    help='Logging level.')

args = parser.parse_args()

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, args.log_level.upper()))
logging.info(args)

try:
    CHECKPOINT_EVERY = int(args.checkpoint_every)
except ValueError:
    CHECKPOINT_EVERY = args.checkpoint_every

config = ConfigParser.ConfigParser()
config.read("examples/%s"%args.config_path)
print config.items(args.model)
TRAIN_PATH = config.get(args.model, "train")
DEV_PATH = config.get(args.model, "dev")
EXPT_PATH = config.get(args.model, "expt")
N_LAYERS = int(config.get(args.model, "n_layers"))
HIDDEN_SIZE = int(config.get(args.model, "hidden_size"))
BATCH_SIZE = int(config.get(args.model, "batch_size"))
TEACHER_FORCING_RATE = float(config.get(args.model, "teacher_forcing_rate"))
LEARNING_RATE = float(config.get(args.model, "learning_rate"))
DROPOUT = float(config.get(args.model, "drop_out"))
WEIGHT_DECAY = float(config.get(args.model, "weight_decay"))
WV_DIM=int(config.get(args.model, "embedding_size"))

ATTENTION = args.attention_type
VOCAB_SIZE = 50000
MAX_LEN = 50
WV_TYPE ='word2vec',
LOCAL_WINDOW_SIZE = 2 


# Prepare dataset
src = SourceField()
tgt = TargetField()

def len_filter(example):
    return len(example.src) <= MAX_LEN and len(example.tgt) <= MAX_LEN

train = torchtext.data.TabularDataset(
    path=TRAIN_PATH, format='tsv',
    fields=[('src', src), ('tgt', tgt)],
    filter_pred=len_filter
)
dev = torchtext.data.TabularDataset(
    path=DEV_PATH, format='tsv',
    fields=[('src', src), ('tgt', tgt)],
    filter_pred=len_filter
)
src.build_vocab(train, wv_type='word2vec', wv_dim=200, fill_from_vectors=True, max_size=VOCAB_SIZE)
#tgt.build_vocab(train, wv_type='glove.6B', fill_from_vectors=True, max_size=VOCAB_SIZE)
#src.build_vocab(train, max_size=VOCAB_SIZE)
tgt.build_vocab(train, max_size=VOCAB_SIZE)
tgt.vocab = src.vocab
input_vocab = src.vocab
output_vocab = tgt.vocab


# Prepare loss
weight = torch.ones(len(tgt.vocab))
pad = tgt.vocab.stoi[tgt.pad_token]
loss = Perplexity(weight, pad)
if torch.cuda.is_available():
    loss.cuda()

seq2seq = None
optimizer = None

if not args.resume:
    # Initialize model
    hidden_size=HIDDEN_SIZE
    encoder = EncoderRNN(len(src.vocab), MAX_LEN, hidden_size,
                         n_layers=N_LAYERS,rnn_cell="LSTM",
                         variable_lengths=True)
    if ATTENTION == "local":
        decoder = AttendedDecoderRNN(len(tgt.vocab), MAX_LEN, hidden_size,
                         n_layers=N_LAYERS,rnn_cell="LSTM", window_size=LOCAL_WINDOW_SIZE,
                         dropout_p=DROPOUT, attention_method=ATTENTION,
                         eos_id=tgt.eos_id, sos_id=tgt.sos_id)
    else:
        decoder = DecoderRNN(len(tgt.vocab), MAX_LEN, hidden_size,
                         n_layers=N_LAYERS,rnn_cell="LSTM",
                         dropout_p=DROPOUT, use_attention=True,
                         eos_id=tgt.eos_id, sos_id=tgt.sos_id)
    seq2seq = Seq2seq(encoder, decoder)
    if torch.cuda.is_available():
        seq2seq.cuda()

    for param in seq2seq.parameters():
        param.data.uniform_(-0.08, 0.08)

	optimizer = Optimizer(torch.optim.Adam(seq2seq.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY), max_grad_norm=5)
# train
t = SupervisedTrainer(loss=loss, batch_size=BATCH_SIZE,
                      checkpoint_every=CHECKPOINT_EVERY,
                      print_every=500, expt_dir=EXPT_PATH)

seq2seq = t.train(seq2seq, train,
        num_epochs=args.n_epoch, dev_data=dev,
        optimizer=optimizer,
        resume=args.resume, teacher_forcing_ratio=TEACHER_FORCING_RATE)

