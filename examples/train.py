import os
import argparse
import logging

import torch
import torchtext
import ConfigParser
from torch.optim.lr_scheduler import StepLR

import seq2seq
from seq2seq.trainer import SupervisedTrainer
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from seq2seq.models.ErrorDecoderRNN import ErrorDecoderRNN
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
                    help='Choose decoder type from: simple, error, attended')
parser.add_argument('--n_epoch', action='store', dest='n_epoch', default=30,
                    help='Number of epoch to train. Default: 30')
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

config = ConfigParser.ConfigParser()
config.read("examples/%s"%args.config_path)
print config.items(args.model)
TRAIN_PATH = config.get(args.model, "train")
DEV_PATH = config.get(args.model, "dev")
EXPT_PATH = config.get(args.model, "expt")
VOCAB_SIZE = int(config.get(args.model, "vocab_size"))
HIDDEN_SIZE = int(config.get(args.model, "hidden_size"))
ATTENTION = config.get(args.model, "attention")
BATCH_SIZE = int(config.get(args.model, "batch_size"))
TEACHER_FORCING_RATE = float(config.get(args.model, "teacher_forcing_rate"))
LEARNING_RATE = float(config.get(args.model, "learning_rate"))
MAX_LEN = int(config.get(args.model, "max_len"))


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
src.build_vocab(train, wv_type='glove.6B', fill_from_vectors=True, max_size=VOCAB_SIZE)
tgt.build_vocab(train, wv_type='glove.6B', fill_from_vectors=True, max_size=VOCAB_SIZE)
#src.build_vocab(train, max_size=VOCAB_SIZE)
#tgt.build_vocab(train, max_size=VOCAB_SIZE)
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
                         variable_lengths=True)
    if args.decoder_type == "error":
    	decoder = ErrorDecoderRNN(len(tgt.vocab), MAX_LEN, hidden_size,
                         dropout_p=0.2, use_attention=True,
                         eos_id=tgt.eos_id, sos_id=tgt.sos_id)
    else:
    	decoder = DecoderRNN(len(tgt.vocab), MAX_LEN, hidden_size,
                         dropout_p=0.2, use_attention=True,
                         eos_id=tgt.eos_id, sos_id=tgt.sos_id)
    seq2seq = Seq2seq(encoder, decoder)
    if torch.cuda.is_available():
        seq2seq.cuda()

    for param in seq2seq.parameters():
        param.data.uniform_(-0.08, 0.08)

optimizer = Optimizer(torch.optim.Adam(seq2seq.parameters(), lr = LEARNING_RATE), max_grad_norm=5)
# train
t = SupervisedTrainer(loss=loss, batch_size=BATCH_SIZE,
                      checkpoint_every=10,
                      print_every=10, expt_dir=EXPT_PATH)

seq2seq = t.train(seq2seq, train,
        num_epochs=args.n_epoch, dev_data=dev,
        optimizer=optimizer,
        resume=args.resume, teacher_forcing_ratio=TEACHER_FORCING_RATE)

