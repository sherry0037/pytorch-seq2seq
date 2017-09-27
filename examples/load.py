import sys
sys.path.append('../')
sys.path.append('../seq2seq/')

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
parser.add_argument('--load_model', action='store', dest='load_model', default="0",
                    help='Name of the model to load')
parser.add_argument('--new_model', action='store', dest='new_model', default="1",
                    help='Name of the model to load')
parser.add_argument('--decoder', action='store', dest='decoder_type', default="simple",
                    help='Choose decoder type from: simple, error, attended')
parser.add_argument('--n_epoch', action='store', type=int, dest='n_epoch', default=30,
                    help='Number of epoch to train. Default: 30')
parser.add_argument('--log-level', dest='log_level',
                    default='info',
                    help='Logging level.')

args = parser.parse_args()

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, args.log_level.upper()))
logging.info(args)

config = ConfigParser.ConfigParser()
config.read("examples/%s"%args.config_path)

print config.items(args.load_model)
EXPT_PATH_0 = config.get(args.load_model, "expt")
TEACHER_FORCING_RATE_0 = float(config.get(args.load_model, "teacher_forcing_rate"))

print config.items(args.new_model)
TRAIN_PATH_1 = config.get(args.new_model, "train")
DEV_PATH_1 = config.get(args.new_model, "dev")
EXPT_PATH_1 = config.get(args.new_model, "expt")
LEARNING_RATE_1 = float(config.get(args.new_model, "learning_rate"))
BATCH_SIZE_1 = int(config.get(args.new_model, "batch_size"))
MAX_LEN_1 = int(config.get(args.new_model, "max_len"))


# Prepare dataset
src = SourceField()
tgt = TargetField()

def len_filter(example):
    return len(example.src) <= MAX_LEN_1 and len(example.tgt) <= MAX_LEN_1

# Get the latest checkpoint
latest_checkpoint_path = Checkpoint.get_latest_checkpoint(EXPT_PATH_0)
checkpoint = Checkpoint.load(latest_checkpoint_path)
seq2seq = checkpoint.model
input_vocab = checkpoint.input_vocab
output_vocab = checkpoint.output_vocab

train = torchtext.data.TabularDataset(
    path=TRAIN_PATH_1, format='tsv',
    fields=[('src', src), ('tgt', tgt)],
    filter_pred=len_filter
)
dev = torchtext.data.TabularDataset(
    path=DEV_PATH_1, format='tsv',
    fields=[('src', src), ('tgt', tgt)],
    filter_pred=len_filter
)

src.vocab = input_vocab
tgt.vocab = output_vocab

# Prepare loss
weight = torch.ones(len(input_vocab))
pad = output_vocab.stoi[tgt.pad_token]
loss = Perplexity(weight, pad)
if torch.cuda.is_available():
    loss.cuda()

# train
optimizer = Optimizer(torch.optim.Adam(seq2seq.parameters(), lr = LEARNING_RATE_1), max_grad_norm=5)

t = SupervisedTrainer(loss=loss, batch_size=BATCH_SIZE_1,
                      checkpoint_every=3000,
                      print_every=50, expt_dir=EXPT_PATH_1)

seq2seq = t.train(seq2seq, train,
        num_epochs=args.n_epoch, dev_data=dev,
        optimizer=optimizer, teacher_forcing_ratio=TEACHER_FORCING_RATE_0)

