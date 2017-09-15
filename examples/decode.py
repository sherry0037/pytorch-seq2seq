import os
import argparse
import logging

import torch
from torch.optim.lr_scheduler import StepLR
import torchtext
import json
import codecs

import seq2seq
from seq2seq.trainer import SupervisedTrainer
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from seq2seq.loss import Perplexity
from seq2seq.optim import Optimizer
from seq2seq.dataset import SourceField, TargetField
from seq2seq.evaluator import Predictor
from seq2seq.util.checkpoint import Checkpoint
 
try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3

# Sample usage:
#      # resuming from a specific checkpoint
#      CHECKPOINT_DIR=2017_09_07_14_45_10 
#      python examples/decode.py --load_checkpoint $CHECKPOINT_DIR
CHECKPOINT_DIR="2017_09_15_08_35_17" 
DEV_PATH="data/nucle/dev/nucle_validation_json.json"
EXPT_PATH="./experiment/nucle"
OUT_PATH=EXPT_PATH+"/outputs"

parser = argparse.ArgumentParser()
parser.add_argument('--dev_path', action='store', dest='dev_path', default=DEV_PATH,
                    help='Path to dev data')
parser.add_argument('--expt_dir', action='store', dest='expt_dir', default=EXPT_PATH,
                    help='Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
parser.add_argument('--out_dir', action='store', dest='out_dir', default=OUT_PATH,
                    help='Path to store predictions')
parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint', default=CHECKPOINT_DIR,
                    help='The name of the checkpoint to load, usually an encoded time string')
parser.add_argument('--resume', action='store_true', dest='resume',
                    default=False,
                    help='Indicates if training has to be resumed from the latest checkpoint')
parser.add_argument('--log-level', dest='log_level',
                    default='info',
                    help='Logging level.')


opt = parser.parse_args()

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, opt.log_level.upper()))
logging.info(opt)

if not os.path.exists(opt.out_dir):
    os.makedirs(opt.out_dir)

logging.info("loading checkpoint from {}".format(os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)))
checkpoint_path = os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)
checkpoint = Checkpoint.load(checkpoint_path)
seq2seq = checkpoint.model
input_vocab = checkpoint.input_vocab
output_vocab = checkpoint.output_vocab

predictor = Predictor(seq2seq, input_vocab, output_vocab)

source_sentences = []
target_sentences = []
output_sentences = []
with open(opt.dev_path, 'r') as f:
    sentences = json.load(f)["data"]
    for s in sentences:
        source_sentences.append(" ".join(s["input_sentence"]))
        target_sentences.append(" ".join(s["corrected_sentence"]))
        output_sentences.append(" ".join(predictor.predict(s["input_sentence"])))

for i in xrange(len(source_sentences)):
    print ("source = " + source_sentences[i] + "\n")
    print ("target = " + target_sentences[i]+ "\n")
    print ("output = " + output_sentences[i] + "\n\n")

"""
with codecs.open(opt.out_dir+"/"+opt.load_checkpoint+".txt", 'w', encoding='utf8') as f:
    for i in xrange(len(source_sentences)):
        f.write("source = " + source_sentences[i] + "\n")
        f.write("target = " + target_sentences[i]+ "\n")
        f.write("output = " + output_sentences[i] + "\n\n")"""
print ("Finish writing.")
