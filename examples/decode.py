import os
import sys
sys.path.append('../')
sys.path.append('../seq2seq/')
import argparse
import logging

import torch
from torch.optim.lr_scheduler import StepLR
import torchtext
import json
import codecs
import ConfigParser

import seq2seq
from seq2seq.trainer import SupervisedTrainer
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from seq2seq.loss import Perplexity
from seq2seq.optim import Optimizer
from seq2seq.dataset import SourceField, TargetField
from seq2seq.evaluator import Predictor
from seq2seq.util.checkpoint import Checkpoint
from seq2seq.util.m2scorer import M2Scorer
 
try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3

# Sample usage:
#      # resuming from a specific checkpoint
#      CHECKPOINT_DIR=2017_09_07_14_45_10 
#      python examples/decode.py --load_checkpoint $CHECKPOINT_DIR
CHECKPOINT_DIR="2017_09_19_15_46_02" 


parser = argparse.ArgumentParser()
parser.add_argument('--config', action='store', dest='config_path', default="config.ini",
                    help='Path to the configuration file')
parser.add_argument('--model', action='store', dest='model', default="0",
                    help='Name of the model')
parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint', default=None,
                    help='The name of the checkpoint to load, usually an encoded time string. Can also be the directory name.')
parser.add_argument('--errors_given', '-e', action='store_true', dest='errors_given', default=False,
                    help='Use known indices for errors to make prediction.')
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
DEV_PATH = config.get(args.model, "dev")[:-3]+"json"
EXPT_PATH = config.get(args.model, "expt")
OUT_PATH=EXPT_PATH + "/outputs"
MAX_LEN = int(config.get(args.model, "max_len"))
   
if not args.load_checkpoint:
    args.load_checkpoint = EXPT_PATH:



checkpoints = [args.load_checkpoint]
if not args.load_checkpoint[:4]=="2017":
    checkpoints = []
    for file in os.listdir(args.load_checkpoint+"/checkpoints"):
        if file[:4]=="2017":
            checkpoints.append(file)

def decode(checkpoint_name, out_path=OUT_PATH, expt_path=EXPT_PATH): 
    logging.info("loading checkpoint from {}".format(os.path.join(expt_path, Checkpoint.CHECKPOINT_DIR_NAME, checkpoint_name)))
    checkpoint_path = os.path.join(expt_path, Checkpoint.CHECKPOINT_DIR_NAME, checkpoint_name)
    checkpoint = Checkpoint.load(checkpoint_path)
    seq2seq = checkpoint.model
    input_vocab = checkpoint.input_vocab
    output_vocab = checkpoint.output_vocab

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    predictor = Predictor(seq2seq, input_vocab, output_vocab)

    def len_filter(source, target):
        return len(source) <= MAX_LEN and len(target) <= MAX_LEN

    source_sentences = []
    target_sentences = []
    output_sentences = []
    with open(DEV_PATH, 'r') as f:
        sentences = json.load(f)["data"]
        for s in sentences:       
            if not len_filter(s["input_sentence"], s["corrected_sentence"]):
                continue
            source_sentences.append(" ".join(s["input_sentence"]))
            target_sentences.append(" ".join(s["corrected_sentence"]))
            if args.errors_given:
                err = []
                for m in s["mistakes"]:
                    err += m["position"]
                output_sentences.append(" ".join(predictor.predict(s["input_sentence"], err)))
                
            else:
                output_sentences.append(" ".join(predictor.predict(s["input_sentence"])))

    assert len(source_sentences) == len(target_sentences) == len(output_sentences)
    if args.errors_given:
	out_path += "/errors"
        if not os.path.exists(out_path):
            os.makedirs(out_path)
    with codecs.open(out_path+"/"+checkpoint_name+".txt", 'w', encoding='utf8') as f:
        for i in xrange(len(source_sentences)):
            f.write("source = " + source_sentences[i] + "\n")
            f.write("target = " + target_sentences[i]+ "\n")
            f.write("output = " + output_sentences[i] + "\n\n")
    print ("Finish decoding. Results saved at %s"%(out_path+"/"+checkpoint_name+".txt"))

for checkpoint_path in checkpoints:
    decode(checkpoint_path)
print ("All finished.")
