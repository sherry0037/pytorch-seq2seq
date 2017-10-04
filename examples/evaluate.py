import os
import sys
sys.path.append('../')
sys.path.append('../seq2seq/')
import argparse
import ConfigParser
import codecs
from seq2seq.util.m2scorer import M2Scorer
 
try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3


parser = argparse.ArgumentParser()
parser.add_argument('--config', action='store', dest='config_path', default="config.ini",
                    help='Path to the configuration file')
parser.add_argument('--model', action='store', dest='model', default="0",
                    help='Name of the model')
parser.add_argument('--out_file', action='store', dest='out_file', default="M2_scores.txt",
                    help='Where to store the results')

parser.add_argument('--errors_given', '-e', action='store_true', dest='errors_given', default=False,
                    help='Use known indices for errors to make prediction.')
args = parser.parse_args()

config = ConfigParser.ConfigParser()
config.read("examples/%s"%args.config_path)
GOLD_PATH = config.get(args.model, "dev")[:-3]+"json"
EXPT_PATH = config.get(args.model, "expt") + "/outputs"
if args.errors_given:
    EXPT_PATH += "/errors"
OUT_PATH=EXPT_PATH + "/scores/"
MAX_LEN = int(config.get(args.model, "max_len"))

if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)

files = []

for file in os.listdir(EXPT_PATH):
    if file.endswith(".txt"):
        files.append(os.path.join(EXPT_PATH, file))

scores = dict()
for system_path in files: 
    m = M2Scorer(system_path, GOLD_PATH, max_len=50)
    scores[system_path[-23:-4]] = m.get_p_r_f1()


with codecs.open(OUT_PATH+args.out_file, 'w', encoding='utf8') as f:
    f.write("name p r f1\n")
    for key, item in scores.items():
        f.write("%s %0.4f %0.4f %0.4f\n"%(key,item[0], item[1], item[2]))
print ("Finish calculating. Results saved at %s"%(OUT_PATH+args.out_file))
