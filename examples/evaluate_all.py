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
parser.add_argument('--input_dir', action='store', dest='input_dir', default="./experiment/nucle",
                    help='Path to the result files')
parser.add_argument('--out_file', action='store', dest='out_file', default="scores.txt",
                    help='Where to store the scores')
parser.add_argument('--gold_file', action='store', dest='gold_file', default="./data/nucle/dev/validation.json",
                    help='Path to gold standard. (must be in json format)')
args = parser.parse_args()


EXPT_PATH = args.input_dir
OUT_PATH = os.path.join(EXPT_PATH, args.out_file)
GOLD_PATH = args.gold_file
MAX_LEN = 50

files = []
for model in os.listdir(EXPT_PATH):
    model_path = os.path.join(EXPT_PATH, model+"/outputs")
    if not os.path.isdir(model_path):
        continue
    for file in os.listdir(model_path):
        if file.endswith(".txt"):
            files.append(os.path.join(model_path, file))

scores = dict()
for system_path in files: 
    folders = system_path.split("/")
    m = M2Scorer(system_path, GOLD_PATH, max_len=MAX_LEN)
    scores[(folders[3], folders[5][-23:-4])] = m.get_p_r_f1()

print ("Start calculating...")
with codecs.open(OUT_PATH, 'w', encoding='utf8') as f:
    f.write("model\tname\tp\tr\tf1\n")
    for key, item in scores.items():
        f.write("%s %s %0.4f %0.4f %0.4f\n"%(key[0],key[1],item[0], item[1], item[2]))
print ("Finish calculating. Results saved at %s"%(OUT_PATH))
