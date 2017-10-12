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
                    help='Path to the result files. Default: "./experiment/nucle"')
parser.add_argument('--out_file_name', action='store', dest='out_file', default="scores.txt",
                    help='Name of the output file. Default: "scores.txt"')
parser.add_argument('--gold_file', action='store', dest='gold_file', default="./data/nucle/dev/validation.json",
                    help='Path to gold standard. (must be in json format). Default: "./data/nucle/dev/valiation.json"')
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
	    print (os.path.join(model_path, file))
            files.append(os.path.join(model_path, file))

print ("Start calculating...")
scores = dict()


with codecs.open(OUT_PATH, 'w', encoding='utf8') as f:
    f.write("model\tname\tp\tr\tf1\n")
    for system_path in files: 
        folders = system_path.split("/")
        m = M2Scorer(system_path, GOLD_PATH, max_len=MAX_LEN)
        s = m.get_p_r_f1()
        print s 
        f.write("%s %s %0.4f %0.4f %0.4f\n"%(folders[3], folders[5][-23:-4],s[0], s[1], s[2]))
print ("Finish calculating. Results saved at %s"%(OUT_PATH))
