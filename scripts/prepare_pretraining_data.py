# Prepare pretraining data by duplicate sentences.
#
# Usage:
# python prepare_pretraining_data.py $INPUT_PATH $OUTPUT_PATH
#


import codecs
import argparse

def make_pre(input_file, output_file):
    lines = open(input_file).read().strip().split('\n')
    if args.preprocess:
        pairs = [[filter_digits(reverse_order(l)), filter_digits(l)] for l in lines]
    else:    
        pairs = [[l, l] for l in lines]
    with open(output_file, "w") as f:
        for p in pairs:
            f.write(p[0] + "\t" + p[1] + "\n")
    print ("Finish writing file.")

def reverse_order(sentence):
    return " ".join(sentence.split()[::-1])

def filter_digits(sentence):
    """
        Replace all the digits in a string to "d". 
    """
    sentence = sentence.lower()
    import re
    return re.sub("\d", "d", sentence)

def get_pretraining_data(input_file, output_dir, sizes=(20000,1000), random_seed=1024):
    import random
    import os
    random.seed(random_seed)
    lines = open(input_file).read().strip().split('\n')
    print lines[0]
    random.shuffle(lines)
    train = lines[:sizes[0]]
    dev = lines[sizes[0]:(sizes[0]+sizes[1])]
    test = lines[(sizes[0]+sizes[1]):]

    train_path = os.path.join(output_dir, "train/data")
    dev_path = os.path.join(output_dir, "dev/data")
    test_path = os.path.join(output_dir, "test/data")
    write_file(train, train_path)
    write_file(dev, dev_path)
    write_file(test, test_path)
    print ("Finish writing all files.")

def write_file(data, data_path):
    with open(data_path, 'w') as f:
        for l in data:
	    f.write(l+"\n")
    print ("Finish writing at %s"%data_path) 	 
def main(argv):
    if argv.split:
        get_pretraining_data("../data/billion/all_data_raw", "../data/billion")
    else:
        make_pre(argv.input_path, argv.output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare pretraining data")
    parser.add_argument("--input", dest="input_path", action="store", default="../data/billion/train/data",
                        help="Define input path. Default: ../data/billion/train/data")
    parser.add_argument("--output", dest="output_path", action="store", default="../data/billion/train/data_p.txt",
                        help="Define output path. Default: ../data/billion/train/data_p.txt")
    parser.add_argument("--preprocess", "-p", dest="preprocess", action="store_true",
                        default = False, 
                        help = "If set true, reverse the order of input sentences and replace all the digits.")
    parser.add_argument("--split", "-s", dest="split", action="store_true",
                        default = False, 
                        help = "If set true, split dataset.")
    args = parser.parse_args()
    main(args)
