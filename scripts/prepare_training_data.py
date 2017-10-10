# Prepare training data from json file.
#
# Usage:
# python prepare_training_data.py $INPUT_PATH $OUTPUT_PATH [-p]
#

import json
import argparse
import codecs


def read_file(input_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
        print ("Finish reading file.")
        return data

def sent2seq(sentences, output_file):
    with codecs.open(output_file, 'w', encoding='utf8') as f:
        for s in sentences:
            input_sentence = " ".join(s["input_sentence"])
            corrected_sentence =  " ".join(s["corrected_sentence"]) 
            if args.preprocess:
                input_sentence = preprocess(s["input_sentence"], reverse=True)
                corrected_sentence = preprocess(s["corrected_sentence"])
            f.write(input_sentence + "\t" + corrected_sentence + "\n")
    print ("Finish writing file at ") + output_file + "."


def preprocess(sentence, reverse=False):
    """ Preprocess a sentence.

        Args:
        sentence(list): list of tokens
        reverse(boolean): wether to reverse the sentence
    """
    if reverse:
        sentence = sentence[::-1]
    sentence = " ".join(sentence)
    sentence = filter_digits(sentence)
    sentence = sentence.lower()
    return sentence
    

def filter_digits(sentence):
    """
        Replace all the digits in a string to "d". 
    """
    import re
    return re.sub("\d", "d", sentence)
    
def main(argv):
    d = read_file(argv.input)   
    sent2seq(d["data"], argv.output)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="convert .json file into .txt file")
    parser.add_argument("--input", dest="input", action="store", default="../data/nucle/train/train.json",
                        help="Define input path. Default: ../data/nucle/train/train.json")
    parser.add_argument("--output", dest="output", action="store", default="../data/nucle/train/data_p.txt",
                        help="Define output path. Default: ../data/nucle/train/data_p.txt")
    parser.add_argument("--preprocess", "-p", dest="preprocess", action="store_true",
                        default = False, 
                        help = "If set true, reverse the order of input sentences and replace all the digits.")
    args = parser.parse_args()
    main(args)
