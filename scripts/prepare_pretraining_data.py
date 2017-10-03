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
    import re
    return re.sub("\d", "d", sentence)
    
def main(argv):
    make_pre(argv.input, argv.output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare pretraining data")
    parser.add_argument("input", default="../data/billion/news.2011.en.shuffled")
    parser.add_argument("output", default="../data/billion/billion2011.txt")
    parser.add_argument("--preprocess", "-p", dest="preprocess", action="store_true",
                        default = False, 
                        help = "If set true, reverse the order of input sentences and replace all the digits.")
    args = parser.parse_args()
    main(args)
