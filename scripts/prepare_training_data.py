# Prepare training data from json file.
#
# Usage:
# python prepare_pretraining_data.py --input $INPUT_PATH --output $OUTPUT_PATH
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
                input_sentence = " ".join(s["input_sentence"][::-1])
                input_sentence = filter_digits(input_sentence)
                corrected_sentence = filter_digits(corrected_sentence)
			f.write(input_sentence + "\t" + corrected_sentence + "\n")
	print ("Finish writing file at ") + output_file + "."

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
	parser.add_argument("input", default="nucle_validation_json.json")
	parser.add_argument("output", default="nucle_validation.txt")
    parser.add_argument("--preprocess", "-p", des="preprocess", action="store_true",
                        default = False, 
                        help = "If set true, reverse the order of input sentences and replace all the digits.")
	args = parser.parse_args()
	main(args)
