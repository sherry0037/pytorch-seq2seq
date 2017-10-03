# Prepare pretraining data by duplicate sentences.
#
# Usage:
# python prepare_training_data.py --input $INPUT_PATH --output $OUTPUT_PATH
#


import codecs
import argparse

def make_pre(input_file, output_file):
    lines = open(input_file).read().strip().split('\n')
    pairs = [[l, l] for l in lines]
    #with codecs.open(output_file, 'w', encoding='utf8') as f:
    with open(output_file, "w") as f:
        for p in pairs:
            f.write(p[0] + "\t" + p[1] + "\n")
    print ("Finish writing file.")
	
def main(argv):
	make_pre(argv.input, argv.output)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Prepare pretraining data")
	parser.add_argument("input", default="../data/billion/news.2011.en.shuffled")
	parser.add_argument("output", default="../data/billion/billion2011.txt")
	args = parser.parse_args()
	main(args)
