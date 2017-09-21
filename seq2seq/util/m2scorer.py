"""
 Score a system's output against a gold reference 

 SYSTEM_PATH: system output, sentence per line
 GOLD_PATH: source sentences with gold token edits

"""


import levenshtein
from util import paragraphs
from util import smart_open
import json

MAX_LEN = 10

class M2Scorer():
    def __init__(self, system_sentences_path, source_gold_path, max_len = MAX_LEN):
        self.max_len = max_len
        self.system_sentences = load_system(system_sentences_path)
        self.source_sentences, self.gold_edits, self.correct_sentences= load_annotation(source_gold_path, max_len)
        #self.source_sentences, self.gold_edits = load_annotation0(source_gold_path)

    def get_p_r_f1(self):
        p, r, f1 = levenshtein.batch_multi_pre_rec_f1(self.system_sentences, self.source_sentences, self.gold_edits)
        return p, r, f1


def len_filter(source, target, max_len):
    return len(source) <= max_len and len(target) <= max_len        

def load_annotation(gold_file, max_len):
    source_sentences = []
    gold_edits = []
    correct_sentences = []
    with open(gold_file, 'r') as f:
        data = json.load(f)
        sentences = data.values()[0]
        for sentence in sentences:
            gold_edit = []
            input_sentence = sentence["input_sentence"]
            correct_sentence = sentence["corrected_sentence"]
            if not len_filter(input_sentence, correct_sentence, max_len):
                continue
            for mistake in sentence["mistakes"]:
                p = mistake["position"]
                if p==[]: continue
                original = " ".join(input_sentence[p[0]:p[-1]+1])
                correction = " ".join(mistake["correction"])
                if mistake["type"] == "insert":
                    correction = correction+" "+original
                m = (p[0], p[-1]+1, original, correction)
                gold_edit.append(m)

            source_sentences.append(u" ".join(input_sentence))
            gold_edits.append({0:gold_edit})
            correct_sentences.append(u" ".join(correct_sentence))
    return source_sentences, gold_edits, correct_sentences


def load_system(system_file):
    fin = smart_open(system_file, 'r')
    system_sentences = [line.decode("utf8").strip() for line in fin.readlines()]
    system_sentences = [s[9:] for s in system_sentences if s[:6] == "output"]
    fin.close()
    return system_sentences

