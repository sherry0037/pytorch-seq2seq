import torch
from torch.autograd import Variable

class Predictor(object):

    def __init__(self, model, src_vocab, tgt_vocab):
        """
        Predictor class to evaluate for a given model.
        Args:
            model (seq2seq.models): trained model. This can be loaded from a checkpoint
                using `seq2seq.util.checkpoint.load`
            src_vocab (seq2seq.dataset.vocabulary.Vocabulary): source sequence vocabulary
            tgt_vocab (seq2seq.dataset.vocabulary.Vocabulary): target sequence vocabulary
        """
        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model.cpu()
        self.model.eval()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab


    def predict(self, src_seq, error_index=None):
        """ Make prediction given `src_seq` as input.

        Args:
            src_seq (list): list of tokens in source language

        Returns:
            tgt_seq (list): list of tokens in target language as predicted
            by the pre-trained model
        """
        src_id_seq = Variable(torch.LongTensor([self.src_vocab.stoi[tok] for tok in src_seq]),
                              volatile=True).view(1, -1) 
        decoder_kick = Variable(torch.LongTensor([self.tgt_vocab.stoi['<sos>']]),
                                volatile=True).view(1, -1)
        if torch.cuda.is_available():
            src_id_seq = src_id_seq.cuda()
            decoder_kick = decoder_kick.cuda()

        if error_index:
            input_variable = src_id_seq
            input_lengths = [len(src_seq)]
            target_variable = decoder_kick
            encoder_outputs, encoder_hidden = self.model.encoder(input_variable, input_lengths)
            softmax_list, _, other = self.model.decoder(inputs=target_variable,
                              encoder_hidden=encoder_hidden,
                              encoder_outputs=encoder_outputs,
                              function=self.model.decode_function,
                              error_index=error_index,
                              src_id_seq=src_id_seq)
        else:
            softmax_list, _, other = self.model(src_id_seq, [len(src_seq)], decoder_kick)
        length = other['length'][0]

        #tgt_id_seq = [other['sequence'][di][0].data[0] for di in range(length)]
        tgt_id_seq = []
        for di in range(len(src_seq)):
            tgt_id_seq.append(other['sequence'][di][0].data[0])
        #tgt_seq = [self.tgt_vocab.itos[tok] for tok in tgt_id_seq]
        tgt_seq = []
        for i in range(len(tgt_id_seq)):
            v = self.src_vocab
            if i in error_index:
                v = self.tgt_vocab
            tgt_seq.append(v.itos[tgt_id_seq[i]])
        return tgt_seq
