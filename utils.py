import string

import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class TransformerLabelConvertor(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token. [pad] for padding
        
        list_token =  ['[GO]','[s]','[PAD]']    #['[GO]', '[s]']  or ['[GO]','[s]','[UNK]','[PAD]']
        list_character = list(character)
        self.character = list_token + list_character
        
        self.dict = {}
        for i, char in enumerate(self.character):
            
            self.dict[char] = i

    def encode(self, text, batch_max_length=25):
        """ convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text : the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
            length : the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
        """
        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.
        # batch_max_length = max(length) # this is not allowed for multi-gpu setting
        batch_max_length += 1
        # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(2)
        for i, t in enumerate(text):
            text = list(t)
            text.append('[s]')
            text = [self.dict[char] for char in text]
            batch_text[i][1:1 + len(text)] = torch.LongTensor(text)  # batch_text[:, 0] = [GO] token
        return (batch_text.to(DEVICE), torch.IntTensor(length).to(DEVICE))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def create_mask(self,src, tgt):
        src_seq_len = src.shape[0]
        tgt_seq_len = tgt.shape[0]
    
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len)
        src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)
        # 2 index of padding token in character list.
        tgt_padding_mask = (tgt == 2).transpose(0, 1)
        return src_mask, tgt_mask, tgt_padding_mask


class AttnLabelConverter(object):
    """ Convert between text-label and text-index """
    
    def __init__(self, character):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.

        # same list_token variable is defined in model.py. If you make changes here, then make changes in that code also.
        list_token =  ['[GO]', '[s]']  # ['[GO]','[s]','[UNK]','[PAD]']
        list_character = list(character)
        self.character = list_token + list_character
        
        self.dict = {}
        for i, char in enumerate(self.character):
            self.dict[char] = i
    
    def encode(self, text, batch_max_length=25):
        """ convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text : the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
            length : the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
        """
        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.
        # batch_max_length = max(length) # this is not allowed for multi-gpu setting
        batch_max_length += 1
        # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text.append('[s]')
            text = [self.dict[char] for char in text]
            batch_text[i][1:1 + len(text)] = torch.LongTensor(text)  # batch_text[:, 0] = [GO] token
        return (batch_text.to(DEVICE), torch.IntTensor(length).to(DEVICE))
    
    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts

class Averager(object):
    """Compute average for torch.Tensor, used for loss average."""

    def __init__(self):
        self.reset()

    def add(self, v):
        count = v.data.numel()
        v = v.data.sum()
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res

#return character list given language name
def getCharacterList(lang):
    punctuation_char = ['!', '#', '%', '&', '\'', '*', ',', '.', '/', ':', ';', '?', '@', '\\']
    character_list = []
    skip_char_list=[]
    if lang=="eng":
        character_list= string.printable[:-6]
        
    if lang=="hi":
        first_char = 'ऀ'
        last_char = 'ॿ'
    
    ch=first_char
    while (first_char <= ch <= last_char):
        x = chr(ord(ch) + 1)
        if ch not in skip_char_list:
            character_list.append(ch)
        ch = x
    
    ret= list(set(character_list+punctuation_char))
    ret = ret+ ['\u200d', '\u200c']
    return ret