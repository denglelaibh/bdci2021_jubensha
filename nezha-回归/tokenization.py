# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import re
import unicodedata
import six
import tensorflow as tf


def validate_case_matches_checkpoint(do_lower_case, init_checkpoint):
    """Checks whether the casing config is consistent with the checkpoint name."""

    # The casing has to be passed in by the user and there is no explicit check
    # as to whether it matches the checkpoint. The casing information probably
    # should have been stored in the bert_config.json file, but it's not, so
    # we have to heuristically detect it to validate.

    if not init_checkpoint:
        return

    m = re.match("^.*?([A-Za-z0-9_-]+)/bert_model.ckpt", init_checkpoint)
    if m is None:
        return

    model_name = m.group(1)

    lower_models = [
        "uncased_L-24_H-1024_A-16", "uncased_L-12_H-768_A-12",
        "multilingual_L-12_H-768_A-12", "chinese_L-12_H-768_A-12"
    ]

    cased_models = [
        "cased_L-12_H-768_A-12", "cased_L-24_H-1024_A-16",
        "multi_cased_L-12_H-768_A-12"
    ]

    is_bad_config = False
    if model_name in lower_models and not do_lower_case:
        is_bad_config = True
        actual_flag = "False"
        case_name = "lowercased"
        opposite_flag = "True"

    if model_name in cased_models and do_lower_case:
        is_bad_config = True
        actual_flag = "True"
        case_name = "cased"
        opposite_flag = "False"

    if is_bad_config:
        raise ValueError(
            "You passed in `--do_lower_case=%s` with `--init_checkpoint=%s`. "
            "However, `%s` seems to be a %s model, so you "
            "should pass in `--do_lower_case=%s` so that the fine-tuning matches "
            "how the model was pre-training. If this error is wrong, please "
            "just comment out this check." % (actual_flag, init_checkpoint,
                                              model_name, case_name, opposite_flag))


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
    #使用的为python3
        if isinstance(text, str):
        #text为判断的对象，str为需要判断的类型
        #python2中的unicode和python3中的str等价，如果为<class 'str'>则为unicode编码
        #及文本数据，如果为<class 'byte'>则为uft-8编码及二进制数据，
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        #以"utf-8"的方式解码对应的text内容，"ignore"代表如果设置不同错误的处理方案
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def printable_text(text):
    """Returns text encoded in a way suitable for print or `tf.logging`."""

    # These functions want `str` for both Python2 and Python3, but in one case
    # it's a Unicode string and in the other it's a byte string.
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text
        elif isinstance(text, unicode):
            return text.encode("utf-8")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with tf.io.gfile.GFile(vocab_file, "r") as reader:
        while True:
            token = convert_to_unicode(reader.readline())
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


def convert_by_vocab(vocab, items):
    """Converts a sequence of [tokens|ids] using the vocab."""
    output = []
    for item in items:
        output.append(vocab[item])
    return output


def convert_tokens_to_ids(vocab, tokens):
    return convert_by_vocab(vocab, tokens)


def convert_ids_to_tokens(inv_vocab, ids):
    return convert_by_vocab(inv_vocab, ids)


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    #把字符串头和尾的空格，以及位于头尾的\n,\t之类的内容删去
    if not text:
        return []
    #如果剩下的是一个空字符串，则直接返回空列表，否则进行split()操作
    tokens = text.split()
    return tokens


class FullTokenizer(object):
    """Runs end-to-end tokenziation."""

    def __init__(self, vocab_file, do_lower_case=True):
        self.vocab = load_vocab(vocab_file)
        r"""
        vocab = OrderedDict([('[PAD]',0),('[unused0]',1),('[unused1]',2)
        ...
        ('##?',30520),('##~',30521)这是对应的vocab单词之中的内容
        """
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        r"""
        将上面vocab的内容形成相应的字典
        {0:'[PAD]',1:'[unused0]',2:'[unused1]',...30521:'##~'}
        """
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        #BasicTokenizer的对应类就在下面
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)
        #WordpieceTokenizer的定义在BasicTokenizer定义的下面

    def tokenize(self, text):
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
        #BasicTokenizer.tokenize(text)
            #print('token = ')
            #print(token)
            #第一次切分：将所有的单词一次性地全部切分出来
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
            #在BasicTokenizer结果的基础上进行再一次切分，得到子词语(subword),
            #词汇表就是在此时引入的(中文不存在字词，因为中文切分出来都是一个字)
                split_tokens.append(sub_token)
        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        return convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids):
        return convert_by_vocab(self.inv_vocab, ids)


class BasicTokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self, do_lower_case=True):
        """Constructs a BasicTokenizer.
        Args:
          do_lower_case: Whether to lower case the input.
        """
        self.do_lower_case = do_lower_case
        #results = self._run_split_on_punc('I am a pig!I love you')
        #返回的数组为['I am a pig','!','I love you']

    def tokenize(self, text):
        """Tokenizes a piece of text.第一个循环之中的tokenize分词操作"""
        text = convert_to_unicode(text)
        text = self._clean_text(text)

        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        text = self._tokenize_chinese_chars(text)
    # 如果是中文的话，每个中文汉字的左右两边都加上空格，(这里只对汉字进行分字，
    # 不对汉字进行分词)，如果是英文的话不需要进行处理，因为英文词与词之间自身
    # 带有相应的空格
        orig_tokens = whitespace_tokenize(text)
        # whitespace_tokenize的定义在FullTokenizer的上面,将对应的句子使用空格进行分割开
        split_tokens = []
        #print('orig_tokens = ')
        #print(orig_tokens)
        
        for token in orig_tokens:
            if self.do_lower_case:
            
                token = token.lower()
                token = self._run_strip_accents(token)
                #_run_strip_accents(text)重点内容，将accents内容去掉
            split_tokens.extend(self._run_split_on_punc(token))
            #将对应的标点符号单独的切分出来，比如['I am a pig!I love you']切分成为
            #['I am a pig','!','I love you']
        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        #这一步先用标准空格拼接上一步的处理结果，再执行空格分词(去除连续空格)
        return output_tokens

    def _run_strip_accents(self, text):
        r"""Strips accents from a piece of text.
        _run_strip_accents(text) 方法用于去除 accents，即变音符号，那么什么是变音符号呢？
        像 Keras 作者 François Chollet 名字中些许奇怪的字符 ç、简历的英文 résumé 中的 é 和中文拼音声调 á 等，这些都是变音符号 accents，
        维基百科中描述如下：附加符号或称变音符号（diacritic、diacritical mark、diacritical point、diacritical sign），
        是指添加在字母上面的符号，以更改字母的发音或者以区分拼写相似词语。例如汉语拼音字母“ü”上面的两个小点，或“á”、“à”字母上面的标调符。"""
        #这里面放入英语单词比如'this','movie'等词语的时候
        #在unicodedata之前与unicodedata之后的内容均为this,movie
        #等内容
        text = unicodedata.normalize("NFD", text)
        #核心:unicodedata.normalize和unicodedata.category,前者返回
        #输入字符串text的规范分解形式，后者返回输入字符char的Unicode类别
        #unicodedata将英文字母和accents的内容切分出来
        output = []
        for char in text:
            cat = unicodedata.category(char)
            #unicodedata.category:返回一个字符在unicode里分类的类型
            #Mn:Nonspace
            if cat == "Mn":
            #cat=="Mn"代表当前的内容为重音符号
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        #print('_run_split_on_punc')
        chars = list(text)
        #输入的字符串为I am a pig!I love you，这里形成相应的数组
        #['I',' ','a','m',' ','a',' ','p','i','g','!','I',' ',
        #'l','o','v','e',' ','y','o','u']
        i = 0
        start_new_word = True
        output = []
        #print('chars = ')
        #print(chars)
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
	  #cp = ord(char)
	  #if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
	  #    (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
	  #  return True
	  #cat = unicodedata.category(char)
	  #if cat.startswith("P"):
	  #带"P"打头的[Pc]:Punctuation,Connector,[Pd]:Punctuation,Dash,
	  #[Pe]:Punctuation,Close...
	  #  return Truereturn False
                output.append([char])
                #当为!的时候，单压入一个[!]的数组'
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                    #前面如果单独压入一个[!]的数组之后，需要重新压入一个新的数组
                start_new_word = False
                output[-1].append(char)
                #只要现在压入数组了start_new_word就必须为False，并且在最后一个数组压入char
                #['I',' ','a','m',' ','a',' ','p','i','g','!','I',' ','l'
                #' ','o','v','e',' ','y','o','u']
                #输出的对应的数组为['I am a pig','!','I love you']
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class WordpieceTokenizer(object):
    """Runs WordPiece tokenziation."""

    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=200):
        #self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)
        self.vocab = vocab
        r"""vocab = OrderedDict([('[PAD]',0),('[unused0]',1),('[unused1]',2)
        ...
        ('##?',30520),('##~',30521)这是对应的vocab单词之中的内容
        """
        self.unk_token = unk_token
        #unk_token="[UNK]"
        self.max_input_chars_per_word = max_input_chars_per_word
        #max_input_chars_per_word = 200
        #unaffable
    def tokenize(self, text):
        """Tokenizes a piece of text into its word pieces.
        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.
        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]
        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer.
        Returns:
          A list of wordpiece tokens.
        """

        text = convert_to_unicode(text)
        #print('text = ')
        #print(text)
        #text = unaffable
        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            #print('~~~chars = ~~~')
            #print(chars)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue
            #单词长度超出200的时候使用["UNK"]标志来代替
            is_bad = False
            start = 0
            sub_tokens = []
            #start为上一次结束的位置，初始化的时候start的值为0
            while start < len(chars):
                end = len(chars)
                #end初始化为字符串最后的位置，这波操作有点类似于双指针操作
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    #注意python中end直接指向结束的位置
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    #没找到对应的substr
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                #if cur_substr in ['\uff01','\uff0c','\uff1f','\u3002']:
                #找到并添加[SEP]间隔标识
                #sub_tokens.append("[SEP]")
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            #没有找到的时候也标记为["UNK"]
            else:
                output_tokens.extend(sub_tokens)
            #这里使用extend的原因在于一个单词被拆分成为了多个部分，
            #每一个部分都是一个拆分过后的数组，比如unaffable被拆分为了
            #['una','##ffa','##ble'],如果多个unaffable就被拆分为了
            #['una','##ffa','##ble','una','##ffa','##ble']
        return output_tokens


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat in ("Cc", "Cf"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False