# coding=UTF-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import re
import os
import sys
import json
import tempfile
import subprocess
import collections

import util
import conll
from bert import tokenization

class DocumentState(object):
  def __init__(self, key):
    self.doc_key = key
    self.sentence_end = []
    self.token_end = []
    self.tokens = []
    self.subtokens = []
    self.info = []
    self.segments = []
    self.subtoken_map = []
    self.segment_subtoken_map = []
    self.sentence_map = []
    self.pronouns = []
    self.clusters = collections.defaultdict(list)
    self.coref_stacks = collections.defaultdict(list)
    self.speakers = []
    self.segment_info = []
    self.pos = []

  def finalize(self):
    # finalized: segments, segment_subtoken_map
    # populate speakers from info
    subtoken_idx = 0
    for segment in self.segment_info:
      speakers = []
      for i, tok_info in enumerate(segment):
        if tok_info is None and (i == 0 or i == len(segment) - 1):      # 头或者尾
          speakers.append('[SPL]')
        elif tok_info is None:
          speakers.append(speakers[-1])
        else:
          speakers.append(tok_info[9])                                  # 获取speaker
          if tok_info[4] == 'PRP':
            self.pronouns.append(subtoken_idx)
        subtoken_idx += 1
      self.speakers += [speakers]
    # populate sentence map

    subtoken_idx = 0
    for segment in self.segment_info:
      postags = []
      for i, tok_info in enumerate(segment):
        if tok_info is None and (i == 0 or i == len(segment) - 1):  # 头或者尾
          postags.append('x')
        elif tok_info is None:
          postags.append(postags[-1])
        else:
          postags.append(tok_info[4])  # 获取speaker
          if tok_info[4] == 'PRP':
            self.pronouns.append(subtoken_idx)
        subtoken_idx += 1
      self.pos += [postags]

    # populate clusters
    first_subtoken_index = -1
    for seg_idx, segment in enumerate(self.segment_info):
      speakers = []
      for i, tok_info in enumerate(segment):
        first_subtoken_index += 1
        coref = tok_info[-2] if tok_info is not None else '-'           # 获取cluster_id
        if coref != "-":
          last_subtoken_index = first_subtoken_index + tok_info[-1] - 1
          for part in coref.split("|"):
            if part[0] == "(":
              if part[-1] == ")":
                cluster_id = int(part[1:-1])
                self.clusters[cluster_id].append((first_subtoken_index, last_subtoken_index))
              else:
                cluster_id = int(part[1:])
                self.coref_stacks[cluster_id].append(first_subtoken_index)
            else:
              cluster_id = int(part[:-1])
              start = self.coref_stacks[cluster_id].pop()
              self.clusters[cluster_id].append((start, last_subtoken_index))
    # merge clusters
    merged_clusters = []
    for c1 in self.clusters.values():
      existing = None
      for m in c1:
        for c2 in merged_clusters:
          if m in c2:
            existing = c2
            break
        if existing is not None:
          break
      if existing is not None:
        print("Merging clusters (shouldn't happen very often.)")
        existing.update(c1)
      else:
        merged_clusters.append(set(c1))
    merged_clusters = [list(c) for c in merged_clusters]
    all_mentions = util.flatten(merged_clusters)
    sentence_map =  get_sentence_map(self.segments, self.sentence_end)
    subtoken_map = util.flatten(self.segment_subtoken_map)
    assert len(all_mentions) == len(set(all_mentions))
    num_words = len(util.flatten(self.segments))
    assert num_words == len(util.flatten(self.speakers))
    assert num_words == len(subtoken_map), (num_words, len(subtoken_map))
    assert num_words == len(sentence_map), (num_words, len(sentence_map))
    return {
      "doc_key": self.doc_key,
      "sentences": self.segments,
      "postags":self.pos,
      "speakers": self.speakers,
      "constituents": [],
      "ner": [],
      "clusters": merged_clusters,
      'sentence_map':sentence_map,
      "subtoken_map": subtoken_map,
      'pronouns': self.pronouns
    }


def normalize_word(word, language):
  if language == "arabic":
    word = word[:word.find("#")]
  if word == "/." or word == "/?":
    return word[1:]
  else:
    return word

# first try to satisfy constraints1, and if not possible, constraints2.
def split_into_segments(document_state, max_segment_len, constraints1, constraints2):
  """
  句子切分, 生成[CLS]....[SEP]的结构
  :param document_state:文本对象
  :param max_segment_len:最大句长[128, 256, 512]
  :param constraints1:  sentences末尾标识
  :param constraints2:  subtoken末尾标识
  :return:
  """
  current = 0
  previous_token = 0
  while current < len(document_state.subtokens):    # 字、以及英文词元素
    end = min(current + max_segment_len - 1 - 2, len(document_state.subtokens) - 1)
    while end >= current and not constraints1[end]:
      end -= 1
    if end < current:
        end = min(current + max_segment_len - 1 - 2, len(document_state.subtokens) - 1)
        while end >= current and not constraints2[end]:
            end -= 1
        if end < current:
            raise Exception("Can't find valid segment")
    document_state.segments.append(['[CLS]'] + document_state.subtokens[current:end + 1] + ['[SEP]'])
    subtoken_map = document_state.subtoken_map[current: end + 1]
    document_state.segment_subtoken_map.append([previous_token] + subtoken_map + [subtoken_map[-1]])
    info = document_state.info[current: end + 1]
    document_state.segment_info.append([None] + info + [None])
    current = end + 1
    previous_token = subtoken_map[-1]

def get_sentence_map(segments, sentence_end):
  current = 0
  sent_map = []
  sent_end_idx = 0
  assert len(sentence_end) == sum([len(s) -2 for s in segments])
  for segment in segments:
    sent_map.append(current)
    for i in range(len(segment) - 2):
      sent_map.append(current)
      current += int(sentence_end[sent_end_idx])
      sent_end_idx += 1
    sent_map.append(current)
  return sent_map


def get_document(document_lines, tokenizer, language, segment_len):
    """
    将v4_gold_conll文件转换为json_lines数据格式

    :param document_lines:一个list的conll文本数据
    :param tokenizer:bert切词器
    :param language:你所要处理的文本语言 chinese or english
    :param segment_len:句子长度128, 256, 512
    :return:
    """
    document_state = DocumentState(document_lines[0])             # 声明一个文本数据对象
    word_idx = -1
    for line in document_lines[1]:
      row = line.split()
      sentence_end = len(row) == 0
      if not sentence_end:
        assert len(row) >= 12
        word_idx += 1
        word = normalize_word(row[3], language)                   # 过滤词汇
        subtokens = tokenizer.tokenize(word)                      # bert处理, 英文被处理为词的元组，中文被切分成单个字
        document_state.tokens.append(word)                        # 词汇保存
        document_state.token_end += ([False] * (len(subtokens) - 1)) + [True]   # token的末尾标识
        for sidx, subtoken in enumerate(subtokens):               # 获取单个字符和id
          document_state.subtokens.append(subtoken)               # 字符保存
          info = None if sidx != 0 else (row + [len(subtokens)])
          document_state.info.append(info)                        # none或者subtokens长度
          document_state.sentence_end.append(False)               # 不是句子末尾，则为False
          document_state.subtoken_map.append(word_idx)            # subtoken所对应的词序
      else:
        document_state.sentence_end[-1] = True

    # split_into_segments(document_state, segment_len, document_state.token_end)
    # split_into_segments(document_state, segment_len, document_state.sentence_end)

    constraints1 = document_state.sentence_end if language != 'arabic' else document_state.token_end
    split_into_segments(document_state, segment_len, constraints1, document_state.token_end)
    stats["max_sent_len_{}".format(language)] = max(max([len(s) for s in document_state.segments]), stats["max_sent_len_{}".format(language)])
    document = document_state.finalize()
    return document


def skip(doc_key):
  # if doc_key in ['nw/xinhua/00/chtb_0078_0', 'wb/eng/00/eng_0004_1']: #, 'nw/xinhua/01/chtb_0194_0', 'nw/xinhua/01/chtb_0157_0']:
    # return True
  return False


def minimize_partition(name, language, extension, labels, stats, tokenizer, seg_len, input_dir, output_dir):
    # input_path：dev.chinese.v4_gold_conll
    input_path = "{}/{}.{}.{}".format(input_dir, name, language, extension)
    # output_path:dev.english.128.jsonlines
    output_path = "{}/{}-bert-wwm-ext.{}.{}.jsonlines".format(output_dir, name, language, seg_len)

    count = 0
    print("Minimizing {}".format(input_path))
    documents = []                                    # 保存一系列元组
    with open(input_path, "r") as input_file:
      for line in input_file.readlines():
        begin_document_match = re.match(conll.BEGIN_DOCUMENT_REGEX, line)     # 找到文章的头部#begin
        if begin_document_match:
          doc_key = conll.get_doc_key(begin_document_match.group(1), begin_document_match.group(2))
          documents.append((doc_key, []))             # 一篇新的文章
        elif line.startswith("#end document"):
          continue
          
          
        else:
          documents[-1][1].append(line)               # str放入list当中
    with open(output_path, "w") as output_file:
      for document_lines in documents:
        if skip(document_lines[0]):
          continue
        try:
          document = get_document(document_lines, tokenizer, language, seg_len)     # 将数据写入文件
          output_file.write(json.dumps(document, ensure_ascii=False))
          output_file.write("\n")
          count += 1
        except:
          continue
    print("Wrote {} documents to {}".format(count, output_path))


def minimize_language(language, labels, stats, vocab_file, seg_len, input_dir, output_dir, do_lower_case):
  tokenizer = tokenization.FullTokenizer(
                vocab_file=vocab_file, do_lower_case=do_lower_case)
  # TODO 处理tagging_dev_pos和tagging_pos两个数据集
  minimize_partition("train", language, "v4_gold_conll", labels, stats, tokenizer, seg_len, input_dir, output_dir)
  minimize_partition("dev", language, "v4_gold_conll", labels, stats, tokenizer, seg_len, input_dir, output_dir)
  minimize_partition("test", language, "v4_gold_conll", labels, stats, tokenizer, seg_len, input_dir, output_dir)

if __name__ == "__main__":
  """
  命令行示例
  python minimize.py bert/chinese_roberta_OntoNote5.0_finetune/vocab.txt conll-2012/ontoNote5.0/ conll-2012/ontoNote5.0/  false
  """
  vocab_file = sys.argv[1]      # 词表
  input_dir = sys.argv[2]       # 输入数据地址
  output_dir = sys.argv[3]      # 输出数据地址
  do_lower_case = sys.argv[4].lower() == 'true'
  print("是否小写化：", do_lower_case)
  labels = collections.defaultdict(set)       # 创建一个字典，里面的值具有默认值
  stats = collections.defaultdict(int)
  if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
  for seg_len in [128, 256, 384, 512]:
    # minimize_language("english", labels, stats, vocab_file, seg_len, input_dir, output_dir, do_lower_case)    # 处理中文数据集
    minimize_language("chinese", labels, stats, vocab_file, seg_len, input_dir, output_dir, do_lower_case)
    # minimize_language("es", labels, stats, vocab_file, seg_len)
    # minimize_language("arabic", labels, stats, vocab_file, seg_len)
  for k, v in labels.items():
    print("{} = [{}]".format(k, ", ".join("\"{}\"".format(label) for label in v)))
  for k, v in stats.items():
    print("{} = {}".format(k, v))
