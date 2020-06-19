#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020-04-08 11:00
# @Author  : henry
# @Site    : http:github.com/henryhust
# @File    : get_pos.py

import os
import json

data_path = "conll-2012"

def flatten(lists):
    return [s for l in lists for s in l]


pos_dict = {"UNK":0}
for file in ["train.chinese.128.jsonlines", "dev.chinese.128.jsonlines", "test.chinese.128.jsonlines"]:
    with open(os.path.join(data_path, file), encoding="utf-8") as fr:
        for line in fr.readlines():
            line = json.loads(line)
            postags = flatten(line["postags"])
            for pos in postags:
                if pos not in pos_dict:
                    pos_dict[pos] = len(pos_dict)
print(pos_dict)