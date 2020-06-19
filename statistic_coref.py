#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020-04-09 18:00
# @Author  : henry
# @Site    : http:github.com/henryhust
# @File    : statistic_coref.py

import os
import json
data_path = "conll-2012"


def flatten(lists):
    return [s for l in lists for s in l]


width_dict = {}
for file in ["train.chinese.128.jsonlines", "dev.chinese.128.jsonlines", "test.chinese.128.jsonlines"]:
    with open(os.path.join(data_path, file), encoding="utf-8") as fr:
        for line in fr.readlines():
            line = json.loads(line)
            clusters = flatten(line["clusters"])
            for span in clusters:
                print(span)
                width = span[1] - span[0] + 1
                width_dict[width] = width_dict.get(width, 0) + 1
print(width_dict)

