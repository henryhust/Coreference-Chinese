# coding=utf-8
# !/usr/bin/python3
# Author:henry
# 2019/11/18
#  ┏┓       ┏┓+ +
# ┏┛┻━━━━━━━┛┻┓ + +
# ┃　　　　　　 ┃
# ┃　　　━　　　┃ ++ + + +
#█████━█████  ┃+
# ┃　　　　　　 ┃ +
# ┃　　　┻　　　┃
# ┗━━┓　　　 ┏━┛
#    ┃　　  ┃
#　 　┃　　  ┃ + + + +
#　 　┃　　　┃　
#　　 ┃　　　┃ + 
#　 　┃　 　 ┗━━━┓ + +
#　 　┃ 　　　　　┣┓
# 　　┃ 　　　　　┏┛
#　 　┗┓┓┏━━━┳┓┏┛ + + + +
#　　  ┃┫┫　 ┃┫┫
#　　  ┗┻┛　 ┗┻┛+ + + +
#  Animal protecting, BUG away
#———————predict后，数据可视化—————————

import metrics
import json
import os
import sys
from bert import tokenization


def flatten(sentences):
    return [word for sen in sentences for word in sen]

flatten_results = []

# inputfile可以使用:conll-2012/tagging_pure/tagging_dev_pos.chinese.128.jsonlines
with open(sys.argv[1]) as fr:
    for line in fr.readlines():
        example = json.loads(line.strip().replace("\'", "\'"))
        flatten_result = flatten(example["sentences"])
        flatten_results.append(flatten_result)


def mention_recall(pre, gold):
    """实体词召回率计算"""
    r = 0
    for p in pre.keys():
        if p in gold.keys():
            r += 1
    num = len(gold.keys())+float("-inf")                          # 加上一个很小的数字，避免分母为零
    print("6.the mention_recall = {}".format((r/num*100)))


tmp = sys.stdout                                                  # 将终端输出结果,保存到out_file当中
num = 0
evaluator0 = metrics.CorefEvaluator()                             # 全局测评器,生成最终evaluation结果


def main0(example, flatten_result):
    global num
    with open(sys.argv[2], "a+") as fa:
        """结果输出文件地址"""
        print(sys.argv[2])
        sys.stdout = fa
        print("the id is {}".format(num))                           # 数据编号0,1,2.....
        num += 1
        print(example["sentences"])
        print("the text is:\n", "".join(flatten_result))
        gold_clusters = []                                          # cluster-span正确结果
        gold_clu_chns = []                                          # cluster-chinese正确结果,即将span转换为对应单词
        for cluster in example["clusters"]:
            span_result = []
            gold_clu_chn = []
            for span in cluster:
                result0 = tuple(flatten_result[int(span[0]):int(span[1])+1])
                result = tuple(span)
                span_result.append(result)
                gold_clu_chn.append(result0)
            span_result = sorted(span_result)
            gold_clusters.append(span_result)       # 文本头尾位置
            gold_clu_chns.append(gold_clu_chn)      # 中文字符
        print("1.clusters:", gold_clusters)
        print("1.clusters_chn:", gold_clu_chns)

        predict_clusters = []                                       # cluster-span预测结果
        predict_clu_chns = []                                       # cluster-chinese预测结果,即将span转换为对应单词
        for cluster in example["predicted_clusters"]:
            span_result = []
            predict_clu_chn = []
            for span in cluster:
                result0 = tuple(flatten_result[int(span[0]):int(span[1])+1])
                result = tuple(span)
                span_result.append(result)
                predict_clu_chn.append(result0)
            predict_clusters.append(span_result)
            predict_clu_chns.append(predict_clu_chn)
        print("2.predicted_clusters:", predict_clusters)
        print("2.predict_clu_chns:", predict_clu_chns)

        top_span = []
        for span in example["top_spans"]:
            span_result = []
            # result = tuple(flatten_result[int(span[0]):int(span[1])+1])
            result = tuple(span)
            top_span.append(result)
        print("3.top_spans:", top_span)                              # top-k span

        sentence_result = []

        # 获取mention_to_gold
        mention_to_gold = {}
        for gc in gold_clusters:
          for mention in gc:
            a = tuple(mention)
            mention_to_gold[a] = tuple(gc)

        # 获取mention_to_predict
        mention_to_predicted = eval(example["mention_to_predict"])
        # print("4.mention_to_predicted={}".format(mention_to_predicted))

        # print("5.mention_to_gold={}".format(mention_to_gold))
        mention_recall(mention_to_predicted, mention_to_gold)
        evaluator = metrics.CorefEvaluator()                            # 局部测评器
        evaluator0.update(predict_clusters, gold_clusters, mention_to_predicted, mention_to_gold)   # 全局测评器
        evaluator.update(predict_clusters, gold_clusters, mention_to_predicted, mention_to_gold)
        p, r, f = evaluator.get_prf()
        print("\ntotal:\nF1={}\nP={}\nR={}".format(f, p, r))
        sys.stdout = tmp


def get_one(id, tuple0):

    print(flatten_results[id][tuple0[0]:tuple0[1]+1])


def main1(input):
    """
    加载predict结果文件,生成predict可视化文件,方便对生成cluster进行进一步查看
    """
    with open(input) as fr:
        """predict预测结果文件"""
        for line in fr.readlines():
            example = json.loads(line.strip().replace("\'", "\""))
            flatten_result = flatten(example["sentences"])
            main0(example, flatten_result)
            print("----------------------------")
        p, r, f = evaluator0.get_prf()
        print("\ntotal:\nF1={}\nP={}\nR={}".format(f, p, r))                        # 打印全局测评结果,即evaluation结果


if __name__ == '__main__':
    """
    命令行示例: python after_predict.py file_input file_output
    """
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    main1(input_file)
