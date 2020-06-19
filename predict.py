from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import json

import tensorflow as tf
import util

import json

except_name = ["公司", "本公司", "该公司", "贵公司", "贵司", "本行", "该行", "本银行", "该集团", "本集团", "集团",
               "它", "他们", "他们", "我们", "该股", "其", "自身"]


def check_span(fla_sentences, span):
    """检查span对应词语是否符合要求"""
    if "".join(fla_sentences[span[0] - 1]) in ["该", "本", "贵"]:      # 对span进行补全
        span = (span[0]-1, span[1])
    return span


def flatten_sentence(sentences):
    """将多维列表展开"""
    return [char for sentence in sentences for char in sentence]


def max_all_count(dict0):
    """获取字典当中的count最大值"""
    a = max([(value, key) for key, value in dict0.items()])
    return a[0]


def cluster_rule(example):
    """
    对模型预测结果进行算法修正
    :param example: 一条json数据
    :return: 纠正后的predicted_cluster
    """
    fla_sentences = flatten_sentence(example["sentences"])
    res_clusters = []
    com2cluster = {}
    except_cluster = {}

    for cluster in example["predicted_clusters"]:
        res_cluster = []
        span_count = {}
        span2pos = {}
        for span in cluster:
            if "".join(fla_sentences[span[0]:span[1] + 1]) in ["公司", "集团"]:  # 对缺失字符进行补充
                span = check_span(fla_sentences, span)
            if "#" in "".join(fla_sentences[span[0]:span[1] + 1]):  # 对不合法单词进行过滤
                continue
            res_cluster.append(span)
            word = "".join(fla_sentences[span[0]:span[1] + 1])
            span_count[word] = span_count.get(word, 0) + 1
            if span2pos.get(word, None) is not None:
                span2pos[word].append(span)
            else:
                span2pos[word] = [span]

        com_name = set(span_count.keys())
        for ex in except_name:
            com_name.discard(ex)
        max_name = ""
        max_count = 0
        for com in com_name:  # 获取cluster当中的频率最大的单词
            if span_count[com] > max_count:
                max_count = span_count[com]
                max_name = com
            elif span_count[com] == max_count and len(com) > len(max_name):
                max_count = span_count[com]
                max_name = com
        print("max_name:{}".format(max_name))

        for com in com_name:  # 公司名称
            if com[:2] == max_name[:2]:  # 头部两个字相同则认为两公司相同
                continue
            elif len(com) < len(max_name) and com in max_name:  # 具有包含关系的两公司,则认为相同
                continue
            elif len(com) > len(max_name) and max_name in com:
                continue
            else:
                print(com)
                # span2pos[com]
                except_cluster[com] = span2pos[com]  # 该公司名
                for n in span2pos[com]:  # 错误预测的span将会筛除
                    res_cluster.remove(n)

        if com2cluster.get(max_name, None) is None:
            com2cluster[max_name] = res_cluster
        else:
            print(res_cluster)
            com2cluster[max_name].extend(res_cluster)

        for key, value in except_cluster.items():  # 这步是十分有用的
            if com2cluster.get(key, None) is None:
                print("该span将被彻底清除:{}".format(key))
                continue
            else:
                print("{}重新融入别的cluster当中".format(key), value)
                com2cluster[key].extend(value)

        # res_clusters.append(res_cluster)
    for v_cluster in com2cluster.values():
        res_clusters.append(v_cluster)
    return res_clusters


if __name__ == "__main__":
    """
    命令行示例
    python predict.py bert_base conll-2012/tagging_pure/tagging_dev_pos.chinese.128.jsonlines result_of_20.txt
    """
    config = util.initialize_from_env()
    log_dir = config["log_dir"]

    # Input file in .jsonlines format.
    input_filename = sys.argv[2]        # 输入数据地址

    # Predictions will be written to this file in .jsonlines format.
    output_filename = sys.argv[3]       # 输出地址

    model = util.get_model(config)
    saver = tf.train.Saver()

    with tf.Session() as session:
      model.restore(session)

      with open(output_filename, "w") as output_file:
        with open(input_filename) as input_file:
          for example_num, line in enumerate(input_file.readlines()):
            example = json.loads(line)
            tensorized_example = model.tensorize_example(example, is_training=False)
            feed_dict = {i: t for i, t in zip(model.input_tensors, tensorized_example)}
            _, _, _, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores = session.run(model.predictions, feed_dict=feed_dict)
            predicted_antecedents = model.get_predicted_antecedents(top_antecedents, top_antecedent_scores)
            example["predicted_clusters"],  mention_to_predict = model.get_predicted_clusters(top_span_starts, top_span_ends, predicted_antecedents)
            example["top_spans"] = list(zip((int(i) for i in top_span_starts), (int(i) for i in top_span_ends)))
            example['head_scores'] = []
            example["mention_to_predict"] = str(mention_to_predict)

            example["predicted_clusters"] = cluster_rule(example)

            output_file.write(str(json.dumps(example, ensure_ascii=False)))
            output_file.write("\n")
            if example_num % 100 == 0:
              print("Decoded {} examples.".format(example_num + 1))
