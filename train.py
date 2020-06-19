#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import tensorflow as tf
import util
import pyhocon
import logging
format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
logging.basicConfig(format=format)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
  """python train.py bert_base"""
  config = util.initialize_from_env()

  report_frequency = config["report_frequency"]
  eval_frequency = config["eval_frequency"]

  model = util.get_model(config)
  saver = tf.train.Saver(max_to_keep=5)         # 仅保存一个checkpoint文件

  log_dir = config["log_dir"]
  max_steps = config['num_epochs'] * config['num_docs']
  writer = tf.summary.FileWriter(log_dir, flush_secs=20)

  max_f1 = 0
  max_test_f1 = 0
  mode = 'w'

  # session_config = tf.ConfigProto(
  #     allow_soft_placement=True,
  #     log_device_placement=True
  # )
  # session_config.gpu_options.allow_growth = True
  with tf.Session() as session:

    session.run(tf.global_variables_initializer())
    model.start_enqueue_thread(session)
    accumulated_loss = 0.0

    ckpt = tf.train.get_checkpoint_state(log_dir)
    if ckpt and ckpt.model_checkpoint_path:
      print("Restoring from: {}".format(ckpt.model_checkpoint_path))
      print(
        type(ckpt.model_checkpoint_path)
      )
      saver.restore(session, ckpt.model_checkpoint_path)
      mode = 'a'
    fh = logging.FileHandler(os.path.join(log_dir, 'stdout.log'), mode=mode)
    fh.setFormatter(logging.Formatter(format))
    logger.addHandler(fh)

    test_flag = 1
    logger.info(pyhocon.HOCONConverter.convert(config, "hocon"))
    initial_time = time.time()
    while True:
        # tf.Session.run()函数返回值为fetches的执行结果。如果fetches是一个元素就返回一个值；
        # 若fetches是一个list，则返回list的值，若fetches是一个字典类型，则返回和fetches同keys的字典。
        
        tf_loss, tf_global_step, _ = session.run([model.loss, model.global_step, model.train_op])
        accumulated_loss += tf_loss
        
        if tf_global_step % report_frequency == 0:
            total_time = time.time() - initial_time
            steps_per_second = tf_global_step / total_time

            average_loss = accumulated_loss / report_frequency
            logger.info("[{}] loss={:.2f}, steps/s={:.2f}".format(tf_global_step, average_loss, steps_per_second))
            writer.add_summary(util.make_summary({"loss": average_loss}), tf_global_step)
            accumulated_loss = 0.0

        if tf_global_step > 0 and tf_global_step % eval_frequency == 0:
            test_flag = -(test_flag)
            eval_summary, eval_f1, eval_p, eval_r, eval_men_r = model.evaluate(session, tf_global_step, mod="eval")
            if eval_f1 > max_f1:
                max_f1 = eval_f1
                saver.save(session, os.path.join(log_dir, "model"), global_step=tf_global_step)
                util.copy_checkpoint(os.path.join(log_dir, "model-{}".format(tf_global_step)), os.path.join(log_dir, "model.max.ckpt"))
            writer.add_summary(eval_summary, tf_global_step)
            writer.add_summary(util.make_summary({"max_eval_f1": max_f1}), tf_global_step)
            logger.info("[{}] evaL_f1={:.4f}, evaL_p={:.4f}, evaL_r={:.4f}, evaL_men_r={:.4f}, max_f1={:.4f}"
                        .format(tf_global_step, eval_f1, eval_p, eval_r, eval_men_r, max_f1))
            if  test_flag > 0:
                test_summary, test_f1, test_p, test_r, test_men_r = model.evaluate(session, tf_global_step, mod="test")
                if test_f1 > max_test_f1:
                    max_test_f1 = test_f1
                writer.add_summary(test_summary, tf_global_step)
                writer.add_summary(util.make_summary({"max_test_f1": max_test_f1}),tf_global_step)
                logger.info("[{}] test_f1={:.4f}, test_p={:.4f}, test_r={:.4f}, test_men_r={:.4f}, max_test_f1={:.4f}"
                            .format(tf_global_step, test_f1, test_p, test_r, test_men_r, max_test_f1))
            if tf_global_step > max_steps:
                logger.info("训练结束")
                break
