### Setup for training
This assumes access to OntoNotes 5.0.
`./setup_training.sh <ontonotes/path/ontonotes-release-5.0> $data_dir`. This preprocesses the OntoNotes corpus, and downloads the original (not finetuned on OntoNotes) BERT models which will be finetuned using `train.py`. 

* Experiment configurations are found in `experiments.conf`. Choose an experiment that you would like to run, e.g. `bert_base`
* Note that configs without the prefix `train_` load checkpoints already tuned on OntoNotes.
* Training: `GPU=0 python train.py <experiment>`
* Results are stored in the `log_root` directory (see `experiments.conf`) and can be viewed via TensorBoard.
* Evaluation: `GPU=0 python evaluate.py <experiment>`. This currently evaluates on the dev set.


## Batched Prediction Instructions

* Create a file where each line similar to `cased_config_vocab/trial.jsonlines` (make sure to strip the newlines so each line is well-formed json):
```
{
  "clusters": [], # leave this blank
  "doc_key": "nw", # key closest to your domain. "nw" is newswire. See the OntoNotes documentation.
  "sentences": [["[CLS]", "subword1", "##subword1", ".", "[SEP]"]], # list of BERT tokenized segments. Each segment should be less than the max_segment_len in your config
  "speakers": [["[SPL]", "-", "-", "-", "[SPL]"]], # speaker information for each subword in sentences
  "sentence_map": [0, 0, 0, 0, 0], # flat list where each element is the sentence index of the subwords
  "subtoken_map": [0, 0, 0, 1, 2]  # flat list containing original word index for each subword. [CLS]  and the first word share the same index
}
```
  * `clusters` should be left empty and is only used for evaluation purposes.
  * `doc_key` indicates the genre, which can be one of the following: `"bc", "bn", "mz", "nw", "pt", "tc", "wb"`
  * `speakers` indicates the speaker of each word. These can be all empty strings if there is only one known speaker.
* Run `GPU=0 python predict.py <experiment> <input_file> <output_file>`, which outputs the input jsonlines with an additional key `predicted_clusters`.

## Notes
* The current config runs the Independent model.
* When running on test, change the `eval_path` and `conll_eval_path` from dev to test.
* The `model_dir` inside the `log_root` contains `stdout.log`. Check the `max_f1` after 57000 steps. For example
``
2019-06-12 12:43:11,926 - INFO - __main__ - [57000] evaL_f1=0.7694, max_f1=0.7697
``
* You can also load pytorch based model files (ending in `.pt`) which share BERT's architecture. See `pytorch_to_tf.py` for details.

### Important Config Keys
* `log_root`: This is where all models and logs are stored. Check this before running anything.
* `bert_learning_rate`: The learning rate for the BERT parameters. Typically, `1e-5` and `2e-5` work well.
* `task_learning_rate`: The learning rate for the other parameters. Typically, LRs between `0.0001` to `0.0003` work well.
* `init_checkpoint`: The checkpoint file from which BERT parameters are initialized. Both TF and Pytorch checkpoints work as long as they use the same BERT architecture. Use `*ckpt` files for TF and `*pt` for Pytorch.
* `max_segment_len`: The maximum size of the BERT context window. Larger segments work better for SpanBERT while BERT suffers a sharp drop at 512.

