一、项目环境
操作系统：Ubuntu 16.04及以上
Python版本：3.6
内存大小：16G及以上
GPU显存:11G
CUDA Version: 10.1
DUDNN Version：7.5.1
Driver Version: 418.74       

二、操作步骤
(1)创建一个空的python环境

(2)安装项目依赖：
 pip install -r requirements.txt

(3)定义环境变量：
 export data_dir=</path/to/data_dir>
 例如：export data_dir=....../coref-mid-master/conll-2012

(4)项目编译
 ./setup_all.sh

三、项目结构
|
|--config 模型配置文件
|--conll-2012指代消解官方数据集存放位置（需要自己下载并预处理）
|--result-of-tagging 模型保存路径
  
|--minimize.py 数据生成
|--independent.py 模型结构
|--train.py 模型训练
 
备注：项目代码着重参考于 《BERT for Coreference Resolution Baselines and Analysis》,可以先看他们的代码。

三、项目训练
(1)超参数
你可以训练和微调自己的bert模型，调整几个config当中的超参数：
 max_seq_length, max_training_sentences, ffnn_size, model_heads = false

(2)数据生成
可以使用minimize.py用于生成训练数据：
 python minimize.py ./conll-2012/char_vocab.chinese.txt inputfile_path outputfile_path false
 Train和dev数据已经生成，保存在conll-2012/tagging_data目录下

(3)模型训练
 每次训练前，修改配置文件config当中的save_path，调整模型保存路径

随后终端运行：
 Python  train.py  bert_base

注：bertbase为11g GPU能够运行的模型版本，如果硬件允许，可尝试bertlarge在至少16gGPU上进行训练

(4) 项目预测
Config文件中save_path选择所需模型
 Python  predict.py  bert_base  inputfile  outputfile
 Inputfile数据格式与train、dev保持一致, 样例参考predict_result_demo.txt

(5)predict结果处理
 Python  after_predict.py  inputfile  outputfile
生成结果参考after_predict_demo.txt，方便查看预测结果
