# aMLP-japanese

Japanese aMLP Pretrained Model --- [Japanese](README.md) / English

aMLP is a Transformer model proposed by [Liu, Dai et al.](Https://arxiv.org/abs/2105.08050).

Roughly speaking, it is a model with better performance that can be used instead of BERT.



# aMLP

***<font color='red'>New</font>***

- 2022/5/1 - The sentence summary model for Japanese has been released.
- 2022/5/1 - Large model for Japanese has been released
- 2022/5/1 - Model conversion and SQuAD executable code for PyTorch have been released.
- 2021/11/13 - Pre-trained base model and SQuAD model have been released



## Public Models

- Japanese Pre-Training Models

| Model Name   | Download URL                                                 | Parameter Number |
| ------------ | ------------------------------------------------------------ | ---------------- |
| aMLP-base-ja | [https://nama.ne.jp/models/aMLP-base-ja.tar.bz2](https://nama.ne.jp/models/aMLP-base-ja.tar.bz2) （[Preliminary](https://s3.ap-northeast-1.amazonaws.com/ailab.nama.ne.jp/models/aMLP-base-ja.tar.bz2)） | 67,923,648     |
| aMLP-large-ja | [https://nama.ne.jp/models/aMLP-large-ja.tar.bz2](https://nama.ne.jp/models/aMLP-large-ja.tar.bz2) （[Preliminary](https://s3.ap-northeast-1.amazonaws.com/ailab.nama.ne.jp/models/aMLP-large-ja.tar.bz2)） | 182,308,032     |

- Japanese SQuAD Models

| Model Name         | Download URL                                                 | Parameter Number | Training Data Size |
| ------------------ | ------------------------------------------------------------ | -------------- | ---------------- |
| aMLP-SQuAD-base-ja | [https://nama.ne.jp/models/aMLP-SQuAD-base-ja.bz2](https://nama.ne.jp/models/aMLP-SQuAD-base-ja.tar.bz2)（[Preliminary](https://s3.ap-northeast-1.amazonaws.com/ailab.nama.ne.jp/models/aMLP-SQuAD-base-ja.tar.bz2)） | 67,924,674     | 200K articles         |
| aMLP-SQuAD-large-ja | [https://nama.ne.jp/models/aMLP-SQuAD-large-ja.bz2](https://nama.ne.jp/models/aMLP-SQuAD-large-ja.tar.bz2)（[Preliminary](https://s3.ap-northeast-1.amazonaws.com/ailab.nama.ne.jp/models/aMLP-SQuAD-large-ja.tar.bz2)） | 182,309,570    | 200K articles         |

- Japanese Text Summarization Models

| Model Name         | Download URL                                                 | Parameter Number | Training Data Size |
| ------------------ | ------------------------------------------------------------ | -------------- | ---------------- |
| aMLP-Summalizer-large-ja | [https://nama.ne.jp/models/aMLP-Summalizer-large-ja.tar.bz2](https://nama.ne.jp/models/aMLP-Summalizer-large-ja.tar.bz2)（[Preliminary](https://s3.ap-northeast-1.amazonaws.com/ailab.nama.ne.jp/models/aMLP-Summalizer-large-ja.tar.bz2)） | 182,309,570    | 200K articles         |


# SQuAD Model

Model Structure:

![squad](squad.png)



## How to use



Clone source from GitHub.

```sh
$ git https://github.com/tanreinama/aMLP-japanese
$ cd aMLP-japanese
```

Download and extract pre-trained model.

```sh
$ wget https://www.nama.ne.jp/models/aMLP-SQuAD-base-ja.tar.bz2
$ tar xvfj aMLP-SQuAD-base-ja.tar.bz2
```

Run "run-squad.py" as below

If you specify the trained model as "--restore_from" and the JSON file in SQuAD format as "--pred_dataset", the answer to the question will be displayed.

All answer suggestions are saved as "squad-predicted.json"

```sh
$ python run-squad.py --restore_from aMLP-SQuAD-base-ja --pred_dataset squad-testdata.json
Question        Answer
ロッキード・マーティン社とボーイング社が共同開発したステルス戦闘機は？  F-22戦闘機
F-22戦闘機の愛称は？    猛禽類の意味のラプター
F-22戦闘機一機あたりの価格は？  1億5千万ドル
F-22戦闘機の航続距離は？        3200km
F-22戦闘機の巡航速度は？        マッハ1.82
F-22の生産数が削減された理由は？        調達コスト
```

Since it is a SQuAD type Q & A model, the JSON file must contain the context.



## Finetune

Prepare a Q & A dataset for fine tuning and save it as a JSON file in SQuAD format.

Then run "run-squad.py" as below

Specify the trained model as "--restore_from" and the JSON file in SQuAD format as "--dataset".

If you have a dataset for evaluation, specify it with "--val_dataset" and the evaluation score will be displayed in the middle of learning.

```sh
$ python run-squad.py --restore_from aMLP-SQuAD-base-ja --dataset squad-testdata.json
```

If you want to train from scratch, specify the pretrained model as "--base_model"

The trained model is saved in the directory specified by "--run_name" under "checkpoint".

The question and answer data set used for learning the public model cannot be published due to copyright issues.



# Extractive Text Summarization

Model Structure:

![summarize](summarize.png)



## How To Use

The text summarization model is created in the same format as the SQuAD task.

Download and extract the trained model file.

```sh
$ wget https://www.nama.ne.jp/models/aMLP-SQuAD-large-ja.tar.bz2
$ tar xvfj aMLP-SQuAD-large-ja.tar.bz2
```

Put the text you want to summarize in the Context in the JSON file in SQuAD format, and leave the question text blank (empty string).

Similar to the Q & A model, running in "run-squad.py" will display a summary instead of the answer.

```sh
$ python run-squad.py --restore_from aMLP-Summalizer-large-ja --pred_dataset summalize-testdata.json --verbose
[Context]
東京株式市場において日経平均株価が値上がりし、3万670円10銭の値で終えた。株高の背景には新型コロナウイルス感染拡大の終息と景気回復への期待感があり、今後は企業業績の回復が焦点になる。日経平均株価が3万円の大台を回復するのは約30年半ぶり。関係者 には過熱感を警戒する見方もあり、しばらくは国内外の感染状況を見ながらの取り引きが続きそう。トピックスも21円16銭値上がりし、2118円87銭で終える。出来高は13億3901万株。
[Answer]
日経平均株価が3万円の大台を回復するのは約30年半ぶり
[Context]
リーガ・エスパニョーラのレガネスはバジャドリードと対戦。23分、エリア内でバジャドリードのハンドによりPKを獲得するも惜しくも外れる。その後の30分にはオスカル・ロドリゲスが先制点を挙げる。1点ビハインドのバジャドリードは49分、エネス・ウナルがゴ ールを決めるがオフサイドの判定でゴールは取り消された。試合はそのままレガネスが1対0で逃げ切る。
[Answer]
試合はそのままレガネスが1対0で逃げ切る
```



# Classification

Model Structure:

![classifier](classificate.png)



## Prepare



Clone source from GitHub.

```sh
$ git https://github.com/tanreinama/aMLP-japanese
$ cd aMLP-japanese
```

Download and extract pre-trained model.

```sh
$ wget https://www.nama.ne.jp/models/aMLP-base-ja.tar.bz2
$ tar xvfj aMLP-base-ja.tar.bz2
```



## Training for classification



In classification task, As shown below, we will learn a model that classifies text files by class on the assumption that the text file is saved in the form of "class name / file".

```
dir/<classA>/textA.txt
dir/<classA>/textB.txt
dir/<classB>/textC.txt
・・・
```


Here is an example of using [livedoor news corpus] (http://www.rondhuit.com/download.html#ldcc) as a sample.

```sh
$ wget https://www.rondhuit.com/download/ldcc-20140209.tar.gz
$ tar xvfz ldcc-20140209.tar.gz
$ ls text/
```

To learn, run "run-classifier.py" as follows

Specify the pretrained model with "--base_model" and the dataset directory with "--dataset"

If you have a dataset for evaluation, specify it with "--val_dataset" and the evaluation score will be displayed in the middle of learning.

```sh
$ python run-classifier.py --dataset text --model aMLP-base-ja --clean_text
```

After the correspondence between the subdirectory name and the class ID is displayed as shown below, learning proceeds.

```sh
Loading dataset...
livedoor-homme mapped index#0
kaden-channel mapped index#1
movie-enter mapped index#2
it-life-hack mapped index#3
topic-news mapped index#4
sports-watch mapped index#5
dokujo-tsushin mapped index#6
peachy mapped index#7
smax mapped index#8
```

The trained model is saved in the directory specified by "--run_name" under "checkpoint".



## Predict for classification

For prediction, run "run-classifier.py" as follows

Specify the trained model as "--restore_from", the dataset directory as "--pred_dataset", and the output file as "--output".

```sh
$ python run-classifier.py --pred_dataset text --output classifier-pred.csv --restore_from checkpoint/aMLP-classifier-ja/checkpoint-XXXX
$ head -n5 classifier-pred.csv
filename,pred,true
text/livedoor-homme/livedoor-homme-4956491.txt,livedoor-homme,livedoor-homme
text/livedoor-homme/livedoor-homme-5492081.txt,livedoor-homme,livedoor-homme
text/livedoor-homme/livedoor-homme-5818455.txt,livedoor-homme,livedoor-homme
text/livedoor-homme/livedoor-homme-6052744.txt,livedoor-homme,livedoor-homme
```

The execution result is saved as a CSV file.



# Run Masked Language Model

Model Structure:

![mlm](mlm.png)

Run as a Masked Language Model. The aMLP model predicts the "[MASK]" part of the entered text

You can solve the MLM with "run-mlm.py".

Since one "[MASK]" is one BPE after encoding, "[MASK]" will be changed from one Japanese character to three Japanese characters.

```sh
$ python run-mlm.py --context "俺の名前は坂本[MASK]。何処にでもいるサラリー[MASK]だ。" --model aMLP-base-ja
俺の名前は坂本だ。何処にでもいるサラリーマンだ。
```



# Text vectorization

Model Structure:

![vectorize](vectorize.png)

Get the vector representation corresponding to the [CLS]token. If "--output" is specified, the file will be saved as a comma-separated text.

```sh
$ python run-vectrize.py --context "こんにちは、世界。" --model aMLP-base-ja
[1.777146577835083, 0.5332596898078918, 0.07858406007289886, 0.5532811880111694, 0.8075544238090515, 1.3260560035705566, 0.6111544370651245, 2.338435173034668, 1.0313552618026733, ・・・
```



# PyTorch Version



The PyTorch version model is created by converting the TensorFlow version model.

Currently, I have a transform program for the trained model and the execution code for the SQuAD task.

Model conversion program is "tfmodel2torch.py".

```sh
$ cd pytorch
$ python tfmodel2torch.py --tf_model_dir ../aMLP-SQuAD-base-ja --output aMLP-SQuAD-base-ja.pt
has_voc: True is_classifier: False is_squad: True
$ ls aMLP-SQuAD-base-ja.*
aMLP-SQuAD-base-ja.json  aMLP-SQuAD-base-ja.pt  aMLP-SQuAD-base-ja.txt
```

The SQuAD task is executed in "test-squad.py".

All answer suggestions are saved as "squad-predicted.json"

```sh
$ python test-squad.py --pred_dataset ../squad-testdata.json --model aMLP-SQuAD-base-ja.pt
Question        Answer
ロッキード・マーティン社とボーイング社が共同開発したステルス戦闘機は？  F-22戦闘機
F-22戦闘機の愛称は？    猛禽類の意味のラプター
F-22戦闘機一機あたりの価格は？  1億5千万ドル
F-22戦闘機の航続距離は？        3200km
F-22戦闘機の巡航速度は？        マッハ1.82
F-22の生産数が削減された理由は？        調達コスト
```



# REFERENCE

[Hanxiao Liu, Zihang Dai, David R. So, Quoc V. Le "Pay Attention to MLPs"  arXiv:2105.08050, 17 May 2021](https://arxiv.org/abs/2105.08050)
