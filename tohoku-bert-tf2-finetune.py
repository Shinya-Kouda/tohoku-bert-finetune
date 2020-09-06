# -*- coding: utf-8 -*-



#%%
""" 0.準備
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import glob
import re
import MeCab
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import TFBertModel
from transformers import BertJapaneseTokenizer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers, losses, metrics
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
# fine-tuning用のconfig
epochs = 1
max_length = 500
train_batch = 32
val_batch = 64
num_classes = 9

#%%
# GPUの確認
print(tf.__version__)
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))



#%%
""" 1.転移学習用データのロード
"""
# livedoorの記事のカテゴリ名
categories = [
    "sports-watch", "topic-news", "dokujo-tsushin", "peachy",
    "movie-enter", "kaden-channel", "livedoor-homme", "smax",
    "it-life-hack",
]
# docsはタプル(category, url, date, title, body)のリスト
docs = []
for category in categories:
    for f in glob.glob(f"./text/{category}/{category}*.txt"):
        # 1ファイルごとに処理
        with open(f, "r") as fin:
            # nextで1行取得する(__next__を呼ぶ)
            url = next(fin).strip()
            date = next(fin).strip()
            title = next(fin).strip()
            body = "\n".join([line.strip() for line in fin if line.strip()])
        docs.append((category, url, date, title, body))
# docsをpd.DataFrame化
df = pd.DataFrame(
        docs,
        columns=["category", "url", "date", "title", "body"],
        dtype="category"
)

#%%
# 日付は日付型に変更（今回使うわけでは無い）
df["date"] = pd.to_datetime(df["date"])
# ラベルエンコーダを作成。これは、ラベルを数値に変換する
le = LabelEncoder()
# ラベルをエンコードし、エンコード結果をyに代入する
df = df.assign(
    label=lambda df: pd.Series(le.fit_transform(df.category))
)

#%%
# datasetをtrainとvalidation用に分割する
idx = df.index.values
idx_train, idx_val = train_test_split(idx, random_state=123)
df_train = df.loc[idx_train, ['body', 'label']]
df_val = df.loc[idx_val, ['body', 'label']]

#%%
# tensorflow.Datasetを使う。カラムはtransformersのTokenizerの規定にあわせる。
data_no_wakati = {
    "train": tf.data.Dataset.from_tensor_slices({
        'sentence1': df_train['body'].tolist(),
        'sentence2': ['', ] * len(df_train),
        'label': df_train['label'].tolist()
    }),
    "validation": tf.data.Dataset.from_tensor_slices({
        'sentence1': df_val['body'].tolist(),
        'sentence2': ['', ] * len(df_val),
        'label': df_val['label'].tolist()
    })
}



#%%
""" 2.日本語BERTトークナイザーの読み込み
glue_convert_examples_to_features関数を参考 ([ソース](https://github.com/huggingface/transformers/blob/e92bcb7eb6c5b9b6ed313cc74abaab50b3dc674f/transformers/data/processors/glue.py)) に、tokenizerをtf.dataのmapに適用するための関数を定義する。

作成するdataは、
- input_ids: テキストをidに変換したもの
- token_type_ids: BERT modelの内部で使用されるもの
- attention_mask: BERT modelの内部で使用されるもの
- label: 教師信号

token_type_ids, attention_maskについては原論文を参考にしてください。
"""
#東北大学の学習済み日本語Bertのトークナイザー。
tokenizer = BertJapaneseTokenizer('./tohoku-bert/vocab.txt')

#%%
# tokenizerからアウトプットを作成する関数を作る関数
def tokenize_map_fn(tokenizer, max_length=100):
    """map function for pretrained tokenizer"""
    def _tokenize(text_a, text_b, label):
        inputs = tokenizer.encode_plus(#TODO:こいつをよく調べる
            text_a.numpy().decode('utf-8'),
            text_b.numpy().decode('utf-8'),
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1] * len(input_ids)
        return input_ids, token_type_ids, attention_mask, label
    
    def _map_fn(data):
        text_a = data['sentence1']
        text_b = data['sentence2']
        label = data['label']
        out = tf.py_function(_tokenize, inp=[text_a, text_b, label], Tout=(tf.int32, tf.int32, tf.int32, tf.int32))
        return (
            {"input_ids": out[0], "token_type_ids": out[1], "attention_mask": out[2]},
            out[3]
        )
    return _map_fn



#%%
""" 3.データのローダー、tohoku-bertとfine-tuning用のモデルを定義
"""
# ローダーの定義
## Load dataset, tokenizer, model from pretrained vocabulary
def load_dataset(data, tokenizer, max_length=128, train_batch=8, val_batch=32):
    # Prepare dataset for BERT as a tf.data.Dataset instance
    #TODO:mapってこんな使い方できるの？調べる
    train_dataset = data['train'].map(tokenize_map_fn(tokenizer, max_length=max_length))
    valid_dataset = data['validation'].map(tokenize_map_fn(tokenizer, max_length=max_length))
    train_dataset = train_dataset.shuffle(100).padded_batch(train_batch, padded_shapes=({'input_ids': max_length, 'token_type_ids': max_length, 'attention_mask': max_length}, []), drop_remainder=True)
    valid_dataset = valid_dataset.padded_batch(val_batch, padded_shapes=({'input_ids': max_length, 'token_type_ids': max_length, 'attention_mask': max_length}, []), drop_remainder=True)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    valid_dataset = valid_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return train_dataset, valid_dataset

#%%
# 日本語tohoku-bert modelをload
bert = TFBertModel.from_pretrained('./tohoku-bert/', from_pt=True)
b = bert.layers[0]

#%%
# fine-tuning用のモデルを定義
def make_model(bert, num_classes, max_length, bert_frozen=True):
    # use only first layer and froze it
    bert.layers[0].trainable = not bert_frozen
    # input
    input_ids = Input(shape=(max_length, ), dtype='int32', name='input_ids')
    attention_mask = Input(shape=(max_length, ), dtype='int32', name='attention_mask')
    token_type_ids = Input(shape=(max_length, ), dtype='int32', name='token_type_ids')
    inputs = [input_ids, attention_mask, token_type_ids]
    # bert
    x = bert.layers[0](inputs)
    # only use pooled_output, x: sequence_output, pooled_output
    out = x[1]
    # fc layer(add layers for transfer learning)
    out = Dropout(0.25)(out)
    out = Dense(128, activation='relu')(out)
    out = Dropout(0.5)(out)
    out = Dense(num_classes, activation='softmax')(out)
    return Model(inputs=inputs, outputs=out)



#%%
""" 4.データのロード、モデルのコンパイル、モデルの学習実施
"""
model = make_model(bert, num_classes , max_length)
## Load dataset, tokenizer, model from pretrained vocabulary
train_dataset, valid_dataset = load_dataset(data_no_wakati, tokenizer, max_length=max_length, train_batch=train_batch, val_batch=val_batch)
## Prepare training: Compile tf.keras model with optimizer, loss and learning rate schedule 
optimizer = optimizers.Adam()
loss = losses.SparseCategoricalCrossentropy()
metric = metrics.SparseCategoricalAccuracy('accuracy')
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
# Train and evaluate using tf.keras.Model.fit()
history = model.fit(train_dataset, epochs=epochs,
                    validation_data=valid_dataset)

# %%
