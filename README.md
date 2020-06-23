# 参考

## tensolflow.js
2次元データ、画像文字データ認識、音声認識があるけど、2次元だけインプット。本当はN次元が欲しい。
https://qiita.com/MATS-Electric-Blue-Industries/items/97f40d43b131f9db1dfb

### sample1
二次元データの予測モデル。

https://qiita.com/MATS-Electric-Blue-Industries/items/1d43bcab76f1d06c81e9



#　よくわからんところ

- createModelの定義のしかた。
- テンソル（=スカラー）（2次元ベクトルでいう絶対値？）N×1の行列
- ユニット
- 正規化の意味＝＞最小値が0最大値が1となるようにし、全てのデータが0or1に治るようにする。

## モデルの学習
- バッチサイズとエポックの適切な値がわからない。ー＞エポックは学習回数。学習するくくり




# こんな感じ？
tfvisが描画用のクラス。
tsはテンソルフロウのオブジェクト


### sample2
N次元データの予測モデル
https://note.com/o_matsuo/n/n75b7e2c9774d




# 仮に採用で使うとしたら
2次元の組み合わせでできるか？
https://avinton.com/blog/2019/07/tensorflow-js/

sudo /Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --args -allow-file-access-from-files