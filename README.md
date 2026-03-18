# graduate_research_original

train.py...パラメータの設定、エポック管理を行っていて、学習の根幹を記述。

test_twtransnet.py...実データによる統合テストを行う。

data_loader.py...元のデータセットから各ユーザごとの移動履歴を作成し、leave-one-out法に基づいてtrain-test-valデータに分割する。

graph_utils.py...各訪問履歴をセットで格納する。後のユーザ-POIグラフの学習で用いる。

model_components.py...POI軌跡グラフ、ユーザ-POIグラフ、近傍集約などの詳細を記述。

twtransnet.py...train.pyの学習中に各関数が呼び出され、model_components.pyの各関数を呼び出す。(train.pyの下流、model_components.pyの上流に相当)

utils.py...評価指標の計算などを記述。
