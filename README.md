# 🎬 映画推薦システム開発プロジェクト（Movie Recommendation System）


# 1. プロジェクト概要

## 1.1 プロジェクトの目的

本プロジェクトの目的は、**ユーザの視聴行動データと映画メタデータを用いて、個人に最適化された映画推薦システムを開発すること**である。

近年のストリーミングサービスでは、非常に多くの映画・映像コンテンツが提供されており、ユーザが自分の好みに合った作品を効率的に発見することが難しくなっている。そのため、ユーザの嗜好に応じて適切な作品を提示できる推薦システムの重要性が高まっている。

本プロジェクトでは、単純な評価履歴に基づく推薦ではなく、ユーザの**実際の行動履歴（クリック・視聴・いいね等）**を活用し、時間とともに変化する嗜好を考慮した映画推薦を実現することを目的とする。



## 1.2 採用するアプローチ

本プロジェクトでは、以下の手法を組み合わせた**ハイブリッド型映画推薦システム**を構築する。

* **協調フィルタリング（SVDモデル）**
  　ユーザと映画のインタラクション行列を低ランク分解し、潜在的な嗜好特徴を学習する。

* **コンテンツベース推薦（ジャンル・キーワード分析）**
  　映画が持つジャンル情報や特徴を用いて、ユーザの興味に近い作品を推薦する。

* **ユーザ行動の時系列変動の考慮**
  　Recency Weight や Trend Detection を導入し、最近の行動ほど強く反映されるよう設計する。

* **AIエージェント（LLM）の導入**
  　自然言語による映画検索を可能にし、ユーザが文章で映画を探せるインターフェースを試作する。

これらの手法を統合することで、精度・柔軟性・ユーザ体験を兼ね備えた推薦システムの実現を目指す。


# 2. プロジェクト計画（概要）

本プロジェクトは、以下の5つの段階に分けて進める。

1. データ収集
2. 推薦モデルの設計
3. パーソナライズおよび適応ロジックの導入
4. ユーザインターフェースとWebアプリ化
5. AIエージェント（LLM）の導入

※ 各段階の詳細は、次節以降で順に説明する。


# 2. プロジェクト計画（詳細）

## 2.1 データ収集

本プロジェクトでは、映画推薦システムを構築するために、**映画メタデータ**および**ユーザ行動ログ**を収集・整備する。

### 2.1.1 映画データセット

映画データには、Kaggle にて公開されている **MovieLens ベースの映画データセット**を使用する。

* データセット名：MovieLens 20M Dataset
* 配布元：Kaggle
* URL：[https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset)

このデータセットには、1940年から2015年までの **26,549 本の映画情報**が含まれており、タイトル・ジャンル・ID などの基本的なメタデータが整備されている。


### 2.1.2 ユーザ行動ログの収集

ユーザの嗜好を推定するため、本プロジェクトでは明示的な評価（★評価など）ではなく、**暗黙的フィードバック（Implicit Feedback）**を中心に扱う。

収集する主な行動は以下の通りである。

* クリック（click）
* ウォッチリストへの追加（added_to_list）
* 視聴（watched）
* いいね（liked）

これらの行動は、ユーザの興味や関心の強さを段階的に表す重要なシグナルとして利用される。



## 2.2 データ構成とファイル設計

本プロジェクトで使用する主要なデータファイル構成を、**PDF と同じ表形式**で以下に示す。

| File Path                 | Contents                                                      | Used For          | Core Fields                                 |
| ------------------------- | ------------------------------------------------------------- | ----------------- | ------------------------------------------- |
| **user_interactions.csv** | ユーザの生行動ログ（seed_like, click, watched, liked, added_to_list など） | モデル学習・ユーザプロファイル構築 | userId, movieId, action, timestamp          |
| **movie.csv**             | 映画メタデータ（1940–2015、26,549本）                                    | 推薦候補生成・検索         | movieId, title, genres, poster_url          |
| **users.csv**             | ユーザアカウント・初期嗜好                                                 | 認証・コールドスタート対応     | userId, username, password, favorite_genres |
| **recommender.pkl**       | 学習済み推薦モデル                                                     | パーソナライズ推薦提供       | SVDモデル / Popularity辞書                       |


### 補足説明

* **user_interactions.csv** は、本プロジェクトにおいて最も重要なデータであり、ユーザの興味の強さや変化を直接反映する。
* **movie.csv** は推薦および検索の候補プールとして機能する。
* **users.csv** の初期ジャンル情報は、コールドスタート問題を緩和するために用いられる。
* **recommender.pkl** は推論時にロードされ、リアルタイム推薦を可能にする。


### 2.2.2 movie.csv

映画のメタデータを管理する CSV ファイルであり、推薦候補の生成および検索機能に利用される。

**主なカラム：**

* movieId：映画ID
* title：映画タイトル
* genres：ジャンル（複数ジャンル可）
* poster_url：ポスター画像URL

本ファイルには、26,549件の映画データが含まれている。



### 2.2.3 users.csv

ユーザアカウント情報および初期嗜好を管理する CSV ファイルである。

**主なカラム：**

* userId：ユーザID
* username：ユーザ名
* password：パスワード
* favorite_genres：サインアップ時に選択した好みのジャンル

この初期ジャンル情報は、コールドスタート対策として利用される。



### 2.2.4 recommender.pkl

学習済みの推薦モデルをシリアライズしたファイルであり、推論時にロードされる。

* Surprise SVD モデル
* もしくは Popularity ベースのフォールバックモデル



## 2.3 推薦モデルの設計（導入）

本プロジェクトでは、推薦モデルの中核として **協調フィルタリング（Collaborative Filtering）** を採用する。

特に、Python ライブラリ **Surprise** に実装されている **SVD（Singular Value Decomposition）モデル**を用いて学習を行う。

SVD は、ユーザ × 映画のインタラクション行列を、

* ユーザの潜在特徴行列 P
* 映画の潜在特徴行列 Q

に分解することで、未視聴映画に対する評価値を予測する手法である。

※ 次節では、暗黙的フィードバックの数値化方法および SVD 学習プロセスについて詳しく説明する。

## 2.3.A データ変換（暗黙的フィードバック → 数値評価）

本プロジェクトでは、明示的な評価（★評価など）ではなく、**暗黙的フィードバック（Implicit Feedback）**を用いてユーザの嗜好を数値化する。

アクションと数値評価の対応表

PDF と同様に、ユーザ行動を段階的な関係強度として数値にマッピングする。

Level	Action	Meaning	Score

1	CLICKED	興味を示した（"I’m curious about this"）	2.0

2	ADDED TO LIST	後で見たい（"I want to watch this later"）	3.0

3	WATCHED	視聴した（"I committed time to this"）	4.0

4	LIKED	非常に気に入った（"I loved this movie!"）	5.0

この構造は、ユーザと映画の関係が段階的に強まっていくエスカレーションモデルとして設計されている。

実世界の行動例（Inception のケース）

ユーザが映画「インセプション」に対して行動を重ねていく過程を以下に示す。

Day	User Action	System Interpretation

Day 1	推薦一覧で映画を見てクリック	Sci-Fi に興味あり

Day 3	再度表示され、ウォッチリストに追加	強い関心がある

Day 7	実際に視聴	視聴意欲が行動に反映

Day 7	視聴後に「いいね」	マインドベンディング系 Sci-Fi が好み

このように、単一の行動ではなく行動の積み重ねによって、ユーザ嗜好がより正確に学習される。

Surprise データセットへの変換

数値化された行動ログは、以下の形式の DataFrame に変換される。

userId	movieId	rating

この DataFrame は、Surprise ライブラリが要求する surprise.Dataset 形式へ変換され、SVD モデルの学習に使用される。

## 🤝 協調フィルタリング（SVD）

Surprise ライブラリを使用

ユーザ×映画行列を低ランク分解

潜在特徴 P（ユーザ）・Q（映画）を学習

<img width="83" height="32" alt="image" src="https://github.com/user-attachments/assets/c398c84a-2438-400c-9189-0533f29381b0" />


予測式：

<img width="224" height="34" alt="image" src="https://github.com/user-attachments/assets/1055e449-e900-4b81-8958-2f2068de8e00" />

## ⏳ Recency Weight（時間減衰）

半減期：14日

古い行動ほど影響を弱める

例：

30日前の「いいね」 < 昨日の「視聴」

## 🎭 ジャンルスコア算出

各映画の重みを全ジャンルに加算

正規化してジャンル嗜好分布を作成

## 🧊 サインアップ嗜好の減衰（Prior Decay）

初期登録ジャンルに +0.3 のブースト

行動数が増えるにつれて段階的に減衰

行動回数	ブースト
0	+0.30
3	+0.24
5	+0.20
10	+0.10
15	0.00
📈 トレンド検出（Trend Detection）

上昇ジャンル：最近 − 全体 > 8%

冷却ジャンル：全体 − 最近 > 10%

## ⚠️ 飽和ペナルティ（Saturation Penalty）

閾値：ジャンルシェア 45%

非線形ペナルティ：excess^1.25

## 🌐 Web アプリ構成
### バックエンド

Flask API

/recommend /action /genres /movie/search

### フロントエンド

React

### ダークテーマ

Watchlist

## 🤖 AI エージェント

### OpenAI API

自然言語検索

summary / trailer / 類似映画

## 📊 進捗状況

現時点で以下が完了している。


### ✔ データ構造

•	movies.csv, user_interactions.csv, users.csv の整備

•	ジャンル正規化処理（genre_set 生成）

✔ 行動データの取得と学習データ作成

•	clicked / watched / liked などのログ記録

•	各操作を重み付けし 暗黙的評価（implicit rating） に変換済み



### ✔ 推薦モデルの実装

•	Surprise SVD による Collaborative Filtering

•	fallback として Popularity ベースモデル も実装

•	Recency・Trend・Saturation を組み込んだ ハイブリッドスコアリング







<img width="467" height="197" alt="image" src="https://github.com/user-attachments/assets/23f9709a-95ef-4be4-aa37-8d0464f842f0" />

or

<img width="462" height="73" alt="image" src="https://github.com/user-attachments/assets/a8333ac2-dfb3-4f62-9d8b-8ea438fe720f" />

### ✔ Web API の構築
•	推薦エンドポイント /recommend/<userId>
•	行動記録 /action
•	ジャンル一覧 /genres
•	映画検索 /movie/search

<img width="1000" height="800" alt="image" src="https://github.com/user-attachments/assets/310d236d-57ad-4111-87fa-a16fb3302c21" />

•	Home page

<img width="1000" height="800"  alt="image" src="https://github.com/user-attachments/assets/fb8a56e5-28bc-4961-9796-cdd779b402f6" />


•Watch List

<img width="1000" height="800" alt="image" src="https://github.com/user-attachments/assets/9043a61e-6f3b-4eed-86a4-a66b8d856d37" />

### ✔ AI エージェントの実装（試作）
•	LLM による映画特定
•	映画説明（summary）と trailer の生成
•	キーワード解析
•	データセット照合と類似作品の抽出

<img width="303" height="429" alt="image" src="https://github.com/user-attachments/assets/97d5e3b8-8c14-47b1-bf96-b5999841c8be" /><img width="360" height="427" alt="image" src="https://github.com/user-attachments/assets/7e05dbd6-3bf4-44a4-8691-2365587cc5da" />

<img width="368" height="444" alt="image" src="https://github.com/user-attachments/assets/b657cf4d-761a-4bf2-b59a-0344096b9933" /><img width="371" height="445" alt="image" src="https://github.com/user-attachments/assets/eeb3e5e6-3d3c-4ada-8ada-a10252a07290" />




