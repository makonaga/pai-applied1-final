# 切った食材を鍋に入れるロボットの模倣学習（Physical AI 応用1講座 最終課題）

切った食材の小片を SO-101 アームで掴んで鍋に入れるタスクを、MuJoCo のシミュレーション上で実現する。
学習データは自作のお手本プログラム（スクリプテッド・エキスパート）で自動生成し、
LeRobot の模倣学習で ACT と VLA（SmolVLA）を学習・評価する。ポリシーの入力はカメラ画像と関節角度だけで、
物体の真の位置は使わない。

- ロボット: SO-101（[so101-nexus](https://github.com/johnsutor/so101-nexus) の MuJoCo モデル・環境を土台に使用）
- 学習: [LeRobot](https://github.com/huggingface/lerobot)（ACT、SmolVLA）
- 実行環境: Amazon SageMaker Studio（データ生成・評価は CPU インスタンス、学習は GPU インスタンス）

## 課題と結果

難易度を 3 段階で上げた。成功率は、学習で見せていない形の片で評価した値（段階1は 50 回、段階2・3は 100 回）。

| 段階 | 課題 | ACT | SmolVLA |
|---|---|---|---|
| 1 | 立方体 1 個を鍋に入れる | 82% | ― |
| 2 | にんじんの片 5〜20 個から、ロボットに一番近い片を鍋に入れる | 56% | 40% |
| 3 | にんじん・じゃがいも・大根の片 7〜26 個から、指示文（例: `Put the potato piece closest to the robot into the pot.`）で指定した食材の一番近い片を鍋に入れる | 4%（指示文を読まない） | 1〜2% |

段階3では、SmolVLA が指示文をほとんど使わず、各食材の候補の真ん中を狙っていたことを、指示文だけを変えたときの
動きの変化と、手先を下ろした位置の分析で確かめた。片を 3〜6 個に減らしたデータで学習すると指示文を使うように
なったことから、SmolVLA が画像を 64 個のトークンにまとめて読むため、片が多いと片どうしを見分けられないことが
原因と推察（詳細はレポート参照）。

## 構成

```
├── src/so101_food/
│   ├── shapes/    # 食材片の形状生成（円柱・楕円体を平面で切った凸多面体。乱切り・角切り・丸みのある乱切り・いちょう切り）
│   ├── env/       # 鍋に入れるタスクの環境（片の並べ方・指示文・成功判定）、カメラ（俯瞰・手首）の描画
│   ├── expert/    # お手本（掴む向きの選択、逆運動学、把持 → 運搬 → 投入 → 帰還の軌道）
│   ├── datagen/   # 成功したエピソードだけを保存するデータ生成（並列、LeRobot v3 形式）、データの結合・動画化
│   ├── eval/      # 学習済みポリシーの評価（並列）、結果の集計、指示文の影響の調査、着地位置の分析、デモ
│   ├── vla/       # SmolVLA の追加学習の補助（カメラ名の対応付け）
│   ├── sim/       # グリッパの開き幅の計測
│   └── utils/     # 描画バックエンドの選択、動画の保存、並列実行
├── assets/pieces/ # 事前に作った食材片の形状セット（3 食材 × 学習用 300 個・評価用 100 個）
└── scripts/       # セットアップ・データ生成・学習・評価・分析のスクリプト
```

実行すると次のフォルダができる（Git には含めない）。

| フォルダ | 中身 |
|---|---|
| `data/` | 生成した学習データ（LeRobot v3 形式） |
| `outputs/` | 学習済みモデル（`train/`）、評価結果と動画（`eval/`）、データの動画（`dataset_videos/`）、デモ動画（`demo/`） |
| `logs/` | スクリプトのログ |

## セットアップ（SageMaker Studio の JupyterLab スペース）

```bash
git clone <このリポジトリの URL>
cd <リポジトリのフォルダ>
bash scripts/setup_sagemaker.sh
```

- 仮想環境を `~/.venvs/pai` に作り、PyTorch の CUDA 版は GPU ドライバのバージョンから自動で選ぶ。
- 描画に使う OS パッケージ（EGL / OSMesa / ffmpeg）はスペースを再起動すると消えるため、起動するたびに実行する。
- 動作を確かめたのは SageMaker Studio だけ。他の Linux では、OS パッケージ（`libegl1 libgl1 libosmesa6 ffmpeg`）を
  入れ、Python 3.12 の仮想環境に `pip install -e .` する。スクリプトは `~/.venvs/pai` の仮想環境を使うので、
  別の場所に作った場合は環境変数 `VENV_DIR` にその場所を指定する。

## 使い方

時間のかかる処理は裏で実行し、ログを `logs/` に書く（`tail -f logs/<ファイル名>` で進み具合を見る）。
データ生成と評価は CPU のプロセスを並列に使うので、vCPU の多いインスタンスが向いている。

| スクリプト | 内容 |
|---|---|
| `scripts/generate_dataset.sh <名前> <本数> [--task cube/multi/lang]` | お手本で学習データを生成する（`--contrast` で同じ配置・指示文違いの手本を 3 本ずつ） |
| `scripts/merge_datasets.sh <名前> <元のデータ名...>` | 学習データを 1 つにまとめる |
| `scripts/export_dataset_videos.sh <名前> [本数]` | 学習データを動画に書き出す |
| `scripts/train_act.sh <データ名> <実行名>` | ACT を学習する（GPU） |
| `scripts/train_smolvla.sh <データ名> <実行名>` | SmolVLA を追加学習する（GPU。`TRAIN_VLM=1` で画像と言語を読む部分も学習） |
| `scripts/evaluate.sh <実行名> <回数> [--task ...]` | 学習済みポリシーを評価する（並列） |
| `scripts/analyze_eval.sh <評価名...>` | 評価結果を掴む片までの距離・片の数ごとに比べる |
| `scripts/probe_language.sh <実行名> <データ名>` | 指示文の食材だけを変えたとき、動きがどれだけ変わるかを調べる |
| `scripts/analyze_landing.sh <評価名...>` | 評価で手先を下ろした位置が、どの片に一番近かったかを調べる |
| `scripts/demo_language.sh <実行名> [配置の数]` | 同じ配置で指示文の食材だけを変えて動かすデモ動画を作る |

### 段階1・2

```bash
# 段階1: 立方体 1 個
bash scripts/generate_dataset.sh pot_cube_v1 300
bash scripts/train_act.sh pot_cube_v1 act_pot_cube_v1
bash scripts/evaluate.sh act_pot_cube_v1 50

# 段階2: にんじんの片 5〜20 個から一番近い片
bash scripts/generate_dataset.sh pot_multi_v1 1000 --task multi
bash scripts/train_act.sh pot_multi_v1 act_pot_multi_v1 --steps=80000
bash scripts/train_smolvla.sh pot_multi_v1 smolvla_pot_multi_v1
bash scripts/evaluate.sh act_pot_multi_v1 100 --task multi
bash scripts/evaluate.sh smolvla_pot_multi_v1 100 --task multi
bash scripts/analyze_eval.sh act_pot_multi_v1 smolvla_pot_multi_v1 --dataset pot_multi_v1
```

### 段階3

```bash
# 学習データ（片 7〜26 個、3 食材、指示文つき）と学習・評価
bash scripts/generate_dataset.sh pot_lang_v1 1500 --task lang
bash scripts/train_smolvla.sh pot_lang_v1 smolvla_pot_lang_v1
bash scripts/train_act.sh pot_lang_v1 act_pot_lang_v1 --steps=80000
bash scripts/evaluate.sh smolvla_pot_lang_v1 100 --task lang
bash scripts/evaluate.sh act_pot_lang_v1 100 --task lang

# 指示文を使っているかの調査と、手先を下ろした位置の分析
bash scripts/probe_language.sh smolvla_pot_lang_v1 pot_lang_v1
bash scripts/analyze_landing.sh smolvla_pot_lang_v1 act_pot_lang_v1

# 原因を確かめるためのデータ
bash scripts/generate_dataset.sh pot_lang_contrast 3000 --task lang --contrast --seed 40000  # 同じ配置で指示文違いの 3 本組
bash scripts/generate_dataset.sh pot_lang_v1b 3000 --task lang --seed 20000
bash scripts/merge_datasets.sh pot_lang_v2 pot_lang_v1 pot_lang_v1b                         # データ 3 倍
bash scripts/merge_datasets.sh pot_lang_v1c pot_lang_v1 pot_lang_contrast
bash scripts/generate_dataset.sh pot_lang_easy 1500 --task lang --min-pieces 3 --max-pieces 6 # 片 3〜6 個
# それぞれ train_smolvla.sh で学習し、evaluate.sh・probe_language.sh・analyze_landing.sh で調べる
# （片 3〜6 個のモデルは evaluate.sh に --min-pieces 3 --max-pieces 6 を付ける）

# 同じ配置で指示文の食材だけを変えるデモ動画
bash scripts/demo_language.sh smolvla_pot_lang_v1 3
```

## 利用しているソフトウェア

- [so101-nexus](https://github.com/johnsutor/so101-nexus)（Apache-2.0）: SO-101 の MJCF、MuJoCo 環境
- [LeRobot](https://github.com/huggingface/lerobot)（Apache-2.0）: データセット形式、ACT・SmolVLA の学習
- [SmolVLA](https://huggingface.co/lerobot/smolvla_base)（Apache-2.0）: 事前学習済みの VLA
- [MuJoCo](https://github.com/google-deepmind/mujoco)（Apache-2.0）
