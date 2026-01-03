# ColPali Work - 画像からマルチベクトル抽出

## 📖 概要

このディレクトリには、ColPali を使って**画像からマルチベクトル（128 次元 ×N トークン または 2,048 次元 ×N トークン）を取得する**ための自己完結型プログラムが含まれています。

ColPali の公式実装から必要な部分を抽出し、`work`ディレクトリ内だけで動作するように修正しています。

---

## 🗺️ 全体の処理フロー

```
┌─────────────────────────────────────────────────────────────────────┐
│                         main.py（実行スクリプト）                      │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
         ┌──────────────────┐           ┌──────────────────┐
         │ modeling_colpali │           │ processing_colpali│
         │   .py (モデル)    │◄──────────│   .py (前処理)    │
         └──────────────────┘           └──────────────────┘
                    │                               │
                    │                               │
    ┌───────────────┼───────────────────────────────┤
    │               │                               │
    ▼               ▼                               ▼
┌────────┐  ┌──────────────┐            ┌─────────────────┐
│ 画像    │  │PaliGemma-3B  │            │ 前処理ユーティリティ│
│入力     │  │(transformers)│            │(processing_utils)│
└────────┘  └──────────────┘            └─────────────────┘
                    │
    ┌───────────────┼───────────────┐
    │               │               │
    ▼               ▼               ▼
┌─────────┐  ┌──────────┐  ┌─────────────┐
│Vision   │  │線形射影① │  │線形射影②    │
│Encoder  │→ │1152→2048 │→ │2048→128     │
│(SigLIP) │  │          │  │(custom_proj)│
└─────────┘  └──────────┘  └─────────────┘
                    │               │
                    ▼               ▼
            ┌──────────────┐  ┌──────────┐
            │2048次元       │  │128次元    │
            │×N トークン    │  │×N トークン│
            │(Gemmaの世界)  │  │(検索用)   │
            └──────────────┘  └──────────┘
                    │               │
                    └───────┬───────┘
                            ▼
                    ┌──────────────┐
                    │  Pooling     │
                    │ (Mean/Max/   │
                    │  Std/Concat) │
                    └──────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │  固定長      │
                    │  ベクトル    │
                    │ (2048 or 128)│
                    └──────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │ LightGBM/SVR │
                    │ (OCR品質予測)│
                    └──────────────┘
```

---

## 🎯 目的

先生の指示に基づき、以下を実現します：

1. **画像 → ColPali → マルチベクトル取得**

   - 128 次元 × 約 1,030 トークン（検索用に最適化）
   - 2,048 次元 × 約 1,030 トークン（Gemma の生の表現）

2. **Pooling 適用で固定長ベクトル化**

   - Mean Pooling / Max Pooling / Std Pooling
   - これらの連結（Concat）

3. **OCR 品質予測への応用**
   - 固定長ベクトル → LightGBM/SVR → OCR 精度予測

---

## 📁 ファイル構成

```
src/colpali/work/
├── README.md                    # このファイル（仕様書）
├── main.py                      # メイン実行スクリプト
├── modeling_colpali.py          # ColPaliモデル定義
├── modeling_colqwen2.py         # ColQwen2モデル定義
├── processing_colpali.py        # ColPaliプロセッサ
├── processing_colqwen2.py       # ColQwen2プロセッサ
└── processing_utils.py          # ユーティリティ関数（MaxSim等）
```

### 各ファイルの役割と関数詳細（実行時系列順）

---

## 📋 実行時の処理フロー詳細（時系列順）

### ステップ 1️⃣: `main.py` - プログラム開始

#### 🔹 関数: `main()`

**役割**: エントリーポイント。コマンドライン引数を解析して全体の処理を制御

**入力**: コマンドライン引数

- `--image`: 画像パス
- `--embedding-type`: "128dim" or "2048dim"
- `--pooling`: "none", "mean", "max", "std", "concat"

**処理**:

1. 引数をパース
2. 画像を PIL.Image として読み込み
3. モデルとプロセッサをロード
4. `get_embeddings()`を呼び出し

**出力**: なし（print で結果を表示）

---

### ステップ 2️⃣: `processing_colpali.py` - プロセッサ初期化

#### 🔹 クラス: `ColPaliProcessor`

**役割**: PaliGemmaProcessor を継承し、ColPali 用の前処理を提供

**継承元**:

- `BaseVisualRetrieverProcessor` (processing_utils.py)
- `PaliGemmaProcessor` (transformers)

**主要属性**:

```python
visual_prompt_prefix = "<image><bos>Describe the image."
```

---

### ステップ 3️⃣: `processing_colpali.py` - 画像前処理

#### 🔹 関数: `ColPaliProcessor.process_images(images)`

**役割**: PIL 画像をモデル入力用のテンソルに変換

**入力**:

- `images`: List[PIL.Image] - PIL 画像のリスト

**処理フロー**:

```python
1. 画像をRGBに変換
   images = [image.convert("RGB") for image in images]

2. テキストプロンプトと画像を結合
   text = ["<image><bos>Describe the image."] * len(images)

3. トークナイザーで処理
   batch_doc = self(
       text=text,
       images=images,
       return_tensors="pt",
       padding="longest"
   )
```

**出力**:

- `BatchFeature` 辞書
  - `input_ids`: トークン ID (batch, seq_len)
  - `attention_mask`: マスク (batch, seq_len)
  - `pixel_values`: 画像テンソル (batch, channels, height, width)

**次へ渡すもの**: この BatchFeature がモデルの入力になる

---

### ステップ 4️⃣: `modeling_colpali.py` - モデル初期化

#### 🔹 クラス: `ColPali`

**役割**: PaliGemma-3B をベースとした検索用モデル

**継承元**: `PaliGemmaPreTrainedModel`

**重要な属性**:

```python
self.dim = 128  # 最終出力次元
self.model = PaliGemmaForConditionalGeneration(config)  # ベースモデル
self.custom_text_proj = nn.Linear(2048, 128)  # 線形射影②
```

**アーキテクチャ**:

```
PaliGemma-3B
├── Vision Tower (SigLIP)
│   └── 出力: 1152次元
├── Multi-modal Projector (線形射影①)
│   └── 1152次元 → 2048次元
└── Language Model (Gemma-2B)
    └── 出力: 2048次元 (hidden_states[-1])
```

---

### ステップ 5️⃣: `modeling_colpali.py` - Forward Pass

#### 🔹 関数: `ColPali.forward(**kwargs)`

**役割**: 画像から 128 次元のマルチベクトル埋め込みを生成

**入力**:

- `kwargs`: BatchFeature 辞書
  - `input_ids`: (batch, seq_len)
  - `attention_mask`: (batch, seq_len)
  - `pixel_values`: (batch, C, H, W)

**処理フロー（詳細）**:

```python
# ステップ1: ベースモデルで処理
outputs = self.model(*args, output_hidden_states=True, **kwargs)
# → PaliGemmaの全層を通過

# ステップ2: Gemmaの最終層を取得
last_hidden_states = outputs.hidden_states[-1]
# shape: (batch_size, sequence_length, 2048)
# ★ここが「Gemmaの世界（2,048次元）」

# ステップ3: 線形射影② (2048→128)
proj = self.custom_text_proj(last_hidden_states)
# shape: (batch_size, sequence_length, 128)

# ステップ4: L2正規化
proj = proj / proj.norm(dim=-1, keepdim=True)
# 各ベクトルを単位ベクトルに

# ステップ5: Attention Maskを適用
proj = proj * kwargs["attention_mask"].unsqueeze(-1)
# パディング部分をゼロに
```

**出力**:

- `proj`: torch.Tensor
  - shape: (batch_size, N_tokens, 128)
  - N_tokens ≈ 1,030 (画像トークン + 特別トークン)

**次へ渡すもの**: この 128 次元マルチベクトルが検索用埋め込み

---

### ステップ 6️⃣: `main.py` - 埋め込み取得

#### 🔹 関数: `get_embeddings(model, processor, images, embedding_type, device)`

**役割**: モデルから埋め込みを取得（128 次元 or 2,048 次元を選択）

**入力**:

- `model`: ColPali モデル
- `processor`: ColPaliProcessor
- `images`: List[PIL.Image]
- `embedding_type`: "128dim" or "2048dim"
- `device`: "auto", "cuda", "mps", "cpu"

**処理フロー**:

```python
# ステップ1: デバイス設定
if device == "auto":
    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

# ステップ2: 画像を前処理
batch_images = processor.process_images(images).to(device)

# ステップ3-A: 2048次元を取得する場合
if embedding_type == "2048dim":
    outputs = model.model(**batch_images, output_hidden_states=True)
    embeddings = outputs.hidden_states[-1]  # (batch, N_tokens, 2048)

    # マスキング
    attention_mask = batch_images["attention_mask"].unsqueeze(-1)
    embeddings = embeddings * attention_mask

# ステップ3-B: 128次元を取得する場合
else:
    embeddings = model(**batch_images)  # (batch, N_tokens, 128)
```

**出力**:

- `embeddings`: torch.Tensor
  - shape: (batch_size, N_tokens, dim)
  - dim = 128 or 2048
- `info`: dict - メタ情報

**データフロー図**:

```
画像 → processor.process_images()
     → BatchFeature
     → model.forward() または model.model.forward()
     → マルチベクトル埋め込み (batch, N_tokens, dim)
```

---

### ステップ 7️⃣: `main.py` - Pooling 適用

#### 🔹 関数: `apply_pooling(embeddings, method)`

**役割**: マルチベクトルを固定長ベクトルに集約

**入力**:

- `embeddings`: torch.Tensor
  - shape: (batch_size, N_tokens, dim)
- `method`: "mean", "max", "std", "concat"

**処理フロー（各手法の詳細）**:

```python
if method == "mean":
    # 平均プーリング
    # 全トークンの平均を取る
    pooled = embeddings.mean(dim=1)
    # shape: (batch_size, dim)
    # 用途: 全体的な画質傾向（明るさ、ぼけ）

elif method == "max":
    # 最大値プーリング
    # 各次元の最大値を取る
    pooled = embeddings.max(dim=1)[0]
    # shape: (batch_size, dim)
    # 用途: 最も顕著な特徴（ノイズ、汚れ）

elif method == "std":
    # 標準偏差プーリング
    # 各次元の標準偏差を計算
    pooled = embeddings.std(dim=1)
    # shape: (batch_size, dim)
    # 用途: 画質のばらつき（コントラスト）

elif method == "concat":
    # 連結プーリング
    # 3つの統計量を結合
    mean_pool = embeddings.mean(dim=1)
    max_pool = embeddings.max(dim=1)[0]
    std_pool = embeddings.std(dim=1)
    pooled = torch.cat([mean_pool, max_pool, std_pool], dim=-1)
    # shape: (batch_size, dim*3)
    # 用途: 多角的な品質評価
```

**出力**:

- `pooled`: torch.Tensor
  - shape: (batch_size, dim) または (batch_size, dim\*3)
  - これが機械学習モデル（LightGBM/SVR）への入力になる

**データ変換**:

```
マルチベクトル (batch, 1030, 2048)
    ↓ Mean Pooling
固定長ベクトル (batch, 2048)
    ↓ .cpu().numpy()
NumPy配列 (batch, 2048)
    ↓
LightGBM/SVR
```

---

### 📊 データ構造の変遷（全体像）

```
1. PIL.Image
   (1200, 1600, 3)  # RGB画像

   ↓ processor.process_images()

2. BatchFeature
   {
     'input_ids': (1, 1030),
     'attention_mask': (1, 1030),
     'pixel_values': (1, 3, 448, 448)
   }

   ↓ model.forward() [128dim] または model.model.forward() [2048dim]

3. マルチベクトル埋め込み
   (1, 1030, 128)  # 128次元の場合
   (1, 1030, 2048) # 2048次元の場合

   ↓ apply_pooling("mean")

4. 固定長ベクトル
   (1, 128)   # 128次元 + mean
   (1, 2048)  # 2048次元 + mean
   (1, 6144)  # 2048次元 + concat

   ↓ .cpu().numpy()

5. NumPy配列（機械学習用）
   shape: (1, dim)
   → LightGBM/SVRへ入力
```

---

### 🔧 補助関数の詳細

#### `processing_utils.py`

##### 🔹 関数: `get_torch_device(device="auto")`

**役割**: 利用可能なデバイスを自動選択

**処理**:

```python
if device == "auto":
    if torch.cuda.is_available():
        return "cuda:0"  # NVIDIA GPU
    elif torch.backends.mps.is_available():
        return "mps"  # Apple Silicon
    else:
        return "cpu"  # CPU
```

##### 🔹 関数: `score_multi_vector(qs, ps, batch_size, device)`

**役割**: Late Interaction (MaxSim) スコア計算（検索用）

**入力**:

- `qs`: クエリ埋め込み List[Tensor] - [(q_len, dim), ...]
- `ps`: ページ埋め込み List[Tensor] - [(p_len, dim), ...]

**処理**:

```python
# クエリとページの全ペアでMaxSim計算
for query_emb in qs:
    for page_emb in ps:
        # 内積行列計算
        sim_matrix = torch.einsum("nd,md->nm", query_emb, page_emb)
        # 各クエリトークンの最大値を取得
        max_sim = sim_matrix.max(dim=1)[0]
        # 合計してスコア化
        score = max_sim.sum()
```

**出力**: スコア行列 (n_queries, n_passages)

---

### ⚙️ 認識の確認

#### あなたの認識は正しいです ✅

1. **画像 → ColPali → マルチベクトル**

   - ✅ 128 次元 × 約 1,030 トークン取得可能
   - ✅ 2,048 次元 × 約 1,030 トークン取得可能

2. **処理の流れ**

   - ✅ 画像 → RGB 変換 → リサイズ(448×448)
   - ✅ → パッチ分割(14×14) → 1,024 パッチ
   - ✅ → Vision Encoder(SigLIP) → 1,152 次元
   - ✅ → 線形射影 ① → 2,048 次元（Gemma の世界）
   - ✅ → Gemma 処理 → 文脈付き 2,048 次元
   - ✅ → 線形射影 ② → 128 次元（検索用）

3. **Pooling**

   - ✅ Mean/Max/Std/Concat すべて実装済み
   - ✅ 固定長ベクトル化により機械学習モデルへ入力可能

4. **OCR 品質予測への応用**
   - ✅ 2,048 次元推奨（画質劣化情報を保持）
   - ✅ Mean Pooling 推奨（全体傾向）
   - ✅ → LightGBM/SVR で学習可能

---

## 🚀 使い方

### 必要なライブラリ

```bash
pip install torch torchvision pillow transformers
```

### 基本的な実行方法

```bash
# 128次元埋め込みを取得（検索用）
python main.py --image /path/to/image.jpg --embedding-type 128dim

# 2,048次元埋め込みを取得（Gemmaの世界）
python main.py --image /path/to/image.jpg --embedding-type 2048dim

# Mean Poolingを適用して固定長ベクトル化
python main.py --image /path/to/image.jpg --embedding-type 2048dim --pooling mean

# Mean + Max + Std を連結（3倍の次元）
python main.py --image /path/to/image.jpg --embedding-type 2048dim --pooling concat
```

### コマンドライン引数

| 引数               | 説明                                   | デフォルト            |
| ------------------ | -------------------------------------- | --------------------- |
| `--image`          | 入力画像のパス（必須）                 | -                     |
| `--model-name`     | HuggingFace のモデル名                 | `vidore/colpali-v1.2` |
| `--embedding-type` | `128dim` または `2048dim`              | `128dim`              |
| `--pooling`        | `none`, `mean`, `max`, `std`, `concat` | `none`                |
| `--device`         | `auto`, `cuda`, `mps`, `cpu`           | `auto`                |

---

## 🔍 技術詳細

### 1. マルチベクトルの取得

#### **128 次元（検索用）**

```python
# modeling_colpali.py の forward メソッド
outputs = self.model(*args, output_hidden_states=True, **kwargs)
last_hidden_states = outputs.hidden_states[-1]  # (batch, N_tokens, 2048)
proj = self.custom_text_proj(last_hidden_states)  # (batch, N_tokens, 128)
proj = proj / proj.norm(dim=-1, keepdim=True)  # L2正規化
```

- **用途**: 検索（Retrieval）に最適化
- **特徴**: ColPali の学習で最適化済み
- **メリット**: 軽量、ストレージ効率が良い
- **デメリット**: 画質劣化情報が希釈されている可能性

#### **2,048 次元（Gemma の世界）**

```python
# main.py の get_embeddings 関数
outputs = model.model(**batch_images, output_hidden_states=True)
embeddings = outputs.hidden_states[-1]  # (batch, N_tokens, 2048)
```

- **用途**: より豊かな特徴表現
- **特徴**: Gemma-2B の生の出力
- **メリット**: 画質劣化情報を保持している可能性が高い
- **デメリット**: ストレージが重い（128 次元の 16 倍）

### 2. トークン数の決定

#### **ColPali (PaliGemma)**

```
入力画像: 448×448 にリサイズ
パッチサイズ: 14×14
パッチ数: 32×32 = 1,024個
特別トークン: BOS等で約6個
合計: 約1,030トークン
```

#### **ColQwen2 (Qwen2-VL)**

```
動的解像度対応
画像サイズに応じてトークン数が変化
例: 512×512 → 約768トークン
```

### 3. Pooling 方法

#### **Mean Pooling（平均）**

```python
pooled = embeddings.mean(dim=1)  # (batch, dim)
```

- 全体的な傾向（暗さ、ボケ）を捉える

#### **Max Pooling（最大値）**

```python
pooled = embeddings.max(dim=1)[0]  # (batch, dim)
```

- 最も強い信号（極端なノイズ、特異な汚れ）を捉える

#### **Std Pooling（標準偏差）**

```python
pooled = embeddings.std(dim=1)  # (batch, dim)
```

- 情報量のムラ（コントラスト差、ノイズ散らばり）を捉える

#### **Concat（連結）**

```python
pooled = torch.cat([mean, max, std], dim=-1)  # (batch, dim*3)
```

- 3 つの統計量を組み合わせて多角的に捉える
- 次元数が 3 倍になる（例: 2,048 → 6,144）

---

## 💡 先生の指示への対応

### 指示内容

```
ColPali（またはそのバックボーンのPaliGemma）に画像を入力し、
LLMの最終層から得られるマルチベクトル（1,030トークン × 128次元）を取得します。
これが取れるかが鍵。

画像サイズによってトークン数が違う気がするので、
128次元×Nトークンが取得できたとしたら、
1,030個のベクトルをどう扱うかがポイントです。

Pooling法: Mean PoolingまたはMax Poolingで集約し、
1つの128次元ベクトルにしてから、軽量なモデル（LightGBMやSVR）に入力する。

Gemmaの世界（2,048次元）が取れるなら、これの方が筋がいい。
やりたいことは、画像からcolpaliを通して、同じ次元のベクトル化したいってだけ。
```

### 実装状況

✅ **完了項目**:

1. 画像 → ColPali → マルチベクトル取得

   - 128 次元 × 約 1,030 トークン ✅
   - 2,048 次元 × 約 1,030 トークン ✅

2. Pooling 実装

   - Mean Pooling ✅
   - Max Pooling ✅
   - Std Pooling ✅
   - Concat（組み合わせ）✅

3. 固定長ベクトル出力
   - (batch_size, 128) または (batch_size, 2048) ✅
   - (batch_size, 384) または (batch_size, 6144) ※Concat 時 ✅

### 推奨設定

```bash
# 品質予測に最適な設定
python main.py \
  --image /path/to/document.jpg \
  --embedding-type 2048dim \
  --pooling mean \
  --device auto
```

**理由**:

- 2,048 次元: 画質劣化情報を保持
- Mean Pooling: 全体的な品質傾向を捉える
- 出力: (1, 2048) の固定長ベクトル → LightGBM へ

---

## 🔧 カスタマイズ方法

### 1. 他のプーリング方法を追加

`main.py`の`apply_pooling`関数に新しいメソッドを追加:

```python
def apply_pooling(embeddings, method):
    # ...既存のコード...

    elif method == "attention":
        # Attention-based pooling
        weights = torch.softmax(embeddings.sum(dim=-1), dim=1)
        return (embeddings * weights.unsqueeze(-1)).sum(dim=1)
```

### 2. ColQwen2 を使用

```python
# main.py の import を変更
from modeling_colqwen2 import ColQwen2
from processing_colqwen2 import ColQwen2Processor

# モデルロード部分を変更
model = ColQwen2.from_pretrained("vidore/colqwen2-v1.0", ...)
processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v1.0")
```

### 3. バッチ処理

```python
# 複数画像を一度に処理
images = [Image.open(f"image{i}.jpg") for i in range(10)]
embeddings, info = get_embeddings(model, processor, images)
# shape: (10, N_tokens, dim)
```

---

## ⚠️ 注意事項

### 1. メモリ使用量

- **128 次元**: 約 256KB/ページ（論文値）
- **2,048 次元**: 約 4MB/ページ（16 倍）

大量の画像を処理する場合はメモリに注意。

### 2. GPU 推奨

- ColPali は比較的大きなモデル（3B parameters）
- CPU 実行は可能だが遅い
- M1/M2 Mac: MPS 対応
- NVIDIA GPU: CUDA 対応

### 3. HuggingFace ログイン

PaliGemma モデルは利用規約への同意が必要:

```bash
huggingface-cli login
```

---

## 📊 出力例

```
📸 画像を読み込み中: document.jpg
   画像サイズ: (1200, 1600)

🤖 モデルをロード中: vidore/colpali-v1.2
   ✅ モデルのロード完了

🔄 2048dim 埋め込みを取得中...

📊 マルチベクトル情報:
   - バッチサイズ: 1
   - トークン数: 1030
   - 埋め込み次元: 2048
   - デバイス: mps
   - shape: (1, 1030, 2048)

🎯 MEAN プーリングを適用中...
   固定長ベクトル shape: (1, 2048)

✅ 完了！

💡 使用例:
   このベクトルをLightGBMやSVRに入力して、OCR品質予測が可能です。
   例: pooled_vector.cpu().numpy() → shape: (1, 2048)
```

---

## 🔗 関連ドキュメント

- [ColPali 論文](https://arxiv.org/abs/2407.01449)
- [PaliGemma](https://huggingface.co/google/paligemma-3b-mix-448)
- [Qwen2-VL](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)

---

## 📝 今後の拡張

### Phase 1: ベクトル取得 ✅（完了）

- ColPali から 128 次元 / 2,048 次元を取得
- Pooling 実装

### Phase 2: 品質予測器構築（次のステップ）

1. 画像データセット準備
2. OCR エンジンで正解ラベル作成（CER/WER）
3. ColPali でベクトル抽出
4. LightGBM/SVR で学習
5. 品質予測モデル完成

### Phase 3: 可視化・解釈性

- Similarity Maps 生成
- どこが品質を下げているか可視化

---

## ✨ まとめ

このディレクトリのファイル群により、以下が実現できます：

✅ **画像 → ColPali → マルチベクトル（128 次元 or 2,048 次元）**  
✅ **Pooling → 固定長ベクトル**  
✅ **→ LightGBM/SVR → OCR 品質予測**

すべてのコードは`work`ディレクトリ内で自己完結しており、他のディレクトリへの依存はありません。

---

**作成日**: 2026/01/03  
**バージョン**: 1.0
