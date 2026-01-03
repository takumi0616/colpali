# ColPali プロジェクト概要

このディレクトリには、ColPali（Efficient Document Retrieval with Vision Language Models）に関する学習・実験用のノートブックと資料が含まれています。

## 📚 ColPali とは

ColPali は、Vision Language Model（VLM）を活用した効率的なドキュメント検索モデルです。従来の OCR ベースの手法とは異なり、ドキュメントページを画像として扱い、テキストだけでなくレイアウト、表、チャート、その他の視覚要素を統合的に理解して、詳細なマルチベクトル埋め込みを作成します。

### 主な特徴

- **OCR 不要**: 複雑で脆弱なレイアウト認識や OCR パイプラインが不要
- **マルチベクトル表現**: Late Interaction 方式により高精度な検索を実現
- **解釈可能性**: 類似度マップによりモデルの注目領域を可視化可能
- **複数のバックボーン**: PaliGemma（ColPali）と Qwen2-VL（ColQwen2）をサポート

---

## 📁 ディレクトリ構造

```
src/colpali/
├── README.md                                    # このファイル
├── work.md                                      # 作業メモ
├── document/                                    # 資料・プログラム集
│   ├── URL.md                                  # 参考リンク集
│   ├── article/                                # 論文と解説記事
│   │   ├── 2407.01449v2.pdf                   # ColPali論文（arXiv）
│   │   ├── memo.md                            # 研究室の先生によるQ&A集
│   │   ├── note記事_実装付きAI論文解説.pdf     # note記事
│   │   └── Zenn記事_テキスト抽出不要のRAG.pdf  # Zenn記事
│   └── program/                               # プログラムリポジトリ
│       ├── colpali/                           # 公式実装リポジトリ
│       └── colpali-cookbooks/                 # サンプルノートブック集
├── finetune_colpali.ipynb                      # ColPaliファインチューニング
├── gen_colpali_similarity_maps.ipynb           # ColPali類似度マップ生成
├── gen_colqwen2_similarity_maps.ipynb          # ColQwen2類似度マップ生成
├── run_e2e_rag_colqwen2_with_adapter_hot_swapping.ipynb  # RAGパイプライン
├── use_transformers_native_colpali.ipynb       # transformers版ColPali使用
└── use_transformers_native_colqwen2.ipynb      # transformers版ColQwen2使用
```

---

## 📓 Jupyter Notebook の説明

### 1. `finetune_colpali.ipynb` - ColPali のファインチューニング 🛠️

**内容**: ColPali モデルをカスタムデータセットでファインチューニングする方法を解説

**主なトピック**:

- LoRA（Low-Rank Adaptation）を用いた効率的なファインチューニング
- 4bit/8bit 量子化（QLoRA）によるメモリ削減
- VDSID-French データセットを使用した多言語対応の実例
- WandB を使用したトレーニング監視
- HuggingFace Hub へのモデルアップロード

**推奨環境**: A100-40GB GPU（より小さい GPU でも量子化により実行可能）

**ユースケース**:

- 特定のドメイン（法律文書、医療記録など）に特化したモデルを作成
- 日本語文書への対応強化

---

### 2. `gen_colpali_similarity_maps.ipynb` - ColPali 類似度マップ生成 👀

**内容**: ColPali の予測を解釈するための類似度マップ（Similarity Maps）の生成方法

**主なトピック**:

- クエリトークンと画像パッチ間の類似度計算
- 各トークンに対応する視覚的証拠のヒートマップ可視化
- OCR 能力とチャート理解能力の確認
- モデルの注目領域の解釈

**推奨環境**: Google Colab（無料の T4 GPU）または M2 Pro Mac

**ユースケース**:

- モデルがドキュメントのどこに注目しているかを理解
- デバッグと精度改善のための診断

**サンプルクエリ**: 「Quelle partie de la production pétrolière du Kazakhstan provient de champs en mer ?」

---

### 3. `gen_colqwen2_similarity_maps.ipynb` - ColQwen2 類似度マップ生成 👀

**内容**: ColQwen2（Qwen2-VL ベース）の類似度マップ生成

**主なトピック**:

- ColPali とほぼ同じだが、ColQwen2 に特化
- 動的解像度対応（任意のアスペクト比の画像を処理可能）
- より高度なチャート理解能力

**推奨環境**: Google Colab（無料の T4 GPU）または M2 Pro Mac

**ユースケース**:

- ColQwen2 の予測を視覚的に解釈
- 複雑なチャートや図表の理解度を確認

**サンプルクエリ**: 「Which hour of the day had the highest overall electricity generation in 2019?」

---

### 4. `run_e2e_rag_colqwen2_with_adapter_hot_swapping.ipynb` - アダプタホットスワッピング RAG 🔥

**内容**: 単一の VLM モデルで検索（Retrieval）と生成（Generation）を両方実行するエンドツーエンド RAG パイプライン

**主なトピック**:

- **アダプタホットスワッピング**: LoRA アダプタの有効/無効を切り替えることで、単一モデルで 2 つの役割を担当
  - 検索モード: ColQwen2 アダプタを有効化してドキュメント検索
  - 生成モード: アダプタを無効化して Qwen2-VL として回答生成
- VRAM の大幅な節約（複数モデル不要）
- Late Interaction による高精度検索

**推奨環境**: Google Colab（無料の T4 GPU）または M2 Pro Mac

**ユースケース**:

- メモリ制約のある環境での RAG システム構築
- 単一モデルでの効率的な文書検索と質問応答

**重要な注意点**: Qwen2-VL-2B は小規模なため、複雑なチャート理解には限界があります。高度な推論が必要な場合は、検索に ColQwen2、生成に大規模 VLM（Qwen2-VL-72B など）を使用することを推奨します。

---

### 5. `use_transformers_native_colpali.ipynb` - Transformers 版 ColPali 🤗

**内容**: HuggingFace Transformers の公式実装を使用した ColPali の基本的な使用方法

**主なトピック**:

- `ColPaliForRetrieval`モデルの読み込み
- 画像とクエリの埋め込み生成
- Late Interaction 類似度スコアリング
- マルチベクトル検索の実装
- ベクトルデータベース（Vespa、Qdrant、Milvus 等）との連携方法
- 類似度マップの生成

**推奨環境**: Google Colab（無料の T4 GPU）または M2 Pro MacBook

**ユースケース**:

- ColPali の基本的な使い方を学ぶ
- プロダクション環境への統合準備
- `colpali-engine`から Transformers 実装への移行

**使用モデル**: `vidore/colpali-v1.3-hf`

---

### 6. `use_transformers_native_colqwen2.ipynb` - Transformers 版 ColQwen2 🤗

**内容**: HuggingFace Transformers の公式実装を使用した ColQwen2 の基本的な使用方法

**主なトピック**:

- `ColQwen2ForRetrieval`モデルの読み込み
- 動的解像度対応の画像処理
- Flash Attention 2 の活用（利用可能な場合）
- マルチベクトル埋め込みの生成とスコアリング
- ベクトルデータベースとの連携
- 類似度マップの生成

**推奨環境**: Google Colab（無料の T4 GPU）または M2 Pro MacBook

**ユースケース**:

- ColQwen2 の基本的な使い方を学ぶ
- 任意のアスペクト比の文書を処理
- Apache 2.0 ライセンスでの商用利用

**使用モデル**: `vidore/colqwen2-v1.0-hf`

---

## 📄 資料（document ディレクトリ）

### `document/URL.md`

ColPali に関する重要なリンク集:

- **Zenn 記事**: テキスト抽出不要の RAG を実現する ColPali
- **colpali-cookbooks**: GitHub リポジトリ
- **colpali**: 公式実装リポジトリ
- **Gemini 見解**: 2025 年 12 月 18 日時点の解説
- **note 記事**: 実装付き AI 論文解説

### `document/article/`

#### `2407.01449v2.pdf`

ColPali の原論文（arXiv）。モデルアーキテクチャ、トレーニング手法、ベンチマーク結果などの詳細が記載されています。

#### `memo.md`

研究室の先生による ColPali に関する詳細な Q&A 集。以下のトピックをカバー:

**Q1**: ColPali は OCR に使えるか？

- **回答要約**: ColPali は OCR の代替として機能しますが、「検索（Retrieval）」に特化。テキストの書き出しには向かない。

**Q2**: ColPali で得られる写像は何次元か？

- **回答要約**: 128 次元のベクトル空間。1 ページあたり約 1,030 個のマルチベクトル（1,024 個のパッチ + 6 個の特別トークン）

**Q3**: バイナリ量子化について

- **回答要約**: リランキングと組み合わせることで、メモリを 32 分の 1 に削減しつつ精度を維持可能

**Q4**: 帳票画像認識の品質予測器の構築

- **回答要約**: ColPali の特徴ベクトルを使用して、OCR 精度を予測するモデルを構築する方法

**Q5**: DiT から ColPali へのパラダイムシフト

- **回答要約**: 大域的埋め込み（Global Embedding）の限界と、ColPali の Late Interaction がなぜ固有名詞の認識に優れているかの理論的解説

#### その他の PDF

- **note 記事\_実装付き AI 論文解説.pdf**: ColPali 論文の日本語解説
- **Zenn 記事\_テキスト抽出不要の RAG を実現する ColPali.pdf**: RAG への実装方法

### `document/program/`

#### `colpali/` - 公式実装

`colpali-engine` パッケージの GitHub リポジトリのクローン。以下を含みます:

- モデル実装（ColPali、ColQwen2 など）
- トレーニングスクリプト
- コレクター、損失関数
- 解釈可能性ツール
- テストコード

**主な機能**:

- ViDoRe ベンチマークでのモデル評価
- LoRA によるファインチューニング
- トークンプーリングによる圧縮
- 類似度マップの生成

#### `colpali-cookbooks/` - サンプル集

公式の ColPali Cookbooks リポジトリのクローン。このディレクトリのトップレベルにあるノートブックはここから取得されています。

---

## 🔧 作業メモ（work.md）

### 2025/12/18「雲居研ゼミ」の課題

1. ColPali のアーキテクチャ図の ViT（Vision Encoder）と LLM の間の細い線（線形投影層）を見つける
2. ドキュメントを 4 分割するとどうなるのかを調査
3. `proj.`（Projection Layer）とは何かを理解する

---

## 🚀 使用開始方法

### 1. 環境構築

```bash
# colpali-engineのインストール
pip install colpali-engine

# ファインチューニング用
pip install "colpali-engine[train]"

# 解釈可能性ツール用
pip install "colpali-engine[interpretability]"
```

### 2. ノートブックの実行

各ノートブックは以下の方法で実行できます:

- **Google Colab**: ノートブック内の Colab ボタンをクリック
- **ローカル**: Jupyter Notebook または VS Code で開く

### 3. モデルの選択

| モデル                   | スコア | ライセンス | 特徴                                       |
| ------------------------ | ------ | ---------- | ------------------------------------------ |
| `vidore/colpali-v1.3`    | 84.8   | Gemma      | PaliGemma-3B ベース、固定解像度            |
| `vidore/colqwen2-v1.0`   | 89.3   | Apache 2.0 | Qwen2-VL-2B ベース、動的解像度、商用利用可 |
| `vidore/colqwen2.5-v0.2` | 89.4   | Apache 2.0 | Qwen2.5-VL-3B ベース、最新版               |

---

## 📊 主なユースケース

1. **文書検索システム**: OCR 不要で視覚的に文書を検索
2. **RAG システム**: ドキュメント検索と質問応答の統合
3. **多言語対応**: 英語以外の言語でのファインチューニング
4. **品質予測**: OCR 結果の信頼度推定
5. **モデル解釈**: 類似度マップによる注目領域の可視化

---

## 🎓 学習リソース

### 論文

- **ColPali 論文**: `document/article/2407.01449v2.pdf`
- **arXiv**: https://arxiv.org/abs/2407.01449

### 公式リンク

- **HuggingFace Organization**: https://huggingface.co/vidore
- **GitHub（公式実装）**: https://github.com/illuin-tech/colpali
- **GitHub（Cookbooks）**: https://github.com/tonywu71/colpali-cookbooks
- **ViDoRe Leaderboard**: https://huggingface.co/spaces/vidore/vidore-leaderboard

### 日本語リソース

- **Zenn 記事**: https://zenn.dev/knowledgesense/articles/08cfc3de7464cb
- **note 記事**: https://note.com/aivalix/n/na8b4120513fe

---

## 💡 技術的なハイライト

### Late Interaction（遅延交差）

- クエリトークンと画像パッチの間で MaxSim 演算を実行
- 各クエリトークンが最も類似する画像パッチを見つける
- 高精度かつ効率的な検索を実現

### マルチベクトル表現

- 1 ページ = 1,030 個の 128 次元ベクトル（ColPali の場合）
- 各パッチが独立した情報を保持
- 情報の損失を最小化

### 効率化技術

- **バイナリ量子化**: メモリ使用量を 97%削減
- **トークンプーリング**: シーケンス長を 66.7%削減
- **LoRA**: 学習可能なパラメータを大幅に削減

---

## 📝 引用

```bibtex
@misc{faysse2024colpaliefficientdocumentretrieval,
      title={ColPali: Efficient Document Retrieval with Vision Language Models},
      author={Manuel Faysse and Hugues Sibille and Tony Wu and Bilel Omrani and Gautier Viaud and Céline Hudelot and Pierre Colombo},
      year={2024},
      eprint={2407.01449},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2407.01449},
}
```

---

## 🤝 コントリビューション

このプロジェクトは学習・研究目的で作成されています。質問や改善提案がある場合は、work.md に記録してください。

---

**最終更新**: 2025 年 12 月 29 日
