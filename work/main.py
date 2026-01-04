"""
ColPali - 画像からマルチベクトル抽出のデモプログラム

このスクリプトは、画像からColPaliを通してマルチベクトル（128次元×Nトークン または 2,048次元×Nトークン）を
取得するデモンストレーションです。

更新 (2026/01/04):
- transformersのネイティブ実装 (ColPaliForRetrieval) を使用するように変更
- モデル名を vidore/colpali-v1.3-hf に変更（-hfサフィックスがtransformersネイティブ版）
"""

import os
import torch
from PIL import Image
from typing import Literal, Tuple
import argparse
from pathlib import Path

# HuggingFace認証（.envファイルからトークンを読み込む）
def setup_hf_auth():
    """HuggingFaceの認証を設定"""
    # .envファイルからHF_TOKENを読み込む
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path, "r") as f:
            for line in f:
                if line.startswith("HF_TOKEN="):
                    token = line.strip().split("=", 1)[1]
                    os.environ["HF_TOKEN"] = token
                    os.environ["HUGGINGFACE_HUB_TOKEN"] = token
                    print(f"   ✅ HF_TOKENを.envファイルから読み込みました")
                    return token
    
    # 環境変数から取得
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        print(f"   ✅ HF_TOKENを環境変数から読み込みました")
        return token
    
    print("   ⚠️ HF_TOKENが見つかりません。huggingface-cli loginでログインしてください。")
    return None


def get_embeddings(
    model,
    processor,
    images: list[Image.Image],
    embedding_type: Literal["128dim", "2048dim"] = "128dim",
    device: str = "auto"
) -> Tuple[torch.Tensor, dict]:
    """
    画像からマルチベクトル埋め込みを取得
    
    Args:
        model: ColPaliForRetrievalモデル
        processor: ColPaliProcessor
        images: PIL画像のリスト
        embedding_type: "128dim" (検索用) または "2048dim" (Gemmaの世界)
        device: デバイス ("auto", "cuda", "mps", "cpu")
    
    Returns:
        embeddings: マルチベクトル埋め込み (batch_size, N_tokens, dim)
        info: 情報辞書（トークン数、次元数など）
    """
    # デバイス設定
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda:0"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    model = model.to(device)
    model.eval()
    
    # 画像を処理
    batch_images = processor(images=images).to(device)
    
    # Forward pass
    with torch.no_grad():
        if embedding_type == "2048dim":
            # Gemmaの2,048次元を取得（proj前）
            # ColPaliForRetrievalの内部構造:
            #   model.vlm = PaliGemmaForConditionalGeneration
            #   model.embedding_proj_layer = Linear(2048, 128)
            # vlmを直接使用してhidden_statesを取得
            outputs = model.vlm(
                **batch_images,
                output_hidden_states=True
            )
            embeddings = outputs.hidden_states[-1]  # (batch_size, N_tokens, 2048)
            
            # マスキング適用
            attention_mask = batch_images["attention_mask"].unsqueeze(-1)
            embeddings = embeddings * attention_mask
            
        else:  # "128dim"
            # 通常のColPali出力（128次元）
            outputs = model(**batch_images)
            embeddings = outputs.embeddings  # (batch_size, N_tokens, 128)
    
    # 情報を収集
    batch_size, n_tokens, dim = embeddings.shape
    info = {
        "batch_size": batch_size,
        "n_tokens": n_tokens,
        "embedding_dim": dim,
        "device": device,
        "embedding_type": embedding_type
    }
    
    return embeddings, info


def apply_pooling(
    embeddings: torch.Tensor,
    method: Literal["mean", "max", "std", "concat"] = "mean"
) -> torch.Tensor:
    """
    マルチベクトル埋め込みをプーリングして固定長ベクトルに変換
    
    Args:
        embeddings: マルチベクトル (batch_size, N_tokens, dim)
        method: プーリング方法
            - "mean": 平均プーリング
            - "max": 最大値プーリング
            - "std": 標準偏差プーリング
            - "concat": Mean + Max + Std を連結（3倍の次元）
    
    Returns:
        pooled: 固定長ベクトル (batch_size, dim) または (batch_size, dim*3)
    """
    if method == "mean":
        return embeddings.mean(dim=1)  # (batch_size, dim)
    
    elif method == "max":
        return embeddings.max(dim=1)[0]  # (batch_size, dim)
    
    elif method == "std":
        return embeddings.std(dim=1)  # (batch_size, dim)
    
    elif method == "concat":
        mean_pool = embeddings.mean(dim=1)
        max_pool = embeddings.max(dim=1)[0]
        std_pool = embeddings.std(dim=1)
        return torch.cat([mean_pool, max_pool, std_pool], dim=-1)  # (batch_size, dim*3)
    
    else:
        raise ValueError(f"Unknown pooling method: {method}")


def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description="ColPali - 画像からマルチベクトル抽出")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="入力画像のパス"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="vidore/colpali-v1.3-hf",
        help="使用するモデル名（HuggingFace）。デフォルト: vidore/colpali-v1.3-hf（transformersネイティブ版）"
    )
    parser.add_argument(
        "--embedding-type",
        type=str,
        choices=["128dim", "2048dim"],
        default="128dim",
        help="埋め込みの次元: 128dim (検索用) または 2048dim (Gemmaの世界)"
    )
    parser.add_argument(
        "--pooling",
        type=str,
        choices=["none", "mean", "max", "std", "concat"],
        default="none",
        help="プーリング方法"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="デバイス (auto, cuda, mps, cpu)"
    )
    
    args = parser.parse_args()
    
    # HuggingFace認証を設定
    print("🔐 HuggingFace認証を設定中...")
    setup_hf_auth()
    
    # 画像を読み込み
    print(f"\n📸 画像を読み込み中: {args.image}")
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"画像が見つかりません: {args.image}")
    
    image = Image.open(image_path).convert("RGB")
    print(f"   画像サイズ: {image.size}")
    
    # モデルとプロセッサをロード
    # transformersのネイティブ実装を使用
    print(f"\n🤖 モデルをロード中: {args.model_name}")
    print("   （transformersネイティブ実装 ColPaliForRetrieval を使用）")
    
    from transformers import ColPaliForRetrieval, ColPaliProcessor
    
    model = ColPaliForRetrieval.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()
    
    processor = ColPaliProcessor.from_pretrained(args.model_name)
    print("   ✅ モデルのロード完了")
    
    # 埋め込みを取得
    print(f"\n🔄 {args.embedding_type} 埋め込みを取得中...")
    embeddings, info = get_embeddings(
        model=model,
        processor=processor,
        images=[image],
        embedding_type=args.embedding_type,
        device=args.device
    )
    
    print(f"\n📊 マルチベクトル情報:")
    print(f"   - バッチサイズ: {info['batch_size']}")
    print(f"   - トークン数: {info['n_tokens']}")
    print(f"   - 埋め込み次元: {info['embedding_dim']}")
    print(f"   - デバイス: {info['device']}")
    print(f"   - shape: {tuple(embeddings.shape)}")
    
    # プーリング適用
    if args.pooling != "none":
        print(f"\n🎯 {args.pooling.upper()} プーリングを適用中...")
        pooled = apply_pooling(embeddings, method=args.pooling)
        print(f"   固定長ベクトル shape: {tuple(pooled.shape)}")
        
        # BFloat16はnumpyでサポートされていないため、float32に変換
        pooled_numpy = pooled.cpu().float().numpy()
        
        print(f"\n✅ 完了！")
        print(f"\n💡 使用例:")
        print(f"   このベクトルをLightGBMやSVRに入力して、OCR品質予測が可能です。")
        print(f"   例: pooled_vector.cpu().float().numpy() → shape: {tuple(pooled_numpy.shape)}")
    else:
        print(f"\n✅ 完了！")
        print(f"\n💡 使用例:")
        print(f"   1. Late Interaction (MaxSim) で検索に使用")
        print(f"   2. Poolingを適用して固定長ベクトル化し、機械学習モデルに入力")


if __name__ == "__main__":
    main()
