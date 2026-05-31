"""
=============================================================================
BÀI 5: LLM as Explainable Re-Ranker for Recommendation System
Tích hợp: SER → gợi ý sản phẩm → LLM re-rank + giải thích
=============================================================================
Cài đặt: pip install openai  (dùng OpenAI API, hoặc thay bằng Anthropic)
=============================================================================
"""

import json
import time
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd


# ============================================================
# PHẦN 1: DỮ LIỆU GIẢ LẬP (Product Catalog)
# ============================================================

@dataclass
class Product:
    id:          str
    name:        str
    category:    str
    price:       float
    description: str
    tags:        list[str]    # mood/emotion tags
    score:       float = 0.0  # base recommendation score


# Catalog sản phẩm mẫu theo cảm xúc
PRODUCT_CATALOG = [
    Product("P001", "Tai nghe chống ồn Sony WH-1000XM5",
            "Điện tử", 8990000,
            "Tai nghe premium giảm tiếng ồn, âm thanh trong trẻo",
            ["calm", "focused", "happy"]),

    Product("P002", "Sách 'Đắc Nhân Tâm'",
            "Sách", 120000,
            "Nghệ thuật giao tiếp và thuyết phục",
            ["neutral", "focused", "sad"]),

    Product("P003", "Túi boxing tập thể thao",
            "Thể thao", 650000,
            "Xả stress hiệu quả, rèn sức bền",
            ["angry", "fearful", "disgust"]),

    Product("P004", "Nến thơm thư giãn Lavender",
            "Chăm sóc cá nhân", 250000,
            "Hương thơm nhẹ nhàng giúp thư giãn, giảm lo âu",
            ["fearful", "sad", "calm"]),

    Product("P005", "Game PS5 – Spider-Man 2",
            "Game", 1590000,
            "Trò chơi hành động phiêu lưu hấp dẫn",
            ["happy", "surprised", "excited"]),

    Product("P006", "Trà thảo mộc giảm stress",
            "Thực phẩm", 85000,
            "Blend thảo mộc tự nhiên giúp bình tâm",
            ["angry", "fearful", "sad"]),

    Product("P007", "Loa Bluetooth JBL Flip 6",
            "Điện tử", 2490000,
            "Âm thanh mạnh mẽ, chống nước IPX7",
            ["happy", "surprised", "calm"]),

    Product("P008", "Nhật ký/Sổ tay bullet journal",
            "Văn phòng phẩm", 95000,
            "Ghi chép cảm xúc, theo dõi tâm trạng hàng ngày",
            ["sad", "neutral", "fearful"]),

    Product("P009", "Khóa học Yoga online",
            "Sức khỏe", 599000,
            "30 ngày yoga từ cơ bản đến nâng cao",
            ["angry", "fearful", "calm", "sad"]),

    Product("P010", "Máy pha cà phê mini",
            "Gia dụng", 1290000,
            "Cà phê sáng tạo năng lượng cho ngày mới",
            ["neutral", "calm", "happy"]),
]


# ============================================================
# PHẦN 2: INITIAL RANKER (Rule-based + Cosine)
# ============================================================

def initial_recommend(emotion: str,
                       catalog: list[Product],
                       top_k: int = 5) -> list[Product]:
    """
    Gợi ý sơ bộ dựa trên emotion tag matching.
    Đây là "initial ranker" trước khi LLM re-rank.
    """
    scored = []
    for p in catalog:
        # Score = số tag khớp / tổng tag
        matches = sum(1 for t in p.tags if t == emotion)
        partial = sum(0.5 for t in p.tags if emotion in t or t in emotion)
        score   = (matches + partial) / len(p.tags)
        p_copy  = Product(**asdict(p))
        p_copy.score = score
        scored.append(p_copy)

    # Sort và lấy top_k (kể cả score = 0 để LLM có đủ context)
    scored.sort(key=lambda x: x.score, reverse=True)
    return scored[:top_k]


# ============================================================
# PHẦN 3: LLM RE-RANKER
# ============================================================

def build_reranker_prompt(emotion: str,
                           context: dict,
                           candidates: list[Product]) -> str:
    """
    Xây dựng prompt cho LLM re-ranker.
    Theo paper: LLM nhận danh sách candidates + user context
    → trả về ranked list kèm explanation.
    """
    candidates_str = "\n".join([
        f"{i+1}. [{p.id}] {p.name} – {p.category} – "
        f"{p.price:,.0f}đ\n   Mô tả: {p.description}"
        for i, p in enumerate(candidates)
    ])

    prompt = f"""Bạn là hệ thống gợi ý sản phẩm thông minh.

THÔNG TIN NGƯỜI DÙNG:
- Cảm xúc phát hiện qua giọng nói: **{emotion}**
- Ngữ cảnh: {json.dumps(context, ensure_ascii=False)}

DANH SÁCH SẢN PHẨM ĐỀ XUẤT BAN ĐẦU:
{candidates_str}

NHIỆM VỤ:
1. Re-rank lại danh sách sản phẩm phù hợp nhất với cảm xúc "{emotion}" của người dùng
2. Giải thích ngắn gọn TẠI SAO mỗi sản phẩm phù hợp/không phù hợp
3. Xem xét yếu tố công bằng (fairness): không ưu tiên quá nhiều sản phẩm đắt tiền

Trả lời theo định dạng JSON:
{{
  "ranked_products": [
    {{
      "rank": 1,
      "product_id": "P001",
      "explanation": "Lý do ngắn gọn...",
      "relevance_score": 0.95
    }},
    ...
  ],
  "overall_reasoning": "Giải thích tổng quát về chiến lược gợi ý...",
  "emotion_analysis": "Phân tích cảm xúc và nhu cầu dự đoán..."
}}"""
    return prompt


def llm_rerank(emotion: str,
               context: dict,
               candidates: list[Product],
               api_key: Optional[str] = None,
               provider: str = "openai") -> dict:
    """
    Gọi LLM API để re-rank candidates.
    Hỗ trợ: openai | anthropic | mock (không cần API key)
    """
    prompt = build_reranker_prompt(emotion, context, candidates)

    if provider == "mock":
        # Mock response để test không cần API key
        return _mock_rerank(candidates, emotion)

    elif provider == "openai":
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system",
                     "content": "Bạn là hệ thống gợi ý sản phẩm dựa trên cảm xúc. "
                                "Luôn trả lời bằng JSON hợp lệ."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"},
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"  [WARN] OpenAI API lỗi: {e}. Dùng mock.")
            return _mock_rerank(candidates, emotion)

    elif provider == "anthropic":
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            text = response.content[0].text
            # Extract JSON từ response
            start = text.find("{")
            end   = text.rfind("}") + 1
            return json.loads(text[start:end])
        except Exception as e:
            print(f"  [WARN] Anthropic API lỗi: {e}. Dùng mock.")
            return _mock_rerank(candidates, emotion)


def _mock_rerank(candidates: list[Product], emotion: str) -> dict:
    """Mock LLM response để test logic không cần API."""
    emotion_explanations = {
        "angry":    "Giúp xả stress và bình tâm",
        "sad":      "Mang lại cảm giác an ủi và ấm áp",
        "happy":    "Tăng cường niềm vui và năng lượng tích cực",
        "fearful":  "Giúp thư giãn và giảm lo âu",
        "neutral":  "Phù hợp cho trạng thái ổn định, tập trung",
        "calm":     "Duy trì sự bình yên và thư thái",
        "disgust":  "Chuyển hướng sự chú ý sang điều tích cực",
        "surprised":"Khai thác sự tò mò và năng lượng mới",
    }

    ranked = []
    for i, p in enumerate(candidates):
        relevance = round(max(0.3, p.score + np.random.uniform(-0.1, 0.1)), 2)
        ranked.append({
            "rank":            i + 1,
            "product_id":      p.id,
            "explanation":     f"{p.name} {emotion_explanations.get(emotion, 'phù hợp với bạn')}",
            "relevance_score": relevance,
        })

    # Sắp xếp lại theo relevance (re-rank thật sự)
    ranked.sort(key=lambda x: x["relevance_score"], reverse=True)
    for i, r in enumerate(ranked):
        r["rank"] = i + 1

    return {
        "ranked_products": ranked,
        "overall_reasoning": f"Dựa trên cảm xúc '{emotion}', "
                             f"hệ thống ưu tiên sản phẩm giúp "
                             f"{emotion_explanations.get(emotion, 'cải thiện tâm trạng')}.",
        "emotion_analysis": f"Người dùng đang cảm thấy '{emotion}'. "
                            f"Nhu cầu dự đoán: "
                            f"{'thư giãn, xả stress' if emotion in ['angry','fearful','sad'] else 'duy trì, tăng cường'}.",
    }


# ============================================================
# PHẦN 4: ĐÁNH GIÁ RE-RANKER (Metrics)
# ============================================================

def compute_ndcg(ranked_ids: list[str],
                 relevant_ids: list[str],
                 k: int = 5) -> float:
    """
    Normalized Discounted Cumulative Gain @k.
    Đây là metric chính trong bài báo để đánh giá re-ranker.
    """
    dcg = 0.0
    for i, pid in enumerate(ranked_ids[:k]):
        if pid in relevant_ids:
            dcg += 1.0 / np.log2(i + 2)

    # IDCG: thứ tự lý tưởng
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_ids), k)))
    return dcg / idcg if idcg > 0 else 0.0


def compute_hit_rate(ranked_ids: list[str],
                     relevant_ids: list[str],
                     k: int = 5) -> float:
    """Hit Rate @k: có ít nhất 1 item relevant trong top-k không."""
    return float(any(pid in relevant_ids for pid in ranked_ids[:k]))


def evaluate_reranker(
    test_cases: list[dict],
    catalog: list[Product],
    use_llm: bool = False,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Đánh giá initial ranker vs LLM re-ranker trên test cases.
    Mỗi test case: {"emotion": str, "context": dict, "relevant_ids": [str]}
    """
    results = []

    for tc in test_cases:
        emotion      = tc["emotion"]
        context      = tc.get("context", {})
        relevant_ids = tc["relevant_ids"]

        # Initial ranking
        candidates   = initial_recommend(emotion, catalog, top_k=5)
        initial_ids  = [p.id for p in candidates]

        # LLM re-ranking
        if use_llm:
            rerank_result = llm_rerank(emotion, context, candidates,
                           api_key=api_key, provider="anthropic")
        else:
            rerank_result = _mock_rerank(candidates, emotion)

        reranked_ids = [r["product_id"]
                        for r in sorted(rerank_result["ranked_products"],
                                        key=lambda x: x["rank"])]

        # Metrics
        initial_ndcg  = compute_ndcg(initial_ids,  relevant_ids)
        reranked_ndcg = compute_ndcg(reranked_ids, relevant_ids)
        initial_hr    = compute_hit_rate(initial_ids,  relevant_ids)
        reranked_hr   = compute_hit_rate(reranked_ids, relevant_ids)

        results.append({
            "Emotion":          emotion,
            "Initial_NDCG@5":   round(initial_ndcg,  4),
            "Reranked_NDCG@5":  round(reranked_ndcg, 4),
            "Improvement_NDCG": round(reranked_ndcg - initial_ndcg, 4),
            "Initial_HR@5":     initial_hr,
            "Reranked_HR@5":    reranked_hr,
            "LLM_Reasoning":    rerank_result.get("overall_reasoning", ""),
        })

    return pd.DataFrame(results)


# ============================================================
# PHẦN 5: PIPELINE TÍCH HỢP SER → GỢI Ý → RE-RANK
# ============================================================

def full_pipeline_demo(emotion: str,
                        catalog: list[Product] = PRODUCT_CATALOG,
                        api_key: Optional[str] = None) -> None:
    """
    Demo pipeline hoàn chỉnh:
    Cảm xúc → Initial Rank → LLM Re-rank → Hiển thị kết quả
    """
    print(f"\n{'='*60}")
    print(f"  PIPELINE: Cảm xúc phát hiện = '{emotion.upper()}'")
    print(f"{'='*60}")

    context = {
        "time_of_day": "evening",
        "device":      "mobile",
        "language":    "vi",
    }

    # Bước 1: Initial recommendation
    candidates = initial_recommend(emotion, catalog, top_k=5)
    print("\n  INITIAL RANKING (rule-based):")
    for i, p in enumerate(candidates):
        print(f"  {i+1}. {p.name} (score={p.score:.2f})")

    # Bước 2: LLM re-rank
    print("\n  LLM RE-RANKING...")
    result = llm_rerank(emotion, context, candidates,
                         api_key=api_key, provider="anthropic")

    print("\n  KẾT QUẢ SAU RE-RANK:")
    print(f"  Phân tích cảm xúc: {result['emotion_analysis']}")
    print(f"  Chiến lược: {result['overall_reasoning']}")
    print()
    for r in result["ranked_products"]:
        pid  = r["product_id"]
        prod = next((p for p in catalog if p.id == pid), None)
        if prod:
            print(f"  #{r['rank']} [{pid}] {prod.name}")
            print(f"      → {r['explanation']}")
            print(f"      → Relevance: {r['relevance_score']:.2f} | "
                  f"Giá: {prod.price:,.0f}đ")


# ============================================================
# MAIN
# ============================================================

def main():
    import os
    API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
    print("\n" + "=" * 65)
    print("  BÀI 5: LLM AS EXPLAINABLE RE-RANKER")
    print("=" * 65)

    # Demo với các cảm xúc từ RAVDESS
    emotions_demo = ["angry", "sad", "happy", "fearful", "neutral"]
    for emotion in emotions_demo:
        full_pipeline_demo(emotion)

    # Đánh giá định lượng
    print("\n" + "=" * 65)
    print("  ĐÁNH GIÁ ĐỊNH LƯỢNG (NDCG@5, HitRate@5)")
    print("=" * 65)

    # Test cases (relevant_ids là ground truth theo cảm xúc)
    test_cases = [
        {"emotion": "angry",   "context": {}, "relevant_ids": ["P003", "P006", "P009"]},
        {"emotion": "sad",     "context": {}, "relevant_ids": ["P004", "P008", "P006"]},
        {"emotion": "happy",   "context": {}, "relevant_ids": ["P005", "P007", "P001"]},
        {"emotion": "fearful", "context": {}, "relevant_ids": ["P004", "P009", "P006"]},
        {"emotion": "neutral", "context": {}, "relevant_ids": ["P002", "P010", "P001"]},
        {"emotion": "calm",    "context": {}, "relevant_ids": ["P001", "P007", "P004"]},
        {"emotion": "disgust", "context": {}, "relevant_ids": ["P003", "P009", "P006"]},
        {"emotion": "surprised","context": {}, "relevant_ids": ["P005", "P007", "P001"]},
    ]

    API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
    df_eval = evaluate_reranker(test_cases, PRODUCT_CATALOG,
                            use_llm=True, api_key=API_KEY)

    print(df_eval[["Emotion", "Initial_NDCG@5",
                    "Reranked_NDCG@5", "Improvement_NDCG",
                    "Initial_HR@5", "Reranked_HR@5"]].to_string(index=False))

    avg_improve = df_eval["Improvement_NDCG"].mean()
    print(f"\n  Cải thiện NDCG@5 trung bình: {avg_improve:+.4f}")

    df_eval.to_csv("results_llm_reranker.csv", index=False)
    print("\n[INFO] Đã lưu: results_llm_reranker.csv")


if __name__ == "__main__":
    import os
    main()