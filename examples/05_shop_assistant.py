"""Example 5: RAG in a sales website — where and how it's used.

Scenario: an online electronics store (data/shop/). RAG powers three
concrete features on a real sales site:

  1. customer-service Q&A — "Is the AirBook 14 good for travel?"
     → retrieves product specs + customer reviews, answers with citations
  2. semantic product search — "noise-cancelling earbuds under 10000"
     → finds matching products without keyword matching
  3. policy Q&A — "Can I return this? Within how many days?"
     → retrieves the shipping & returns policy

Tip: the shop docs are in Traditional Chinese. For Chinese documents,
use a multilingual embedding model, e.g.:
    RAG_EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2 \\
        python examples/05_shop_assistant.py

Run from the repo root:
    python examples/05_shop_assistant.py                # preset demo
    python examples/05_shop_assistant.py --ask "..."    # your own question
    python examples/05_shop_assistant.py --ask "..." --answer   # + LLM answer (needs Ollama)
"""
from __future__ import annotations

import argparse

from src.rag_example.config import Settings
from src.rag_example.pipeline import RAGPipeline

# (scenario title, how the website uses it, demo questions)
SCENARIOS: list[tuple[str, str, list[str]]] = [
    (
        "場景一：客服問答",
        "網站這樣用：客服聊天機器人收到商品問題時，先檢索規格＋評價再回答，"
        "答案附上商品連結，答不出來就轉真人。",
        [
            "AirBook 14 適合帶出門工作嗎？",
            "SoundPods Pro 有什麼缺點？",
        ],
    ),
    (
        "場景二：語意商品搜尋",
        "網站這樣用：搜尋框不再只比對關鍵字。「輕便、電池久的筆電」這種描述，"
        "直接對應到商品，適合放在首頁搜尋或「幫我挑」功能。",
        [
            "預算一萬以內、降噪好的耳機",
            "剪影片用的筆電",
        ],
    ),
    (
        "場景三：政策問答",
        "網站這樣用：運費、退貨、保固這類重複問題，機器人直接引用政策文件回答，"
        "減少客服工單。回答必須附出處，政策更新只要換文件重建索引。",
        [
            "買了不喜歡可以退嗎？幾天內？",
            "耳機保固多久？",
        ],
    ),
]


def show_hits(pipeline: RAGPipeline, question: str) -> None:
    print(f"Q: {question}")
    for hit in pipeline.retrieve(question):
        print(f"  [{hit.score:.2f}] {hit.chunk.title}（{hit.chunk.source}）")
        print(f"  {hit.chunk.text[:120]}...")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG shop-assistant demo.")
    parser.add_argument("--ask", help="ask your own question instead of the preset demo")
    parser.add_argument("--answer", action="store_true",
                        help="also generate an LLM answer (needs Ollama)")
    args = parser.parse_args()

    settings = Settings()
    settings.docs_dir = "data/shop"
    pipeline = RAGPipeline(settings)
    print(f"已索引 {len(pipeline.ingest())} 個 chunks（賣場文件）\n")
    if settings.embedding_model == "all-MiniLM-L6-v2":
        print("提示：賣場文件是中文，建議用多語言 embedding 模型效果更好：")
        print("  RAG_EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2 "
              "python examples/05_shop_assistant.py\n")

    if args.ask:
        questions = [("你的問題", "", [args.ask])]
    else:
        questions = SCENARIOS

    for title, usage, qs in questions:
        if not args.ask:
            print(f"{title}\n{usage}\n")
        for q in qs:
            if args.answer:
                answer = pipeline.ask(q)
                print(f"Q: {answer.question}\nA: {answer.text}\n")
                for i, hit in enumerate(answer.sources, start=1):
                    print(f"  [{i}] {hit.chunk.title}（{hit.chunk.source}）")
                print()
            else:
                show_hits(pipeline, q)


if __name__ == "__main__":
    main()
