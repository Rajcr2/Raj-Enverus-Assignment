"""Hybrid RAG: semantic search (ChromaDB) + BM25 keyword search, fused by rank.

Run `python src/rag.py` to answer the whole question bank and write answers.md.
"""
import math
import os
import re
import time
from collections import Counter
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

from prompts import SYSTEM_PROMPT, build_user_prompt

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_PATH = BASE_DIR / "chroma_embeddings"
COLLECTION_NAME = "agent_as_a_judge"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
TOP_K = int(os.getenv("TOP_K", "8"))  # chunks sent to the LLM
POOL = 20  # candidates taken from each search before fusing
RRF_K = 60  # standard reciprocal-rank-fusion constant

STOPWORDS = set(
    "a an the of in on and or is are was were what which how does do did for to as by "
    "with it its that this from at be been when who many much than using use".split()
)


def tokenize(text):
    words = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?%?", text.lower())  # keeps 44.80% together
    return [w for w in words if w not in STOPWORDS]


class BM25:
    """Tiny BM25 keyword index (no extra library needed)."""

    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(tokenize(d)) for d in docs]
        self.lengths = [sum(t.values()) for t in self.tf]
        self.avg_len = sum(self.lengths) / len(docs)
        df = Counter(w for t in self.tf for w in t)
        n = len(docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def scores(self, query):
        words = tokenize(query)
        out = []
        for tf, length in zip(self.tf, self.lengths):
            score = 0.0
            for w in words:
                if w in tf:
                    norm = tf[w] + self.k1 * (1 - self.b + self.b * length / self.avg_len)
                    score += self.idf[w] * tf[w] * (self.k1 + 1) / norm
            out.append(score)
        return out


class RAG:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        self.collection = client.get_collection(COLLECTION_NAME)

        data = self.collection.get(include=["documents", "metadatas"])
        self.docs = data["documents"]
        self.metas = data["metadatas"]
        self.position = {chunk_id: i for i, chunk_id in enumerate(data["ids"])}
        self.bm25 = BM25(self.docs)
        self.llm = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    def retrieve(self, question, k=TOP_K):
        # 1) semantic search
        query_embedding = self.model.encode([question], normalize_embeddings=True)[0].tolist()
        found = self.collection.query(query_embeddings=[query_embedding], n_results=POOL)
        semantic = [self.position[chunk_id] for chunk_id in found["ids"][0]]

        # 2) keyword search
        scores = self.bm25.scores(question)
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:POOL]
        keyword = [i for i in ranked if scores[i] > 0]

        # 3) reciprocal rank fusion scores every candidate (used for ordering)
        fused = {}
        for ranking in (semantic, keyword):
            for rank, i in enumerate(ranking):
                fused[i] = fused.get(i, 0) + 1 / (RRF_K + rank + 1)

        # each search is guaranteed half of the slots; the rest are filled by fused rank
        best = []
        for i in semantic[: k // 2] + keyword[: k // 2] + sorted(fused, key=fused.get, reverse=True):
            if i not in best and len(best) < k:
                best.append(i)
        best.sort(key=fused.get, reverse=True)

        evidence = []
        for i in best:
            found_by = "both" if i in semantic and i in keyword else ("semantic" if i in semantic else "keyword")
            evidence.append({
                "text": self.docs[i],
                "page": self.metas[i]["page"],
                "chunk": self.metas[i]["chunk"],
                "score": fused[i],
                "found_by": found_by,
            })
        return evidence

    def answer(self, question, k=TOP_K):
        evidence = self.retrieve(question, k=k)
        context = "\n\n".join(
            f"[Page {e['page']}, Chunk {e['chunk']}]\n{e['text']}" for e in evidence
        )
        response = self.llm.chat.completions.create(
            model=LLM_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(context, question)},
            ],
        )
        return {"answer": response.choices[0].message.content.strip(), "evidence": evidence}


# ---- question bank (Q4-Q18 of the form), used by `python src/rag.py` ----
QUESTIONS = {
    4: "What is the DevAI dataset, and how many tasks, requirements, and preferences does it contain?",
    5: "What percentage of evaluation time and cost does Agent-as-a-Judge save compared to using three human experts?",
    6: "According to Section 4.4 (Cost Analysis), how much did Agent-as-a-Judge cost and how long did it take, compared to Human-as-a-Judge?",
    7: "Which three open-source agentic frameworks were benchmarked on DevAI?",
    8: "What is the average cost and average time for OpenHands?",
    9: "Which system is the most cost-efficient and which is the most expensive?",
    10: "What was GPT-Pilot's \"Requirements Met (Independent)\" percentage under Human-as-a-Judge?",
    11: "What was MetaGPT's Task Solve Rate?",
    12: "In the black-box setting, what Alignment Rate did Agent-as-a-Judge achieve when evaluating OpenHands?",
    13: "What alignment rate does Agent-as-a-Judge achieve using only the \"task component,\" and after adding graph \"read\" and \"locate\"?",
    14: "Which search algorithm (BM25, Sentence-BERT, Fuzzy Search, or no search module) gave the best alignment rate?",
    15: "Which two model architectures are mentioned most frequently in the DevAI user queries?",
    16: "What is requirement R1 in the \"Devin AI Software Engineer Plants Secret Messages in Images\" task?",
    17: "Which of the three human evaluators (231a, 38bb, cn90) made the most errors when judging GPT-Pilot, and what was the error rate?",
    18: "In the diagram comparing LLM-as-a-Judge, Agent-as-a-Judge, and Human-as-a-Judge, what key drawback is highlighted for Human-as-a-Judge?",
}
PAUSE_SECONDS = 12  # stay under the free-tier tokens-per-minute limit

if __name__ == "__main__":
    rag = RAG()
    with open(BASE_DIR / "Answers.md", "w", encoding="utf-8") as f:
        f.write("# Answers from the RAG chatbot\n")
        for number, question in QUESTIONS.items():
            try:
                result = rag.answer(question)
            except Exception as exc:  # most likely a rate limit: wait and retry once
                print(f"Q{number} failed ({exc}); retrying in 40s")
                time.sleep(40)
                result = rag.answer(question)
            print(f"Q{number}: {result['answer']}")
            f.write(f"\n## Q{number}. {question}\n\n**Answer:** {result['answer']}\n\n**Retrieved chunks:**\n")
            for i, e in enumerate(result["evidence"], start=1):
                f.write(f"\n{i}. Page {e['page']}, Chunk {e['chunk']} ({e['found_by']})\n   > {e['text']}\n")
            f.flush()
            time.sleep(PAUSE_SECONDS)
    print(f"Saved {BASE_DIR / 'answers.md'}")
