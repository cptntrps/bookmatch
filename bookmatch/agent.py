"""Embedded chat agent — reasons about which books a reader would like.

Runs on the livingos-api gateway (bare-opus, OpenAI schema). The agent is grounded
two ways: (1) the taste profile + neighbor-gap recs are in its system prompt, and
(2) it has a live `project_axis` tool that ranks the reader's actual shelf along any
semantic axis it names ("more french", "summery", "melancholic") — the June
"impossible insight" trick, one dot product per book, exposed interactively.
"""
from __future__ import annotations
import json
import os
import numpy as np
from openai import OpenAI

from . import analyze, embed

GATEWAY_URL = os.environ.get("BOOKMATCH_GATEWAY_URL", "http://127.0.0.1:8510/v1")
CHAT_MODEL = os.environ.get("BOOKMATCH_CHAT_MODEL", "bare-opus")

_TOOL = {
    "type": "function",
    "function": {
        "name": "project_axis",
        "description": (
            "Rank the reader's OWN shelf along a semantic axis you define, e.g. "
            "'more french', 'summery', 'melancholic', 'experimental'. Pure geometry "
            "over the books they actually own — use it to ground any claim about "
            "their taste in specific titles, and to reason about what adjacent books "
            "they'd like. Returns the books at the positive and negative poles."),
        "parameters": {
            "type": "object",
            "properties": {
                "label": {"type": "string", "description": "short axis name, e.g. 'french'"},
                "positive": {"type": "array", "items": {"type": "string"},
                             "description": "concept phrases for the positive pole, "
                             "e.g. ['French literature','a novel set in Paris']"},
                "negative": {"type": "array", "items": {"type": "string"},
                             "description": "optional phrases for the opposite pole, "
                             "e.g. ['American literature']"},
            },
            "required": ["label", "positive"],
        },
    },
}


class BookAgent:
    def __init__(self, books: list[dict], embeddings: np.ndarray,
                 profile: dict, recs: list[dict], label: str = "this reader"):
        self.books = books
        self.E = embeddings
        self.profile = profile
        self.recs = recs
        self.label = label
        self._client_ = None

    @property
    def _client(self) -> OpenAI:
        if self._client_ is None:
            self._client_ = OpenAI(base_url=GATEWAY_URL,
                                   api_key=os.environ["BOOKMATCH_GATEWAY_KEY"])
        return self._client_

    # ---- the live tool ----
    def project_axis(self, positive: list[str], negative: list[str] | None = None,
                     top: int = 12) -> dict:
        pos = embed.embed_terms(positive)
        neg = embed.embed_terms(negative) if negative else None
        axis = analyze.axis_from_vectors(pos, neg)
        proj = analyze.project_axis(self.E, axis)
        order = np.argsort(-proj)
        fmt = lambda i: f"{self.books[i]['title']} — {self.books[i].get('author','')}"
        return {"positive_pole": [fmt(i) for i in order[:top]],
                "negative_pole": [fmt(i) for i in order[::-1][:top]]}

    # ---- system grounding ----
    def _system(self) -> str:
        p = self.profile
        def names(key, k=6):
            return "; ".join(f"{b['title']} ({b['author']})" for b in p.get(key, [])[:k])
        recs = "; ".join(f"{r['title']} ({r['author']})" for r in self.recs[:12])
        return (
            f"You are a well-read literary friend helping think about what {self.label} "
            f"would enjoy reading next. You know their Goodreads shelf intimately.\n\n"
            f"TASTE PROFILE ({p.get('n_rated')} rated, mean {p.get('mean_rating')}):\n"
            f"- Loves most: {names('loved')}\n"
            f"- Merely tolerates: {names('tolerated')}\n"
            f"- Loves more than the crowd does: {names('loves_more_than_world')}\n"
            f"- Blind spot (unlike what they love): {names('blind_spot')}\n\n"
            f"NEIGHBOR-GAP RECOMMENDATIONS (books that readers nearest them in a "
            f"876k-user corpus love, that they don't own yet): {recs}\n\n"
            f"Ground your suggestions in these and in the shelf. When a question is "
            f"about a mood, vibe, or direction ('more french', 'something for summer', "
            f"'a poetry book she'd like'), CALL project_axis to see where their own "
            f"books fall on that axis before answering, then recommend accordingly. "
            f"Be specific and concise; name real books and say why they fit.")

    def chat(self, history: list[dict], user_msg: str, max_steps: int = 4) -> str:
        messages = [{"role": "system", "content": self._system()}]
        messages += history
        messages.append({"role": "user", "content": user_msg})
        for _ in range(max_steps):
            resp = self._client.chat.completions.create(
                model=CHAT_MODEL, messages=messages, tools=[_TOOL], temperature=0.7)
            msg = resp.choices[0].message
            if not msg.tool_calls:
                return msg.content or ""
            messages.append(msg.model_dump(exclude_none=True))
            for call in msg.tool_calls:
                args = json.loads(call.function.arguments or "{}")
                try:
                    result = self.project_axis(args.get("positive", []),
                                               args.get("negative"))
                except Exception as e:  # surface tool failure to the model, don't crash
                    result = {"error": str(e)}
                messages.append({"role": "tool", "tool_call_id": call.id,
                                 "content": json.dumps(result)})
        # ran out of steps — ask once more without tools for a final answer
        resp = self._client.chat.completions.create(
            model=CHAT_MODEL, messages=messages, temperature=0.7)
        return resp.choices[0].message.content or ""
