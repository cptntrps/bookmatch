"""Embedded chat agent: reasons about which books a reader would like.

Any OpenAI-compatible chat endpoint with tool calling works (OpenAI, a local
Ollama, a gateway). The agent is grounded two ways: (1) the taste profile and the
neighbor-gap recommendations are in its system prompt, and (2) it has one live
tool, `project_axis`, that ranks the reader's own shelf AND the candidate
recommendations along any semantic axis it names ("more french", "summery",
"less violent"), and reports whether the reader's own ratings follow that axis.
One dot product per book, exposed interactively. See METHOD.md.

Environment:
  BOOKMATCH_CHAT_URL    base URL; unset = api.openai.com
  BOOKMATCH_CHAT_KEY    bearer key (falls back to OPENAI_API_KEY)
  BOOKMATCH_CHAT_MODEL  model name (default gpt-4o-mini)
"""
from __future__ import annotations
import json
import os
import numpy as np
from openai import OpenAI

from . import analyze, embed

CHAT_URL = os.environ.get("BOOKMATCH_CHAT_URL") or os.environ.get("BOOKMATCH_GATEWAY_URL")
CHAT_KEY = (os.environ.get("BOOKMATCH_CHAT_KEY") or os.environ.get("BOOKMATCH_GATEWAY_KEY")
            or os.environ.get("OPENAI_API_KEY"))
CHAT_MODEL = os.environ.get("BOOKMATCH_CHAT_MODEL", "gpt-4o-mini")

_TOOL = {
    "type": "function",
    "function": {
        "name": "project_axis",
        "description": (
            "Rank the reader's OWN shelf and the CANDIDATE recommendations along a "
            "semantic axis you define, e.g. 'more french', 'summery', 'melancholic', "
            "'less violent'. Pure geometry over the books they actually own: use it "
            "to ground any claim about their taste in specific titles, to pick which "
            "candidates fit a direction, and to check (rating_agreement) whether "
            "their own ratings follow the axis. Give a negative pole for 'less X' "
            "or 'X rather than Y' requests. Returns the books at both poles of the "
            "shelf, the candidates at both poles, and the rating agreement."),
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
        self._C: np.ndarray | None = None      # candidate embeddings, built on first use

    @property
    def _client(self) -> OpenAI:
        if self._client_ is None:
            if not CHAT_KEY:
                raise RuntimeError("set BOOKMATCH_CHAT_KEY (or OPENAI_API_KEY) for chat")
            self._client_ = OpenAI(base_url=CHAT_URL, api_key=CHAT_KEY)
        return self._client_

    def _candidates(self) -> np.ndarray:
        if self._C is None:
            self._C = (embed.embed_books(self.recs) if self.recs
                       else np.zeros((0, self.E.shape[1])))
        return self._C

    # ---- the live tool ----
    def project_axis(self, positive: list[str], negative: list[str] | None = None,
                     top: int = 12, label: str = "axis") -> dict:
        pos = embed.embed_terms(positive)
        neg = embed.embed_terms(negative) if negative else None
        C = self._candidates()
        r = analyze.axis_positions(self.books, self.E, pos, neg, label,
                                   self.recs if C.size else None, C if C.size else None)
        name = lambda b: f"{b['title']} — {b.get('author', '')}"
        out = {"positive_pole": [name(b) for b in r["shelf"][:top]],
               "negative_pole": [name(b) for b in r["shelf"][::-1][:top]],
               "rating_agreement": r["rating_agreement"]}
        if r["candidates"]:
            # projected with the shelf's center, so "toward the positive pole" means the same thing
            out["candidates_toward_positive"] = [name(b) for b in r["candidates"][:top]]
            out["candidates_toward_negative"] = [name(b) for b in r["candidates"][::-1][:top]]
        return out

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
            f"about a mood, vibe, or direction ('more french', 'less violent', "
            f"'something for summer', 'a poetry book she'd like'), CALL project_axis "
            f"to see where their own books and the candidates fall on that axis before "
            f"answering; quote the rating_agreement reading when it is informative. "
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
                    result = self.project_axis(args.get("positive", []), args.get("negative"),
                                               label=args.get("label", "axis"))
                except Exception as e:  # surface tool failure to the model, don't crash
                    result = {"error": str(e)}
                messages.append({"role": "tool", "tool_call_id": call.id,
                                 "content": json.dumps(result)})
        # ran out of steps: ask once more without tools for a final answer
        resp = self._client.chat.completions.create(
            model=CHAT_MODEL, messages=messages, temperature=0.7)
        return resp.choices[0].message.content or ""
