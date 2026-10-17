# Method: conversational concept axes over a personal library

First implemented 2026-07-05 (private repository). Published 2026-10-17 with this
document. Reference implementation: `bookmatch/analyze.py`, `bookmatch/agent.py`,
`bookmatch/cli.py`.

## Problem

A reader asks a recommender for a *direction*, not an item: "something more
French", "less violent than what I usually read", "a summer book she would like".
Similarity search answers "more like X" well and "less X" badly, because a single
query vector has no opposite. Attribute filters need labels nobody has written for
a private shelf.

## Method

1. **Embed the shelf.** Every book the reader has read is embedded once with a
   pretrained text-embedding model, from `title by author. description`. No
   training, no labels. The same model embeds everything that follows.
2. **Build the taste fingerprint.** From the embeddings and the reader's own
   ratings: the rating-weighted *taste axis* (loved vs tolerated), *contrarian*
   books (own rating minus community average, both signs), the shelf's principal
   *content axes* (SVD), and the *antipode* (the region least like what they
   love). The fingerprint goes into a chat model's system prompt together with
   collaborative-filtering candidates (books that the reader's nearest neighbours
   in a large public corpus love and the reader does not own).
3. **The model names an axis in conversation.** When the dialogue turns to a
   direction or mood, the model calls one tool, `project_axis(label, positive,
   negative)`, choosing the concept phrases itself: for "less violent",
   `positive = ["gentle, quiet domestic fiction"]`, `negative = ["graphic
   violence", "war novel"]`.
4. **Axis from phrases, no training.** Each phrase is embedded with the same
   model. The axis is the unit vector `mean(positive) - mean(negative)`; with no
   negative pole it is `mean(positive)` alone. This takes milliseconds.
5. **Project in the shelf's frame.** Every shelf book is scored by one dot
   product of its mean-centered embedding with the axis. The *candidates* are
   projected with the **same center** (the shelf mean), so "toward the positive
   pole" means the same thing for a book the reader owns and a book being
   proposed. Both lists are returned, at both poles.
6. **Check the axis against the reader's own ratings.** The Spearman rank
   correlation between projection and rating, over rated books, is returned with
   the ranking and a one-sentence reading ("the reader rates books toward the
   'french' pole higher: a weak relation, Spearman +0.21, n=281"). Too few rated
   books, or no variance, is reported as unknown, never as zero.
7. **Feed it back.** The tool result returns to the conversation. The model grounds
   its next recommendation in where the reader's own books and the candidates fall
   on the axis, and in whether the axis predicts the reader's ratings at all.

## Properties

- **Unsupervised and instant.** Any concept the model can phrase becomes an axis.
  No per-user training, no attribute labels, no retrieval index.
- **Two poles handle "less X".** A contrast axis ranks in both directions; a
  single query vector cannot express "decrease".
- **One frame for shelf and candidates.** Centering the candidates on the shelf
  mean keeps the ranking comparable across the two sets.
- **A self-check travels with the ranking.** The rating agreement tells the model,
  and the reader, whether the axis explains anything about this person before it
  is used to recommend.

## Relation to prior work

Concept directions in embedding spaces are established: word-vector analogies,
concept activation vectors, and personalized "soft attribute" critiquing on
collaborative-filtering embeddings (Göpfert et al., WWW 2022), latent linear
critiquing (Luo et al., WWW 2020). Conversational recommenders that give a language
model tools and the user's history are also established (for example
TalkPlay-Tools, 2025). What this repository combines, and what we have not found
elsewhere: the model names the poles mid-conversation; the axis is built from those
phrases in a pretrained text-embedding space without training; the reader's own
library and the candidates are ranked in one frame; the ranking carries its
agreement with the reader's own ratings; and the result returns into the dialogue.
SteerEval (2026) reports that similarity-only rankers handle "decrease" requests
poorly; the two-pole axis is a direct answer to that case.

## Verification in this repository

- `tests/test_more_french.py`: on a real 309-book shelf, an axis built only from the
  phrases "French literature" vs "American literature" ranks the unambiguously
  French author (Duras) in the top 10% and surfaces the European literary authors
  near the top, with zero labels.
- `tests/test_analyze.py`, `tests/test_cli.py`: projection in a shared frame and
  the rating agreement on synthetic geometry, including the honest-unknown cases.
- `tests/test_agent.py`: the live tool ranks candidates in the shelf's frame and
  returns a quotable agreement reading.
