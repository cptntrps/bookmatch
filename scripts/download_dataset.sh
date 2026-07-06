#!/usr/bin/env bash
# Download the UCSD Goodreads corpus (Wan & McAuley 2018) — direct HTTP, resumable.
# Target: ~/data/goodreads-ucsd/raw  (gitignored, ~6.5 GB)
set -euo pipefail
BASE="https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads"
DEST="${1:-$HOME/data/goodreads-ucsd/raw}"
mkdir -p "$DEST"
FILES=(
  goodreads_interactions.csv        # ~4.3 GB  user_id,book_id,is_read,rating,is_reviewed (remapped ids)
  book_id_map.csv                   # ~38 MB   remapped book_id <-> real goodreads book_id
  goodreads_books.json.gz           # ~2.1 GB  real book_id -> title, author_ids
  goodreads_book_authors.json.gz    # ~18 MB   author_id -> name
)
for f in "${FILES[@]}"; do
  echo "[$(date +%H:%M:%S)] fetching $f"
  curl -fL -C - --retry 5 --retry-delay 5 -m 7200 -o "$DEST/$f" "$BASE/$f"
  echo "[$(date +%H:%M:%S)] done $f -> $(du -h "$DEST/$f" | cut -f1)"
done
echo "ALL DOWNLOADS COMPLETE -> $DEST"
ls -la "$DEST"
