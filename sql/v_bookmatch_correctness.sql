-- books.v_bookmatch_correctness — bookmatch's CORRECTNESS surface (ADR-0071 reference-slot,
-- registry.product.bookmatch.correctness_view_ref; ops.actionables 54975 / portfolio gap 54429).
--
-- Correctness here means: for each Goodreads account bookmatch has ingested, did the recommender
-- pipeline actually complete and produce a GROUNDED result — shelf ingested, taste-axis embeddings
-- present for every shelf row, an analysis computed, the UCSD corpus actually matched something,
-- and a non-empty recommendation set + taste profile persisted. A row that answers false on any
-- assertion is a silent-wrong result: the service still returns 200 and the UI still renders.
--
-- NOT health. Health is a live probe (systemd bookmatch.service + GET :8092/api/health with
-- ok:true AND corpus_ready:true) and is deliberately declared not-applicable for the view slot.
--
-- match_rate note: since ops.actionables:131533 the fetcher deduplicates editions BEFORE
-- embedding and matching (rss.dedupe_editions), so match_rate and books.shelf_books are now
-- computed over the same list and the delta should be ~0. It stays EXPOSED as a column rather
-- than asserted on: a shelf ingested before that change still carries the old skew, and a tight
-- equality would fire falsely on it.
--
-- Owner: bookmatch-po. Read-only view; no dependency on service code.

CREATE OR REPLACE VIEW books.v_bookmatch_correctness AS
WITH shelf AS (
    SELECT account_id,
           count(*)               AS shelf_books,
           count(embedding)       AS shelf_embedded,
           count(matched_book_id) AS shelf_corpus_matched
      FROM books.shelf_books
     GROUP BY account_id
)
SELECT
    a.id                                                   AS account_id,
    a.user_id,
    a.label,
    COALESCE(sh.shelf_books, 0)                            AS shelf_books,
    COALESCE(sh.shelf_embedded, 0)                         AS shelf_embedded,
    COALESCE(sh.shelf_corpus_matched, 0)                   AS shelf_corpus_matched,
    an.match_rate,
    jsonb_array_length(COALESCE(an.recs, '[]'::jsonb))     AS n_recs,
    an.computed_at,

    -- falsifiable per-account assertions
    COALESCE(sh.shelf_books, 0) > 0                        AS ok_shelf_ingested,
    COALESCE(sh.shelf_books, 0) > 0
        AND sh.shelf_embedded = sh.shelf_books             AS ok_embeddings_complete,
    an.account_id IS NOT NULL                              AS ok_analysis_present,
    COALESCE(an.match_rate, 0) > 0
        AND COALESCE(sh.shelf_corpus_matched, 0) > 0       AS ok_corpus_grounded,
    jsonb_array_length(COALESCE(an.recs, '[]'::jsonb)) > 0 AS ok_recs_present,
    jsonb_typeof(an.profile) = 'object'
        AND an.profile <> '{}'::jsonb                      AS ok_profile_present,

    -- informational: raw-vs-deduped match accounting (see header note)
    round((COALESCE(an.match_rate, 0)
           - COALESCE(sh.shelf_corpus_matched, 0)::numeric
             / NULLIF(sh.shelf_books, 0))::numeric, 4)     AS match_rate_dedup_delta,

    (COALESCE(sh.shelf_books, 0) > 0
     AND sh.shelf_embedded = sh.shelf_books
     AND an.account_id IS NOT NULL
     AND COALESCE(an.match_rate, 0) > 0
     AND COALESCE(sh.shelf_corpus_matched, 0) > 0
     AND jsonb_array_length(COALESCE(an.recs, '[]'::jsonb)) > 0
     AND jsonb_typeof(an.profile) = 'object'
     AND an.profile <> '{}'::jsonb)                        AS is_correct,

    NULLIF(concat_ws('; ',
        CASE WHEN COALESCE(sh.shelf_books, 0) = 0
             THEN 'shelf not ingested' END,
        CASE WHEN COALESCE(sh.shelf_books, 0) > 0 AND sh.shelf_embedded < sh.shelf_books
             THEN 'taste-axis embeddings missing on '
                  || (sh.shelf_books - sh.shelf_embedded) || ' shelf rows' END,
        CASE WHEN an.account_id IS NULL
             THEN 'no analysis computed' END,
        CASE WHEN an.account_id IS NOT NULL
              AND (COALESCE(an.match_rate, 0) = 0 OR COALESCE(sh.shelf_corpus_matched, 0) = 0)
             THEN 'ungrounded: UCSD corpus matched nothing' END,
        CASE WHEN an.account_id IS NOT NULL
              AND jsonb_array_length(COALESCE(an.recs, '[]'::jsonb)) = 0
             THEN 'empty recommendation set' END,
        CASE WHEN an.account_id IS NOT NULL
              AND (jsonb_typeof(an.profile) IS DISTINCT FROM 'object'
                   OR an.profile = '{}'::jsonb)
             THEN 'empty taste profile' END
    ), '')                                                 AS defect
FROM books.accounts a
LEFT JOIN shelf sh ON sh.account_id = a.id
LEFT JOIN books.analyses an ON an.account_id = a.id;

COMMENT ON VIEW books.v_bookmatch_correctness IS
    'bookmatch correctness surface (ADR-0071 correctness_view_ref, row 54975): per Goodreads '
    'account, did the ingest -> embed -> corpus-match -> recommend pipeline produce a grounded, '
    'non-empty result. is_correct=false with a defect string = a silent-wrong recommendation set. '
    'Health is NOT here (live systemd + /api/health ok+corpus_ready probe). Owner: bookmatch-po.';
