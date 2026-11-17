# Language models inside links: project notes

## Goal
Shayon's project: "How good can a language model be at each size a link allows?"
One model per link size, each a clickable link that runs in the browser; the
smallest is inside a QR code. It is live at https://q.shayon.work and is meant
to reach the Hacker News front page and bring job offers. Every post and the
site must say he is available for hire ("I'm looking for my next role, Berlin
or remote: hi@shayon.work").

## Current state (all measured)
| Link | Model | URL chars | bits/char |
|---|---|---:|---|
| QR v30, 137 modules (`out/qr_names_url.txt`) | names MLP, 1,984 weights, 31 levels | 3,429 | 2.313 (validation split, also used for selection) |
| QR-sized (`out/qrstory_url.txt`) | story MLP, 1,895 weights | 3,352 | 2.480 (reduced alphabet) |
| 64 KB (`out/t64kfix_url.txt`) | story transformer, 90,420 weights | 64,529 | 1.610 |
| 1 MB (`out/big2_url.txt`) | story transformer, 1,498,640 weights | 1,010,424 | 1.104 |
Story scores are on the fixed 500k-character TinyStories validation slice
(`data/heldout.u8`, from `stories/get_data.py`). What was tried and lost is in
`docs/findings.md`; the QR transformer record is `docs/RESULTS_qrtf.md`.

## Layout
- `site/`: served as-is by Cloudflare Pages (no build step). `index.html` is the
  loader (both link formats), `home/` the launch page, `data/payloads/` copies of
  the four links, `img/qr.png` and `img/og.png`, `_headers` (CSP etc.).
- `names/`: `chatnames.txt` (from LiteLLM via `clean2.py`), `train2.py` (split,
  data, GRU/MLP), `teacher.py` (teacher, its samples, distilled init),
  `qrtf.py` (QR-size MLP/attention training and exact export),
  `qrtf_mlp_best.pt` (the QR model), `qrtf_best.pt` (best v30 attention model),
  `payload_url_mlp_qrtf.txt` (temperature-1 export), `verify_qrtf.js`,
  `qrtf_table.json` (all 56 exports).
- `stories/`: `stories.py` (transformer train/export), `qrstory.py` (QR-sized
  MLP), `get_data.py`, `score_story.py`, `stories_runtime.js`, verifiers, tests.
- `out/`: the four links, their `*_meta.json`, `qrstory_L31_h32_k3_e6.pt`.
- `tools/`: `check.py` (run before every deploy), `browser_test.py`,
  `make_qr.py` (temperature 0.8 folded in; writes the QR link and images),
  `qr.py`, `host.json` (single URL prefix source) with `host.py`/`host.cjs`.

## Invariants
- `uv run python tools/check.py` must pass (static, Node, Firefox + Chromium).
- `tools/make_qr.py` must reproduce `out/qr_names_url.txt` and `site/img/qr.png`
  byte-identically; `names/qrtf.py export names/qrtf_mlp_best.pt` must reproduce
  `names/payload_url_mlp_qrtf.txt`; `stories/qrstory.py export ... --publish`
  must reproduce `out/qrstory_url.txt`.
- Do not change model fragment bytes or `site/` payloads without regenerating
  and re-checking them.

## Deployment
Cloudflare Pages project `link-models` serves `site/` from the `main` branch.
Run `tools/check.py` before pushing.

## Working notes
- Report measured numbers only; label estimates as estimates.
- The names validation split is seeded in `names/train2.py`; keep it fixed so
  results stay comparable. It was also used for selection; say so.
- The end-of-name token is `\n` (index 0).
- Laptop has frozen from memory pressure: torch threads <= 2, one browser test
  process at a time.
- Not done: no real-phone scan results are recorded in the repo; browser URL limits for the
  1 MB link vary by browser and app; no rude-word filter on names.
- Older material (CV scripts, 40 KB tier, n-gram tables, scan simulations, lab
  notes, first prototype) was pruned before going public; it is in git history.
