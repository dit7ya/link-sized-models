# What I tried, and what it scored

All names scores are bits per character on the fixed validation split in
`names/train2.py` (151 held-out names, 2,521 characters including the end of
each name). Lower is better.

**Caveat.** That validation split was also used to choose checkpoints,
architectures and quantisers, including follow-up runs picked after seeing
earlier results. So the QR-size comparisons below are validation scores with
selection bias, not clean test scores. Read small gaps (a few hundredths of a
bit) with that in mind.

## First round: the original QR model (1,689 weights, 61 levels per weight)

| Experiment | Result |
|---|---|
| Same-size letter-pair table vs neural net (first, noisier list) | 3.60 vs 2.75: the net wins |
| Letter-triple table | 2.51, but about 6.9 KB, 3x the whole QR code |
| Cleaning the list to chat models only (1,215 names) | biggest gain: 2.75 to about 2.36 (different test sets) |
| Recurrent nets (RNN, GRU) at the same size | slightly worse: 2.42 to 2.54 |
| Longer window (4 or 5 previous characters) | worse |
| Position counter (+24 weights) | over-long names from about 20% to 4-7% |
| 200 judge-picked names added to training | no measurable gain (small batch) |
| 19k unjudged teacher samples added | small gain: 2.39 to 2.36 |
| Word pieces (BPE, 24 to 64 merges) | worse: 2.46 to 2.53 |

Conclusion at the time: at about 1.7k weights the limit is capacity, not
training tricks.

## Second round: transformers at QR size, and a retuned MLP

Every row is the exact exported program, run in Node and checked against the
Python model (largest log-probability difference about 1e-14). The table of
all 56 exports is in `RESULTS_qrtf.md` and `names/qrtf_table.json`.

| Model | QR version | Weights | URL chars | bits/char |
|---|---:|---:|---:|---:|
| MLP, 29 hidden, 31 levels, learned scales (**in the QR code now**) | 30 | 1,984 | 3,429 | 2.313 |
| MLP, 32 hidden, same recipe | 31 | 2,161 | 3,693 | 2.295 |
| Original MLP, 24 hidden, 61 levels | 30 | 1,689 | 3,492 | 2.355 |
| Best attention model that fits version 30 (`tap21_h25_lsq_fresh_s0`) | 30 | 1,772 | 3,102 | 2.417 |
| Best attention model up to version 31 (`tap21_lsq_fine_s1`) | 31 | 1,949 | 3,337 | 2.352 |
| Best standard transformer block (`tf12f16_value_norm45_s0`) | 31 | 1,171 | 2,880 | 2.635 |

- **Attention lost at this size.** The attention program is about 290
  characters longer than the MLP program, and program bytes cost more QR bits
  than weight digits, so it leaves less room for weights.
- **Coarser weights with learned scales helped most.** 31 levels instead of 61,
  with a learned scale per tensor, made room for 29 hidden units instead of 24.
  Matched control (28 hidden, 31 levels, seed 0): max-based quantisation 2.423,
  learned scales 2.319.
- **Seed spread is a few hundredths:** 2.319, 2.331 and 2.386 for three
  from-scratch seeds of the 28-hidden MLP.

## Against a table, with cross-validation

An earlier five-fold cross-validation on names (scripts no longer in the tree,
still in git history) put the original QR network at 2.40 bits/char against
3.40 for a character table of the same link size. That was the earlier QR
model, not the one in the QR code now.

## Story models

Scored on a fixed 500,000-character slice of the TinyStories validation set by
running the exported program. A character n-gram table packed into the same
link sizes (code also only in git history) scored 1.877 at 64 KB and 1.347 at
1 MB, against 1.610 and 1.104 for the transformers. The QR-sized story MLP
scores 2.480 on a reduced alphabet, so it is not comparable with the others.
