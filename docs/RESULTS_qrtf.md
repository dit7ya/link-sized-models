# QR transformer experiment

Detailed record of the QR-size attention runs and the retuned MLP that is now in the QR code. Written when the run happened; file names are as they are now, and files that were pruned from the tree (per-run JSON, v31 exports, PNGs, the ablation and protocol records, the Firefox check) are still in git history. Short summary: `findings.md`.

Preferred v30 attention model: **tap21_h25_lsq_fresh_s0**, **2.417461 bits/char**, QR **v30-L**, 3,102 URL characters, 732 program bytes, 2 slack bits. It is 0.061998 bpc behind the exact published MLP and meets the stated 0.10-bpc replacement threshold. 

Best accuracy up to v31: `tap21_lsq_fine_s1` at **2.352224 bpc**, QR v31, exported separately at the time (now only in git history). The primary artifact prefers v30 because it satisfies the stated original-MLP + 0.10 bpc rule.

The best conventional block transformer within v31 is `tf12f16_value_norm45_s0` at **2.634988 bpc**. The winning `tap` model is a two-head, strongly local causal attention encoder followed by a tanh MLP classifier; the preferred v30 model was trained from scratch with distillation and learned quantization scales. The v31 accuracy alternative transfers and fine-tunes a QAT MLP checkpoint. It reads a single current character embedding, retains all previous embeddings, and computes content-dependent causal softmax attention with Gaussian relative biases centered one and two positions back. It does not read an explicit three-character window before attention. `skip=0` omits the optional classifier residual.

Best checked MLP: `mlp31_h32_lsq_fine_s0` at **2.294589 bpc**, v31. The published model's exact rounded runtime scores **2.355463422651**, versus the historical unrounded-scale score **2.355525732102**. These differ by 0.00006231 bpc; the main table consistently scores the actual executable.

The improved MLP gains 0.060875 validation bpc. The primary improved-MLP artifact preserves v30 (`names/payload_url_mlp_qrtf.txt`, readable program `names/runtime_mlp_qrtf.js`). These remain validation-selected gains.

At v30, the MLP is 0.104209 bpc better than the preferred transformer. Up to v31, the MLP is 0.057636 bpc better than the best attention model. The transformer meets the original replacement rule; the MLP has lower loss at both QR tiers.

Best v30 MLP: `mlp31_h29_lsq_fine_s0`, **2.313252 bpc**, selected as the primary improved-MLP artifact `names/payload_url_mlp_qrtf.txt`; checkpoint `names/qrtf_mlp_best.pt`. `tools/make_qr.py` then folds in temperature 0.8 to make the link in the QR code. Both improved MLP choices and both transformer choices retain all validation probability checks.

The preferred v30 transformer uses 13,878 QR data bits versus 13,624 for the v30 MLP, despite URLs of 3,102 versus 3,429 characters. Program bytes cost more QR bits than decimal weight digits.

## Protocol

The original `names/train2.py` split is unchanged: 1064 training names, 151 validation names, and 2,521 validation targets including EOS. Ordered validation-name SHA-256: `c7b005d55786b84376ea0282b71534b2d2a7cb08bcd17127839d94c93b3599dd`. Teacher data are `names/teacher.pt` and 19,125 saved samples in `names/teacher_samples.json` (made by `names/teacher.py`). Validation names are asserted absent from the sampled augmentation. Soft targets are cached in float16 only for training names and teacher samples, then converted to float32 for the loss.

Parameter counts are quantized weights; embedded scale constants are counted in program bytes. All reported artifact losses use float64 dequantized weights with the exact scale literals used by JavaScript, at temperature 1. Training uses AdamW, cosine decay, modest batches of 64, gradient clipping, max-based QAT or learned positive quantization scales, and hard/teacher-soft loss mixtures. Biases are quantized during training. Learned scales use no weight decay; their initial values are reset after float warmup. Earlier runs used four Torch threads; later runs use two. Each run is individually bounded at 870 seconds.

Checkpoint selection and architecture/quantizer choices used this same fixed validation split, including adaptive follow-up experiments. These are selection-biased validation measurements, not independent test scores or the site's five-fold CV results. Initial states are eligible in later runs; `--no-initial` exactly reproduces earlier selection behavior. The teacher cache now preserves RNG state; `--legacy-cache-rng` reproduces the first run's historical RNG consumption.

Sampling uses the exact exported `g()`, temperature 0.8, 2,000 unfiltered draws, LCG seed 1. Novel means a draw does not exactly equal any of the 1,215 real names; repeated novel draws still count. Names are uncapped. High novelty alone does not establish plausibility. Payload bytes exclude the 23-byte uppercase host prefix and include the three-digit length header, program and decimal weights; URL bytes equal URL characters because the payloads are ASCII. QR versions/slack include mode headers and the host.

## Quantization and seed controls

Matched h=28, 31-level, seed-0 controls with the same soft-target weight and training schedule: max-based QAT **2.422809 bpc**, learned scales/clipping **2.319355**. The stored-weight count is identical; the measured gain for this pair is 0.103454 bpc.

Seeds 0/1/2, v31 attention fine-tuning: **2.354723, 2.352224, 2.353515 bpc** after checkpoint selection. Fine-tuning seeds share a common initialization; from-scratch seeds vary both initialization and minibatches.

Seeds 0/1/2, h=32 MLP fine-tuning: **2.294589, 2.308294, 2.299934 bpc** after checkpoint selection. Fine-tuning seeds share a common initialization; from-scratch seeds vary both initialization and minibatches.

Seeds 0/1/2, h=28 MLP from scratch: **2.319355, 2.330529, 2.385694 bpc** after checkpoint selection. Fine-tuning seeds share a common initialization; from-scratch seeds vary both initialization and minibatches.

The search was not exhaustive: no new mixed-radix per-tensor-level, pruning or entropy-coded training sweep was performed. All measured exports here use one uniform radix with per-tensor scales. Independent CV/test evaluation, deployment, site integration and real-phone scans were not performed.

## Measured artifact table

`tf`: conventional causal block, identity or learned value projection, optional RMSNorm/tied output/shared depth. `tap`: two full-embedding attention heads with local relative bias and a wider classifier. `hybrid`: an explicit 3-gram MLP plus attention control. `*_init` and `*_transfer` runs have zero training steps; they are initialization/transfer measurements. Configuration and export details for every row are in `names/qrtf_table.json`.

| model | params | levels | program chars | weight digits | payload bytes | URL chars | QR v | slack bits | val bpc | novel % / 2,000 | >24 chars % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mlp31_h32_lsq_fine_s0 | 2161 | 31 | 444 | 3223 | 3670 | 3693 | 31 | 240 | 2.294589 | 96.95 | 5.00 |
| mlp31_h32_lsq_fine_s2 | 2161 | 31 | 444 | 3223 | 3670 | 3693 | 31 | 240 | 2.299934 | 95.90 | 2.50 |
| mlp31_h32_lsq_fine_s1 | 2161 | 31 | 443 | 3223 | 3669 | 3692 | 31 | 248 | 2.308294 | 96.85 | 3.65 |
| mlp31_h29_lsq_fine_s0 | 1984 | 31 | 444 | 2959 | 3406 | 3429 | 30 | 256 | 2.313252 | 96.90 | 5.15 |
| mlp31_h28_lsq_s0 | 1925 | 31 | 444 | 2871 | 3318 | 3341 | 30 | 550 | 2.319355 | 96.35 | 2.70 |
| mlp31_h28_lsq_s1 | 1925 | 31 | 444 | 2871 | 3318 | 3341 | 30 | 550 | 2.330529 | 97.10 | 4.50 |
| mlp61_soft3 | 1689 | 61 | 445 | 3016 | 3464 | 3487 | 30 | 58 | 2.355463 | 96.80 | 3.35 |
| published_mlp | 1689 | 61 | 450 | 3016 | 3469 | 3492 | 30 | 18 | 2.355463 | 96.80 | 3.35 |
| mlp45_lsq_fine_s0 | 1689 | 45 | 444 | 2793 | 3240 | 3263 | 30 | 810 | 2.359242 | 96.75 | 5.60 |
| mlp61_soft7 | 1689 | 61 | 445 | 3016 | 3464 | 3487 | 30 | 58 | 2.360951 | 96.55 | 4.10 |
| mlp45_h26_soft3_s0 | 1807 | 45 | 442 | 2988 | 3433 | 3456 | 30 | 176 | 2.369230 | 96.50 | 2.00 |
| mlp45_soft7 | 1689 | 45 | 443 | 2793 | 3239 | 3262 | 30 | 818 | 2.384423 | 96.60 | 3.80 |
| mlp45_soft3 | 1689 | 45 | 443 | 2793 | 3239 | 3262 | 30 | 818 | 2.385284 | 97.45 | 4.10 |
| mlp31_h28_lsq_s2 | 1925 | 31 | 443 | 2872 | 3318 | 3341 | 30 | 554 | 2.385694 | 97.10 | 4.05 |
| mlp45_lsq_s0 | 1689 | 45 | 445 | 2793 | 3241 | 3264 | 30 | 802 | 2.407637 | 98.60 | 4.30 |
| mlp31_h28_s0 | 1925 | 31 | 443 | 2871 | 3317 | 3340 | 30 | 558 | 2.408967 | 97.50 | 7.55 |
| mlp31_h28_soft3_max_s0 | 1925 | 31 | 443 | 2871 | 3317 | 3340 | 30 | 558 | 2.422809 | 98.20 | 11.55 |
| mlp31_h24_soft3_s0 | 1689 | 31 | 444 | 2520 | 2967 | 2990 | 28 | 88 | 2.460262 | 97.75 | 6.10 |
| mlp31_init | 1689 | 31 | 444 | 2520 | 2967 | 2990 | 28 | 88 | 2.460262 | 97.75 | 6.10 |
| tap21_lsq_fine_s1 | 1949 | 21 | 733 | 2578 | 3314 | 3337 | 31 | 78 | 2.352224 | 96.40 | 3.90 |
| tap21_lsq_fine_s2 | 1949 | 21 | 734 | 2578 | 3315 | 3338 | 31 | 70 | 2.353515 | 97.00 | 3.15 |
| tap21_lsq_fine_s0 | 1949 | 21 | 733 | 2578 | 3314 | 3337 | 31 | 78 | 2.354723 | 98.50 | 5.60 |
| tap21_lsq_qonly_s0 | 1949 | 21 | 730 | 2578 | 3311 | 3334 | 31 | 102 | 2.381690 | 97.05 | 5.70 |
| tap21_h28_lsq_transfer | 1949 | 21 | 731 | 2578 | 3312 | 3335 | 31 | 94 | 2.382842 | 97.10 | 5.10 |
| tap45_s6_s0 | 1713 | 45 | 745 | 2833 | 3581 | 3604 | 32 | 28 | 2.387949 | 97.15 | 4.35 |
| tap31_qonly_s0 | 1713 | 31 | 733 | 2555 | 3291 | 3314 | 31 | 155 | 2.410834 | 97.15 | 3.30 |
| tap21_h25_lsq_fresh_s0 | 1772 | 21 | 732 | 2344 | 3079 | 3102 | 30 | 2 | 2.417461 | 97.90 | 4.30 |
| tap31_from31_init | 1713 | 31 | 734 | 2555 | 3292 | 3315 | 31 | 147 | 2.418155 | 96.35 | 3.20 |
| tap45_h22_skip0_soft3_s0 | 1595 | 45 | 733 | 2637 | 3373 | 3396 | 32 | 778 | 2.445458 | 98.15 | 5.95 |
| tap31_h24_skip0_soft3_s0 | 1713 | 31 | 733 | 2555 | 3291 | 3314 | 31 | 155 | 2.460191 | 98.10 | 5.05 |
| tap31_init_skip0 | 1713 | 31 | 733 | 2555 | 3291 | 3314 | 31 | 155 | 2.460191 | 98.10 | 5.05 |
| tap31_s6_s0 | 1713 | 31 | 746 | 2555 | 3304 | 3327 | 31 | 51 | 2.460581 | 97.95 | 5.55 |
| hybrid31_h24_q1 | 1728 | 31 | 819 | 2578 | 3400 | 3423 | 32 | 286 | 2.465371 | 97.10 | 3.50 |
| tap45_h21_s6_s0 | 1536 | 45 | 745 | 2540 | 3288 | 3311 | 31 | 109 | 2.482645 | 98.15 | 4.50 |
| tap31_s1_s0 | 1713 | 31 | 746 | 2555 | 3304 | 3327 | 31 | 51 | 2.486312 | 99.40 | 16.80 |
| tap23_h27_lsq_transfer | 1890 | 23 | 734 | 2574 | 3311 | 3334 | 31 | 84 | 2.496062 | 98.50 | 2.15 |
| tap31_h22_s6_s0 | 1595 | 31 | 746 | 2379 | 3128 | 3151 | 31 | 638 | 2.496863 | 98.40 | 8.35 |
| tap23_h22_s6_s0 | 1595 | 23 | 746 | 2173 | 2922 | 2945 | 30 | 460 | 2.587477 | 97.75 | 2.95 |
| tap23_h22_skip0_soft3_s0 | 1595 | 23 | 732 | 2173 | 2908 | 2931 | 30 | 572 | 2.602166 | 98.45 | 7.50 |
| tf12f16_value_norm45_s0 | 1171 | 45 | 918 | 1936 | 2857 | 2880 | 31 | 738 | 2.634988 | 98.40 | 6.85 |
| tf12f20_win3_s0 | 1127 | 61 | 904 | 2013 | 2920 | 2943 | 31 | 594 | 2.637274 | 98.10 | 2.80 |
| tf12f24_norm45_h1_s0 | 1227 | 45 | 897 | 2029 | 2929 | 2952 | 31 | 596 | 2.669452 | 97.65 | 3.20 |
| tf8f32_out_norm45_s0 | 1287 | 45 | 960 | 2128 | 3091 | 3114 | 32 | 658 | 2.679087 | 98.75 | 12.10 |
| tf8f24_out_s0 | 1151 | 61 | 902 | 2056 | 2961 | 2984 | 31 | 466 | 2.704240 | 98.65 | 3.20 |
| tf16f16_sin_lsq31_warm_s0 | 1351 | 31 | 917 | 2015 | 2935 | 2958 | 31 | 483 | 2.711355 | 99.45 | 2.60 |
| tf12f20_norm_s0 | 1127 | 61 | 965 | 2013 | 2981 | 3004 | 31 | 106 | 2.716501 | 98.25 | 3.35 |
| tf12f16_norm91_s0 | 1027 | 91 | 900 | 2013 | 2916 | 2939 | 31 | 626 | 2.718863 | 98.20 | 2.80 |
| tf16f16_sin_lsq31_s0 | 1351 | 31 | 916 | 2015 | 2934 | 2957 | 31 | 491 | 2.735768 | 98.70 | 2.60 |
| tf12f20_l2_s0 | 1127 | 61 | 943 | 2013 | 2959 | 2982 | 31 | 282 | 2.755984 | 97.60 | 3.40 |
| tf12f20_win3_norm31_s0 | 1127 | 31 | 972 | 1681 | 2656 | 2679 | 30 | 292 | 2.780758 | 98.00 | 8.80 |
| tf16f16_norm31_s0 | 1351 | 31 | 897 | 2015 | 2915 | 2938 | 31 | 643 | 2.794018 | 99.15 | 2.40 |
| tf12f20_s0 | 1127 | 61 | 893 | 2013 | 2909 | 2932 | 31 | 682 | 2.796283 | 98.60 | 4.80 |
| tf12f24_sin45_warm_s0 | 1227 | 45 | 914 | 2029 | 2946 | 2969 | 31 | 460 | 2.868576 | 99.05 | 4.65 |
| tf12f24_sin_norm31_s0 | 1227 | 31 | 913 | 1831 | 2747 | 2770 | 30 | 264 | 2.868953 | 99.05 | 1.85 |
| tf12f16_l2_norm45_h1_s0 | 1027 | 45 | 947 | 1698 | 2648 | 2671 | 30 | 436 | 2.891751 | 98.90 | 1.65 |
| tf12f16_l2_norm31_s0 | 1027 | 31 | 999 | 1532 | 2534 | 2557 | 30 | 573 | 2.896995 | 99.45 | 2.35 |

## What the attention contributes

Same packed weights, no retraining: full model **2.417461135 bpc**; content q/k zeroed **2.557252831**; hard previous-two/previous-one retrieval **2.558919853**; both context heads removed **4.477187627**. The latter retains the current embedding and length feature. This distinguishes retrieval from learned content selection for these frozen weights; retraining ablated models was not done.

## Verification and artifacts

The chosen attention model is `names/qrtf_best.pt` with its export record `names/verification_qrtf.json`; the MLP in the QR code is `names/qrtf_mlp_best.pt` with `names/verification_mlp_qrtf.json`. The readable program differs from the golfed executable only in whitespace. The renderer uses `qrcode` optimize=20 mixed alphanumeric/byte/numeric segmentation (`tools/qr.py`); zxing-cpp decodes each exported PNG back to its exact URL.

For the chosen transformer, Node checks every integer plus all **98,319** validation log-probabilities across all 151 names, using the exact raw URL and indirect non-strict eval. Maximum absolute log-probability error is **1.07e-14**. Compact generators are scored by intercepting their actual probability array at `Math.random()` and forcing the held-out characters; the verifier contains no replacement forward implementation.

Physical phone scanning was not measured as part of this experiment.

## Reproduce

Export and reverify without retraining (from the repo root; the fixture is written next to the output):

```bash
uv run python names/qrtf.py export names/qrtf_mlp_best.pt --output /tmp/names
cmp /tmp/names.url.txt names/payload_url_mlp_qrtf.txt
node names/verify_qrtf.js names/payload_url_mlp_qrtf.txt /tmp/names.fixture.json
uv run python names/qrtf.py export names/qrtf_best.pt --output /tmp/attention
```

Training commands for the two kept models (each run is under 15 minutes on a laptop CPU; `--init` is the distilled student from `names/teacher.py`):

```bash
uv run python names/qrtf.py train --tag mlp31_h29_lsq_fine_s0 --kind mlp --threads 2 --width 6 --ff 29 --heads 2 --q 2 --levels 31 --context 3 --layers 1 --window 1 --slope 1.0 --skip 0.1 --steps 24000 --eval-every 1000 --batch 64 --warmup 2000 --seed 0 --lr 0.0006 --wd 0.01 --soft 0.3 --extra 0.7 --max-seconds 870 --lsq --init names/student_real_plus_20k_teacher_samples.pt
uv run python names/qrtf.py train --tag tap21_h25_lsq_fresh_s0 --kind tap --threads 2 --width 6 --ff 25 --heads 2 --q 1 --levels 21 --context 3 --layers 1 --window 1 --slope 6.0 --skip 0.0 --steps 24000 --eval-every 1000 --batch 64 --warmup 500 --seed 0 --lr 0.004 --wd 0.01 --soft 0.3 --extra 0.6 --max-seconds 870 --lsq
```

The exact command for every other run was in `names/qrtf_runs/reproduce_commands.txt` (git history).

Measured training-loop time over 50 positive-step runs: 9621.6 summed seconds, longest run 338.7 seconds. All were below 15 minutes.

## Ten samples from each executable

<details><summary>mlp31_h32_lsq_fine_s0</summary>

```text
glm-3.6-13b
llama-7b-char-largp4-vaeite7-s-a
deepsedellama-250
lla-2p5-7b-instruct
claude-opus-10b
kim-5.1-70b-instruct
llama-7-flaseenimo-fus-40
claude-llama-24b
kiep4-coder-3.5-iso
qwen-3.2-mediu2
```

</details>

<details><summary>mlp31_h32_lsq_fine_s2</summary>

```text
gpt-3.6-13b
ministral-2204222b-fpvb
kwonite-medium-1p5-1-1
deepsee-2p5-code-3
qwen3-70b
olmuct-1.6
lla-7.0-70b
mitabricks-gpla-1017
contar-nspot
date-10b
```

</details>

<details><summary>mlp31_h32_lsq_fine_s1</summary>

```text
glm-4.5-116-flash
llama-2.5-8b-chge
gutral-7b-instras-g
gem-5.1-mia-xt
gpt-5.1-flash
grok-coder
llama-1-fast
ninhrd-veeo-47b
deepseeanite-a3b
nlma-3.1
```

</details>

<details><summary>mlp31_h29_lsq_fine_s0</summary>

```text
glm-3.5-12b-instruct
claude-3.7
grok-v4pbo-code
llame-k2-1-2b-barn
ctainite-chat-6
claude-spar-1.5-fladast
glm-5.8-coder-1p5
fw
llama3.3-maxn
opeg-m3-1b-chat
```

</details>

<details><summary>mlp31_h28_lsq_s0</summary>

```text
glm-4.6-1254b-itpus-5
claude-ourorch-v3.8-30b
gemma-v2.0-1-260
llalaude-mode-32b
llama3-fistruct-v1-h-
gpt-5-pro-v2-turbo
ro-ap3-8b-25b-instruct
data-13b-instruct
glm-4p5-code-der-v3-lat
claude-opul-340
```

</details>

<details><summary>mlp31_h28_lsq_s1</summary>

```text
gpt-3.5-128b-instruct
claude-otabt-instruini
deepsederk
gemini
mise
yos-4p5-chat-largect
grok-4-flash
aina-7b-instrurn
gpt-6-4-0-8b
cotoini-thos-2-8b
```

</details>

<details><summary>mlp61_soft3</summary>

```text
glm-3.7-12b-instruct
databricks-grch-v32b-chaickith
rax
gemba-5.5-0
ypas-v3.1
qwen3-70b-instrucpl
gpt-0.5
gemma-3b-it-opus-4
gpt-5.3
motiseek-v4.1
```

</details>

<details><summary>published_mlp</summary>

```text
glm-3.7-12b-instruct
databricks-grch-v32b-chaickith
rax
gemba-5.5-0
ypas-v3.1
qwen3-70b-instrucpl
gpt-0.5
gemma-3b-it-opus-4
gpt-5.3
motiseek-v4.1
```

</details>

<details><summary>mlp45_lsq_fine_s0</summary>

```text
glm-2.5-13b
llama-4-1-11b-instruct
kitrosv
gpt-5.6-193b
gem-ma-3.8
cugminstraf-mem-1.2-5-27b
glemarba-1-87b
gpt-5ab-spero
qwan3-8b-imax-lir-v202b
com-1p5-5-flash
```

</details>

<details><summary>mlp61_soft7</summary>

```text
glm-3.7-0-72b-it-lat-5
gemma-7b-it
llranitabricks-g
qwen3-5-mini-p-0qc2-grmin
qwen3-72b-instrucpl-lat
2-chat-opustrus-4-flase
lara-lathinkinw
nllama3-1b-instruct
kimax
qwen3-claudeep8-7b
```

</details>

<details><summary>mlp45_h26_soft3_s0</summary>

```text
glm-3.5-12b-instruct
clateranite
testralium-3.3
mos
rax
gemhan-3.8
cugmity-coder-4.1
mugpt-onne-4.1
qwan3-7-5-7b-sonstruct
p3-chat-l2-mini
```

</details>

<details><summary>mlp45_soft7</summary>

```text
glm-2.5-13b-instruct
chateree-s2-kimixnstruct
gemma-v2p-laracr-vle
cude-ous-4-ning-3-7b
mistral
mistra-4.1
fw-gpterculminimo
deapero
motron-veepseege
code-llama-4.2
```

</details>

<details><summary>mlp45_soft3</summary>

```text
gemma-4-0325b-ale-r2
databhe-7b
grok-v4.it
gpt-5.6-17b-14
geminimo
cur-mite-4-pret-1-fus-4kst
gemma-lafama-2-8b-it
grok-4-prae-1.2
gemma-7b-onstruct-2
sma3-flangrok-1.1
```

</details>

<details><summary>mlp31_h28_lsq_s2</summary>

```text
glm-3-flash
llama-7b
databricks-gute-yuser
gpt-5.5-1b
o-3-3-flash-1-uhsonr
glm-5.3-13b-instruct
mistra-3.1-5-minite-vet
llama-4-code
mistral-7b-inn-5-1b
gimi-k2-7b-i.sarbe-chat
```

</details>

<details><summary>mlp45_lsq_s0</summary>

```text
glm-4-meai-gpt-thini
claude-3.7-cybhat-low
gpt-5.6-1b
qwen2-flashe
cus-4-turb-m5-8b
het-max-m3-13b
llada-1-3-minist
vipto-v4-22-70b
comma-3.7-flash
qwen-3-7b-glm-5-4
```

</details>

<details><summary>mlp31_h28_s0</summary>

```text
glm-4.6-13b
llama-4.1
claude-opint-5.staus
gpt-4.7-14b-a3
gesimo.s-2p5-72b-instruct
gpt-5.6-flakistrucen
deepseeks-gut-7b
ro
elaseek
momiornustar2-7b
```

</details>

<details><summary>mlp31_h28_soft3_max_s0</summary>

```text
glm-4.5-13b-instruct
dat-ul-8b-chnus-wpstruct
hernaas-latba
peepseedti-aburo.
qwen3-72b-instruct
minie-3-2-2-8b-it
grok-4-pr-5-12b-imink-pr-8b-cl
cos-1wan3-flanh-ons
merast-4.2-flatest
mab
```

</details>

<details><summary>mlp31_h24_soft3_s0</summary>

```text
glm-2.5-12b
montocodeel-a3b
gpt-54-32p4-vx-3-1-20b
code-larast
mot-4p5-como-4
qwen3-7b-instruct
motron
gpt-4-flastrutemin
qwen3-8b-imist
qwes-m-m-8b-a1b
```

</details>

<details><summary>mlp31_init</summary>

```text
glm-2.5-12b
montocodeel-a3b
gpt-54-32p4-vx-3-1-20b
code-larast
mot-4p5-como-4
qwen3-7b-instruct
motron
gpt-4-flastrutemin
qwen3-8b-imist
qwes-m-m-8b-a1b
```

</details>

<details><summary>tap21_lsq_fine_s1</summary>

```text
glm-4.7-2307b-chat
jai-glm-4.7-aistrunmetuct
deepse-sonar
geminimo
cys-cono-fat-med
gpt-onus-4-13b-it
6eb
glm-5.3b-axo-4b
ro-3-025b-amustruct
nita-llamba-3-flash
```

</details>

<details><summary>tap21_lsq_fine_s2</summary>

```text
glm-4.7-1265-situ
mini-3-lash-v1-th-fustpun
maranit-ult
gemanei.istruct
mus-4-nro-latest
grok-4-ash
qwan3-5-asonstrune-chat
aba
mini-4b-instruct
data-14b
```

</details>

<details><summary>tap21_lsq_fine_s0</summary>

```text
gpt-3-flash
granite-2-104b-instrun-3-v3-rt
gpt-5.5-17b-72b
deem-ct0-vl-7b-instruct
claude-ouro-1.5-fla
gpt-5-pro-v2-turbo
qwen3-71b-8b-instruct
gemma-2p2-6-opseen-3-04
qwen3-8b-instruct-2-2b
llama-3-305
```

</details>

<details><summary>tap21_lsq_qonly_s0</summary>

```text
gpt-3-llama-42bhi-7-clarch
glm-4.sneryvhlr-7b
glm-4-tlr
gemini-4.8
cysn-rvas-20b-instruct
qwen3
minist-llama-clary-1b
mistraaki
llama-v3p6-fp8-lat
llame-7-llama-7b
```

</details>

<details><summary>tap21_h28_lsq_transfer</summary>

```text
gpt-3-llama-42bhi-chat
gistral-2-ton-xonstruct
gpt-27b-aaak-v2-cor
cysknistract-med
gpt-onstral
minist-llama-clary-1b
mistraaki
llama-v3p6-fp8-lat
llame-7-llama-7b
gemh-mini-fini-pyt
```

</details>

<details><summary>tap45_s6_s0</summary>

```text
deepmere-s2.6-8b-instruct
llame-lpp7-vl-star-largk-1us-
gem-5.2-miaeym-5p7-flash
llama3-haulera3
nimo
2-v2-20b-instruct
llama-4-8b-instruct
seek-cipo-17b
fw-glm-5.3-122b
llama-3-flmaes-5-1
```

</details>

<details><summary>tap31_qonly_s0</summary>

```text
glm-4.5-1b
gpt-vmab-hea-245-cor
gtm-5p7-vp
gpt-5.6-1b
qwen3-medius
cyv-4.8-codevinstrus
onstral
mistra-4.1-5
nllargevma-7.8-0
lite-3-flash
```

</details>

<details><summary>tap21_h25_lsq_fresh_s0</summary>

```text
flas5-kimi-k24-tuh
kimax-mini-o-ow-sotast
deepseek-2eis
gemistral-1-vl-3b-instruct
claude-ount-1.7
gro-code-25b-it
grok-4-ss
deepseed-spard-r1-707
llamba-3-coder-4-3
mistral-mediumtrbn2
```

</details>

<details><summary>tap31_from31_init</summary>

```text
glm-4.5-1b
gpt-vmab-hea-245-cor
gtm-5p7-vp
gpt-5.6-1b
qwen3-medius
cyv-4npt-5
qwen3-8b-instruct
mistra-4.1-5
nllargevma-7.8-0
lite-3-flash
```

</details>

<details><summary>tap45_h22_skip0_soft3_s0</summary>

```text
glm-3.5-14b-instruct
databricks-gs-4-vl-st
gpt-5.5-1lash
claude-opa
yrdelpome-tla-gemitb-26b
ji-k3-cod-5.1
geministruct-late
gmand-c-lpterest
pla-1-plaxd-2.5
gemma-3
```

</details>

<details><summary>tap31_h24_skip0_soft3_s0</summary>

```text
glm-2.5-12b
montobpik-1.1-flast
grok-v4.8-7b-flasr1
qwen3-72b
deepe-v3-47b-instruct
gpt-5-mini-1.5-flaceeh
glm-4p7-cure-late
gpt-5.2
mus-5-7lestruct-3
qwen3.2-13b
```

</details>

<details><summary>tap31_init_skip0</summary>

```text
glm-2.5-12b
montobpik-1.1-flast
grok-v4.8-7b-flasr1
qwen3-72b
deepe-v3-47b-instruct
gpt-5-mini-1.5-flaceeh
glm-4p7-cure-late
gpt-5.2
mus-5-7lestruct-3
qwen3.2-13b
```

</details>

<details><summary>tap31_s6_s0</summary>

```text
glm-2.5-122b-instruct
code-a3b-fewes29b
qwen3-70b-instra-1.1-70b-l
ctlaietrel
qwen3-5-soninr
glm-5.3-4
gpt-5-instruct-latest
gpobar
contore-v2p5-1
llame-3-425-13b
```

</details>

<details><summary>hybrid31_h24_q1</summary>

```text
glm-3.6-13b-instruct
dat-malraicht
llrab-slle-gemo
code-laraninkis-1t0
gpt-5.1-mir-1-flash
qwen-llama-1-fasr-nis-r
gustral-n2-8
llama3.5-plus
qwen-ara-v3-5-onm
mistral-pheem-5.1
```

</details>

<details><summary>tap45_h21_s6_s0</summary>

```text
glm-3.5-125b-instruct
cma-4.2-sinus-4p6-sul
mistron-t-t
gemellama4
cugmarneri-pro-f3-co
pext-laienk-6.0
meepseek-v4-turba-v4
gpt-5.3
motestructar
com-1p3-42b-chat
```

</details>

<details><summary>tap31_s1_s0</summary>

```text
glm-2.5-1-70b-ith-latest
minimo-4b-instrunsnel-56b
cpio-m2-24b
mish-teminstrh
qwen3-7ba
mistral-medita
gpt-5-flash-lptest
qwand-rcoderok-plus-tom-1-f0
sina-maab-inst
metab-fesongroninh
```

</details>

<details><summary>tap23_h27_lsq_transfer</summary>

```text
glm-4.7-2.2-8b
ssata-2-3-codeconus-5-v5-st
hemma-v3-sha
hagatd-llamatlerglon
deepseek-v3.1-pro
command
gpgigatd-lro7b
mes-llama
mhan-rvatba-onus-1
mo-4b
```

</details>

<details><summary>tap31_h22_s6_s0</summary>

```text
glm-3.6-13b-instruct
clagrm-3.7-fus-4p6-vr
gpt-5.6-1b-inshe-qw-nus
cyllama-3-2b-instrun-prop
glm-5.2-7
gpt-5-minis.4p7-ge-v3
gpt-5-8-gpt-5-flash
d-tp5
sma0-pla-v3-flash-pro-20b
llama-v2.0
```

</details>

<details><summary>tap23_h22_s6_s0</summary>

```text
glm-3.5-12b-it
sonni-2-2-70b
momita-gutetuiri-k24b
codeak-v3b
deeph-vini-rob-2lirge
glm-5.8-70b
minima
gpt-4.5-core-vis1
dee-4.0-8-70b-instruct
llama-1p3-20b
```

</details>

<details><summary>tap23_h22_skip0_soft3_s0</summary>

```text
gemma-lalerast
sonk-4-21b-instruct
llripks-glm-4ma
bugr-5-8-heanst-5wf
gpt-5-5.7-lare-v2unt-fas
minimo
gpt-5-mistrr4-71b-49b
gpt-5-fastrum-7p5-min
llame-chat-ceet-710b
qwen3-flastrud-r
```

</details>

<details><summary>tf12f16_value_norm45_s0</summary>

```text
glm-4-qw03-5-plus
noi-2.0-flash
granixtros-7b
lami-2p5-1-11-flash-1
xlamistral-nma-1-code-von2
codept-0-inqw-4.4-v6
viphicod-7b
mini
nostral-flash
mid-v2-3-flash
```

</details>

<details><summary>tf12f20_win3_s0</summary>

```text
deep
gpt-5-9b-itect
deal-pgeem-1-v2-7ut
swel3-fast
cv-s-fast-fast-1
xwel3-flasp
misgelp-okrtrea
mistra-cla3-224b
tgeum4-max-0b
code-4.lini-kx
```

</details>

<details><summary>tf12f24_norm45_h1_s0</summary>

```text
glm-2-ma-l-flastrok-4
claude-mt
gremaxinstral-g
llama-sal
gemi-k42
cyrinnqw-getbinspsh-5-7b
glm-5.3-pr4arcer
omnixt-vflashet
gplus-7
morininstral-1-pl
```

</details>

<details><summary>tf8f32_out_norm45_s0</summary>

```text
glm-3.6-12-7b-gt-4b
gemini-3-fp2
vaimoniro-flash
qwen3.1-1.1-70b-s
cyrastra-70b-ins-hininist
glam-4.6-0-lash
qwen4.52.5-fast
flash-5
motrchax-ppraipt
code-code-lates
```

</details>

<details><summary>tf8f24_out_s0</summary>

```text
fw-k-ins-gpt-shluct
elama-coder3-vlewuurucbash
derh
rat
gema-4-5.7
cyiietote-36b
llamaba-constraruct
gab-3-3-fpp8b
gtema-2.5
deak-ll-frieremrconca-7b
```

</details>

<details><summary>tf16f16_sin_lsq31_warm_s0</summary>

```text
deni-4.5
llama-v5-marcat
minins-isstrus-stainct
llama-sam-instestl
cwndaturde.midfin
llamatri-8
llama--4.1-3-flast
grok-4.7-00b
llam-kma-pprt-pre.
llama-3-7b-instrucg
```

</details>

<details><summary>tf12f20_norm_s0</summary>

```text
ek-mani-1.2
llatepest
claude-llansh
gunituarie-3.7-1b
qwen2-ma-mit
cxm2-5b-init-ins-8b-cinst
gllma-5.2
gpt-5.5-cotrax
llama3-3.0-7b
coodeps-v2.5-2
```

</details>

<details><summary>tf12f16_norm91_s0</summary>

```text
glm-2.5-130b-latruct
dagemininp5-v2-3fp8b-a-7
gemmi-tat
gak-mi-gpt
cyiiinter-30b-inge-t
okllama-llami-00b
gemmistrall-feshat
gpt-5.3
monistron-stralr
cqwen3.5-263b-i
```

</details>

<details><summary>tf16f16_sin_lsq31_s0</summary>

```text
glm-4.5-150b-inxtruct
codestrardev2.2fpext
gpt-4.6-1b-l5b
glm-4.5-1-visept-5-2b-ins-4-5
mistral-larlib
gpt-4.5-hitentro
llam-4.1-8b-15b-flast
mish
ki-1pckonli-1b-ins
matapiede-3.5-visres
```

</details>

<details><summary>tf12f20_l2_s0</summary>

```text
glm-115-06-flast-lash
coden-dlla-v3.3.8-st
gpt-4-praume-4
gemma-fp1
yininto-5-pro
gemmma-3.8
glm-5-5.2
gpt-4
noinstruites
se
```

</details>

<details><summary>tf12f20_win3_norm31_s0</summary>

```text
glm-2.6-12000-ct-mt----mhst
gpt-52b-ixtrlo-mint-500
seeama-222
llamaxunepq-latest
latash-flash
coimini
gpt-5.7-mix-7w52240b
7w-3.5-latopcoasuk-mine-oaet
gemma-12b-indp4p8-code
gpt-usdp5.1
```

</details>

<details><summary>tf16f16_norm31_s0</summary>

```text
deeprok-1.0
llarmani-2-5
misont-5027bb-its
graniooh
sek-4-5-medr-v0
yrok-v3-70b-insronn
ogstral
mistra-3-1-3-chatruct
llaman2-7-flaserustruct
oni-1-8-1b
```

</details>

<details><summary>tf12f20_s0</summary>

```text
glm-4.5-13-7b-3b-322
das-llamaralufl-xlated-iastruchi
qwen3-ingpt-1
xophitre-3b-gesph
qwen3p5-5
minite
gpt-4-miniuphos-at
qwen3-72b-14b-fp2
seek-3
llami-4.21-70b
```

</details>

<details><summary>tf12f24_sin45_warm_s0</summary>

```text
glm-2-code-5.6
vok-lagalasoderr-3p4-fr
qwen3.5-27b
code-lama2-26b-1-v34b-p5
kistral-7b-initruct
omima-1-b-de-ninstrur-insmin
gpmini-1.7-flatr-sep
code-v3-33b
gek-lam-5-prnistructruut
qwen-3.20b-insrctrubminst
```

</details>

<details><summary>tf12f24_sin_norm31_s0</summary>

```text
demi-k3-12
gpt-psek2-1
gemi-k3.2
sermses33
granitra
qwek-1.1-7-hit
cyonitracodet-flasot
mistral-lerdba
gpt-5.5-int-4pronidpt
dedema-12b-fp8-p5-gen2-f
```

</details>

<details><summary>tf12f16_l2_norm45_h1_s0</summary>

```text
glm-1-li
iimistral-fas
gemimal-3-v4.5k-itrept
gemma-pla
gemar.1-8b
cyrepst-ineston-1.7b
mitral-1.6
gpt
gpt-5-8b-fp8
tenmi-te
```

</details>

<details><summary>tf12f16_l2_norm31_s0</summary>

```text
glm-2-lacodeestus
phamam-k3-33b
grem-vlmix-m4
lama-14-m.5
geprasis
cwea250b-ict-flanteu
qwen3-3
npsosn
gpham-k3-8ck3b-insttr
gp8-77b
```

</details>

## Three-sentence conclusion

The preferred v30 attention model achieves 2.417461 bits/char in QR v30, 0.061998 bpc behind the published MLP, so it meets the stated replacement rule. The best v31 attention model reaches 2.352224 bpc, and the retuned MLP reaches 2.294589 bpc and remains the quality winner. These validation-selected results establish an executable size/quality point on the fixed names split and do not establish a general transformer advantage.
