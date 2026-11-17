"""Score a checked-in story URL without its untracked training checkpoint.

uv run python stories/get_data.py --heldout-only --vocab-from out/t64kfix_url.txt
OMP_NUM_THREADS=2 uv run python stories/score_story.py out/t64kfix_url.txt
"""
import argparse
import base64
import json
import math
from pathlib import Path
import re
import sys
import time
from urllib.parse import unquote

import numpy as np
import torch

torch.set_num_threads(2)


def unpack_url(path):
    fragment = unquote(Path(path).read_text().strip().split('#', 1)[1])
    program, stream = fragment.split('.', 1)
    source = base64.urlsafe_b64decode(program + '=' * (-len(program) % 4)).decode('utf-8')
    match = re.match(r'^C=(\[.*?\]);K=(\[.*?\]);X=("(?:\\.|[^"\\])*");', source)
    if not match:
        raise ValueError('Not a supported story export')
    config, scales, alphabet = [json.loads(s) for s in match.groups()]
    d, layers, heads, context, bits, temperature, row = config
    vocab = '\0' + alphabet
    shapes = [(len(vocab), d)] + [(r*d, c*d) for _ in range(layers) for r, c in [(3, 1), (1, 1), (4, 1), (1, 4)]]
    count = sum(r for r, _ in shapes) if row else 0
    codes_chars = math.ceil(count * 8 / 6)
    codes = base64.urlsafe_b64decode(stream[:codes_chars] + '=' * (-codes_chars % 4))[:count]
    weights = stream[codes_chars:]
    chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_'
    position = accumulator = remaining = 0

    def read():
        nonlocal position, accumulator, remaining
        while remaining < bits:
            accumulator = accumulator * 64 + chars.index(weights[position]); position += 1; remaining += 6
        remaining -= bits
        value = accumulator >> remaining
        accumulator &= (1 << remaining) - 1
        return value - ((1 << (bits - 1)) - 1)

    matrices, u = [], 0
    for (r, c), scale in zip(shapes, scales, strict=True):
        matrix = np.empty((r, c), dtype=np.float32)
        for i in range(r):
            multiplier = scale * 2 ** ((codes[u] - 128) / 32) if row else scale
            if row: u += 1
            for j in range(c): matrix[i, j] = read() * multiplier
        matrices.append(torch.from_numpy(matrix))
    if position != len(weights):
        raise ValueError('Unexpected weight-stream length')
    return config, vocab, matrices


def load_model(path):
    import stories
    config, vocab, matrices = unpack_url(path)
    if stories.VOCAB != vocab:
        raise ValueError('Vocabulary differs: rerun stories/get_data.py --heldout-only --vocab-from ' + str(path))
    d, layers, heads, context, bits, _, row = config
    model = stories.GPT(d, layers, heads, context, bits, 'row' if row else 'tensor')
    with torch.no_grad():
        for target, packed in zip(model.tensors(), matrices, strict=True): target.copy_(packed)
    model.q = False  # We loaded the decoded weights; never quantise them again.
    return model.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('payload', type=Path)
    parser.add_argument('--chars', type=int, default=500000)
    args = parser.parse_args()
    model = load_model(args.payload)
    import stories
    held = np.fromfile('data/heldout.u8', dtype=np.uint8, count=args.chars)
    if len(held) != args.chars or args.chars <= 0:
        raise ValueError('Requested benchmark length is unavailable')
    start = time.monotonic()
    loss = stories.evaluate(model, torch.from_numpy(held.astype(np.int64)))
    result = dict(payload=str(args.payload), heldout_chars=len(held), bits_per_char=loss,
                  seconds=time.monotonic()-start, url_chars=len(args.payload.read_text().strip()),
                  fragment_bytes=len(args.payload.read_text().strip().split('#', 1)[1].encode()))
    print(json.dumps(result, indent=2))
    meta = args.payload.with_name(args.payload.name.replace('_url.txt', '_meta.json'))
    if meta != args.payload and meta.exists():
        reference = json.loads(meta.read_text())
        key = 'heldout_bpc' if args.chars == reference['heldout_chars'] else 'heldout_first20k_bpc' if args.chars == 20000 else None
        if key and abs(loss-reference[key]) > (0.00006 if key == 'heldout_bpc' else 0.00002):
            raise AssertionError(f'Packed URL loss {loss} differs from {key}={reference[key]}')


if __name__ == '__main__':
    main()
