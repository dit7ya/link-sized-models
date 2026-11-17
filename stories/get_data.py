"""Stream TinyStories V2 GPT-4 into the original fixed character benchmark.

uv run python stories/get_data.py  # 40 MiB training prefix plus validation
uv run python stories/get_data.py --heldout-only --vocab-from out/t64kfix_url.txt

The latter reproduces a published URL's 500k benchmark without training data.
"""
import argparse
import base64
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote

BASE = 'https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/'
TRAIN_MB, HELD, EOT = 40, 500_000, '<|endoftext|>'


def fetch(name, out, nbytes=None):
    out = Path(out)
    if out.exists(): return
    partial = out.with_suffix(out.suffix + '.part')
    command = ['curl', '-sSfL', '--retry', '2']
    if nbytes: command += ['-r', f'0-{nbytes-1}']
    # Bound bytes written even when a server ignores the Range request.
    process = subprocess.Popen(command + [BASE + name], stdout=subprocess.PIPE)
    total = 0
    try:
        with partial.open('wb') as dest:
            while not nbytes or total < nbytes:
                block = process.stdout.read(min(65536, nbytes-total) if nbytes else 65536)
                if not block: break
                dest.write(block); total += len(block)
        if nbytes and total == nbytes and process.poll() is None: process.terminate()
        status = process.wait()
        if (status and not (nbytes and total == nbytes)) or not total:
            raise RuntimeError(f'Download failed: {name} (curl exit {status})')
        partial.replace(out)
    finally:
        process.stdout.close()
        if process.poll() is None: process.kill(); process.wait()


def iter_stories(path, complete_only=False):
    """Same strip/CR/EOT mapping as the original loader, one story at a time."""
    pending = ''
    with open(path, encoding='utf-8', errors='ignore', newline='') as source:
        while chunk := source.read(65536):
            parts = (pending + chunk.replace('\r', '')).split(EOT)
            pending = parts.pop()
            for story in parts:
                if story.strip(): yield story.strip()
        if not complete_only and pending.strip(): yield pending.strip()


def encode_to(path, source, vocab, limit=None, complete_only=False):
    stoi = {c: i for i, c in enumerate(vocab)}
    space = stoi[' ']
    size = bad = utf8 = count = 0
    with open(path, 'wb') as dest:
        for story in iter_stories(source, complete_only):
            count += 1
            encoded = bytearray()
            for char in story + '\0':
                if limit is not None and size >= limit: break
                token = stoi.get(char, space)
                bad += char not in stoi
                encoded.append(token); size += 1; utf8 += len(vocab[token].encode())
            dest.write(encoded)
            if limit is not None and size >= limit: break
    with open(path, 'rb') as encoded_file:
        digest = hashlib.file_digest(encoded_file, 'sha256').hexdigest()
    return dict(chars=size, replaced=bad, stories=count, utf8_bytes=utf8, sha256=digest)


def payload_vocab(path):
    fragment = unquote(Path(path).read_text().strip().split('#', 1)[1])
    program = fragment.split('.', 1)[0]
    source = base64.urlsafe_b64decode(program + '=' * (-len(program) % 4)).decode('utf-8')
    match = re.search(r';X=("(?:\\.|[^"\\])*");', source)
    if not match: raise ValueError('Story payload has no vocabulary')
    return '\0' + json.loads(match[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--heldout-only', action='store_true')
    parser.add_argument('--vocab-from', type=Path)
    args = parser.parse_args()
    Path('data').mkdir(exist_ok=True)
    fetch('TinyStoriesV2-GPT4-valid.txt', 'data/valid.txt')
    if args.heldout_only:
        vocab = payload_vocab(args.vocab_from) if args.vocab_from else json.loads(Path('data/vocab.json').read_text())['vocab']
    else:
        fetch('TinyStoriesV2-GPT4-train.txt', 'data/train_raw.txt', TRAIN_MB << 20)
        counts = Counter()
        for story in iter_stories('data/train_raw.txt', complete_only=True): counts.update(story)
        vocab = '\0' + ''.join(sorted(c for c, n in counts.items() if n >= 200))
        train = encode_to('data/train.u8', 'data/train_raw.txt', vocab, complete_only=True)
        print('train:', json.dumps(train))
    if len(vocab) > 256: raise ValueError('Vocabulary does not fit uint8')
    held = encode_to('data/heldout.u8', 'data/valid.txt', vocab, limit=HELD)
    if held['chars'] != HELD: raise ValueError('Incomplete validation download')
    Path('data/vocab.json').write_text(json.dumps({'vocab': vocab, 'heldout_utf8_bytes': held['utf8_bytes']}))
    print('heldout:', json.dumps(held)); print('vocab', len(vocab), repr(vocab))


if __name__ == '__main__':
    main()
