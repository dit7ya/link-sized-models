"""Canonical public URL; host.json is shared with the Node verifiers.

Keep the explicit root slash: browsers normalize an empty HTTPS path to '/'.
Only scheme/host are uppercased for QR alphanumeric encoding; fragments never are.
"""
import json
from pathlib import Path

PREFIX = json.loads(Path(__file__).with_suffix('.json').read_text())['prefix']
QR_PREFIX = PREFIX.upper()


def check_prefix(url):
    if not url.startswith((PREFIX, QR_PREFIX)):
        raise ValueError(f'Expected canonical model prefix {PREFIX}')

