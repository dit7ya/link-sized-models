"""Build the featured QR artifact from the retuned names model.

Takes names/payload_url_mlp_qrtf.txt (names/qrtf.py export, temperature 1), folds a
sampling temperature of 0.8 into the two output-layer constants, and writes
the link, the QR image and the social card.

  uv run python tools/make_qr.py
"""
from pathlib import Path
import re
import sys

from PIL import Image
import zxingcpp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))
from qr import qr_measure  # noqa: E402

TEMPERATURE = 0.8


def lit(x):
    return f'{x:.3g}'.lstrip('0')


def main():
    url = (ROOT / 'names/payload_url_mlp_qrtf.txt').read_text().strip()
    head, frag = url.split('#', 1)
    n = int(frag[:3])
    program, digits = frag[3:3 + n], frag[3 + n:]
    # Output bias and output weight scales: a = W[..]*B, a += W[..]*H[i]*S.
    m = re.search(r'for\(a=W\[(\d+)\+j\]\*([.\d]+),i=0;i<(\d+);i\+\+\)a\+=W\[(\d+)\+j\*\3\+i\]\*H\[i\]\*([.\d]+);t\+=p\[j\]=exp\(a\)', program)
    assert m, 'unexpected program shape'
    bias, scale = float(m.group(2)), float(m.group(5))
    new = m.group(0).replace('*' + m.group(2) + ',', '*' + lit(bias / TEMPERATURE) + ',').replace('*' + m.group(5) + ';', '*' + lit(scale / TEMPERATURE) + ';')
    program = program.replace(m.group(0), new)
    assert len(program) < 1000 and digits.isdigit()
    url = f'{head}#{len(program):03d}{program}{digits}'

    qr, version, bits, capacity, modes = qr_measure(url, 20)
    assert version <= 30, version
    image = qr.make_image().convert('RGB')
    assert zxingcpp.read_barcodes(image)[0].text == url
    for path in ['out/qr_names_url.txt', 'site/data/payloads/qr_names_url.txt']:
        (ROOT / path).write_text(url + '\n')
    image.save(ROOT / 'site/img/qr.png')

    # Social card: swap the code in place, at 3 px per module.
    card_path = ROOT / 'site/img/og.png'
    card = Image.open(card_path).convert('RGB')
    old = zxingcpp.read_barcodes(card)[0].position
    side = (qr.modules_count + 8) * 3
    small = image.resize((side, side), Image.NEAREST)
    card.paste(small, (old.top_left.x - 12, old.top_left.y - 12))
    assert zxingcpp.read_barcodes(card)[0].text == url
    card.save(card_path)

    print(f'{len(url)} URL chars, program {len(program)}, {len(digits)} digits, QR v{version}, '
          f'{qr.modules_count} modules, {capacity - bits} slack bits, image {image.size}')
    print(program)


if __name__ == '__main__':
    main()
