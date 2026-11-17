"""QR helpers shared by the names and story exports and tools/make_qr.py."""
import qrcode
import qrcode.util as qutil
from qrcode.constants import ERROR_CORRECT_L
from PIL import Image
import zxingcpp

MODE_NAME = {1: 'numeric', 2: 'alnum', 4: 'byte'}


def qr_measure(data, optimize):
    """Build an error-correction-L QR code; return it with version, data bits, capacity and segments."""
    qr = qrcode.QRCode(error_correction=ERROR_CORRECT_L, box_size=6, border=4)
    qr.add_data(data, optimize=optimize)
    ver = qr.version
    mode_sizes = qutil.mode_sizes_for_version(ver)
    buf = qutil.BitBuffer()
    modes = []
    for chunk in qr.data_list:
        buf.put(chunk.mode, 4)
        buf.put(len(chunk.data), mode_sizes[chunk.mode])
        chunk.write(buf)
        modes.append(f'{MODE_NAME.get(chunk.mode, chunk.mode)}:{len(chunk.data)}')
    cap = qutil.BIT_LIMIT_TABLE[ERROR_CORRECT_L][ver]
    return qr, ver, len(buf), cap, modes


def zxing_ok(qr, url, png_path):
    """Save the QR image and check that zxing-cpp decodes it back to exactly url."""
    img = qr.make_image().convert('RGB')
    img.save(png_path)
    res = zxingcpp.read_barcodes(Image.open(png_path))
    text = res[0].text if res else None
    if text != url:
        raise AssertionError(f'zxing mismatch {png_path}: got {text[:80] if text else None!r}')
    return True
