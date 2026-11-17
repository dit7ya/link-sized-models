"""Run every check for the published site. No training, no network.

    uv run python tools/check.py               # static checks, loader unit test, Firefox + Chromium
    uv run python tools/check.py --no-browser  # skip the browser checks

The browser checks need Playwright and its browsers. If the project environment
does not have Playwright, they run under `uv run --with playwright`.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import unittest

from PIL import Image
import zxingcpp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from host import PREFIX, QR_PREFIX  # noqa: E402
from static_host import headers_for  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / 'site'
# Each file the site serves, and the exported artifact it must equal.
PAYLOADS = {'qr_names_url.txt': 'out/qr_names_url.txt', 'qrstory_url.txt': 'out/qrstory_url.txt',
            't64kfix_url.txt': 'out/t64kfix_url.txt', 'big2_url.txt': 'out/big2_url.txt'}


def read(path):
    return (ROOT / path).read_text().strip()


class Site(unittest.TestCase):
    def test_payloads_match_exports(self):
        for served, source in PAYLOADS.items():
            with self.subTest(served=served):
                url = read(f'site/data/payloads/{served}')
                self.assertEqual(url, read(source))
                self.assertTrue(url.startswith((PREFIX, QR_PREFIX)))

    def test_qr_codes_decode_to_the_names_link(self):
        url = read('site/data/payloads/qr_names_url.txt')
        self.assertTrue(url.startswith(QR_PREFIX), 'QR link must use the uppercase prefix (alphanumeric mode)')
        for image in ['site/img/qr.png', 'site/img/og.png']:
            with self.subTest(image=image):
                found = zxingcpp.read_barcodes(Image.open(ROOT / image))
                self.assertEqual(len(found), 1)
                self.assertEqual(found[0].text, url)
                self.assertLessEqual(int(found[0].extra['Version']), 30)

    def test_launch_page_files_exist(self):
        for path in ['index.html', '404.html', '_headers', 'home/index.html', 'home/page.css',
                     'home/page.js', 'img/qr.png', 'img/og.png', 'fonts/space-grotesk-latin.woff2']:
            self.assertTrue((SITE / path).is_file(), path)

    def test_headers(self):
        html = headers_for('/')
        policy = html['content-security-policy']
        self.assertIn("script-src 'self' 'unsafe-inline' 'unsafe-eval'", policy)
        self.assertIn("connect-src 'self'", policy)
        self.assertNotIn('https:', policy)
        self.assertNotIn('*', policy)
        self.assertEqual(html['cache-control'], 'public, max-age=300, must-revalidate')
        self.assertEqual(headers_for('/data/payloads/big2_url.txt')['cache-control'], 'public, max-age=86400, must-revalidate')

    def test_python_and_node_share_prefix(self):
        node = subprocess.check_output(['node', '-e', "process.stdout.write(require('./tools/host.cjs').prefix)"], cwd=ROOT, text=True)
        self.assertEqual(node, PREFIX)
        self.assertEqual(json.loads(read('tools/host.json'))['prefix'], PREFIX)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    results = []

    print('== static checks', flush=True)
    outcome = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Site))
    results.append(('static checks', outcome.wasSuccessful()))

    print('== loader DEFLATE fallback (Node)', flush=True)
    results.append(('loader DEFLATE fallback', subprocess.run(['node', 'tools/runtime_test.js'], cwd=ROOT).returncode == 0))

    if not args.no_browser:
        print('== browser checks', flush=True)
        probe = subprocess.run([sys.executable, '-c', 'import playwright'], capture_output=True)
        command = [sys.executable] if probe.returncode == 0 else ['uv', 'run', '--with', 'playwright', 'python']
        results.append(('browser checks', subprocess.run(command + ['tools/browser_test.py'], cwd=ROOT).returncode == 0))

    print('\n== summary')
    for name, ok in results:
        print(('PASS ' if ok else 'FAIL ') + name)
    sys.exit(0 if all(ok for _, ok in results) else 1)


if __name__ == '__main__':
    main()
