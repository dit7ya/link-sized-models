"""Browser checks for the live site: the loader and the launch page.

    uv run --with playwright python tools/browser_test.py [--engines firefox chromium]

Serves site/ over local HTTP with the response headers from site/_headers
(so the real Content-Security-Policy applies), then runs one browser at a time.
"""
import argparse
import base64
import functools
import http.server
import json
from pathlib import Path
import sys
import threading
import time
from urllib.parse import urlsplit

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from static_host import headers_for  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / 'site'
PAYLOADS = ['qr_names_url.txt', 'qrstory_url.txt', 't64kfix_url.txt', 'big2_url.txt']
SEED = "let seed=3;Math.random=()=>{seed=(1664525*seed+1013904223)>>>0;return seed/4294967296};"
WATCH_CSP = "window.__csp=[];addEventListener('securitypolicyviolation',e=>__csp.push(e.effectiveDirective+' '+e.blockedURI))"
ERRORS = ('This link looks damaged', 'This link carries no model', 'failed to initialise')


def serve():
    class Handler(http.server.SimpleHTTPRequestHandler):
        def end_headers(self):
            for key, value in headers_for(urlsplit(self.path).path).items():
                self.send_header(key, value)
            super().end_headers()

        def log_message(self, *_):
            pass
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(Handler, directory=str(SITE)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f'http://127.0.0.1:{server.server_address[1]}'


class Checks:
    def __init__(self, engine):
        self.engine, self.passed, self.failed = engine, [], []

    def run(self, name, fn):
        started = time.perf_counter()
        try:
            fn()
        except Exception as error:  # report every check, then fail at the end
            self.failed.append((name, f'{type(error).__name__}: {error}'.splitlines()[0][:300]))
            print(f'FAIL {self.engine}: {name}: {self.failed[-1][1]}', flush=True)
        else:
            self.passed.append(name)
            print(f'ok   {self.engine}: {name} ({time.perf_counter() - started:.1f}s)', flush=True)


def body(page):
    return page.locator('body').inner_text().strip()


def open_fresh(page, url):
    # A new fragment on the same document reloads; start from a blank page instead.
    page.goto('about:blank')
    page.goto(url, wait_until='load')


def loader_checks(checks, page, base):
    def payload(filename):
        def check():
            fragment = (SITE / 'data/payloads' / filename).read_text().strip().split('#', 1)[1]
            open_fresh(page, base + '/#' + fragment)
            page.wait_for_function("window.__linkModel?.complete && document.querySelector('#ft') && document.body.innerText.trim().length>0", timeout=180000)
            first = body(page)
            assert not any(e in first for e in ERRORS), first[:200]
            assert page.locator('#ft').is_visible(), 'footer missing'
            page.click('#again')
            page.wait_for_function("before=>window.__linkModel.complete && document.body.innerText.trim().length>0 && document.body.innerText.trim()!==before",
                                   arg=first, timeout=180000)
            second = body(page)
            assert not any(e in second for e in ERRORS), second[:200]
            assert page.evaluate('__csp') == [], page.evaluate('__csp')
        return check
    for filename in PAYLOADS:
        checks.run(f'loader runs {filename} and "Write another" gives new text', payload(filename))

    def empty_hash():
        open_fresh(page, base + '/')
        page.wait_for_url(base + '/home/')
    checks.run('empty hash redirects to /home/', empty_hash)

    def damaged():
        for fragment, expected in [('missing', 'This link carries no model'), ('!!!!.abc', 'This link looks damaged'),
                                   ('999abc', 'This link looks damaged'), ('000123', 'This link looks damaged'),
                                   ('%ZZ', 'This link looks damaged')]:
            open_fresh(page, base + '/#' + fragment)
            assert expected in body(page), (fragment, body(page)[:200])
        program = base64.urlsafe_b64encode(b"Promise.reject(Error('bad'))").decode().rstrip('=')
        open_fresh(page, base + '/#' + program + '.x')
        page.wait_for_function("document.body.innerText.includes('failed to initialise')")
    checks.run('damaged links show the error message', damaged)

    def raw_non_strict():
        raw = "g=()=>{with(Math)return d+'>'+floor(2.9)}"
        for program in [raw, raw.replace('>', '%3E')]:
            open_fresh(page, base + '/#' + f'{len(raw):03d}' + program + '123')
            assert body(page) == '123>2', body(page)
    checks.run('numeric format: non-strict eval, length measured after URI decoding', raw_non_strict)

    def csp_blocks_outside():
        open_fresh(page, base + '/#ZG9jdW1lbnQudGl0bGU9ZA.xyz')
        assert page.title() == 'xyz'
        blocked = page.evaluate("async()=>{try{await fetch('https://outside.invalid/m');return false}catch{return true}}")
        page.wait_for_function("__csp.some(v=>v.startsWith('connect-src'))")
        assert blocked
    checks.run('CSP header allows the loader but blocks outside connections', csp_blocks_outside)


def home_checks(checks, page, base):
    def load():
        open_fresh(page, base + '/home/')
        assert page.title() == 'This QR code is a language model', page.title()
        page.wait_for_function("document.querySelectorAll('#names li').length>=3", timeout=60000)
    checks.run('launch page title and at least 3 names', load)

    def toy():
        page.wait_for_function("!document.getElementById('toy-go').disabled", timeout=60000)
        page.fill('#toy-input', 'gpt-')
        page.click('#toy-go')
        page.wait_for_function("document.getElementById('toy-name').textContent.startsWith('gpt-') && document.querySelectorAll('#odds-rows li').length>=6", timeout=60000)
    checks.run('start-a-name box finishes "gpt-" and shows odds', toy)

    def packing():
        page.wait_for_function("document.getElementById('pack-number').textContent.length>100 && document.querySelectorAll('#link-block span').length===4", timeout=30000)
        page.click('#run-all')
    checks.run('weights number and full link are filled', packing)

    def ladder():
        stories = page.locator('.rung[data-model] .story')
        assert stories.count() == 3, stories.count()
        page.locator('.rung[data-model]').last.scroll_into_view_if_needed()
        page.wait_for_function("[...document.querySelectorAll('.rung[data-model] .story')].every(s=>!s.classList.contains('waiting') && s.textContent.trim().length>40 && !s.textContent.includes('Loading'))", timeout=180000)
        hrefs = page.locator('.rung .open').evaluate_all('links=>links.map(a=>a.getAttribute("href"))')
        assert len(hrefs) == 3 and all(h.startswith('/#') for h in hrefs), [h[:20] for h in hrefs]
    checks.run('all three ladder stories write text; "Open" links start with /#', ladder)

    def mobile():
        page.set_viewport_size({'width': 390, 'height': 844})
        page.wait_for_timeout(300)
        width = page.evaluate('[document.documentElement.scrollWidth, innerWidth]')
        page.set_viewport_size({'width': 1280, 'height': 900})
        assert width[0] <= width[1], width
    checks.run('no horizontal overflow at 390 px', mobile)

    def no_violations():
        violations = page.evaluate('__csp')
        assert not violations, violations
    checks.run('no CSP violations on the launch page', no_violations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engines', nargs='+', default=['firefox', 'chromium'])
    args = parser.parse_args()
    server, base = serve()
    results = {}
    try:
        with sync_playwright() as playwright:
            for engine in args.engines:  # one browser at a time
                browser = getattr(playwright, engine).launch()
                checks = Checks(engine)
                page = browser.new_page(viewport={'width': 1280, 'height': 900})
                page.set_default_timeout(60000)
                page.add_init_script(SEED)
                page.add_init_script(WATCH_CSP)
                errors = []
                page.on('pageerror', lambda error: errors.append(str(error)))
                loader_checks(checks, page, base)
                home_checks(checks, page, base)

                def no_errors():
                    assert not errors, errors[:3]
                checks.run('no uncaught page errors', no_errors)
                results[engine] = dict(version=browser.version, passed=len(checks.passed), failed=checks.failed)
                browser.close()
    finally:
        server.shutdown()
    print(json.dumps(results, indent=2))
    failed = sum(len(r['failed']) for r in results.values())
    passed = sum(r['passed'] for r in results.values())
    print(f'Browser checks: {passed} passed, {failed} failed')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
