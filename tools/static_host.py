"""Apply this project's Cloudflare _headers rules in local HTTP/browser checks."""
from fnmatch import fnmatchcase
from pathlib import Path

SITE = Path(__file__).resolve().parent.parent / 'site'


def headers_for(path):
    result, matches = {}, False
    for line in (SITE / '_headers').read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        if not line.startswith((' ', '\t')):
            matches = fnmatchcase(path, line.strip())
        elif matches:
            value = line.strip()
            if value.startswith('! '):
                result.pop(value[2:].lower(), None)
            else:
                key, value = value.split(':', 1)
                key = key.lower()
                result[key] = result[key] + ', ' + value.strip() if key in result else value.strip()
    return result
