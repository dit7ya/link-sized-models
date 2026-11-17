// Runs each model from its own link inside a sandboxed frame. The frame is the
// same loader a phone opens; it reports its text back with postMessage.
(() => {
  'use strict';
  const QR_LINK = '../data/payloads/qr_names_url.txt';
  const engines = new Map(); // frame window -> handler

  const linkCache = new Map();
  function fetchLink(path) {
    if (!linkCache.has(path)) {
      linkCache.set(path, fetch(path).then(r => {
        if (!r.ok) throw Error('Could not load ' + path);
        return r.text();
      }).then(t => t.trim()));
    }
    return linkCache.get(path);
  }

  // The loader lives at the site root and reads everything after '#'.
  const localHref = link => '/#' + link.slice(link.indexOf('#') + 1);
  // Frames ask for the current loader, not one the browser cached before an update.
  const frameHref = link => '/?v=4#' + link.slice(link.indexOf('#') + 1);

  function startEngine(link, onOutput) {
    const frame = document.createElement('iframe');
    frame.className = 'engine';
    frame.title = 'model';
    frame.tabIndex = -1;
    frame.setAttribute('aria-hidden', 'true');
    frame.setAttribute('sandbox', 'allow-scripts');
    document.body.append(frame);
    frame.src = frameHref(link);
    const engine = {
      frame,
      onOutput,
      generate() { frame.contentWindow.postMessage({type: 'link-model-generate'}, '*'); },
      // Ask for one name with its first characters fixed, plus the odds at every step.
      trace(force) { frame.contentWindow.postMessage({type: 'link-model-generate', trace: true, force}, '*'); },
    };
    engines.set(frame.contentWindow, engine);
    return engine;
  }

  addEventListener('message', event => {
    const data = event.data;
    if (!data) return;
    const engine = engines.get(event.source);
    if (!engine) return;
    if (data.type === 'link-model-trace' && engine.onTrace && Array.isArray(data.steps)) engine.onTrace(data.steps);
    if (data.type !== 'link-model-output' || typeof data.text !== 'string') return;
    engine.onOutput(data.text, !!data.complete, data.generation);
  });

  // Hero: a steady stream of invented names.
  const list = document.getElementById('names');
  const MAX_NAMES = 8;
  let heroVisible = true;
  let lastGeneration = -1;
  let heroEngine = null;
  let timer = 0;

  function addName(text) {
    const name = text.replace(/[_\s]+$/, '').trim();
    if (!name) return;
    for (const li of list.children) { li.classList.remove('new'); li.classList.add('old'); }
    const li = document.createElement('li');
    li.className = 'new';
    li.textContent = name;
    list.prepend(li);
    while (list.children.length > MAX_NAMES) list.lastElementChild.remove();
  }

  function scheduleNext(delay) {
    clearTimeout(timer);
    timer = setTimeout(() => {
      if (heroVisible && !document.hidden) heroEngine.generate();
      else scheduleNext(800);
    }, delay);
  }

  fetchLink(QR_LINK).then(link => {
    heroEngine = startEngine(link, (text, complete, generation) => {
      if (!complete || generation === lastGeneration) return;
      lastGeneration = generation;
      addName(text);
      // Fill the list quickly at first, then slow to a readable pace.
      scheduleNext(list.children.length < 5 ? 250 : 1400);
    });
    renderLink(link);
    renderPacking(link);
    setupToy(link);
  }).catch(err => { list.innerHTML = ''; const li = document.createElement('li'); li.textContent = err.message; list.append(li); });

  if ('IntersectionObserver' in window) {
    new IntersectionObserver(entries => { heroVisible = entries[0].isIntersecting; })
      .observe(document.querySelector('.hero'));
  }

  // The full link, coloured by part: address, program length, program, weights.
  function renderLink(link) {
    const block = document.getElementById('link-block');
    const hash = link.indexOf('#') + 1;
    const frag = link.slice(hash);
    const n = Number(frag.slice(0, 3));
    const parts = [
      ['seg-host', link.slice(0, hash)],
      ['seg-len', frag.slice(0, 3)],
      ['seg-prog', frag.slice(3, 3 + n)],
      ['seg-data', frag.slice(3 + n)],
    ];
    block.textContent = '';
    for (const [cls, text] of parts) {
      const span = document.createElement('span');
      span.className = cls;
      span.textContent = text;
      block.append(span);
    }
  }

  // The packing picture, filled in from the real link.
  function renderPacking(link) {
    const frag = link.slice(link.indexOf('#') + 1);
    const n = Number(frag.slice(0, 3));
    const digits = frag.slice(3 + n);
    document.getElementById('pack-number').textContent = digits.slice(0, 96) + ' … ' + digits.slice(-12);
  }

  // A network drawn from its link: one line per weight.
  function drawNet(id, link, cfg) {
    const svg = document.getElementById(id);
    const frag = link.slice(link.indexOf('#') + 1);
    const n = Number(frag.slice(0, 3));
    const m = frag.slice(3, 3 + n).match(/i=(\d+);.*?\/=(\d+)n\)\*\d+n\)-(\d+)/);
    if (!svg || !m || typeof BigInt !== 'function') return;
    const count = Number(m[1]), base = BigInt(m[2]), shift = Number(m[3]);
    const W = new Array(count);
    for (let v = BigInt(frag.slice(3 + n)), i = count; i--;) { W[i] = Number(v % base) - shift; v /= base; }

    const NS = 'http://www.w3.org/2000/svg';
    const el = (name, attrs, text) => {
      const e = document.createElementNS(NS, name);
      for (const k in attrs) e.setAttribute(k, attrs[k]);
      if (text != null) e.textContent = text;
      return e;
    };
    const width = 640, left = 14, right = 626, yIn = 62, yHid = 172, yOut = 282;
    const spread = (count, a, b) => Array.from({length: count}, (_, i) => a + (b - a) * (count === 1 ? .5 : i / (count - 1)));
    // Inputs: three groups of six, then the length input if the model has one.
    const inX = [];
    const groupW = cfg.length ? 150 : 170, gap = (right - left - 3 * groupW - (cfg.length ? 40 : 0)) / (cfg.length ? 3 : 2);
    for (let g = 0; g < 3; g++) {
      const x0 = left + g * (groupW + gap);
      spread(6, x0 + 10, x0 + groupW - 10).forEach(x => inX.push(x));
      svg.append(el('text', {x: x0 + groupW / 2, y: yIn - 30, class: 'mid'}, ['3 characters back', '2 back', 'the last character'][g]));
      svg.append(el('text', {x: x0 + groupW / 2, y: yIn - 18, class: 'ch'}, cfg.example[g]));
    }
    if (cfg.length) { inX.push(right - 20); svg.append(el('text', {x: right - 20, y: yIn - 18, class: 'mid'}, 'length')); }
    const hidX = spread(cfg.hidden, left + 10, right - 10);
    const outX = spread(cfg.labels.length, left + 6, right - 6);

    const lines = document.createDocumentFragment();
    const edge = (x1, y1, x2, y2, w) => {
      const o = Math.min(1, Math.abs(w) / shift);
      lines.append(el('line', {x1, y1, x2, y2, class: w < 0 ? 'neg' : 'pos', 'stroke-opacity': (.04 + .5 * o * o).toFixed(3), 'stroke-width': (.3 + o).toFixed(2)}));
    };
    for (let j = 0; j < cfg.hidden; j++) {
      for (let i = 0; i < inX.length; i++) edge(inX[i], yIn + 4, hidX[j], yHid - 6, W[cfg.hiddenAt + j * inX.length + i]);
      for (let k = 0; k < outX.length; k++) edge(hidX[j], yHid + 6, outX[k], yOut - 6, W[cfg.outAt + k * cfg.hidden + j]);
    }
    svg.append(lines);
    inX.forEach(x => svg.append(el('circle', {cx: x, cy: yIn, r: 3.5})));
    hidX.forEach(x => svg.append(el('circle', {cx: x, cy: yHid, r: 5.5, class: 'hid'})));
    outX.forEach((x, k) => {
      svg.append(el('rect', {x: x - 5, y: yOut - 5, width: 10, height: 10}));
      svg.append(el('text', {x, y: yOut + 20, class: 'ch'}, cfg.labels[k]));
    });
    svg.append(el('text', {x: left, y: yHid - 14}, cfg.hidden + ' hidden units'));
    svg.append(el('text', {x: left, y: yOut + 36}, cfg.labels.length + ' outputs, one per character'));
  }

  const LETTERS = Array.from({length: 26}, (_, i) => String.fromCharCode(97 + i));
  fetchLink(QR_LINK).then(link => drawNet('net-names', link, {
    hidden: 29, hiddenAt: 234, outAt: 814, length: true, example: ['m', 'i', 'n'],
    labels: ['end', '-', '.'].concat(Array.from('0123456789'), LETTERS),
  })).catch(() => {});
  fetchLink('../data/payloads/qrstory_url.txt').then(link => drawNet('net-story', link, {
    hidden: 32, hiddenAt: 198, outAt: 806, length: false, example: ['t', 'h', 'e'],
    labels: ['␣', '.', ',', '"', "'", '!', '?'].concat(LETTERS),
  })).catch(() => {});

  // Give it a start: the same link, with its first choices made for it.
  const CHARS = ['', '-', '.'].concat(Array.from({length: 36}, (_, i) => i.toString(36)));
  function setupToy(link) {
    const input = document.getElementById('toy-input');
    const go = document.getElementById('toy-go');
    const nameEl = document.getElementById('toy-name');
    const odds = document.getElementById('odds');
    const title = document.getElementById('odds-title');
    const rows = document.getElementById('odds-rows');
    const hint = document.getElementById('toy-hint');
    const still = matchMedia('(prefers-reduced-motion: reduce)').matches;
    let steps = [], given = 0, ready = false, pending = false, timer = 0;

    const label = i => i ? CHARS[i] : 'end';

    function showStep(k) {
      const step = steps[k];
      if (!step) return;
      const before = steps.slice(0, k).map(s => CHARS[s.i]).join('');
      title.textContent = '';
      title.append(before ? 'After ' : 'At the very start', ...(before ? [Object.assign(document.createElement('code'), {textContent: before}), ', its odds for the next character:'] : [', its odds for the first character:']));
      const order = step.p.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]);
      const shown = order.slice(0, 6);
      if (!shown.some(([, i]) => i === step.i)) shown.push([step.p[step.i], step.i]);
      const max = shown[0][0] || 1;
      rows.textContent = '';
      for (const [p, i] of shown) {
        const li = document.createElement('li');
        if (i === step.i) li.className = 'picked';
        const ch = document.createElement('span'); ch.textContent = label(i);
        const track = document.createElement('span'); track.className = 'track';
        const fill = document.createElement('span'); fill.className = 'fill'; fill.style.width = (100 * p / max).toFixed(1) + '%';
        track.append(fill);
        const pct = document.createElement('span'); pct.className = 'pct';
        pct.textContent = p >= .0995 ? Math.round(p * 100) + '%' : p >= .001 ? (p * 100).toFixed(1) + '%' : '<0.1%';
        li.append(ch, track, pct);
        rows.append(li);
      }
      odds.hidden = false;
      nameEl.querySelectorAll('button').forEach((b, j) => b.classList.toggle('on', j === k));
    }

    function reveal(upTo) {
      nameEl.textContent = '';
      steps.slice(0, upTo).forEach((step, k) => {
        if (!step.i) {
          const stop = document.createElement('button');
          stop.type = 'button'; stop.className = 'stop'; stop.textContent = 'end';
          stop.setAttribute('aria-label', 'End of the name');
          stop.addEventListener('click', () => showStep(k));
          nameEl.append(stop);
          return;
        }
        const b = document.createElement('button');
        b.type = 'button';
        b.textContent = CHARS[step.i];
        if (k < given) b.className = 'given';
        b.addEventListener('click', () => showStep(k));
        nameEl.append(b);
      });
      if (!upTo) nameEl.textContent = '\u00a0';
    }

    function play() {
      clearInterval(timer);
      if (still) { reveal(steps.length); showStep(Math.min(given, steps.length - 1)); return; }
      let k = Math.min(given, steps.length - 1);
      reveal(k + 1); showStep(k);
      timer = setInterval(() => {
        if (++k >= steps.length) { clearInterval(timer); hint.hidden = false; return; }
        reveal(k + 1); showStep(k);
      }, 140);
    }

    const engine = startEngine(link, () => { if (!ready) { ready = true; if (pending) run(); } });
    engine.onTrace = traced => {
      // Keep the forced start even if the name ended early.
      steps = traced;
      go.disabled = false;
      play();
    };

    function run() {
      const start = input.value.toLowerCase().replace(/[^a-z0-9.-]/g, '').slice(0, 14);
      input.value = start;
      if (!ready) { pending = true; return; }
      pending = false;
      given = start.length;
      go.disabled = true;
      engine.trace(Array.from(start, c => CHARS.indexOf(c, 1)));
    }

    go.addEventListener('click', run);
    input.addEventListener('keydown', e => { if (e.key === 'Enter') run(); });
    document.getElementById('presets').addEventListener('click', e => {
      const b = e.target.closest('button[data-start]');
      if (!b) return;
      input.value = b.dataset.start;
      run();
    });
    run();
  }

  // Story models start when their row scrolls into view.
  function setupRung(rung) {
    const story = rung.querySelector('.story');
    const again = rung.querySelector('.again');
    const open = rung.querySelector('.open');
    let engine = null;
    let started = false;

    const preview = rung.querySelector('.rung-link');
    const speed = rung.querySelector('.speed');
    let t0 = 0;
    fetchLink(rung.dataset.model).then(link => {
      open.href = localHref(link);
      const b = document.createElement('b');
      b.textContent = link.slice(0, 44);
      const rest = document.createElement('span');
      rest.textContent = '+ ' + (link.length - 44).toLocaleString('en') + ' more characters';
      preview.append(b, rest);
    })
      .catch(() => { open.remove(); });

    function start() {
      if (started) return;
      started = true;
      story.textContent = 'Loading the model…';
      fetchLink(rung.dataset.model).then(link => {
        // Keep the start of the story in view; the reader scrolls if they want more.
        engine = startEngine(link, (text, complete) => {
          story.classList.remove('waiting');
          story.classList.toggle('streaming', !complete);
          story.textContent = text.replace(/_$/, '');
          again.disabled = !complete;
          if (!t0) t0 = performance.now();
          const seconds = (performance.now() - t0) / 1000;
          if (seconds > .5 && text.length > 20) speed.textContent = Math.round(text.length / seconds).toLocaleString('en') + ' characters a second';
        });
      }).catch(err => { story.textContent = err.message; });
    }

    again.addEventListener('click', () => {
      if (!engine) return;
      again.disabled = true;
      story.scrollTop = 0;
      t0 = 0;
      engine.generate();
    });
    runners.push(() => { if (engine) again.click(); else start(); });

    if ('IntersectionObserver' in window) {
      const io = new IntersectionObserver(entries => {
        if (entries.some(e => e.isIntersecting)) { io.disconnect(); start(); }
      }, {rootMargin: '200px 0px'});
      io.observe(rung);
    } else {
      start();
    }
  }

  const runners = [];
  document.querySelectorAll('.rung[data-model]').forEach(setupRung);
  document.getElementById('run-all').addEventListener('click', () => runners.forEach(run => run()));
})();
