/* ROCm perf metrics trends.
 *
 * Plain SVG built in the DOM: no chart library, no CDN, no build step -- the
 * page is three static files next to a trends.json, which is what the Pages
 * job uploads. Every definition lives in perf-metrics/trends.py; nothing is
 * recomputed here beyond axis scaling and the percentages the tooltips show.
 *
 * The chart primitives (lineChart, stackPanel, kpiRow, ticks, the crosshair)
 * are the ci-metrics ones with the day axis generalised to a run axis: a perf
 * point is one nightly against one pinned XLA commit, and two nightlies can
 * land on the same day.
 */
'use strict';

const SVG_NS = 'http://www.w3.org/2000/svg';

const state = {
  data: null,
  tab: 'models',
  modes: new Set(),
  metric: 'p50_us',
  module: null,
  table: false,
};

/* ---- formatting ---------------------------------------------------------- */

/* Module times span three orders of magnitude (detr at 1.4ms, alphafold3 at
 * 1597ms), so the unit follows the value rather than being fixed. */
const fmtMs = (v) => {
  if (v == null) return '--';
  if (v >= 10000) return `${(v / 1000).toFixed(1)}s`;
  if (v >= 100) return `${Math.round(v)}ms`;
  if (v >= 10) return `${v.toFixed(1)}ms`;
  return `${v.toFixed(2)}ms`;
};

const fmtUs = (v) => (v == null ? '--' : v >= 1000 ? `${(v / 1000).toFixed(2)}ms` : `${Math.round(v)}µs`);
const fmtQps = (v) => (v == null ? '--' : Math.round(v).toLocaleString());
const fmtIndex = (v) => (v == null ? '--' : v.toFixed(1));
const fmtInt = (v) => (v == null ? '--' : Math.round(v).toLocaleString());

/* A change, always with its sign: these are read as directions first. */
const fmtDelta = (v) => (v == null ? '--' : `${v > 0 ? '+' : ''}${(v * 100).toFixed(1)}%`);

const MONTHS = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

const fmtDay = (iso) => {
  if (!iso) return '--';
  const [, m, d] = iso.split('-');
  return `${MONTHS[+m]} ${+d}`;
};

const hhmm = (iso) => (iso ? iso.slice(11, 16) : '');

/**
 * The x-axis label for one run.
 *
 * The date alone, except where another run shares it -- the nightly is also
 * run by hand, so two landed on Nov 4 and two on Nov 6. Those get the UTC time
 * as well, which is the only thing that tells them apart on an axis.
 */
function runLabel(i) {
  const runs = state.data.runs;
  const r = runs[i];
  const shared = runs.some((o, k) => k !== i && o.day === r.day);
  return shared ? `${fmtDay(r.day)} ${hhmm(r.iso)}` : fmtDay(r.day);
}

const modeColor = (i) => `var(--series-${i + 1})`;
/* Group colours continue after the modes, so a category band is never the same
 * hue as a stack's line on the chart above it. */
const groupColor = (i) => `var(--series-${((i + 3) % 7) + 1})`;

/* The dash patterns the GPU counts take, in order. Colour stays with the
 * stack, so filtering a stack off never repaints the rest. */
const DASHES = [null, '6 4', '2 3', '10 3 2 3'];

/* ---- tiny DOM helpers ---------------------------------------------------- */

function el(tag, attrs = {}, children = []) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    // An absent attribute is absent, not the string "undefined".
    if (v === null || v === undefined) continue;
    if (k === 'class') n.className = v;
    else if (k === 'text') n.textContent = v;
    else if (k === 'html') n.innerHTML = v;
    else n.setAttribute(k, v);
  }
  for (const c of [].concat(children)) if (c) n.append(c);
  return n;
}

function svg(tag, attrs = {}) {
  const n = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v !== null && v !== undefined) n.setAttribute(k, v);
  }
  return n;
}

function swatch(color, dashed) {
  return el('span', { class: dashed ? 'swatch dashed' : 'swatch', style: `--c:${color}` });
}

/** A link, or plain text when there is nothing to point at. */
function link(href, text, title, cls) {
  return href ? el('a', { href, text, title, class: cls, target: '_blank', rel: 'noopener' })
              : el('span', { text, title, class: cls });
}

/* ---- scales -------------------------------------------------------------- */

/** Round a maximum up to a readable tick step, and return the tick values. */
function ticks(max, count = 4, min = 0) {
  if (!(max > min)) return { min: 0, max: max > 0 ? max : 1, values: [0, max > 0 ? max : 1] };
  const raw = (max - min) / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? 10 * mag;
  const bottom = Math.floor(min / step) * step;
  const top = Math.ceil(max / step) * step;
  const values = [];
  for (let v = bottom; v <= top + step / 2; v += step) values.push(+v.toFixed(6));
  return { min: bottom, max: top, values };
}

/** Which run indices get an x-axis label, without colliding with the last. */
function labelIndices(n, x, minGap) {
  const every = Math.max(1, Math.ceil(n / Math.max(1, Math.floor(x(n - 1) / minGap))));
  const idxs = [];
  for (let i = 0; i < n; i += every) idxs.push(i);
  const last = n - 1;
  if (idxs[idxs.length - 1] !== last) {
    // Drop any label the forced final one would sit on top of.
    while (idxs.length && x(last) - x(idxs[idxs.length - 1]) < minGap * 0.8) idxs.pop();
    idxs.push(last);
  }
  return idxs;
}

/* ---- line chart ---------------------------------------------------------- */

/**
 * A multi-series line chart over the shared run axis.
 * `series` is [{label, color, values[], dash?, n?, ok?}]; nulls break the line.
 * `ok[i]` false draws that point hollow -- a run that did not measure the whole
 * suite is still plotted, because the gap is itself worth seeing, but it is
 * never drawn as if it were a clean reading.
 *
 * `zeroBased` is false for indices and latencies, whose interesting range is a
 * few percent above a large floor: a zero-based axis would flatten every one
 * of them into the same straight line.
 */
function lineChart(host, { series, fmt, zeroBased = true, directLabels = true }) {
  const runs = state.data.runs;
  host.textContent = '';
  const width = Math.max(320, host.clientWidth);
  const M = { top: 14, right: directLabels ? 92 : 16, bottom: 30, left: 62 };
  const H = 252;
  const plotW = width - M.left - M.right;
  const plotH = H - M.top - M.bottom;

  const flat = series.flatMap((s) => s.values).filter((v) => v != null);
  if (!flat.length) {
    host.append(el('p', { class: 'note empty', text: 'Nothing measured for this selection.' }));
    return;
  }
  const hi = Math.max(...flat);
  const lo = Math.min(...flat);
  // A little headroom below the lowest point, so a line never rides the axis.
  // A series that never moves has no range to clip to, so it falls back to a
  // zero-based axis rather than to a band of zero height.
  const floor = zeroBased || !(hi > lo) ? 0 : Math.max(0, lo - (hi - lo) * 0.25);
  const scale = ticks(hi, 4, floor);
  const span = scale.max - scale.min || 1;
  const x = (i) => (runs.length === 1 ? plotW / 2 : (i / (runs.length - 1)) * plotW);
  const y = (v) => plotH - ((v - scale.min) / span) * plotH;

  const root = svg('svg', { viewBox: `0 0 ${width} ${H}`, height: H, role: 'img' });
  const g = svg('g', { transform: `translate(${M.left},${M.top})` });
  root.append(g);

  // Gridlines: solid hairlines, one step off the surface.
  for (const v of scale.values) {
    g.append(svg('line', {
      x1: 0, x2: plotW, y1: y(v), y2: y(v), stroke: 'var(--grid)', 'stroke-width': 1,
    }));
    const t = svg('text', {
      x: -10, y: y(v) + 4, 'text-anchor': 'end',
      fill: 'var(--text-muted)', 'font-size': 11, 'font-variant-numeric': 'tabular-nums',
    });
    t.textContent = fmt(v);
    g.append(t);
  }

  for (const i of labelIndices(runs.length, x, 74)) {
    const t = svg('text', {
      x: x(i), y: plotH + 20, 'text-anchor': 'middle',
      fill: 'var(--text-muted)', 'font-size': 11,
    });
    t.textContent = runLabel(i);
    g.append(t);
  }
  g.append(svg('line', {
    x1: 0, x2: plotW, y1: plotH, y2: plotH, stroke: 'var(--axis)', 'stroke-width': 1,
  }));

  // Lines, broken across null runs.
  for (const s of series) {
    let run = [];
    const flush = () => {
      if (run.length > 1) {
        g.append(svg('path', {
          d: run.map((p, k) => `${k ? 'L' : 'M'}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(''),
          fill: 'none', stroke: s.color, 'stroke-width': 2,
          'stroke-dasharray': s.dash || null,
          'stroke-linejoin': 'round', 'stroke-linecap': 'round',
        }));
      }
      run = [];
    };
    s.values.forEach((v, i) => (v == null ? flush() : run.push([x(i), y(v)])));
    flush();
  }

  // Markers. Perf runs are few, so every point gets one: a lone measurement
  // with no neighbours would otherwise draw nothing at all.
  for (const s of series) {
    s.values.forEach((v, i) => {
      if (v == null) return;
      const thin = s.ok ? s.ok[i] === false : false;
      g.append(svg('circle', {
        cx: x(i), cy: y(v), r: 4,
        fill: thin ? 'var(--surface-1)' : s.color,
        stroke: thin ? s.color : 'var(--surface-1)', 'stroke-width': 2,
      }));
    });
  }

  // Direct labels at the line ends, skipped where they would collide.
  if (directLabels) {
    const ends = series
      .map((s) => {
        for (let i = s.values.length - 1; i >= 0; i--) {
          if (s.values[i] != null) return { s, i, v: s.values[i] };
        }
        return null;
      })
      .filter(Boolean)
      .sort((a, b) => y(a.v) - y(b.v));

    let lastY = -Infinity;
    for (const e of ends) {
      const yy = y(e.v);
      if (yy - lastY < 14) continue;
      lastY = yy;
      const t = svg('text', {
        x: x(e.i) + 10, y: yy + 4, fill: 'var(--text-secondary)',
        'font-size': 11.5, 'font-variant-numeric': 'tabular-nums',
      });
      t.textContent = fmt(e.v);
      g.append(t);
    }
  }

  // Crosshair + tooltip layer.
  const rule = svg('line', { y1: 0, y2: plotH, stroke: 'var(--axis)', 'stroke-width': 1, opacity: 0 });
  g.append(rule);
  const dots = svg('g', { opacity: 0 });
  g.append(dots);

  host.append(root);
  const tip = el('div', { class: 'tip' });
  host.append(tip);

  const show = (i) => {
    rule.setAttribute('x1', x(i));
    rule.setAttribute('x2', x(i));
    rule.setAttribute('opacity', 1);
    dots.textContent = '';
    for (const s of series) {
      if (s.values[i] == null) continue;
      dots.append(svg('circle', {
        cx: x(i), cy: y(s.values[i]), r: 4.5, fill: s.color,
        stroke: 'var(--surface-1)', 'stroke-width': 2,
      }));
    }
    dots.setAttribute('opacity', 1);

    const r = runs[i];
    tip.textContent = '';
    tip.append(el('div', { class: 'tip-head' }, [
      el('span', { text: `${fmtDay(r.day)} ${hhmm(r.iso)}` }),
      el('span', { class: 'tip-flag', text: r.commit ? `xla ${r.commit}` : '' }),
    ]));
    for (const s of series) {
      const flags = [];
      if (s.ok && s.ok[i] === false) flags.push('partial');
      if (s.n && s.n[i] != null && s.n[i] !== (s.core ?? s.n[i])) flags.push(`n=${s.n[i]}`);
      tip.append(el('div', { class: 'tip-row' }, [
        swatch(s.color, s.dash),
        el('span', { text: s.label }),
        el('span', { class: 'v', text: fmt(s.values[i]) }),
        el('span', { class: 'n', text: flags.join(' · ') }),
      ]));
    }
    tip.classList.add('on');

    const tw = tip.offsetWidth || 200;
    tip.style.left = `${Math.min(Math.max(M.left + x(i) - tw / 2, 4), width - tw - 4)}px`;
    tip.style.top = `${M.top + 4}px`;
  };

  const hide = () => {
    rule.setAttribute('opacity', 0);
    dots.setAttribute('opacity', 0);
    tip.classList.remove('on');
  };

  const nearest = (clientX) => {
    const box = root.getBoundingClientRect();
    const px = ((clientX - box.left) / box.width) * width - M.left;
    return Math.max(0, Math.min(runs.length - 1, Math.round((px / plotW) * (runs.length - 1))));
  };

  // Hit surface spans the whole plot, so there is nothing to aim at.
  g.append(svg('rect', {
    x: 0, y: 0, width: plotW, height: plotH, fill: 'transparent', style: 'cursor:crosshair',
  }));
  root.addEventListener('mousemove', (e) => show(nearest(e.clientX)));
  root.addEventListener('mouseleave', hide);

  // Keyboard reaches the same values as hover.
  let focus = runs.length - 1;
  root.setAttribute('tabindex', '0');
  root.addEventListener('focus', () => show(focus));
  root.addEventListener('blur', hide);
  root.addEventListener('keydown', (e) => {
    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
    e.preventDefault();
    focus = Math.max(0, Math.min(runs.length - 1, focus + (e.key === 'ArrowRight' ? 1 : -1)));
    show(focus);
  });
}

/* ---- stacked area (small multiples) -------------------------------------- */

/** Total ms per run for one mode; also the small-multiple scale. */
function modeTotals(mode) {
  const d = state.data;
  return d.runs.map((_, i) =>
    d.groups.reduce((a, g) => a + (d.series[mode][`group_${g.key}`]?.[i] ?? 0), 0));
}

function stackPanel(host, mode, sharedMax) {
  const d = state.data;
  const runs = d.runs;
  const row = d.series[mode.key];

  const width = Math.max(260, host.clientWidth);
  const M = { top: 12, right: 12, bottom: 26, left: 52 };
  const H = 190;
  const plotW = width - M.left - M.right;
  const plotH = H - M.top - M.bottom;

  // Every panel shares one y scale -- panels on their own scales would make
  // stacks with very different totals look identical.
  const totals = modeTotals(mode.key);
  const scale = ticks(sharedMax, 3);
  const x = (i) => (runs.length === 1 ? plotW / 2 : (i / (runs.length - 1)) * plotW);
  const y = (v) => plotH - (v / scale.max) * plotH;

  const root = svg('svg', { viewBox: `0 0 ${width} ${H}`, height: H, role: 'img' });
  const g = svg('g', { transform: `translate(${M.left},${M.top})` });
  root.append(g);

  for (const v of scale.values) {
    g.append(svg('line', {
      x1: 0, x2: plotW, y1: y(v), y2: y(v), stroke: 'var(--grid)', 'stroke-width': 1,
    }));
    const t = svg('text', {
      x: -9, y: y(v) + 4, 'text-anchor': 'end', fill: 'var(--text-muted)',
      'font-size': 10.5, 'font-variant-numeric': 'tabular-nums',
    });
    t.textContent = fmtMs(v);
    g.append(t);
  }

  // Bands, bottom-up. Gaps split a band into separate runs so a missing run is
  // never bridged with a straight line.
  const base = runs.map(() => 0);
  d.groups.forEach((p, pi) => {
    const vals = row[`group_${p.key}`] || runs.map(() => null);
    let run = [];
    const flush = () => {
      if (run.length > 1) {
        const up = run.map((r) => `${r.x.toFixed(1)},${y(r.top).toFixed(1)}`);
        const down = run.slice().reverse().map((r) => `${r.x.toFixed(1)},${y(r.bot).toFixed(1)}`);
        g.append(svg('path', {
          d: `M${up.join('L')}L${down.join('L')}Z`,
          fill: groupColor(pi), 'fill-opacity': 0.85,
        }));
        // 2px surface gap between touching segments -- never a border stroke.
        g.append(svg('path', {
          d: `M${up.join('L')}`, fill: 'none',
          stroke: 'var(--surface-1)', 'stroke-width': 2,
        }));
      }
      run = [];
    };
    vals.forEach((v, i) => {
      if (v == null) { flush(); return; }
      run.push({ x: x(i), bot: base[i], top: base[i] + v });
      base[i] += v;
    });
    flush();
  });

  for (const i of labelIndices(runs.length, x, 70)) {
    const t = svg('text', {
      x: x(i), y: plotH + 18, 'text-anchor': 'middle',
      fill: 'var(--text-muted)', 'font-size': 10.5,
    });
    t.textContent = runLabel(i);
    g.append(t);
  }
  g.append(svg('line', {
    x1: 0, x2: plotW, y1: plotH, y2: plotH, stroke: 'var(--axis)', 'stroke-width': 1,
  }));

  host.append(root);

  const tip = el('div', { class: 'tip' });
  host.append(tip);
  const rule = svg('line', { y1: 0, y2: plotH, stroke: 'var(--axis)', 'stroke-width': 1, opacity: 0 });
  g.append(rule);

  const show = (i) => {
    rule.setAttribute('x1', x(i));
    rule.setAttribute('x2', x(i));
    rule.setAttribute('opacity', 1);
    tip.textContent = '';
    tip.append(el('div', { class: 'tip-head' }, [
      el('span', { text: `${fmtDay(runs[i].day)} ${hhmm(runs[i].iso)}` }),
      el('span', { class: 'tip-flag', text: totals[i] ? `total ${fmtMs(totals[i])}` : 'no total' }),
    ]));
    // Largest group first: the reader wants the dominant one, not the order.
    d.groups
      .map((p, pi) => ({ p, pi, v: row[`group_${p.key}`]?.[i] }))
      .filter((r) => r.v != null && r.v > 0)
      .sort((a, b) => b.v - a.v)
      .forEach((r) => {
        tip.append(el('div', { class: 'tip-row' }, [
          swatch(groupColor(r.pi)),
          el('span', { text: r.p.label }),
          el('span', { class: 'v', text: fmtMs(r.v) }),
        ]));
      });
    tip.classList.add('on');
    const tw = tip.offsetWidth || 200;
    tip.style.left = `${Math.min(Math.max(M.left + x(i) - tw / 2, 4), width - tw - 4)}px`;
    tip.style.top = `${M.top + 4}px`;
  };

  g.append(svg('rect', { x: 0, y: 0, width: plotW, height: plotH, fill: 'transparent' }));
  root.addEventListener('mousemove', (e) => {
    const box = root.getBoundingClientRect();
    const px = ((e.clientX - box.left) / box.width) * width - M.left;
    show(Math.max(0, Math.min(runs.length - 1, Math.round((px / plotW) * (runs.length - 1)))));
  });
  root.addEventListener('mouseleave', () => {
    rule.setAttribute('opacity', 0);
    tip.classList.remove('on');
  });
}

/* ---- pieces -------------------------------------------------------------- */

function activeModes() {
  return state.data.modes
    .map((m, i) => ({ ...m, color: modeColor(i) }))
    .filter((m) => state.modes.has(m.key));
}

/** The index/total series for the active stacks, with their coverage flags. */
function modeSeries(field) {
  return activeModes().map((m) => ({
    key: m.key,
    label: m.label,
    color: m.color,
    values: state.data.series[m.key][field],
    ok: state.data.series[m.key].complete,
    n: state.data.series[m.key].n,
    core: state.data.series[m.key].core,
  }));
}

const lastIndex = (vals) => {
  for (let i = vals.length - 1; i >= 0; i--) if (vals[i] != null) return i;
  return -1;
};

const firstIndex = (vals) => {
  for (let i = 0; i < vals.length; i++) if (vals[i] != null) return i;
  return -1;
};

/**
 * One tile per series: where it stands now and how far that is from where the
 * window started. `better` names the direction that is good news, so a rising
 * throughput reads green where a rising latency reads red.
 */
function kpiRow(host, series, { fmt, better, fallback }) {
  host.textContent = '';
  for (const s of series) {
    const last = lastIndex(s.values);
    const first = firstIndex(s.values);
    const cur = last >= 0 ? s.values[last] : null;
    const was = first >= 0 && first !== last ? s.values[first] : null;

    let delta = null;
    if (cur != null && was != null && was > 0) {
      const change = (cur - was) / was;
      const rising = change > 0;
      delta = el('div', { class: 'kpi-delta' }, [
        el('span', {
          class: rising === (better === 'up') ? 'good' : 'bad',
          text: `${rising ? '▲' : '▼'} ${Math.abs(change * 100).toFixed(1)}%`,
        }),
        el('span', { text: ` vs ${fmtDay(state.data.runs[first].day)}` }),
      ]);
    }

    host.append(el('div', { class: 'kpi' }, [
      el('div', { class: 'kpi-label' }, [swatch(s.color, s.dash), el('span', { text: s.label })]),
      el('div', { class: 'kpi-value', text: fmt(cur) }),
      delta || el('div', { class: 'kpi-delta', text: fallback }),
    ]));
  }
}

/**
 * What a view says when its `trends.json` does not carry it.
 *
 * The page and its data deploy together but are built by different steps, and
 * a trends.json written by an older trends.py simply lacks the newer keys.
 * Saying so beats hiding the view: a missing card reads as "the change was
 * never deployed", which is indistinguishable from "the data is stale".
 */
function emptyState(host, what) {
  host.textContent = '';
  host.append(el('p', { class: 'note empty' }, [
    el('span', { text: `No ${what} in this trends.json. ` }),
    el('span', {
      text: `It was generated ${state.data.generated} by a trends.py older than this page `
        + '— the dashboard is ahead of its data. Re-running trends.py over the '
        + 'perf-metrics-data runs fills this in.',
    }),
  ]));
}

/* ---- models tab ---------------------------------------------------------- */

function renderGroupLegend() {
  const host = document.getElementById('group-legend');
  host.textContent = '';
  state.data.groups.forEach((p, i) => {
    host.append(el('span', { class: 'legend-item' }, [
      swatch(groupColor(i)), el('span', { text: p.label }),
    ]));
  });
}

/** The headline the index card leads with. */
function indexStat() {
  const host = document.getElementById('index-stat');
  host.textContent = '';
  const modes = activeModes();
  if (!modes.length) return;
  const m = modes[modes.length - 1];
  const vals = state.data.series[m.key].index;
  const a = firstIndex(vals);
  const b = lastIndex(vals);
  if (a < 0 || b < 0 || a === b) return;
  const change = vals[b] - vals[a];
  const word = Math.abs(change) < 0.5 ? 'level with'
    : change < 0 ? 'faster than' : 'slower than';
  host.append(el('span', {
    html: `<span class="hl">${m.label}</span> is at <span class="hl">${fmtIndex(vals[b])}</span>, `
      + `${Math.abs(change).toFixed(1)}% ${word} ${fmtDay(state.data.runs[a].day)}.`,
  }));
}

/** The noise floor, in the card that needs it to be read correctly. */
function noiseNote() {
  const host = document.getElementById('noise-note');
  host.textContent = '';
  const parts = activeModes()
    .map((m) => [m, state.data.series[m.key].noise_pct])
    .filter(([, v]) => v != null)
    .map(([m, v]) => `${m.label} ±${(v * 100).toFixed(1)}%`);
  if (!parts.length) return;
  host.append(el('span', {
    text: `Run-to-run noise floor, the 90th percentile of every module's change between `
      + `consecutive runs: ${parts.join(', ')}. A move on this chart smaller than that is `
      + `the measurement, not the stack.`,
  }));
}

function renderGroups() {
  const host = document.getElementById('chart-groups');
  host.textContent = '';
  const modes = activeModes();
  // One scale for every panel, so the panels can be read against each other.
  const sharedMax = Math.max(
    ...modes.flatMap((m) => modeTotals(m.key)).filter((v) => v > 0), 1);

  // Build every panel before drawing any: a panel measured while it is the
  // grid's only child reports the full width, and the grid then reflows to two
  // columns under it, leaving that panel's viewBox scaled down.
  const plots = modes.map((m) => {
    const panel = el('div', {});
    panel.append(el('p', { class: 'panel-title', text: m.label }));
    const plot = el('div', { class: 'plot' });
    panel.append(plot);
    host.append(panel);
    return plot;
  });
  modes.forEach((m, i) => stackPanel(plots[i], m, sharedMax));
}

function renderModels() {
  kpiRow(document.getElementById('kpis'), modeSeries('index'),
         { fmt: fmtIndex, better: 'down', fallback: 'execution time index' });
  indexStat();
  noiseNote();
  lineChart(document.getElementById('chart-index'), {
    series: modeSeries('index'), fmt: fmtIndex, zeroBased: false,
  });
  lineChart(document.getElementById('chart-total'), {
    series: modeSeries('total_ms'), fmt: fmtMs, zeroBased: false,
  });
  renderGroupLegend();
  renderGroups();
}

/* ---- modules tab --------------------------------------------------------- */

const moduleById = (key) => state.data.modules.find((m) => m.key === key);

/** `model · module`, stacked, because the module id alone is not unique. */
function moduleName(m) {
  return el('div', { class: 'mod' }, [
    el('span', { class: 'mod-model', text: m.model }),
    el('span', { class: 'mod-id', text: m.module || '(single module)' }),
  ]);
}

function buildModulePicker() {
  const host = document.getElementById('module-pick');
  host.textContent = '';
  const byGroup = new Map();
  for (const m of state.data.modules) {
    if (!byGroup.has(m.group)) byGroup.set(m.group, []);
    byGroup.get(m.group).push(m);
  }
  const labels = Object.fromEntries(state.data.groups.map((g) => [g.key, g.label]));
  for (const [group, mods] of byGroup) {
    const og = el('optgroup', { label: labels[group] || group });
    for (const m of mods) {
      og.append(el('option', {
        value: m.key,
        text: `${m.model} · ${m.leaf} · ${m.module || 'single module'}`,
        selected: m.key === state.module ? '' : null,
      }));
    }
    host.append(og);
  }
  host.value = state.module;
  // render(), not renderModules(): the data table below is scoped by the
  // picked module too, and only render() knows whether it is open.
  host.onchange = () => { state.module = host.value; render(); };
}

function renderModuleChart() {
  const host = document.getElementById('chart-module');
  const d = state.data;
  const mod = moduleById(state.module);
  const stat = document.getElementById('module-stat');
  stat.textContent = '';
  if (!mod) { emptyState(host, 'module series'); return; }

  const series = activeModes().map((m) => ({
    label: m.label,
    color: m.color,
    values: d.module_values[m.key]?.[state.module] ?? d.runs.map(() => null),
  }));
  stat.append(el('span', {
    html: `<span class="hl">${mod.model}</span> · ${mod.leaf}`
      + (mod.module ? ` · <span class="hl">${mod.module}</span>` : ''),
  }));
  lineChart(host, { series, fmt: fmtMs, zeroBased: false });
}

/**
 * Where a module's move stands on the other stacks in the same run.
 *
 * All three stacks replay the same modules against the same XLA commit, so
 * this separates a change in XLA -- which moves every stack at once -- from
 * one in a single ROCm stack. The second is the row to chase, and the only one
 * that gets a mark.
 */
function agreement(also) {
  if (also === 'all') return el('span', { class: 'muted', text: 'every stack' });
  if (also === 'only') {
    return el('span', {
      class: 'diff', text: 'this stack only',
      title: 'The other stacks measured this module in the same run and did not move with it',
    });
  }
  return el('span', { class: 'muted', text: 'not measured elsewhere' });
}

const deltaCell = (v) => el('span', {
  class: v == null ? 'delta' : `delta ${v > 0 ? 'slower' : 'faster'}`,
  text: fmtDelta(v),
});

/**
 * What moved between each run and the one before it, newest run first.
 *
 * Grouped by run rather than by stack: the question this answers is "what
 * changed last night", and the chart above it is already per-module.
 */
function renderMovers() {
  const host = document.getElementById('movers-host');
  host.textContent = '';
  const d = state.data;
  const modes = activeModes();

  const cols = [
    { h: 'Stack', lead: true }, { h: 'Module', lead: true }, { h: 'Now' },
    { h: 'vs previous run' }, { h: 'vs start' }, { h: 'Other stacks', lead: true },
  ];
  const cell = (i, children) => el('td', { class: cols[i].lead ? 'lead' : null },
                                   typeof children === 'string' ? [el('span', { text: children })] : children);

  const table = el('table');
  const thead = el('thead');
  thead.append(el('tr', {}, cols.map((c) =>
    el('th', { text: c.h, scope: 'col', class: c.lead ? 'lead' : null }))));
  table.append(thead);

  const tbody = el('tbody');
  let any = false;
  for (let i = d.runs.length - 1; i >= 0; i--) {
    const rows = modes.map((m) => [m, d.movers[m.key]?.[i]]).filter(([, t]) => t);
    if (!rows.length) continue;
    any = true;
    const r = d.runs[i];

    tbody.append(el('tr', {}, [el('td', { class: 'run-head', colspan: cols.length }, [
      el('span', { text: `${fmtDay(r.day)} ${hhmm(r.iso)}` }),
      link(r.sha ? `https://github.com/${d.repository}/commit/${r.sha}` : null,
           `xla ${r.commit || ''}`, 'The upstream commit this run benchmarked', 'sha'),
    ])]));

    for (const [m, t] of rows) {
      if (!t.modules.length) {
        tbody.append(el('tr', {}, [
          cell(0, [el('div', {}, [swatch(m.color), el('span', { text: m.label })])]),
          el('td', { class: 'quiet', colspan: cols.length - 1 },
             [el('span', { text: `nothing above ±${(t.noise * 100).toFixed(1)}%` })]),
        ]));
        continue;
      }
      t.modules.forEach((c, k) => {
        const tr = el('tr', { class: c.key === state.module ? 'picked' : null }, [
          // The stack is named once per group of its own movers, and carries
          // the floor those movers cleared: a per-run fact, not a per-row one.
          cell(0, k ? [] : [
            el('div', {}, [swatch(m.color), el('span', { text: m.label })]),
            el('div', {
              class: 'muted',
              text: `${t.n} above ±${(t.noise * 100).toFixed(1)}%`,
            }),
          ]),
          cell(1, [moduleName(c), el('span', { class: 'leaf', text: c.leaf })]),
          cell(2, [
            el('span', { text: fmtMs(c.ms) }),
            el('span', { class: 'tip-flag', text: ` was ${fmtMs(c.was)}` }),
          ]),
          cell(3, [deltaCell(c.delta)]),
          cell(4, [deltaCell(c.vs_base)]),
          cell(5, [agreement(c.also)]),
        ]);
        // The table is the index into the chart above it.
        tr.style.cursor = 'pointer';
        tr.title = 'Plot this module';
        tr.addEventListener('click', () => {
          state.module = c.key;
          document.getElementById('module-pick').value = c.key;
          render();
        });
        tbody.append(tr);
      });
    }
  }
  // A change is a comparison, so the first run of a window has nothing to
  // report and a one-run window has nothing at all. An empty table with a
  // header and no rows reads as broken; this says which it is.
  if (!any) {
    host.append(el('p', { class: 'note empty' }, [el('span', {
      text: d.runs.length < 2
        ? 'Only one run published so far. A change is measured against the run before it, '
          + 'so this fills in from the second run on.'
        : 'No stack in view reported a comparison on any run. Turn a stack back on above.',
    })]));
    return;
  }
  table.append(tbody);
  host.append(table);
}

function renderModules() {
  if (!state.data.modules || !state.data.module_values) {
    emptyState(document.getElementById('chart-module'), 'per-module series');
  } else {
    renderModuleChart();
  }
  if (state.data.movers) renderMovers();
  else emptyState(document.getElementById('movers-host'), 'module changes');
}

/* ---- latency tab --------------------------------------------------------- */

/** One series per (stack, GPU count): colour is the stack, dash is the count. */
function latencySeries() {
  const d = state.data;
  if (!d.latency) return [];
  return activeModes().flatMap((m) =>
    d.latency.gpus
      .map((g, gi) => ({ g, gi, row: d.latency.series[`${m.key}:${g}`] }))
      .filter((s) => s.row)
      .map((s) => ({
        key: `${m.key}:${s.g}`,
        label: `${m.label} · ${s.g} GPU${s.g === 1 ? '' : 's'}`,
        color: m.color,
        dash: DASHES[s.gi % DASHES.length],
        values: s.row[state.metric],
      })));
}

function renderLatencyLegend(series) {
  const host = document.getElementById('latency-legend');
  host.textContent = '';
  for (const s of series) {
    host.append(el('span', { class: 'legend-item' }, [
      swatch(s.color, !!s.dash), el('span', { text: s.label }),
    ]));
  }
}

const metricDef = () =>
  state.data.latency?.metrics.find((m) => m.key === state.metric)
  ?? { key: state.metric, label: state.metric, unit: 'us' };

const latencyFmt = () => (metricDef().unit === 'qps' ? fmtQps : fmtUs);

/**
 * The headline the latency card leads with: how the figure scales with GPUs.
 *
 * Both halves come from the *same* stack, so the sentence compares GPU counts
 * and nothing else; mixing stacks would read as a scaling result when it was
 * really a version difference.
 */
function latencyStat(series) {
  const host = document.getElementById('latency-stat');
  host.textContent = '';
  const now = (s) => {
    const i = lastIndex(s.values);
    return i < 0 ? null : s.values[i];
  };
  const measured = series.filter((s) => now(s) != null);
  const mode = measured.map((s) => s.key.split(':')[0]).pop();
  const mine = measured.filter((s) => s.key.split(':')[0] === mode);
  const one = mine.find((s) => s.key.endsWith(':1'));
  const many = mine[mine.length - 1];
  if (!one || !many || one === many) return;
  const fmt = latencyFmt();
  const gpus = many.key.split(':')[1];
  host.append(el('span', {
    html: `Latest on <span class="hl">${many.label.split(' · ')[0]}</span>: `
      + `<span class="hl">${fmt(now(one))}</span> at 1 GPU against `
      + `<span class="hl">${fmt(now(many))}</span> at ${gpus}.`,
  }));
}

function renderLatency() {
  const d = state.data;
  const host = document.getElementById('chart-latency');
  if (!d.latency || !Object.keys(d.latency.series).length) {
    document.getElementById('latency-kpis').hidden = true;
    document.getElementById('latency-legend').textContent = '';
    emptyState(host, 'latency measurements');
    return;
  }
  document.getElementById('latency-kpis').hidden = false;
  const series = latencySeries();
  renderLatencyLegend(series);
  latencyStat(series);
  kpiRow(document.getElementById('latency-kpis'), series, {
    fmt: latencyFmt(),
    better: metricDef().unit === 'qps' ? 'up' : 'down',
    fallback: metricDef().label,
  });
  lineChart(host, { series, fmt: latencyFmt(), zeroBased: false });
}

/* ---- tables -------------------------------------------------------------- */

function renderModelsTable(host) {
  const d = state.data;
  const cols = [
    ['Run', (m, i) => `${fmtDay(d.runs[i].day)} ${hhmm(d.runs[i].iso)}`],
    ['XLA', () => null],  // filled as a link below
    ['Modules', (m, i) => fmtInt(d.series[m].n[i])],
    ['Index', (m, i) => fmtIndex(d.series[m].index[i])],
    ['Total', (m, i) => fmtMs(d.series[m].total_ms[i])],
    ...d.groups.map((g) => [g.label, (m, i) => fmtMs(d.series[m][`group_${g.key}`]?.[i])]),
  ];

  const table = el('table');
  const thead = el('thead');
  thead.append(el('tr', {}, cols.map(([h]) => el('th', { text: h, scope: 'col' }))));
  table.append(thead);

  const tbody = el('tbody');
  for (const m of activeModes()) {
    tbody.append(el('tr', {}, [
      el('td', { class: 'lane-head', colspan: cols.length, text: m.label }),
    ]));
    d.runs.forEach((r, i) => {
      if (d.series[m.key].n[i] == null) return;
      const cls = d.series[m.key].complete[i] ? '' : 'partial';
      tbody.append(el('tr', {}, cols.map(([h, f], ci) => el('td', { class: cls }, ci === 1
        ? [link(r.sha ? `https://github.com/${d.repository}/commit/${r.sha}` : null,
                r.commit || '--', 'The upstream commit this run benchmarked', 'sha')]
        : [el('span', { text: f(m.key, i) })]))));
    });
  }
  table.append(tbody);
  host.append(table);
  host.append(el('p', {
    class: 'note',
    text: 'Greyed rows are runs that did not measure the whole suite; those report no '
      + 'total, because a partial sum would read as a speed-up.',
  }));
}

function renderModulesTable(host) {
  const d = state.data;
  const mod = moduleById(state.module);
  if (!mod) { emptyState(host, 'per-module series'); return; }
  const modes = activeModes();

  const table = el('table');
  const thead = el('thead');
  thead.append(el('tr', {}, [el('th', { text: 'Run', scope: 'col' }),
                             el('th', { text: 'XLA', scope: 'col' })].concat(
    modes.map((m) => el('th', { text: m.label, scope: 'col' })))));
  table.append(thead);

  const tbody = el('tbody');
  d.runs.forEach((r, i) => {
    tbody.append(el('tr', {}, [
      el('td', { text: `${fmtDay(r.day)} ${hhmm(r.iso)}` }),
      el('td', {}, [link(r.sha ? `https://github.com/${d.repository}/commit/${r.sha}` : null,
                         r.commit || '--', null, 'sha')]),
    ].concat(modes.map((m) =>
      el('td', { text: fmtMs(d.module_values[m.key]?.[state.module]?.[i]) })))));
  });
  table.append(tbody);
  host.append(el('p', { class: 'note', text: `${mod.model} · ${mod.leaf} · ${mod.module || 'single module'}` }));
  host.append(table);
}

function renderLatencyTable(host) {
  const d = state.data;
  const series = latencySeries();
  if (!series.length) { emptyState(host, 'latency measurements'); return; }
  const metrics = d.latency.metrics;

  // Every metric for every series, so the table is not scoped by the chip the
  // chart is scoped by: the p99 beside the p50 is the point of reading it.
  const table = el('table');
  const thead = el('thead');
  thead.append(el('tr', {}, [el('th', {})].concat(
    series.map((s) => el('th', { text: s.label, scope: 'col', colspan: metrics.length })))));
  thead.append(el('tr', {}, [el('th', { text: 'Run', scope: 'col' })].concat(
    series.flatMap(() => metrics.map((m) => el('th', { text: m.label, scope: 'col' }))))));
  table.append(thead);

  const tbody = el('tbody');
  d.runs.forEach((r, i) => {
    const row = series.flatMap((s) => metrics.map((m) => {
      const v = d.latency.series[s.key][m.key][i];
      return el('td', { text: m.unit === 'qps' ? fmtQps(v) : fmtUs(v) });
    }));
    if (row.every((td) => td.textContent === '--')) return;
    tbody.append(el('tr', {}, [el('td', { text: `${fmtDay(r.day)} ${hhmm(r.iso)}` })].concat(row)));
  });
  table.append(tbody);
  host.append(table);
}

/** The table the open tab's charts are made of. */
function renderTable() {
  const host = document.getElementById('table-host');
  host.textContent = '';
  if (state.tab === 'modules') renderModulesTable(host);
  else if (state.tab === 'latency') renderLatencyTable(host);
  else renderModelsTable(host);
}

/* ---- controls ------------------------------------------------------------ */

function render() {
  // Only the open tab is drawn: a chart measured while its panel is hidden
  // reports zero width and lays itself out against that.
  if (state.tab === 'modules') renderModules();
  else if (state.tab === 'latency') renderLatency();
  else renderModels();
  if (state.table) renderTable();
}

function buildModeChips() {
  const host = document.getElementById('mode-chips');
  host.textContent = '';
  state.data.modes.forEach((m, i) => {
    const on = state.modes.has(m.key);
    const hip = state.data.hip?.[m.key]?.filter(Boolean).pop();
    const b = el('button', {
      type: 'button', class: `chip${on ? ' on' : ''}`, 'aria-pressed': String(on),
      // The only place the page can say which ROCm a stack actually was: the
      // latency tool records it, and nothing else in the run does.
      title: hip ? `HIP ${hip}` : null,
    }, [swatch(modeColor(i)), el('span', { text: m.label })]);
    b.addEventListener('click', () => {
      if (state.modes.has(m.key)) {
        // Never leave every chart empty.
        if (state.modes.size === 1) return;
        state.modes.delete(m.key);
      } else {
        state.modes.add(m.key);
      }
      buildModeChips();
      render();
    });
    host.append(b);
  });
}

function buildMetricChips() {
  const host = document.getElementById('metric-chips');
  host.textContent = '';
  for (const m of state.data.latency?.metrics ?? []) {
    const on = state.metric === m.key;
    const b = el('button', {
      type: 'button', class: `chip${on ? ' on' : ''}`, 'aria-pressed': String(on),
      text: m.label,
    });
    b.addEventListener('click', () => {
      state.metric = m.key;
      buildMetricChips();
      render();
    });
    host.append(b);
  }
}

/** Show one tab's panel, and only the filters that scope it. */
function showTab(tab) {
  state.tab = tab;
  for (const b of document.querySelectorAll('.tab')) {
    const on = b.dataset.tab === tab;
    b.classList.toggle('on', on);
    b.setAttribute('aria-selected', String(on));
    document.getElementById(`view-${b.dataset.tab}`).hidden = !on;
  }
  // The metric chips pick between the latency tool's own figures; on the other
  // tabs there is one figure per chart and they would scope nothing.
  document.getElementById('metric-filter').hidden = tab !== 'latency';
  render();
}

function wireControls() {
  const tabs = [...document.querySelectorAll('.tab')];
  for (const b of tabs) {
    b.addEventListener('click', () => showTab(b.dataset.tab));
    // A tablist is driven by the arrow keys, not by Tab.
    b.addEventListener('keydown', (e) => {
      const step = { ArrowRight: 1, ArrowLeft: -1 }[e.key];
      if (!step) return;
      e.preventDefault();
      const next = tabs[(tabs.indexOf(b) + step + tabs.length) % tabs.length];
      next.focus();
      showTab(next.dataset.tab);
    });
  }

  const tb = document.getElementById('table-toggle');
  tb.addEventListener('click', () => {
    state.table = !state.table;
    tb.classList.toggle('on', state.table);
    tb.setAttribute('aria-pressed', String(state.table));
    tb.textContent = state.table ? 'Hide table' : 'Show table';
    document.getElementById('table-card').hidden = !state.table;
    if (state.table) renderTable();
  });

  const theme = document.getElementById('theme');
  theme.addEventListener('click', () => {
    const cur = document.documentElement.getAttribute('data-theme');
    const sysDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const next = cur ? (cur === 'dark' ? 'light' : 'dark') : (sysDark ? 'light' : 'dark');
    document.documentElement.setAttribute('data-theme', next);
    try { localStorage.setItem('ci-metrics-theme', next); } catch (_) { /* private mode */ }
    render();
  });

  let t;
  window.addEventListener('resize', () => {
    clearTimeout(t);
    t = setTimeout(render, 150);
  });
}

/* ---- boot ---------------------------------------------------------------- */

/** The module the Modules tab opens on: the largest mover of the latest run. */
function openingModule(data) {
  for (let i = data.runs.length - 1; i >= 0; i--) {
    for (const m of data.modes) {
      const row = data.movers?.[m.key]?.[i];
      if (row && row.modules.length) return row.modules[0].key;
    }
  }
  return data.modules?.[0]?.key ?? null;
}

(async function main() {
  try {
    // The same key as the CI dashboard: one site, one theme choice.
    const saved = localStorage.getItem('ci-metrics-theme');
    if (saved) document.documentElement.setAttribute('data-theme', saved);
  } catch (_) { /* private mode */ }

  let data;
  try {
    const res = await fetch('trends.json', { cache: 'no-cache' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    data = await res.json();
  } catch (e) {
    document.getElementById('subtitle').textContent = `Could not load trends.json: ${e.message}`;
    return;
  }

  state.data = data;
  for (const m of data.modes) state.modes.add(m.key);
  state.module = openingModule(data);

  const plural = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`;
  const first = fmtDay(data.runs[0].day);
  const last = fmtDay(data.runs[data.runs.length - 1].day);
  const models = new Set((data.modules ?? []).map((m) => m.model)).size;
  document.getElementById('subtitle').textContent =
    `${first === last ? first : `${first} – ${last}`} · ${plural(data.runs.length, 'run')} · `
    + `${plural((data.modules ?? []).length, 'module')} from ${plural(models, 'model')} · `
    + `${plural(data.modes.length, 'ROCm stack')}`;
  document.getElementById('provenance').textContent =
    `Generated ${data.generated}. Times are UTC.`;

  buildModeChips();
  buildMetricChips();
  buildModulePicker();
  wireControls();
  showTab(state.tab);
})();
