/* ROCm CI metrics trends.
 *
 * Plain SVG built in the DOM: no chart library, no CDN, no build step -- the
 * page is three static files next to a trends.json, which is what the Pages
 * job uploads. The metric definitions all live in ci-metrics/*.py; nothing is
 * recomputed here beyond percentages and axis scaling.
 */
'use strict';

const SVG_NS = 'http://www.w3.org/2000/svg';

/* A day with few commits still gets a percentile, but it is not worth reading:
 * weekends drop to a handful of commits, where a p90 is one slow run. Those
 * points are drawn hollow and called out in the tooltip rather than hidden,
 * because the low volume is itself real. */
const LOW_N = 20;

const state = {
  data: null,
  tab: 'timing',
  lanes: new Set(),
  pct: 'p50',
  table: false,
};

/* ---- formatting ---------------------------------------------------------- */

const fmtDur = (s) => {
  if (s == null) return '--';
  if (s < 60) return `${Math.round(s)}s`;
  const m = s / 60;
  return m < 10 ? `${m.toFixed(1)}m` : `${Math.round(m)}m`;
};
const fmtInt = (v) => (v == null ? '--' : Math.round(v).toLocaleString());
const fmtPct = (v) => (v == null ? '--' : `${Math.round(v * 100)}%`);

const fmtDay = (iso) => {
  const [, m, d] = iso.split('-');
  return `${['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][+m]} ${+d}`;
};

const laneColor = (i) => `var(--series-${i + 1})`;

/* ---- tiny DOM helpers ---------------------------------------------------- */

function el(tag, attrs = {}, children = []) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    // An absent attribute is absent, not the string "undefined".
    if (v === null || v === undefined) continue;
    if (k === 'class') n.className = v;
    else if (k === 'text') n.textContent = v;
    else if (k === 'html') n.innerHTML = v;
    else if (v !== null && v !== undefined) n.setAttribute(k, v);
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

/* ---- scales -------------------------------------------------------------- */

/* Durations need their own tick ladder: decimal steps land on 1000s, which
 * reads as "16.7m" on the axis. These are the steps that are round in minutes
 * and hours. */
const NICE_SECONDS = [5, 10, 15, 20, 30, 60, 120, 180, 300, 600, 900, 1200,
                      1800, 2700, 3600, 5400, 7200, 10800, 14400];

/** Round a maximum up to a readable tick step, and return the tick values. */
function ticks(max, count = 4, kind = 'num') {
  if (!(max > 0)) return { max: 1, values: [0, 1] };
  let step;
  if (kind === 'dur') {
    step = NICE_SECONDS.find((s) => max / s <= count) ?? Math.ceil(max / count / 3600) * 3600;
  } else {
    const raw = max / count;
    const mag = 10 ** Math.floor(Math.log10(raw));
    step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? 10 * mag;
  }
  const top = Math.ceil(max / step) * step;
  const values = [];
  for (let v = 0; v <= top + step / 2; v += step) values.push(+v.toFixed(6));
  return { max: top, values };
}

/** Which day indices get an x-axis label, without colliding with the last. */
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
 * A multi-series line chart over the shared day axis.
 * `series` is [{key, label, color, values[], n?, dash?}]; nulls break the line.
 * `n` is the sample behind each point -- the lane's commits unless the series
 * carries its own -- and decides which markers are drawn hollow.
 */
function lineChart(host, { series, fmt, yMax, tickKind = 'num', directLabels = true }) {
  const days = state.data.days;
  const complete = state.data.complete;
  const sampleOf = (s, i) => (s.n ? s.n[i] : state.data.series[s.key].n[i]);

  host.textContent = '';
  const width = Math.max(320, host.clientWidth);
  const M = { top: 14, right: directLabels ? 88 : 16, bottom: 30, left: 54 };
  const H = 252;
  const plotW = width - M.left - M.right;
  const plotH = H - M.top - M.bottom;

  const flat = series.flatMap((s) => s.values).filter((v) => v != null);
  const scale = yMax != null
    ? { max: yMax, values: ticks(yMax, 4, tickKind).values.filter((v) => v <= yMax) }
    : ticks(Math.max(...flat, 0), 4, tickKind);
  const x = (i) => (days.length === 1 ? plotW / 2 : (i / (days.length - 1)) * plotW);
  const y = (v) => plotH - (v / scale.max) * plotH;

  const root = svg('svg', { viewBox: `0 0 ${width} ${H}`, height: H, role: 'img' });
  const g = svg('g', { transform: `translate(${M.left},${M.top})` });
  root.append(g);

  // Gridlines: solid hairlines, one step off the surface.
  for (const v of scale.values) {
    g.append(svg('line', {
      x1: 0, x2: plotW, y1: y(v), y2: y(v),
      stroke: 'var(--grid)', 'stroke-width': 1,
    }));
    const t = svg('text', {
      x: -10, y: y(v) + 4, 'text-anchor': 'end',
      fill: 'var(--text-muted)', 'font-size': 11, 'font-variant-numeric': 'tabular-nums',
    });
    t.textContent = fmt(v);
    g.append(t);
  }

  // X axis: label a subset so ticks never collide.
  for (const i of labelIndices(days.length, x, 58)) {
    const t = svg('text', {
      x: x(i), y: plotH + 20, 'text-anchor': 'middle',
      fill: 'var(--text-muted)', 'font-size': 11,
    });
    t.textContent = fmtDay(days[i]);
    g.append(t);
  }
  g.append(svg('line', {
    x1: 0, x2: plotW, y1: plotH, y2: plotH, stroke: 'var(--axis)', 'stroke-width': 1,
  }));

  // Lines, broken across null days.
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

  // Markers. Hollow where the day is provisional or the sample is thin; the
  // 2px surface ring keeps them legible where lines cross.
  for (const s of series) {
    s.values.forEach((v, i) => {
      if (v == null) return;
      const thin = (sampleOf(s, i) ?? 0) < LOW_N || !complete[i];
      if (!thin && days.length > 20) return;
      g.append(svg('circle', {
        cx: x(i), cy: y(v), r: 4,
        fill: thin ? 'var(--surface-1)' : s.color,
        stroke: thin ? s.color : 'var(--surface-1)', 'stroke-width': 2,
      }));
    });
  }

  // Direct labels at the line ends, skipped where they would collide (the
  // legend already carries identity; stacked labels would detach from lines).
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
      g.append(svg('circle', {
        cx: x(e.i), cy: yy, r: 4, fill: e.s.color,
        stroke: 'var(--surface-1)', 'stroke-width': 2,
      }));
    }
  }

  // Crosshair + tooltip layer.
  const rule = svg('line', {
    y1: 0, y2: plotH, stroke: 'var(--axis)', 'stroke-width': 1, opacity: 0,
  });
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
      const v = s.values[i];
      if (v == null) continue;
      dots.append(svg('circle', {
        cx: x(i), cy: y(v), r: 4.5, fill: s.color,
        stroke: 'var(--surface-1)', 'stroke-width': 2,
      }));
    }
    dots.setAttribute('opacity', 1);

    const rows = series.map((s) => {
      const n = sampleOf(s, i);
      return el('div', { class: 'tip-row' }, [
        swatch(s.color, s.dash),
        el('span', { text: s.label }),
        el('span', { class: 'v', text: fmt(s.values[i]) }),
        el('span', { class: 'n', text: n == null ? '' : `n=${n}` }),
      ]);
    });
    const flags = [];
    if (!complete[i]) flags.push('partial day');
    tip.textContent = '';
    tip.append(el('div', { class: 'tip-head' }, [
      el('span', { text: fmtDay(days[i]) }),
      flags.length ? el('span', { class: 'tip-flag', text: flags.join(' · ') }) : null,
    ]));
    for (const r of rows) tip.append(r);
    tip.classList.add('on');

    const tw = tip.offsetWidth || 200;
    const left = Math.min(Math.max(M.left + x(i) - tw / 2, 4), width - tw - 4);
    tip.style.left = `${left}px`;
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
    return Math.max(0, Math.min(days.length - 1, Math.round((px / plotW) * (days.length - 1))));
  };

  // Hit surface spans the whole plot, so there is nothing to aim at.
  const hit = svg('rect', {
    x: 0, y: 0, width: plotW, height: plotH, fill: 'transparent', style: 'cursor:crosshair',
  });
  g.append(hit);
  root.addEventListener('mousemove', (e) => show(nearest(e.clientX)));
  root.addEventListener('mouseleave', hide);

  // Keyboard reaches the same values as hover.
  let focus = days.length - 1;
  root.setAttribute('tabindex', '0');
  root.addEventListener('focus', () => show(focus));
  root.addEventListener('blur', hide);
  root.addEventListener('keydown', (e) => {
    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
    e.preventDefault();
    focus = Math.max(0, Math.min(days.length - 1, focus + (e.key === 'ArrowRight' ? 1 : -1)));
    show(focus);
  });
}

/* ---- stacked area (small multiples) -------------------------------------- */

/** Total mean seconds per day for one lane; also the small-multiple scale. */
function laneTotals(lane) {
  const d = state.data;
  const row = d.series[lane];
  return d.days.map((_, i) =>
    d.phases.reduce((a, p) => a + (row[`phase_${p.key}`][i] ?? 0), 0));
}

function stackPanel(host, lane, sharedMax) {
  const d = state.data;
  const days = d.days;
  const phases = d.phases;
  const row = d.series[lane.key];

  const width = Math.max(260, host.clientWidth);
  const M = { top: 12, right: 12, bottom: 26, left: 46 };
  const H = 190;
  const plotW = width - M.left - M.right;
  const plotH = H - M.top - M.bottom;

  // Every panel shares one y scale -- panels on their own scales would make
  // four lanes with very different totals look identical.
  const totals = laneTotals(lane.key);
  const scale = ticks(sharedMax, 3, 'dur');
  const x = (i) => (days.length === 1 ? plotW / 2 : (i / (days.length - 1)) * plotW);
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
    t.textContent = fmtDur(v);
    g.append(t);
  }

  // Bands, bottom-up. Gaps in the data split a band into separate runs so a
  // missing day is never bridged with a straight line.
  const base = days.map(() => 0);
  phases.forEach((p, pi) => {
    const vals = row[`phase_${p.key}`];
    let run = [];
    const flush = () => {
      if (run.length > 1) {
        const up = run.map((r) => `${r.x.toFixed(1)},${y(r.top).toFixed(1)}`);
        const down = run.slice().reverse().map((r) => `${r.x.toFixed(1)},${y(r.bot).toFixed(1)}`);
        g.append(svg('path', {
          d: `M${up.join('L')}L${down.join('L')}Z`,
          fill: laneColor(pi), 'fill-opacity': 0.85,
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

  for (const i of labelIndices(days.length, x, 54)) {
    const t = svg('text', {
      x: x(i), y: plotH + 18, 'text-anchor': 'middle',
      fill: 'var(--text-muted)', 'font-size': 10.5,
    });
    t.textContent = fmtDay(days[i]);
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
      el('span', { text: fmtDay(days[i]) }),
      el('span', { class: 'tip-flag', text: `total ${fmtDur(totals[i])}` }),
    ]));
    // Largest phase first: the reader wants the dominant one, not the order.
    phases
      .map((p, pi) => ({ p, pi, v: row[`phase_${p.key}`][i] }))
      .filter((r) => r.v != null && r.v > 0)
      .sort((a, b) => b.v - a.v)
      .forEach((r) => {
        tip.append(el('div', { class: 'tip-row' }, [
          swatch(laneColor(r.pi)),
          el('span', { text: r.p.label }),
          el('span', { class: 'v', text: fmtDur(r.v) }),
        ]));
      });
    tip.classList.add('on');
    const tw = tip.offsetWidth || 200;
    tip.style.left = `${Math.min(Math.max(M.left + x(i) - tw / 2, 4), width - tw - 4)}px`;
    tip.style.top = `${M.top + 4}px`;
  };

  const hit = svg('rect', { x: 0, y: 0, width: plotW, height: plotH, fill: 'transparent' });
  g.append(hit);
  root.addEventListener('mousemove', (e) => {
    const box = root.getBoundingClientRect();
    const px = ((e.clientX - box.left) / box.width) * width - M.left;
    show(Math.max(0, Math.min(days.length - 1, Math.round((px / plotW) * (days.length - 1)))));
  });
  root.addEventListener('mouseleave', () => {
    rule.setAttribute('opacity', 0);
    tip.classList.remove('on');
  });
}

/* ---- pieces -------------------------------------------------------------- */

function activeLanes() {
  return state.data.lanes
    .map((l, i) => ({ ...l, color: laneColor(i) }))
    .filter((l) => state.lanes.has(l.key));
}

function seriesFor(field) {
  return activeLanes().map((l) => ({
    key: l.key,
    label: l.label,
    color: l.color,
    values: state.data.series[l.key][field],
  }));
}

/** The last day with a usable reading, preferring complete days. */
function lastSolid(vals, ns) {
  const d = state.data;
  for (let i = vals.length - 1; i >= 0; i--) {
    if (vals[i] != null && d.complete[i] && (ns[i] ?? 0) >= LOW_N) return i;
  }
  for (let i = vals.length - 1; i >= 0; i--) if (vals[i] != null) return i;
  return -1;
}

function firstSolid(vals, ns) {
  const d = state.data;
  for (let i = 0; i < vals.length; i++) {
    if (vals[i] != null && d.complete[i] && (ns[i] ?? 0) >= LOW_N) return i;
  }
  return -1;
}

/**
 * One tile per series: where it stands now and how far that is from where the
 * window started. `better` names the direction that is good news, so a rising
 * suite reads green where a rising wait reads red.
 */
function kpiRow(host, series, { fmt, better, fallback }) {
  host.textContent = '';
  for (const s of series) {
    const last = lastSolid(s.values, s.ns);
    const first = firstSolid(s.values, s.ns);
    const cur = last >= 0 ? s.values[last] : null;
    const was = first >= 0 && first !== last ? s.values[first] : null;

    let delta = null;
    if (cur != null && was != null && was > 0) {
      const change = (cur - was) / was;
      const rising = change > 0;
      delta = el('div', { class: 'kpi-delta' }, [
        el('span', {
          class: rising === (better === 'up') ? 'good' : 'bad',
          text: `${rising ? '▲' : '▼'} ${Math.abs(change * 100).toFixed(0)}%`,
        }),
        el('span', { text: ` vs ${fmtDay(state.data.days[first])}` }),
      ]);
    }

    host.append(el('div', { class: 'kpi' }, [
      el('div', { class: 'kpi-label' }, [swatch(s.color, s.dash), el('span', { text: s.label })]),
      el('div', { class: 'kpi-value', text: fmt(cur) }),
      delta || el('div', { class: 'kpi-delta', text: fallback }),
    ]));
  }
}

function renderKpis() {
  const field = `ttl_${state.pct}`;
  kpiRow(document.getElementById('kpis'), activeLanes().map((l) => ({
    label: l.label,
    color: l.color,
    values: state.data.series[l.key][field],
    ns: state.data.series[l.key].n,
  })), { fmt: fmtDur, better: 'down', fallback: `${state.pct} time to signal` });
}

/* ---- tests --------------------------------------------------------------- */

/**
 * One series per test suite: a lane, or a job family inside a lane where the
 * repo runs more than one suite in a workflow. Colour stays with the lane --
 * the same hue it has on the timing tab -- and the family is the dash, so a
 * filter that drops a lane never repaints the rest.
 */
function testSeries() {
  const d = state.data;
  const labels = Object.fromEntries(d.lanes.map((l, i) => [l.key, [l.label, laneColor(i)]]));
  return d.test_cats
    .filter((c) => state.lanes.has(c.lane))
    .map((c) => {
      const family = d.test_cats.filter((x) => x.lane === c.lane);
      const [laneLabel, color] = labels[c.lane];
      const nth = family.indexOf(c);
      return {
        key: c.key,
        label: family.length > 1 ? `${laneLabel} · ${c.label}` : laneLabel,
        color,
        dash: nth > 0 ? '6 4' : null,
        values: d.tests[c.key].tests_max,
        median: d.tests[c.key].tests_p50,
        n: d.tests[c.key].n,
      };
    });
}

function renderTestLegend(series) {
  const host = document.getElementById('test-legend');
  host.textContent = '';
  for (const s of series) {
    host.append(el('span', { class: 'legend-item' }, [
      swatch(s.color, s.dash), el('span', { text: s.label }),
    ]));
  }
}

/** The headline the tests card leads with: the widest ROCm/NV gap today. */
function testsStat(series) {
  const host = document.getElementById('tests-stat');
  host.textContent = '';
  const now = (s) => {
    const i = lastSolid(s.values, s.n);
    return i < 0 ? null : s.values[i];
  };
  // Pair each ROCm suite with the NV suite of the same repo and family.
  const pairs = series
    .filter((s) => s.key.includes('_rocm'))
    .map((s) => [s, series.find((o) => o.key === s.key.replace('_rocm', '_nv'))])
    .filter(([a, b]) => b && now(a) != null && now(b) != null && now(b) > 0)
    .sort((x, y) => now(x[0]) / now(x[1]) - now(y[0]) / now(y[1]));
  if (!pairs.length) return;
  const [rocm, nv] = pairs[0];
  host.append(el('span', {
    html: `<span class="hl">${rocm.label}</span> runs <span class="hl">${fmtInt(now(rocm))}</span> `
      + `tests to the matching NV lane's <span class="hl">${fmtInt(now(nv))}</span> `
      + `(${Math.round((now(rocm) / now(nv)) * 100)}%).`,
  }));
}

/** A link, or plain text when there is nothing to point at. */
function link(href, text, title, cls) {
  return href ? el('a', { href, text, title, class: cls, target: '_blank', rel: 'noopener' })
              : el('span', { text, title, class: cls });
}

/** ROCm or NV: the side of a repo's presubmit a lane key names. */
const sideOf = (key) => (key.endsWith('_rocm') ? 'ROCm' : 'NV');

/**
 * Where a tail commit sits in the other lane of the same repo.
 *
 * Both lanes run on the same head commits, so this separates a heavy PR --
 * slow on ROCm and on NV alike -- from a lane that is slow on its own. The
 * second is the row to chase, and the only one that gets a mark. It is
 * computed against the whole other lane, so a filter that hides that lane does
 * not change what this says.
 */
function agreement(also, lane, other) {
  if (!other) return el('span', { class: 'muted', text: '--' });
  if (also === 'both') {
    return el('span', { class: 'muted', text: `shared with ${sideOf(other)}` });
  }
  if (also === 'only') {
    return el('span', {
      class: 'diff', text: `${sideOf(lane)} only`,
      title: `This commit is in the ${sideOf(lane)} tail and not in the ${sideOf(other)} one`,
    });
  }
  return el('span', { class: 'muted', text: `no ${sideOf(other)} signal` });
}

/**
 * The commits that put each day's p90 where it is, newest day first.
 *
 * Grouped by day rather than by lane: the question this answers is "what
 * happened on the day the tail moved", and the chart above it is already
 * per-lane.
 */
function renderTail() {
  const host = document.getElementById('tail-host');
  host.textContent = '';
  const d = state.data;
  const lanes = activeLanes();
  const phaseLabel = Object.fromEntries(d.phases.map((p, i) => [p.key, [p.label, laneColor(i)]]));

  // `lead` marks the columns that read as text rather than as figures, so the
  // numbers stay right-aligned under each other and the names stay left.
  const cols = [
    { h: `Lane, and its p${d.tail_pct} that day`, lead: true },
    { h: 'PR', lead: true }, { h: 'Commit', lead: true },
    { h: 'Time to signal' }, { h: 'Other lane', lead: true }, { h: 'Mostly', lead: true },
  ];
  const cell = (i, children) => el('td', { class: cols[i].lead ? 'lead' : null },
                                   typeof children === 'string' ? [el('span', { text: children })] : children);
  const table = el('table');
  const thead = el('thead');
  thead.append(el('tr', {}, cols.map((c) =>
    el('th', { text: c.h, scope: 'col', class: c.lead ? 'lead' : null }))));
  table.append(thead);

  const tbody = el('tbody');
  for (let i = d.days.length - 1; i >= 0; i--) {
    const rows = lanes
      .map((l) => [l, d.tail[l.key][i]])
      .filter(([, t]) => t && t.commits.length);
    if (!rows.length) continue;

    tbody.append(el('tr', {}, [el('td', {
      class: 'lane-head', colspan: cols.length,
    }, [
      el('span', { text: fmtDay(d.days[i]) }),
      d.complete[i] ? null : el('span', { class: 'tip-flag', text: ' partial day' }),
    ])]));

    for (const [l, t] of rows) {
      t.commits.forEach((c, k) => {
        const [phase, phaseColor] = phaseLabel[c.phase] || ['--', 'var(--text-muted)'];
        const base = `https://github.com/${l.repo}`;
        tbody.append(el('tr', {}, [
          // The lane is named once per group of its own commits, and carries
          // the cut those commits are above: a per-day fact, not a per-row one.
          cell(0, k ? [] : [
            el('div', {}, [swatch(l.color), el('span', { text: l.label })]),
            el('div', {
              class: 'muted',
              text: `p${d.tail_pct} ${fmtDur(t.cut)} · ${t.n} at or above`
                + (t.other ? `, ${t.shared} shared` : ''),
            }),
          ]),
          cell(1, [c.pr
            ? link(`${base}/pull/${c.pr}`, `#${c.pr}`)
            : link(c.run ? `${base}/actions/runs/${c.run}` : null,
                   c.branch || `run ${c.run}`, 'No PR number on this run; opens the run')]),
          cell(2, [link(`${base}/commit/${c.sha}`, c.sha, 'The head commit that got this signal', 'sha')]),
          cell(3, [
            el('span', { text: fmtDur(c.total) }),
            c.rerun ? el('span', { class: 'tip-flag', text: ' re-run' }) : null,
            c.passed ? null : el('span', { class: 'tip-flag', text: ' failed' }),
          ]),
          cell(4, [agreement(c.also, l.key, t.other)]),
          cell(5, [swatch(phaseColor), el('span', { text: `${phase} ${fmtDur(c.phase_s)}` })]),
        ]));
      });
    }
  }
  table.append(tbody);
  host.append(table);
}

/**
 * What a view says when its `trends.json` does not carry it.
 *
 * The page and its data deploy together but are built by different steps, and
 * a trends.json written by an older trends.py simply lacks the newer keys.
 * Saying so beats hiding the view: a missing tab reads as "the change was
 * never deployed", which is indistinguishable from "the data is stale".
 */
function emptyState(host, what) {
  host.textContent = '';
  host.append(el('p', { class: 'note empty' }, [
    el('span', { text: `No ${what} in this trends.json. ` }),
    el('span', {
      text: `It was generated ${state.data.generated} by a trends.py older than this page `
        + '— the dashboard is ahead of its data. Re-running trends.py over the '
        + 'ci-metrics-data snapshots fills this in.',
    }),
  ]));
}

function renderTests() {
  const d = state.data;
  // An empty KPI row would still hold its own gap above the first card.
  document.getElementById('test-kpis').hidden = !(d.tests && d.test_cats);

  if (d.tests && d.test_cats) {
    const series = testSeries();
    renderTestLegend(series);
    testsStat(series);
    kpiRow(document.getElementById('test-kpis'),
           series.map((s) => ({ ...s, ns: s.n })),
           { fmt: fmtInt, better: 'up', fallback: 'tests in the suite' });
    lineChart(document.getElementById('chart-tests'), { series, fmt: fmtInt });
  } else {
    for (const id of ['test-kpis', 'test-legend', 'tests-stat']) {
      document.getElementById(id).textContent = '';
    }
    emptyState(document.getElementById('chart-tests'), 'test counts');
  }

  // The two cards came from different commits, so they are missing
  // independently: an older file can carry the suites and not the tail.
  if (d.tail) renderTail();
  else emptyState(document.getElementById('tail-host'), `p${d.tail_pct || 90} tail`);
}

function renderTestTable() {
  const host = document.getElementById('table-host');
  host.textContent = '';
  const d = state.data;
  if (!d.tests || !d.test_cats) {
    emptyState(host, 'test counts');
    return;
  }
  const series = testSeries();

  // Three columns a suite: the plotted suite, the median job, and the sample.
  // A median far below the suite is the day's truncation rate, not a smaller
  // suite -- jobs that failed early, or never reached the second invocation.
  const sub = ['suite', 'median', 'jobs'];
  const table = el('table');
  const thead = el('thead');
  thead.append(el('tr', {}, [el('th', {})].concat(
    series.map((s) => el('th', { text: s.label, scope: 'col', colspan: sub.length })))));
  thead.append(el('tr', {}, [el('th', { text: 'Day', scope: 'col' })].concat(
    series.flatMap(() => sub.map((h) => el('th', { text: h, scope: 'col' }))))));
  table.append(thead);

  const tbody = el('tbody');
  d.days.forEach((_, i) => {
    if (series.every((s) => s.values[i] == null)) return;
    const cls = d.complete[i] ? '' : 'partial';
    tbody.append(el('tr', {}, [el('td', { class: cls, text: fmtDay(d.days[i]) })].concat(
      series.flatMap((s) => [
        el('td', { class: cls, text: fmtInt(s.values[i]) }),
        el('td', { class: cls, text: fmtInt(s.median[i]) }),
        el('td', { class: cls, text: fmtInt(s.n[i]) }),
      ]))));
  });
  table.append(tbody);
  host.append(table);
  host.append(el('p', {
    class: 'note',
    text: 'Suite is the day\'s largest sampled job and is what the chart plots; '
      + 'median is the middle one, lower wherever jobs ended early. '
      + 'Greyed rows are days the snapshots do not fully cover.',
  }));
}

function renderTimingTable() {
  const host = document.getElementById('table-host');
  host.textContent = '';
  const d = state.data;
  const cols = [
    ['Day', (l, i) => fmtDay(d.days[i])],
    ['Commits', (l, i) => fmtInt(d.series[l].n[i])],
    ['Signal p50', (l, i) => fmtDur(d.series[l].ttl_p50[i])],
    ['Signal p90', (l, i) => fmtDur(d.series[l].ttl_p90[i])],
    ['Queue p50', (l, i) => fmtDur(d.series[l].queue_p50[i])],
    ['Queue p90', (l, i) => fmtDur(d.series[l].queue_p90[i])],
    ['Passed', (l, i) => fmtPct(d.series[l].pass_rate[i])],
    ...d.phases.map((p) => [p.label, (l, i) => fmtDur(d.series[l][`phase_${p.key}`][i])]),
  ];

  const table = el('table');
  const thead = el('thead');
  thead.append(el('tr', {}, cols.map(([h]) => el('th', { text: h, scope: 'col' }))));
  table.append(thead);

  const tbody = el('tbody');
  for (const l of activeLanes()) {
    tbody.append(el('tr', {}, [
      el('td', { class: 'lane-head', colspan: cols.length, text: l.label }),
    ]));
    d.days.forEach((_, i) => {
      if (d.series[l.key].n[i] == null) return;
      tbody.append(el('tr', {}, cols.map(([, f]) =>
        el('td', { class: d.complete[i] ? '' : 'partial', text: f(l.key, i) }))));
    });
  }
  table.append(tbody);
  host.append(table);
  host.append(el('p', {
    class: 'note',
    text: 'Greyed rows are days the snapshots do not fully cover.',
  }));
}

function renderPhaseLegend() {
  const host = document.getElementById('phase-legend');
  host.textContent = '';
  state.data.phases.forEach((p, i) => {
    host.append(el('span', { class: 'legend-item' }, [
      swatch(laneColor(i)), el('span', { text: p.label }),
    ]));
  });
}

function renderPhases() {
  const host = document.getElementById('chart-phases');
  host.textContent = '';
  const lanes = activeLanes();
  // One scale for every panel, so the panels can be read against each other.
  const sharedMax = Math.max(
    ...lanes.flatMap((l) => laneTotals(l.key)).filter((v) => v > 0), 1);

  // Build every panel before drawing any: a panel measured while it is the
  // grid's only child reports the full width, and the grid then reflows to two
  // columns under it, leaving that panel's viewBox scaled down.
  const plots = lanes.map((l) => {
    const panel = el('div', {});
    panel.append(el('p', { class: 'panel-title', text: l.label }));
    const plot = el('div', { class: 'plot' });
    panel.append(plot);
    host.append(panel);
    return plot;
  });
  lanes.forEach((l, i) => stackPanel(plots[i], l, sharedMax));
}

/** The headline the time-to-signal card leads with. */
function ttlStat() {
  const host = document.getElementById('ttl-stat');
  const lane = state.data.lanes.find((l) => state.lanes.has(l.key));
  if (!lane) { host.textContent = ''; return; }
  const field = `ttl_${state.pct}`;
  const a = firstSolid(lane.key, field);
  const b = lastSolid(lane.key, field);
  if (a < 0 || b < 0 || a === b) { host.textContent = ''; return; }
  const va = state.data.series[lane.key][field][a];
  const vb = state.data.series[lane.key][field][b];
  const word = vb < va ? 'down' : 'up';
  host.textContent = '';
  host.append(el('span', {
    html: `<span class="hl">${lane.label}</span> is ${word} from `
      + `<span class="hl">${fmtDur(va)}</span> to <span class="hl">${fmtDur(vb)}</span> `
      + `(${state.pct}) between ${fmtDay(state.data.days[a])} and ${fmtDay(state.data.days[b])}.`,
  }));
}

/** The table the open tab's charts are made of. */
function renderTable() {
  if (state.tab === 'tests') renderTestTable();
  else renderTimingTable();
}

function renderTiming() {
  renderKpis();
  ttlStat();
  lineChart(document.getElementById('chart-ttl'), {
    series: seriesFor(`ttl_${state.pct}`), fmt: fmtDur, tickKind: 'dur',
  });
  lineChart(document.getElementById('chart-queue'), {
    series: seriesFor(`queue_${state.pct}`), fmt: fmtDur, tickKind: 'dur',
  });
  renderPhases();
  lineChart(document.getElementById('chart-volume'), {
    series: seriesFor('n'), fmt: fmtInt,
  });
  lineChart(document.getElementById('chart-pass'), {
    series: seriesFor('pass_rate'), fmt: fmtPct, yMax: 1,
  });
}

function render() {
  // Only the open tab is drawn: a chart measured while its panel is hidden
  // reports zero width and lays itself out against that.
  if (state.tab === 'tests') renderTests();
  else renderTiming();
  if (state.table) renderTable();
}

/* ---- controls ------------------------------------------------------------ */

function buildLaneChips() {
  const host = document.getElementById('lane-chips');
  host.textContent = '';
  state.data.lanes.forEach((l, i) => {
    const on = state.lanes.has(l.key);
    const b = el('button', {
      type: 'button', class: `chip${on ? ' on' : ''}`, 'aria-pressed': String(on),
    }, [swatch(laneColor(i)), el('span', { text: l.label })]);
    b.addEventListener('click', () => {
      if (state.lanes.has(l.key)) {
        // Never leave every chart empty.
        if (state.lanes.size === 1) return;
        state.lanes.delete(l.key);
      } else {
        state.lanes.add(l.key);
      }
      buildLaneChips();
      render();
    });
    host.append(b);
  });
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
  // The percentile is a property of a duration distribution; the suite size is
  // a median by definition, so the chips would scope nothing on the tests tab.
  document.getElementById('pct-filter').hidden = tab !== 'timing';
  render();
}

function wireControls() {
  const tabs = [...document.querySelectorAll('.tab')].filter((t) => !t.hidden);
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

  for (const b of document.querySelectorAll('#pct-chips .chip')) {
    b.addEventListener('click', () => {
      state.pct = b.dataset.pct;
      for (const o of document.querySelectorAll('#pct-chips .chip')) {
        o.classList.toggle('on', o === b);
        o.setAttribute('aria-pressed', String(o === b));
      }
      render();
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

(async function main() {
  try {
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
  for (const l of data.lanes) state.lanes.add(l.key);

  const first = fmtDay(data.days[0]);
  const last = fmtDay(data.days[data.days.length - 1]);
  const total = Object.values(data.unique_commits).reduce((a, b) => a + b, 0);
  const snaps = `${data.snapshots} nightly snapshot${data.snapshots === 1 ? '' : 's'}`;
  document.getElementById('subtitle').textContent =
    `${first} – ${last} · ${total.toLocaleString()} commits across ${snaps}`;
  document.getElementById('provenance').textContent =
    `Generated ${data.generated}. Days are ${data.timezone}.`;

  // Both tabs are always offered, including against a trends.json written
  // before these series existed: that view then says so (`emptyState`) rather
  // than vanishing, because a tab that is simply absent looks exactly like a
  // deploy that never happened.
  for (const n of document.querySelectorAll('.tail-pct')) n.textContent = `p${data.tail_pct || 90}`;

  buildLaneChips();
  renderPhaseLegend();
  wireControls();
  showTab(state.tab);
})();
