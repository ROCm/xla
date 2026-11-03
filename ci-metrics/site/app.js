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

function swatch(color) {
  return el('span', { class: 'swatch', style: `--c:${color}` });
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
 * `series` is [{key, label, color, values[]}]; nulls break the line.
 */
function lineChart(host, { series, fmt, yMax, tickKind = 'num', directLabels = true }) {
  const days = state.data.days;
  const complete = state.data.complete;
  const counts = state.data.series;

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
      const thin = (counts[s.key].n[i] ?? 0) < LOW_N || !complete[i];
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
      const n = counts[s.key].n[i];
      return el('div', { class: 'tip-row' }, [
        swatch(s.color),
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
function lastSolid(lane, field) {
  const d = state.data;
  const vals = d.series[lane][field];
  const ns = d.series[lane].n;
  for (let i = vals.length - 1; i >= 0; i--) {
    if (vals[i] != null && d.complete[i] && (ns[i] ?? 0) >= LOW_N) return i;
  }
  for (let i = vals.length - 1; i >= 0; i--) if (vals[i] != null) return i;
  return -1;
}

function firstSolid(lane, field) {
  const d = state.data;
  const vals = d.series[lane][field];
  const ns = d.series[lane].n;
  for (let i = 0; i < vals.length; i++) {
    if (vals[i] != null && d.complete[i] && (ns[i] ?? 0) >= LOW_N) return i;
  }
  return -1;
}

function renderKpis() {
  const host = document.getElementById('kpis');
  host.textContent = '';
  const field = `ttl_${state.pct}`;
  for (const l of activeLanes()) {
    const last = lastSolid(l.key, field);
    const first = firstSolid(l.key, field);
    const cur = last >= 0 ? state.data.series[l.key][field][last] : null;
    const was = first >= 0 && first !== last ? state.data.series[l.key][field][first] : null;

    let delta = null;
    if (cur != null && was != null && was > 0) {
      const change = (cur - was) / was;
      // Shorter is better, so a fall is the good direction.
      const dir = change < 0 ? 'down' : 'up';
      const arrow = change < 0 ? '▼' : '▲';
      delta = el('div', { class: 'kpi-delta' }, [
        el('span', { class: dir, text: `${arrow} ${Math.abs(change * 100).toFixed(0)}%` }),
        el('span', { text: ` vs ${fmtDay(state.data.days[first])}` }),
      ]);
    }

    host.append(el('div', { class: 'kpi' }, [
      el('div', { class: 'kpi-label' }, [swatch(l.color), el('span', { text: l.label })]),
      el('div', { class: 'kpi-value', text: fmtDur(cur) }),
      delta || el('div', { class: 'kpi-delta', text: `${state.pct} time to signal` }),
    ]));
  }
}

function renderTable() {
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

function render() {
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

function wireControls() {
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

  buildLaneChips();
  renderPhaseLegend();
  wireControls();
  render();
})();
