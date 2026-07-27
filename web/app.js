/* Results dashboard: fetch analytics.json and render KPIs + charts (light theme). */
(function () {
  const params = new URLSearchParams(location.search);
  const jobId = params.get('job');

  const COL = {
    red: '#e30613', redSoft: 'rgba(227,6,19,.14)', ink: '#14161a',
    green: '#17b26a', amber: '#f79009', orange: '#ff7a3d', blue: '#1f5fd0',
    grid: 'rgba(20,22,26,.08)', text: '#767d89',
  };
  // Mentor class colours, keyed by class code.
  const CLASS_COLOR = {
    A: '#e30613', C: '#ff7a3d', D: '#b3000f', E: '#1f5fd0',
    G: '#17b26a', V: '#f79009', F: '#8a8f98',
  };
  const codeOf = (label) => (label || '').split('·')[0].trim();
  const LEVELS = ['Free-flow', 'Moderate', 'Heavy', 'Jam'];
  const LEVEL_CLASS = { 'Free-flow': 'free', 'Moderate': 'mod', 'Heavy': 'heavy', 'Jam': 'jam' };

  if (window.Chart) {
    Chart.defaults.color = COL.text;
    Chart.defaults.font.family = 'Segoe UI, Inter, system-ui, sans-serif';
    Chart.defaults.borderColor = COL.grid;
    // THE fix for the charts overflowing their cards: Chart.js defaults to
    // maintainAspectRatio:true, which ignores the container's CSS height and
    // sizes the canvas to width/2 — on a wide card that is ~800px tall, so the
    // congestion chart grew past the footer. Honour the .chart-box height.
    Chart.defaults.responsive = true;
    Chart.defaults.maintainAspectRatio = false;
    // Mutate the animation defaults, do NOT replace the object. Assigning a fresh
    // {duration,easing} wipes Chart.js's internal per-property animation registry
    // and throws "this._fn is not a function" on the next animation tick — which
    // on resize left canvases mis-sized and could overflow the page.
    Chart.defaults.animation.duration = 700;
    Chart.defaults.animation.easing = 'easeOutQuart';
    Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(20,22,26,.92)';
    Chart.defaults.plugins.tooltip.padding = 10;
    Chart.defaults.plugins.tooltip.cornerRadius = 8;
    Chart.defaults.plugins.tooltip.displayColors = false;
  }

  if (!jobId) { document.getElementById('loading').innerHTML = '<p>No job specified.</p>'; return; }

  const bust = '?_=' + Date.now();
  fetch('/api/results/' + jobId + bust, { cache: 'no-store' })
    .then(r => { if (!r.ok) throw new Error('Report not ready (' + r.status + ')'); return r.json(); })
    .then(render)
    .catch(err => {
      document.getElementById('loading').innerHTML =
        '<p>' + err.message + '</p><a class="btn secondary" href="/">Back</a>';
    });

  // Warning text comes from the pipeline and is interpolated into innerHTML, so
  // escape it. It can contain a filename, and a filename is user-supplied.
  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[c]));
  }

  function render(a) {
    document.getElementById('loading').classList.add('hidden');
    document.getElementById('content').classList.remove('hidden');

    // ---- meta + video ----
    const v = a.video;
    document.getElementById('fileMeta').textContent =
      `${v.filename} · ${v.width}×${v.height} · ${v.duration_sec}s · ${v.fps} fps`;
    const player = document.getElementById('player');
    player.src = '/api/video/' + jobId + bust;
    const tabOrig = document.getElementById('tabOrig');
    document.getElementById('tabAnnot').onclick = () => switchVid(true);
    tabOrig.onclick = () => switchVid(false);
    // Only offer "Original" if the server can actually serve it. Jobs produced by
    // the CLI have no uploaded copy, and the tab used to silently show nothing.
    fetch('/api/original/' + jobId, { method: 'HEAD' })
      .then(r => { if (!r.ok) throw new Error('unavailable'); })
      .catch(() => {
        tabOrig.disabled = true;
        tabOrig.title = 'Original not stored for this job';
        tabOrig.style.opacity = 0.45;
        tabOrig.style.cursor = 'not-allowed';
        tabOrig.onclick = null;
      });
    function switchVid(annot) {
      document.getElementById('tabAnnot').classList.toggle('active', annot);
      document.getElementById('tabOrig').classList.toggle('active', !annot);
      const t = player.currentTime;
      player.src = (annot ? '/api/video/' + jobId : '/api/original/' + jobId) + bust;
      // Seeking has to wait for the new source's metadata — assigning
      // currentTime straight after setting src is discarded (readyState 0),
      // which silently restarts playback from 0.
      player.addEventListener('loadedmetadata', function seek() {
        player.removeEventListener('loadedmetadata', seek);
        if (t) player.currentTime = Math.min(t, player.duration || t);
        player.play().catch(() => {});
      });
    }

    // ---- KPIs ----
    const c = a.counting, sp = a.speed, cg = a.congestion;
    const worst = cg.worst_level || 'Free-flow';
    kpis([
      { label: 'Vehicles counted', value: c.total_vehicles, cls: 'red',
        sub: `${c.by_direction.in} in · ${c.by_direction.out} out` },
      { label: 'Avg speed', value: sp.avg_kmh, unit: 'km/h', cls: 'ink',
        sub: `median ${sp.median_kmh} · ${sp.n_vehicles_timed} timed` },
      { label: 'Peak congestion', value: worst, cls: worstCls(worst),
        sub: `${cg.avg_occupancy != null ? Math.round(cg.avg_occupancy * 100) : 0}% avg occupancy`, badge: true },
      { label: 'Pedestrians', value: (c.pedestrians.in + c.pedestrians.out), cls: 'amber',
        sub: 'crossings detected' },
      { label: 'Throughput', value: (a.throughput ? a.throughput.veh_per_hour : 0), unit: 'veh/h', cls: 'ink',
        sub: a.throughput && a.throughput.busiest_lane ? `busiest: Lane ${a.throughput.busiest_lane}` : 'projected flow' },
      { label: 'Duration', value: v.duration_sec, unit: 's', cls: 'blue',
        sub: `${v.frames_processed} frames analyzed` },
    ]);

    const tr = a.tracking || {};
    const trBits = tr.stable_ids
      ? `Vehicle IDs persist through occlusions (ByteTrack + re-identification): ` +
        `<b>${tr.gap_reattachments || 0}</b> re-attachments after a vehicle was lost, ` +
        `<b>${tr.duplicate_merges || 0}</b> duplicate detections collapsed — so a car that ` +
        `disappears behind a truck keeps its original ID instead of being counted twice.`
      : 'Stable IDs are disabled; a vehicle lost to occlusion may be counted twice.';
    // A calibration mismatch has to be impossible to miss: the counts, lanes and
    // speeds below are all wrong when it fires, and a small confident number is
    // more damaging than a visible error.
    const cal = a.calibration || {};
    const health = (cal.health === 'MISMATCHED')
      ? '<div class="note" style="border-left:4px solid #e4002b;background:rgba(228,0,43,.08);">' +
        '<b>⚠ Scene calibration does not match this video.</b><br>' +
        escapeHtml(cal.health_note || '') +
        '<br><br><b>The vehicle count, lane analytics and speeds on this page are ' +
        'not reliable for this clip.</b></div>'
      : '';
    const strideWarn = (a.tracking && a.tracking.stride_health &&
                        a.tracking.stride_health !== 'OK')
      ? '<div class="note" style="border-left:4px solid #ff7a3d;background:rgba(255,122,61,.08);">' +
        '<b>⚠ Frame stride too coarse.</b><br>' +
        escapeHtml(a.tracking.stride_note || '') + '</div>'
      : '';
    document.getElementById('calNote').innerHTML = health + strideWarn +
      '<div class="note">' + trBits + '<br><br>' +
      'Lane boundaries are measured from the painted stripes, not estimated: a median-background ' +
      'fit recovers each line, and the road plane is rectified so the dash pitch comes out constant ' +
      '(5.9% spread). Speeds inherit that rectification and assume a ' +
      '<code>dash_pitch_m</code> of 12&nbsp;m — change only that value if this road uses a different ' +
      'marking standard, or set <code>target_size_m</code> from a surveyed ground distance to remove ' +
      'the assumption.</div>' + classNote(a);

    // ---- congestion timeline (rendered FIRST, before any Chart.js chart) ----
    // A state ribbon, not a bar chart: congestion is a categorical state over
    // time (status ramp Free-flow -> Jam), so encoding severity as bar HEIGHT
    // made 98% Free-flow read as a flat green block with one confusing spike.
    //
    // Rendered before the charts on purpose. It is pure HTML/CSS and cannot fail
    // from a Chart.js problem; putting it first means a chart error can never
    // leave the congestion timeline blank.
    const ts = a.timeseries;
    congestionRibbon(ts.t_sec, ts.congestion_level);
    const pct = cg.levels_pct || {};
    document.getElementById('congestionSummary').innerHTML =
      '<div class="ribbon-legend">' +
      LEVELS.map(l => `<span class="badge ${LEVEL_CLASS[l]}">${l}<b>${pct[l] || 0}%</b></span>`).join('') +
      '</div>' +
      (cg.peak_periods && cg.peak_periods.length
        ? `<div class="tag" style="margin-top:12px">⚠ Peak congestion: ` +
          cg.peak_periods.map(p => `${p.start_sec}–${p.end_sec}s (${p.level})`).join(', ') + '</div>'
        : '');

    // ---- vehicle mix doughnut ----
    const mix = c.by_class || {};
    const mixLabels = Object.keys(mix), mixVals = Object.values(mix);
    const mixColors = mixLabels.map(l => CLASS_COLOR[codeOf(l)] || COL.text);
    doughnut('classChart', mixLabels, mixVals, mixColors);
    document.getElementById('classTable').innerHTML =
      mixLabels.map((k, i) =>
        `<tr><td><span style="color:${mixColors[i]}">●</span> ${k}</td><td>${mixVals[i]}</td></tr>`
      ).join('') + `<tr><td>Total</td><td>${c.total_vehicles}</td></tr>`;

    // ---- classification scheme legend ----
    const scheme = a.class_scheme || {};
    const codeCounts = c.by_class_code || {};
    document.getElementById('schemeLegend').innerHTML = Object.entries(scheme).map(([code, name]) => {
      const n = codeCounts[code] || 0;
      return `<span class="chip ${n ? '' : 'off'}"><span class="dot" style="background:${CLASS_COLOR[code] || COL.text}"></span>` +
             `<span class="code">${code}</span> ${name} <span class="cnt">· ${n}</span></span>`;
    }).join('');

    // ---- lane-by-lane analytics ----
    const lanes = a.lanes || [];
    if (lanes.length) {
      const laneLabels = lanes.map(l => 'Lane ' + l.lane);
      const laneColors = lanes.map(l => l.color);
      new Chart(el('laneChart'), { type: 'bar',
        data: { labels: laneLabels, datasets: [{ label: 'Vehicles', data: lanes.map(l => l.count),
          backgroundColor: laneColors, borderRadius: 6 }] },
        options: { plugins: { legend: { display: false } }, scales: { x: noGrid, y: yGrid } } });
      document.getElementById('laneTable').innerHTML =
        '<tr><td style="color:var(--muted)">Lane</td><td style="color:var(--muted);text-align:left">Vehicles</td>' +
        '<td style="color:var(--muted);text-align:left">Avg speed</td><td style="color:var(--muted);text-align:right">Mostly</td></tr>' +
        lanes.map(l =>
          `<tr><td><span style="color:${l.color}">●</span> Lane ${l.lane}</td>` +
          `<td style="text-align:left;color:var(--ink)">${l.count} <span style="color:var(--muted)">(${l.pct}%)</span></td>` +
          `<td style="text-align:left;color:var(--ink)">${l.avg_speed_kmh ? l.avg_speed_kmh + ' km/h' : '—'}</td>` +
          `<td style="text-align:right">${l.dominant_class}</td></tr>`
        ).join('');
      const tp = a.throughput || {};
      document.getElementById('throughputLine').innerHTML =
        `Projected throughput <b style="color:var(--ink)">${tp.veh_per_hour || 0} veh/h</b>` +
        (tp.busiest_lane ? ` · busiest <b style="color:var(--ink)">Lane ${tp.busiest_lane}</b>` : '') +
        (tp.lane_balance_pct != null ? ` · lane balance ${tp.lane_balance_pct}%` : '');
    }

    // ---- volume over time ----
    line('volumeChart', ts.t_sec.map(s => s + 's'), ts.vehicle_count, 'Vehicles', COL.red, COL.redSoft);

    // ---- speed histogram ----
    document.getElementById('speedHint').textContent =
      sp.n_vehicles_timed ? `— ${sp.n_vehicles_timed} vehicles, median ${sp.median_kmh} km/h` : '';
    if (sp.histogram && sp.histogram.bins.length) {
      bar('speedChart', sp.histogram.bins, sp.histogram.counts, 'Vehicles', COL.red);
    } else { empty('speedChart', 'No speed samples'); }

    // ---- directional flow ----
    const dir = c.by_class_direction || { in: {}, out: {} };
    const allClasses = Array.from(new Set([...Object.keys(dir.in || {}), ...Object.keys(dir.out || {})]));
    groupedBar('dirChart', allClasses.map(codeOf),
      allClasses.map(k => (dir.in || {})[k] || 0),
      allClasses.map(k => (dir.out || {})[k] || 0));

  }

  // Build the congestion state ribbon: one flex row of coloured segments sized
  // by how long each state ran, with time ticks below. Adjacent same-state
  // seconds are merged into one run so the DOM stays small and the segment
  // boundaries mean something (a state change), not an arbitrary 1s grid.
  function congestionRibbon(tSec, levels) {
    const host = document.getElementById('congestionRibbon');
    if (!host || !levels || !levels.length) return;
    const runs = [];
    for (let i = 0; i < levels.length; i++) {
      const last = runs[runs.length - 1];
      if (last && last.level === levels[i]) last.end = tSec[i];
      else runs.push({ level: levels[i], start: tSec[i], end: tSec[i] });
    }
    const total = (tSec[tSec.length - 1] - tSec[0]) || 1;
    const seg = runs.map(r => {
      const w = Math.max(((r.end - r.start + 1) / (total + 1)) * 100, 0.4);
      return `<div class="seg ${LEVEL_CLASS[r.level]}" style="width:${w}%"
        title="${r.level}: ${r.start}s–${r.end}s"></div>`;
    }).join('');
    // A handful of evenly spaced time ticks — not one per second, which collided.
    const nTicks = 8, ticks = [];
    for (let k = 0; k <= nTicks; k++) {
      ticks.push(`<span>${Math.round(tSec[0] + (total * k / nTicks))}s</span>`);
    }
    host.innerHTML = `<div class="ribbon">${seg}</div>
      <div class="ribbon-axis">${ticks.join('')}</div>`;
  }

  // Where the class labels on THIS report came from. Previously hard-coded to
  // "C and V need a fine-tuned model", which stopped being true the moment one
  // was wired in — the page denied using the thing it was using. Reports have to
  // describe their own run.
  function classNote(a) {
    const c = a.classification;
    if (!c) return '';   // report predates the field
    const trained = !!c.model;
    return '<div class="note"' +
      (trained ? '' : ' style="border-left:4px solid #ffbe3d;background:rgba(255,190,61,.08);"') +
      '><b>Vehicle class source:</b> ' + escapeHtml(c.source) +
      (trained ? ' — <code>' + escapeHtml(c.model) + '</code>' : '') +
      '<br>' + escapeHtml(c.note || '') + '</div>';
  }

  function worstCls(l) { return { 'Free-flow': 'green', 'Moderate': 'amber', 'Heavy': 'amber', 'Jam': 'red' }[l] || 'ink'; }

  function kpis(items) {
    document.getElementById('kpis').innerHTML = items.map(k => {
      const val = k.badge
        ? `<span class="badge ${LEVEL_CLASS[k.value] || 'free'}">${k.value}</span>`
        : `${k.value}${k.unit ? ' <small>' + k.unit + '</small>' : ''}`;
      return `<div class="kpi ${k.cls}"><div class="label">${k.label}</div>
        <div class="value">${val}</div><div class="sub">${k.sub || ''}</div></div>`;
    }).join('');
  }

  const noGrid = { grid: { display: false }, ticks: { maxRotation: 0, autoSkip: true } };
  const yGrid = { grid: { color: COL.grid }, ticks: { precision: 0 }, beginAtZero: true };

  function doughnut(id, labels, data, colors) {
    new Chart(el(id), { type: 'doughnut',
      data: { labels, datasets: [{ data, backgroundColor: colors, borderColor: '#fff', borderWidth: 2 }] },
      options: { plugins: { legend: { position: 'bottom' } }, cutout: '62%' } });
  }
  function line(id, labels, data, label, border, fill) {
    new Chart(el(id), { type: 'line',
      data: { labels, datasets: [{ label, data, borderColor: border, backgroundColor: fill,
        fill: true, tension: .35, pointRadius: 0, borderWidth: 2.5 }] },
      options: { plugins: { legend: { display: false } }, scales: { x: noGrid, y: yGrid } } });
  }
  function bar(id, labels, data, label, color) {
    new Chart(el(id), { type: 'bar',
      data: { labels, datasets: [{ label, data, backgroundColor: color, borderRadius: 5 }] },
      options: { plugins: { legend: { display: false } }, scales: { x: noGrid, y: yGrid } } });
  }
  function groupedBar(id, labels, inData, outData) {
    new Chart(el(id), { type: 'bar',
      data: { labels, datasets: [
        { label: 'In', data: inData, backgroundColor: COL.red, borderRadius: 5 },
        { label: 'Out', data: outData, backgroundColor: COL.ink, borderRadius: 5 } ] },
      options: { plugins: { legend: { position: 'bottom' } }, scales: { x: noGrid, y: yGrid } } });
  }
  function empty(id, msg) { el(id).parentElement.innerHTML = `<p class="tag">${msg}</p>`; }
  function el(id) { return document.getElementById(id); }

  const pdf = document.getElementById('pdfBtn');
  if (pdf) pdf.onclick = () => window.print();
})();
