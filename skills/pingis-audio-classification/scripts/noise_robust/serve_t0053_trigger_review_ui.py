"""
T0053 local Fable trigger review UI.

Serves a small browser UI for the T0052 continuous debug WAV. The UI lets a
reviewer listen around each saved native trigger, label it, and add manual
markers for missed bounces. Labels are saved under ignored /data outputs.

No app runtime, model JSON, APK, training, or export change happens here.
"""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import sys
import wave
from array import array
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT_DIR = Path(__file__).resolve().parents[4]
SESSION_ID = "fable_live_session_2026-06-28T16-26-01-662Z"
RAW_DIR = ROOT_DIR / "data" / "audio" / "raw" / "t0052_fable_continuous_debug_round" / "fable_live_debug"
EVAL_DIR = ROOT_DIR / "data" / "audio" / "models" / "evaluations" / "t0052_fable_continuous_debug_round"
OUT_DIR = ROOT_DIR / "data" / "audio" / "models" / "evaluations" / "t0053_fable_trigger_review_ui"
LABELS_PATH = OUT_DIR / f"{SESSION_ID}_review_labels.json"
EXPECTED_COUNT = 30
REPORTED_APP_COUNT = 0
TRIGGER_CSV: Path | None = None
MANUAL_ONLY = False
GATE_NOTE = (
    "Saved phone native candidates. The native gate watches 10 ms bandpass-RMS frames and triggers when RMS >= "
    "max(background * 1.5, 0.0015), then applies a 120 ms cooldown."
)

HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Fable Trigger Review</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f5ef;
      --panel: #ffffff;
      --text: #1e2528;
      --muted: #667176;
      --line: #d9ddd7;
      --strong: #0f766e;
      --racket: #18864b;
      --noise: #5f6770;
      --duplicate: #7c3aed;
      --unclear: #b7791f;
      --danger: #b42318;
      --ink: #111827;
      --soft: #edf7f4;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Arial, Helvetica, sans-serif;
      letter-spacing: 0;
    }
    button, input, textarea, select {
      font: inherit;
      letter-spacing: 0;
    }
    button {
      min-height: 34px;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--text);
      border-radius: 6px;
      padding: 6px 10px;
      cursor: pointer;
      white-space: nowrap;
    }
    button:hover { border-color: #98a5a4; }
    button.primary {
      background: var(--strong);
      border-color: var(--strong);
      color: #fff;
    }
    button.active {
      outline: 2px solid var(--ink);
      outline-offset: 1px;
    }
    .app {
      display: grid;
      grid-template-rows: auto auto 1fr;
      min-height: 100vh;
    }
    header {
      padding: 16px 20px 10px;
      border-bottom: 1px solid var(--line);
      background: #fff;
    }
    h1 {
      margin: 0;
      font-size: 22px;
      line-height: 1.2;
    }
    .subtitle {
      margin-top: 6px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.4;
    }
    .metrics {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 12px;
    }
    .metric {
      border: 1px solid var(--line);
      background: #fafafa;
      border-radius: 6px;
      padding: 6px 8px;
      min-width: 104px;
    }
    .metric b {
      display: block;
      font-size: 16px;
      color: var(--ink);
    }
    .metric span {
      color: var(--muted);
      font-size: 11px;
    }
    .toolbar {
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
      padding: 10px 20px;
      border-bottom: 1px solid var(--line);
      background: #fbfaf6;
    }
    .toolbar-group {
      display: flex;
      gap: 6px;
      align-items: center;
      flex-wrap: wrap;
      border-left: 1px solid var(--line);
      padding-left: 8px;
    }
    .toolbar label {
      color: var(--muted);
      font-size: 12px;
      display: inline-flex;
      align-items: center;
      gap: 5px;
    }
    select {
      min-height: 34px;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--text);
      border-radius: 6px;
      padding: 5px 8px;
    }
    audio { width: min(520px, 100%); height: 36px; }
    .status {
      margin-left: auto;
      color: var(--muted);
      font-size: 12px;
    }
    main {
      display: grid;
      grid-template-columns: minmax(520px, 1fr) 410px;
      gap: 12px;
      padding: 12px;
      min-height: 0;
    }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      min-width: 0;
    }
    .wave-section {
      display: grid;
      grid-template-rows: auto auto 1fr;
      min-height: 0;
    }
    .section-head {
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      display: flex;
      gap: 8px;
      align-items: center;
      justify-content: space-between;
    }
    .section-head h2 {
      margin: 0;
      font-size: 15px;
    }
    .gate-note {
      padding: 8px 12px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.4;
      border-bottom: 1px solid var(--line);
      background: var(--soft);
    }
    .canvas-wrap {
      position: relative;
      padding: 10px 12px 0;
    }
    canvas {
      width: 100%;
      height: 220px;
      display: block;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fbfbf8;
      cursor: crosshair;
      touch-action: none;
    }
    .detail {
      padding: 10px 12px;
      display: grid;
      grid-template-columns: 1fr;
      gap: 8px;
      border-top: 1px solid var(--line);
    }
    .detail-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
      font-size: 12px;
    }
    .detail-grid div {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 6px 8px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .detail-grid span {
      display: block;
      color: var(--muted);
      font-size: 11px;
      margin-bottom: 2px;
    }
    .label-buttons {
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
    }
    .label-racket { border-color: var(--racket); color: var(--racket); }
    .label-noise { border-color: var(--noise); color: var(--noise); }
    .label-duplicate { border-color: var(--duplicate); color: var(--duplicate); }
    .label-unclear { border-color: var(--unclear); color: var(--unclear); }
    textarea {
      width: 100%;
      min-height: 58px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
    }
    .side {
      display: grid;
      grid-template-rows: auto auto 1fr;
      min-height: 0;
    }
    .filters {
      padding: 8px 10px;
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
      border-bottom: 1px solid var(--line);
    }
    .list {
      overflow: auto;
      min-height: 0;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }
    th, td {
      border-bottom: 1px solid #ecefec;
      padding: 6px 7px;
      text-align: left;
      vertical-align: top;
    }
    th {
      position: sticky;
      top: 0;
      background: #fff;
      z-index: 1;
      color: var(--muted);
      font-weight: 700;
    }
    tr {
      cursor: pointer;
    }
    tr.selected {
      background: #e7f5f1;
      outline: 2px solid #99d6c8;
      outline-offset: -2px;
    }
    tr.reference {
      color: var(--muted);
    }
    .badge {
      display: inline-block;
      border-radius: 5px;
      border: 1px solid var(--line);
      padding: 2px 5px;
      font-size: 11px;
      max-width: 90px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .badge.racket { color: var(--racket); border-color: var(--racket); }
    .badge.noise { color: var(--noise); border-color: var(--noise); }
    .badge.duplicate { color: var(--duplicate); border-color: var(--duplicate); }
    .badge.unclear { color: var(--unclear); border-color: var(--unclear); }
    body.manual-only .label-buttons {
      display: none;
    }
    body.manual-only #addNoiseBtn,
    body.manual-only #addUnclearBtn {
      display: none;
    }
    body.manual-only button[data-filter="unreviewed"],
    body.manual-only button[data-filter="racket"],
    body.manual-only button[data-filter="noise"] {
      display: none;
    }
    .manual-row {
      border-top: 1px solid var(--line);
      padding: 8px 10px;
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
      align-items: center;
    }
    @media (max-width: 980px) {
      main { grid-template-columns: 1fr; }
      .side { min-height: 520px; }
      .status { margin-left: 0; width: 100%; }
    }
  </style>
</head>
<body>
  <div class="app">
    <header>
      <h1>Fable Trigger Review</h1>
      <div class="subtitle" id="sourceLine">Loading...</div>
      <div class="metrics" id="metrics"></div>
    </header>
    <div class="toolbar">
      <audio id="audio" controls preload="metadata" src="/audio.wav"></audio>
      <button id="playBtn" class="primary">Play</button>
      <button id="loopBtn">Loop Off</button>
      <div class="toolbar-group">
        <button id="prevBtn">Prev Label</button>
        <button id="nextBtn">Next Label</button>
        <button id="addRacketTopBtn" class="primary">Add Racket</button>
        <button id="deleteLabelTopBtn">Delete Label</button>
      </div>
      <div class="toolbar-group">
        <label>Speed
          <select id="speedSelect">
            <option value="0.5">0.5x</option>
            <option value="0.75">0.75x</option>
            <option value="1" selected>1x</option>
            <option value="1.25">1.25x</option>
            <option value="1.5">1.5x</option>
          </select>
        </label>
      </div>
      <div class="toolbar-group">
        <button id="zoomOutBtn">Zoom Out</button>
        <button id="zoomFitBtn">Fit</button>
        <button id="zoomInBtn">Zoom In</button>
        <button id="panLeftBtn">Left</button>
        <button id="panRightBtn">Right</button>
        <span class="status" id="zoomReadout">1x</span>
      </div>
      <button id="saveBtn">Save Labels</button>
      <button id="exportBtn">Export JSON</button>
      <span class="status" id="saveStatus">Not loaded</span>
    </div>
    <main>
      <section class="wave-section">
        <div class="section-head">
          <h2>Continuous WAV</h2>
          <div id="timeReadout">0.000 s</div>
        </div>
        <div class="gate-note" id="gateNote"></div>
        <div class="canvas-wrap">
          <canvas id="wave" width="1400" height="220"></canvas>
        </div>
        <div class="detail">
          <div class="detail-grid" id="selectedDetails"></div>
          <div class="label-buttons">
            <button class="label-racket" data-label="racket">Racket</button>
            <button class="label-noise" data-label="noise">Noise</button>
            <button class="label-duplicate" data-label="duplicate">Duplicate</button>
            <button class="label-unclear" data-label="unclear">Unclear</button>
            <button id="clearLabelBtn">Clear</button>
            <button id="resetTimeBtn">Reset Time</button>
          </div>
          <textarea id="noteBox" placeholder="Note for selected trigger"></textarea>
          <div class="manual-row">
            <button id="addRacketBtn" class="primary">Add Racket</button>
            <button id="addNoiseBtn">Add Noise at Playhead</button>
            <button id="addUnclearBtn">Add Unclear at Playhead</button>
            <button id="deleteManualBtn">Delete Label</button>
          </div>
        </div>
      </section>
      <section class="side">
        <div class="section-head">
          <h2 id="triggerTitle">Trigger Candidates</h2>
          <div id="reviewCount">0 reviewed</div>
        </div>
        <div class="filters">
          <button data-filter="all" class="active">All</button>
          <button data-filter="unreviewed">Unreviewed</button>
          <button data-filter="racket">Racket</button>
          <button data-filter="noise">Noise</button>
          <button data-filter="manual">Manual</button>
        </div>
        <div class="list">
          <table>
            <thead>
              <tr>
                <th>#</th>
                <th>Time</th>
                <th>Model</th>
                <th>Reject</th>
                <th>Your label</th>
              </tr>
            </thead>
            <tbody id="triggerBody"></tbody>
          </table>
        </div>
      </section>
    </main>
  </div>
  <script>
    const state = {
      session: null,
      selectedId: null,
      filter: 'all',
      dirty: false,
      loop: false,
      loopStart: 0,
      loopEnd: 0,
      saveTimer: null,
      drag: null,
      manualOnly: false,
      zoom: 1,
      viewStart: 0,
      playheadTime: 0,
      pendingSeekTime: null,
      pendingSeekStartedAt: 0
    };

    const audio = document.getElementById('audio');
    const canvas = document.getElementById('wave');
    const ctx = canvas.getContext('2d');
    const body = document.getElementById('triggerBody');
    const saveStatus = document.getElementById('saveStatus');

    function fmtTime(seconds) {
      if (!Number.isFinite(seconds)) return '-';
      return seconds.toFixed(3) + ' s';
    }

    function labelFor(id) {
      return state.session.review.trigger_labels[id] || { label: '', note: '' };
    }

    function manualFor(id) {
      return state.session.review.manual_markers.find(m => m.id === id);
    }

    function clampTime(seconds) {
      const duration = state.session?.duration_s || 0;
      return Math.max(0, Math.min(duration, Number(seconds) || 0));
    }

    function setPlayhead(seconds, syncAudio = true) {
      const next = clampTime(seconds);
      state.playheadTime = next;
      if (syncAudio) {
        state.pendingSeekTime = next;
        state.pendingSeekStartedAt = Date.now();
        try {
          audio.currentTime = next;
        } catch (err) {
          state.pendingSeekTime = null;
        }
      }
      document.getElementById('timeReadout').textContent = fmtTime(next);
      ensurePlayheadVisible();
      drawWaveform();
    }

    function syncPlayheadFromAudio(force = false) {
      const audioTime = clampTime(audio.currentTime);
      if (state.pendingSeekTime !== null) {
        const ageMs = Date.now() - state.pendingSeekStartedAt;
        const delta = Math.abs(audioTime - state.pendingSeekTime);
        if (delta > 0.08 && ageMs < 2500) {
          if (force && ageMs > 150) {
            try {
              audio.currentTime = state.pendingSeekTime;
            } catch (err) {
              state.pendingSeekTime = null;
            }
          }
          document.getElementById('timeReadout').textContent = fmtTime(state.playheadTime);
          drawWaveform();
          return;
        }
        state.pendingSeekTime = null;
      }
      state.playheadTime = audioTime;
      document.getElementById('timeReadout').textContent = fmtTime(state.playheadTime);
      ensurePlayheadVisible();
      drawWaveform();
    }

    function viewDuration() {
      const duration = state.session?.duration_s || 0;
      return Math.max(0.35, duration / Math.max(1, state.zoom));
    }

    function maxViewStart() {
      const duration = state.session?.duration_s || 0;
      return Math.max(0, duration - viewDuration());
    }

    function clampViewStart(value) {
      return Math.max(0, Math.min(maxViewStart(), Number(value) || 0));
    }

    function viewEnd() {
      return Math.min(state.session?.duration_s || 0, state.viewStart + viewDuration());
    }

    function timeToCanvasX(time, width) {
      return (time - state.viewStart) / Math.max(0.001, viewDuration()) * width;
    }

    function canvasXToTime(xRatio) {
      return clampTime(state.viewStart + xRatio * viewDuration());
    }

    function centerViewOn(time) {
      state.viewStart = clampViewStart(clampTime(time) - viewDuration() / 2);
    }

    function ensurePlayheadVisible() {
      const now = clampTime(state.playheadTime);
      if (now < state.viewStart || now > viewEnd()) centerViewOn(now);
    }

    function setZoom(nextZoom) {
      const center = clampTime(state.playheadTime || (state.viewStart + viewDuration() / 2));
      state.zoom = Math.max(1, Math.min(32, nextZoom));
      centerViewOn(center);
      renderAll();
    }

    function panView(direction) {
      state.viewStart = clampViewStart(state.viewStart + direction * viewDuration() * 0.65);
      renderAll();
    }

    function itemTime(item) {
      if (!item) return clampTime(state.playheadTime);
      if (item.kind === 'manual') return clampTime(manualFor(item.id)?.time_s ?? item.time_s);
      const adjusted = Number(labelFor(item.id).adjusted_time_s);
      return Number.isFinite(adjusted) ? clampTime(adjusted) : clampTime(item.time_s);
    }

    function setItemTime(item, seconds) {
      const nextTime = Number(clampTime(seconds).toFixed(3));
      if (item.kind === 'manual') {
        const marker = manualFor(item.id);
        if (marker) marker.time_s = nextTime;
        return;
      }
      if (state.manualOnly) return;
      const existing = labelFor(item.id);
      state.session.review.trigger_labels[item.id] = {
        ...existing,
        time_s: nextTime,
        original_time_s: existing.original_time_s ?? item.time_s,
        adjusted_time_s: nextTime,
        event_index: item.event_index,
        updated_at: new Date().toISOString()
      };
    }

    function nearestItemAtTime(time) {
      const items = state.manualOnly ? allItems().filter(item => item.kind === 'manual') : allItems();
      return items.reduce((best, item) => {
        const delta = Math.abs(itemTime(item) - time);
        return !best || delta < best.delta ? { item, delta } : best;
      }, null);
    }

    function canvasTime(e) {
      const rect = canvas.getBoundingClientRect();
      const x = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      return canvasXToTime(x);
    }

    function markUnsavedDuringDrag() {
      state.dirty = true;
      saveStatus.textContent = 'Unsaved';
      if (state.saveTimer) clearTimeout(state.saveTimer);
    }

    function allItems() {
      const triggers = state.session.triggers.map(t => ({ ...t, kind: 'trigger' }));
      const manual = state.session.review.manual_markers.map(m => ({
        id: m.id,
        event_index: 'M',
        time_s: Number(m.time_s),
        model_label: 'manual',
        model_confidence: '',
        reject_reason: '',
        native_rms: '',
        prob_racket_bounce: '',
        nearest_offline_trigger_delta_ms: '',
        kind: 'manual'
      }));
      return triggers.concat(manual).sort((a, b) => itemTime(a) - itemTime(b));
    }

    function currentItem() {
      if (!state.selectedId) return null;
      return allItems().find(item => item.id === state.selectedId) || null;
    }

    function colorFor(item) {
      const saved = item.kind === 'manual'
        ? (state.session.review.manual_markers.find(m => m.id === item.id)?.label || 'manual')
        : (labelFor(item.id).label || '');
      if (saved === 'racket') return '#18864b';
      if (saved === 'noise') return '#5f6770';
      if (saved === 'duplicate') return '#7c3aed';
      if (saved === 'unclear') return '#b7791f';
      if (item.kind === 'manual') return '#0f766e';
      if (item.reject_reason === 'stale_backlog') return '#b42318';
      if (item.model_label === 'noise') return '#d97706';
      return '#667176';
    }

    function renderMetrics() {
      const labels = state.session.review.trigger_labels;
      const manual = state.session.review.manual_markers;
      const values = Object.values(labels).map(x => x.label).filter(Boolean);
      const reviewed = values.length;
      const manualRacket = manual.filter(x => x.label === 'racket').length;
      const racket = values.filter(x => x === 'racket').length + manualRacket;
      const noise = values.filter(x => x === 'noise').length + manual.filter(x => x.label === 'noise').length;
      const duplicate = values.filter(x => x === 'duplicate').length;
      const unclear = values.filter(x => x === 'unclear').length + manual.filter(x => x.label === 'unclear').length;
      const metrics = [
        ['Expected', state.session.expected_count],
        ['App Count', state.session.reported_app_count],
        ['Reference Triggers', state.session.triggers.length],
        [state.manualOnly ? 'Manual Racket' : 'Reviewed', state.manualOnly ? manualRacket : reviewed],
        ['Racket Labels', racket],
        ['Noise Labels', noise],
        ['Duplicate', duplicate],
        ['Unclear', unclear],
        ['Manual', manual.length]
      ];
      document.getElementById('metrics').innerHTML = metrics.map(([label, value]) =>
        `<div class="metric"><b>${value}</b><span>${label}</span></div>`
      ).join('');
      document.getElementById('reviewCount').textContent = state.manualOnly
        ? `${manualRacket}/${state.session.expected_count} manual racket`
        : `${reviewed} reviewed`;
    }

    function renderDetails() {
      const item = currentItem();
      if (!item) {
        document.getElementById('selectedDetails').innerHTML = [
          ['Selection', 'None'],
          ['Playhead', fmtTime(state.playheadTime || 0)],
          ['Visible', `${fmtTime(state.viewStart)} - ${fmtTime(viewEnd())}`],
          ['Tip', state.manualOnly ? 'Click waveform, then Add Racket' : 'Click or drag a marker']
        ].map(([k, v]) => `<div><span>${k}</span>${v}</div>`).join('');
        document.getElementById('noteBox').value = '';
        document.querySelectorAll('[data-label]').forEach(btn => btn.classList.remove('active'));
        return;
      }
      const saved = item.kind === 'manual'
        ? manualFor(item.id)
        : labelFor(item.id);
      const displayTime = itemTime(item);
      const originalTime = item.kind === 'trigger' ? fmtTime(item.time_s) : '-';
      const adjustedTime = item.kind === 'trigger' && Number.isFinite(Number(saved?.adjusted_time_s))
        ? fmtTime(Number(saved.adjusted_time_s))
        : '-';
      const details = [
        ['ID', item.event_index],
        ['Time', fmtTime(displayTime)],
        ['Original', state.manualOnly && item.kind !== 'manual' ? 'read-only' : originalTime],
        ['Adjusted', state.manualOnly && item.kind !== 'manual' ? 'disabled' : adjustedTime],
        ['Model', item.model_label || '-'],
        ['Confidence', item.model_confidence ? Number(item.model_confidence).toFixed(3) : '-'],
        ['Reject', item.reject_reason || '-'],
        ['Racket Prob', item.prob_racket_bounce ? Number(item.prob_racket_bounce).toFixed(3) : '-'],
        ['RMS', item.native_rms ? Number(item.native_rms).toFixed(4) : '-'],
        ['Offline Delta', item.nearest_offline_trigger_delta_ms !== '' ? Number(item.nearest_offline_trigger_delta_ms).toFixed(1) + ' ms' : '-']
      ];
      document.getElementById('selectedDetails').innerHTML = details.map(([k, v]) =>
        `<div><span>${k}</span>${v}</div>`
      ).join('');
      document.getElementById('noteBox').value = saved?.note || '';
      document.querySelectorAll('[data-label]').forEach(btn => {
        btn.classList.toggle('active', saved?.label === btn.dataset.label);
      });
    }

    function renderTable() {
      const rows = allItems().filter(item => {
        const saved = item.kind === 'manual'
          ? manualFor(item.id)?.label
          : labelFor(item.id).label;
        if (state.filter === 'all') return true;
        if (state.manualOnly && state.filter === 'unreviewed') return false;
        if (state.filter === 'unreviewed') return item.kind === 'trigger' && !saved;
        if (state.filter === 'manual') return item.kind === 'manual';
        return saved === state.filter;
      });
      body.innerHTML = rows.map(item => {
        const saved = item.kind === 'manual'
          ? manualFor(item.id)?.label
          : labelFor(item.id).label;
        const conf = item.model_confidence ? Number(item.model_confidence).toFixed(2) : '';
        const model = item.model_label === 'manual' ? 'manual' : `${item.model_label || '-'} ${conf}`;
        const badge = saved
          ? `<span class="badge ${saved}">${saved}</span>`
          : (state.manualOnly && item.kind === 'trigger' ? '<span class="badge">reference</span>' : '<span class="badge">open</span>');
        const classes = [
          item.id === state.selectedId ? 'selected' : '',
          state.manualOnly && item.kind === 'trigger' ? 'reference' : ''
        ].filter(Boolean).join(' ');
        return `<tr data-id="${item.id}" class="${classes}">
          <td>${item.event_index}</td>
          <td>${fmtTime(itemTime(item))}</td>
          <td>${model}</td>
          <td>${item.reject_reason || '-'}</td>
          <td>${badge}</td>
        </tr>`;
      }).join('');
      body.querySelectorAll('tr').forEach(row => {
        row.addEventListener('click', () => {
          const item = allItems().find(candidate => candidate.id === row.dataset.id);
          if (!item) return;
          setPlayhead(itemTime(item));
          centerViewOn(state.playheadTime);
          if (state.manualOnly && item.kind !== 'manual') {
            state.selectedId = null;
            renderAll();
            return;
          }
          selectItem(row.dataset.id, false);
        });
      });
    }

    function drawWaveform() {
      if (!state.session) return;
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      const width = Math.max(300, Math.floor(rect.width * dpr));
      const height = Math.max(160, Math.floor(rect.height * dpr));
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = '#fbfbf8';
      ctx.fillRect(0, 0, width, height);
      const mid = height / 2;
      ctx.strokeStyle = '#d5dbd4';
      ctx.beginPath();
      ctx.moveTo(0, mid);
      ctx.lineTo(width, mid);
      ctx.stroke();
      const peaks = state.session.waveform;
      ctx.strokeStyle = '#3f4a4d';
      ctx.lineWidth = Math.max(1, dpr);
      for (let i = 0; i < peaks.length; i += 1) {
        const time = i / Math.max(1, peaks.length - 1) * state.session.duration_s;
        if (time < state.viewStart || time > viewEnd()) continue;
        const x = timeToCanvasX(time, width);
        const y1 = mid - peaks[i][1] * mid * 0.92;
        const y2 = mid - peaks[i][0] * mid * 0.92;
        ctx.beginPath();
        ctx.moveTo(x, y1);
        ctx.lineTo(x, y2);
        ctx.stroke();
      }
      for (const item of allItems()) {
        const t = itemTime(item);
        if (t < state.viewStart || t > viewEnd()) continue;
        const x = timeToCanvasX(t, width);
        ctx.strokeStyle = colorFor(item);
        ctx.globalAlpha = state.manualOnly && item.kind === 'trigger' ? 0.35 : 1.0;
        ctx.lineWidth = item.id === state.selectedId ? 4 * dpr : (item.kind === 'manual' ? 3 * dpr : 1.5 * dpr);
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
      }
      ctx.globalAlpha = 1.0;
      const playX = timeToCanvasX(state.playheadTime, width);
      ctx.strokeStyle = '#e11d48';
      ctx.lineWidth = 3 * dpr;
      if (playX >= 0 && playX <= width) {
        ctx.beginPath();
        ctx.moveTo(playX, 0);
        ctx.lineTo(playX, height);
        ctx.stroke();
        ctx.fillStyle = '#e11d48';
        ctx.beginPath();
        ctx.moveTo(playX, 0);
        ctx.lineTo(playX - 7 * dpr, 13 * dpr);
        ctx.lineTo(playX + 7 * dpr, 13 * dpr);
        ctx.closePath();
        ctx.fill();
      }
      document.getElementById('zoomReadout').textContent =
        `${state.zoom}x | ${fmtTime(state.viewStart)} - ${fmtTime(viewEnd())}`;
    }

    function selectItem(id, seek) {
      state.selectedId = id;
      const item = currentItem();
      if (seek && item) {
        setPlayhead(Math.max(0, itemTime(item) - 0.15));
      }
      ensurePlayheadVisible();
      renderAll();
    }

    function markDirty() {
      state.dirty = true;
      saveStatus.textContent = 'Unsaved';
      if (state.saveTimer) clearTimeout(state.saveTimer);
      state.saveTimer = setTimeout(saveLabels, 800);
    }

    function setLabel(label) {
      const item = currentItem();
      if (!item) return;
      if (item.kind === 'manual') {
        const marker = manualFor(item.id);
        if (marker) marker.label = label;
      } else {
        if (state.manualOnly) return;
        const existing = labelFor(item.id);
        state.session.review.trigger_labels[item.id] = {
          ...existing,
          label,
          time_s: itemTime(item),
          original_time_s: existing.original_time_s ?? item.time_s,
          event_index: item.event_index,
          updated_at: new Date().toISOString()
        };
      }
      markDirty();
      renderAll();
    }

    function clearLabel() {
      const item = currentItem();
      if (!item) return;
      if (item.kind === 'manual') {
        state.session.review.manual_markers = state.session.review.manual_markers.filter(m => m.id !== item.id);
        state.selectedId = null;
      } else {
        if (state.manualOnly) return;
        delete state.session.review.trigger_labels[item.id];
      }
      markDirty();
      renderAll();
    }

    function updateNote(value) {
      const item = currentItem();
      if (!item) return;
      if (item.kind === 'manual') {
        const marker = manualFor(item.id);
        if (marker) marker.note = value;
      } else {
        if (state.manualOnly) return;
        const existing = labelFor(item.id);
        state.session.review.trigger_labels[item.id] = {
          ...existing,
          note: value,
          time_s: itemTime(item),
          original_time_s: existing.original_time_s ?? item.time_s,
          event_index: item.event_index,
          updated_at: new Date().toISOString()
        };
      }
      markDirty();
    }

    function addManual(label) {
      const id = 'manual_' + Date.now();
      state.session.review.manual_markers.push({
        id,
        time_s: Number(state.playheadTime.toFixed(3)),
        label,
        note: '',
        created_at: new Date().toISOString()
      });
      state.selectedId = id;
      centerViewOn(state.playheadTime);
      markDirty();
      renderAll();
    }

    function resetSelectedTime() {
      const item = currentItem();
      if (!item || item.kind !== 'trigger') return;
      const existing = labelFor(item.id);
      if (!Number.isFinite(Number(existing.adjusted_time_s))) return;
      state.session.review.trigger_labels[item.id] = {
        ...existing,
        time_s: item.time_s,
        original_time_s: existing.original_time_s ?? item.time_s,
        adjusted_time_s: undefined,
        event_index: item.event_index,
        updated_at: new Date().toISOString()
      };
      delete state.session.review.trigger_labels[item.id].adjusted_time_s;
      markDirty();
      renderAll();
    }

    function deleteSelectedManual() {
      const item = currentItem();
      if (!item || item.kind !== 'manual') return;
      state.session.review.manual_markers = state.session.review.manual_markers.filter(m => m.id !== item.id);
      state.selectedId = null;
      markDirty();
      renderAll();
    }

    function playWindow() {
      if (state.loop) {
        const center = clampTime(state.playheadTime);
        state.loopStart = Math.max(0, center - 0.22);
        state.loopEnd = Math.min(state.session.duration_s, center + 0.58);
        setPlayhead(state.loopStart);
      } else {
        state.loopEnd = 0;
        setPlayhead(state.playheadTime);
      }
      audio.play();
    }

    function moveSelection(delta) {
      const items = state.manualOnly ? allItems().filter(item => item.kind === 'manual') : allItems();
      if (!items.length) return;
      let index = items.findIndex(item => item.id === state.selectedId);
      if (index < 0) {
        const playhead = clampTime(state.playheadTime);
        index = items.findIndex(item => itemTime(item) >= playhead);
        if (index < 0) index = items.length - 1;
        if (delta < 0 && index > 0 && itemTime(items[index]) > playhead) index -= 1;
      }
      const next = items[Math.min(items.length - 1, Math.max(0, index + delta))];
      if (next) selectItem(next.id, true);
    }

    async function saveLabels() {
      if (!state.session) return;
      const payload = {
        session_id: state.session.session_id,
        expected_count: state.session.expected_count,
        reported_app_count: state.session.reported_app_count,
        manual_only: state.session.manual_only,
        trigger_labels: state.session.review.trigger_labels,
        manual_markers: state.session.review.manual_markers
      };
      saveStatus.textContent = 'Saving...';
      const res = await fetch('/api/labels', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      if (!res.ok) {
        saveStatus.textContent = 'Save failed';
        return;
      }
      state.dirty = false;
      const saved = await res.json();
      saveStatus.textContent = 'Saved ' + new Date(saved.saved_at).toLocaleTimeString();
    }

    function exportLabels() {
      const payload = {
        session_id: state.session.session_id,
        source_wav: state.session.wav_file,
        expected_count: state.session.expected_count,
        reported_app_count: state.session.reported_app_count,
        manual_only: state.session.manual_only,
        trigger_labels: state.session.review.trigger_labels,
        manual_markers: state.session.review.manual_markers,
        exported_at: new Date().toISOString()
      };
      const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = state.session.session_id + '_review_labels.json';
      a.click();
      URL.revokeObjectURL(url);
    }

    function renderAll() {
      renderMetrics();
      renderDetails();
      renderTable();
      drawWaveform();
    }

    async function init() {
      const res = await fetch('/api/session');
      state.session = await res.json();
      state.manualOnly = Boolean(state.session.manual_only);
      state.selectedId = null;
      state.zoom = 1;
      state.viewStart = 0;
      state.playheadTime = 0;
      document.body.classList.toggle('manual-only', state.manualOnly);
      document.getElementById('playBtn').textContent = 'Play';
      document.getElementById('triggerTitle').textContent = state.manualOnly
        ? 'Reference Triggers + Manual Labels'
        : 'Trigger Candidates';
      document.getElementById('sourceLine').textContent =
        `${state.session.session_id} | ${state.session.duration_s.toFixed(3)} s | labels: ${state.session.labels_file}`;
      document.getElementById('gateNote').textContent = state.session.gate_note || '';
      saveStatus.textContent = 'Loaded';
      renderAll();
    }

    document.querySelectorAll('[data-label]').forEach(btn => {
      btn.addEventListener('click', () => setLabel(btn.dataset.label));
    });
    document.querySelectorAll('[data-filter]').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('[data-filter]').forEach(x => x.classList.remove('active'));
        btn.classList.add('active');
        state.filter = btn.dataset.filter;
        renderTable();
      });
    });
    document.getElementById('clearLabelBtn').addEventListener('click', clearLabel);
    document.getElementById('resetTimeBtn').addEventListener('click', resetSelectedTime);
    document.getElementById('noteBox').addEventListener('input', e => updateNote(e.target.value));
    document.getElementById('prevBtn').addEventListener('click', () => moveSelection(-1));
    document.getElementById('nextBtn').addEventListener('click', () => moveSelection(1));
    document.getElementById('playBtn').addEventListener('click', playWindow);
    document.getElementById('speedSelect').addEventListener('change', e => {
      audio.playbackRate = Number(e.target.value) || 1;
    });
    document.getElementById('zoomInBtn').addEventListener('click', () => setZoom(state.zoom * 2));
    document.getElementById('zoomOutBtn').addEventListener('click', () => setZoom(state.zoom / 2));
    document.getElementById('zoomFitBtn').addEventListener('click', () => setZoom(1));
    document.getElementById('panLeftBtn').addEventListener('click', () => panView(-1));
    document.getElementById('panRightBtn').addEventListener('click', () => panView(1));
    document.getElementById('loopBtn').addEventListener('click', () => {
      state.loop = !state.loop;
      document.getElementById('loopBtn').textContent = state.loop ? 'Loop On' : 'Loop Off';
      document.getElementById('loopBtn').classList.toggle('active', state.loop);
    });
    document.getElementById('saveBtn').addEventListener('click', saveLabels);
    document.getElementById('exportBtn').addEventListener('click', exportLabels);
    document.getElementById('addRacketBtn').addEventListener('click', () => addManual('racket'));
    document.getElementById('addRacketTopBtn').addEventListener('click', () => addManual('racket'));
    document.getElementById('addNoiseBtn').addEventListener('click', () => addManual('noise'));
    document.getElementById('addUnclearBtn').addEventListener('click', () => addManual('unclear'));
    document.getElementById('deleteManualBtn').addEventListener('click', deleteSelectedManual);
    document.getElementById('deleteLabelTopBtn').addEventListener('click', deleteSelectedManual);
    canvas.addEventListener('pointerdown', e => {
      if (!state.session) return;
      const time = canvasTime(e);
      const nearest = nearestItemAtTime(time);
      if (nearest && nearest.delta <= 0.3) {
        state.selectedId = nearest.item.id;
        setPlayhead(itemTime(nearest.item));
        state.drag = {
          id: nearest.item.id,
          pointerId: e.pointerId,
          startX: e.clientX,
          startY: e.clientY,
          moved: false
        };
        canvas.setPointerCapture(e.pointerId);
      } else {
        state.selectedId = null;
        setPlayhead(time);
        state.loopEnd = 0;
      }
      renderAll();
    });
    canvas.addEventListener('pointermove', e => {
      if (!state.drag || state.drag.pointerId !== e.pointerId) return;
      const movedEnough = Math.abs(e.clientX - state.drag.startX) > 3 || Math.abs(e.clientY - state.drag.startY) > 3;
      if (!state.drag.moved && !movedEnough) return;
      state.drag.moved = true;
      const item = currentItem();
      if (!item) return;
      const time = canvasTime(e);
      setItemTime(item, time);
      setPlayhead(time);
      markUnsavedDuringDrag();
      renderAll();
      e.preventDefault();
    });
    canvas.addEventListener('pointerup', e => {
      if (!state.drag || state.drag.pointerId !== e.pointerId) return;
      const wasMoved = state.drag.moved;
      state.drag = null;
      canvas.releasePointerCapture(e.pointerId);
      if (wasMoved) markDirty();
      renderAll();
    });
    canvas.addEventListener('pointercancel', e => {
      if (!state.drag || state.drag.pointerId !== e.pointerId) return;
      const wasMoved = state.drag.moved;
      state.drag = null;
      canvas.releasePointerCapture(e.pointerId);
      if (wasMoved) markDirty();
      renderAll();
    });
    audio.addEventListener('timeupdate', () => {
      syncPlayheadFromAudio();
      if (state.loopEnd && state.playheadTime >= state.loopEnd) {
        if (state.loop) setPlayhead(state.loopStart);
        else {
          audio.pause();
          state.loopEnd = 0;
        }
      }
    });
    audio.addEventListener('seeked', () => syncPlayheadFromAudio(true));
    audio.addEventListener('playing', () => syncPlayheadFromAudio(true));
    window.addEventListener('resize', drawWaveform);
    window.addEventListener('keydown', e => {
      const tag = (e.target?.tagName || '').toLowerCase();
      if (tag === 'textarea' || tag === 'input' || tag === 'select') return;
      if (e.code === 'Space') {
        e.preventDefault();
        if (audio.paused) playWindow();
        else audio.pause();
      } else if (e.key.toLowerCase() === 'r') {
        addManual('racket');
      } else if (e.key === 'Delete' || e.key === 'Backspace') {
        deleteSelectedManual();
      } else if (e.key === '+' || e.key === '=') {
        setZoom(state.zoom * 2);
      } else if (e.key === '-' || e.key === '_') {
        setZoom(state.zoom / 2);
      }
    });
    window.addEventListener('beforeunload', e => {
      if (!state.dirty) return;
      e.preventDefault();
      e.returnValue = '';
    });
    init().catch(err => {
      saveStatus.textContent = 'Load failed';
      document.getElementById('sourceLine').textContent = String(err);
    });
  </script>
</body>
</html>
"""


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        out = float(value)
    except Exception:
        return default
    return out if out == out else default


def read_wav_peaks(path: Path, bins: int = 1800) -> tuple[list[list[float]], float, int]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
        frame_count = wav.getnframes()
    if channels != 1 or width != 2:
        raise ValueError(f"Expected mono 16-bit WAV, got channels={channels}, width={width}")
    samples = array("h")
    samples.frombytes(frames)
    if sys.byteorder != "little":
        samples.byteswap()
    if not samples:
        return [], 0.0, sample_rate
    step = max(1, len(samples) // bins)
    peaks: list[list[float]] = []
    for start in range(0, len(samples), step):
        chunk = samples[start:start + step]
        if not chunk:
            continue
        peaks.append([min(chunk) / 32768.0, max(chunk) / 32768.0])
    return peaks, frame_count / float(sample_rate), sample_rate


def load_review() -> dict[str, Any]:
    if LABELS_PATH.exists():
        data = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
        data.setdefault("trigger_labels", {})
        data.setdefault("manual_markers", [])
        return data
    return {
        "session_id": SESSION_ID,
        "expected_count": EXPECTED_COUNT,
        "reported_app_count": REPORTED_APP_COUNT,
        "trigger_labels": {},
        "manual_markers": [],
    }


def nearest_trigger_delta(time_ms: float | None, offline_times: list[float]) -> float | None:
    if time_ms is None or not offline_times:
        return None
    nearest = min(offline_times, key=lambda value: abs(value - time_ms))
    return nearest - time_ms


def build_replay_csv_triggers(path: Path) -> list[dict[str, Any]]:
    rows = read_csv(path)
    triggers: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        event_index = str(row.get("trigger_index") or row.get("event_index") or index)
        time_ms = (
            safe_float(row.get("onset_ms"))
            or safe_float(row.get("onset_ms_wav"))
            or safe_float(row.get("estimated_wav_ms"))
            or 0.0
        )
        prob_racket = row.get("prob_racket_bounce") or row.get("prob_racket") or row.get("model_confidence") or ""
        prob_noise = row.get("prob_noise") or ""
        model_label = row.get("model_label") or row.get("prediction") or "replay_candidate"
        reject_reason = row.get("reject_reason") or ("counted" if str(row.get("counted")).lower() == "true" else "")
        triggers.append(
            {
                "id": f"replay_{event_index}",
                "event_index": event_index,
                "time_s": round(float(time_ms) / 1000.0, 6),
                "model_label": model_label,
                "model_confidence": prob_racket,
                "reject_reason": reject_reason,
                "native_rms": row.get("frame_rms") or row.get("native_rms") or "",
                "native_background_rms": row.get("background_rms") or row.get("native_background_rms") or "",
                "prob_racket_bounce": prob_racket,
                "prob_noise": prob_noise,
                "nearest_offline_trigger_delta_ms": "",
                "received_minus_onset_ms": "",
            }
        )
    triggers.sort(key=lambda item: item["time_s"])
    return triggers


def build_t0052_saved_event_triggers(saved_events_path: Path, offline_path: Path) -> tuple[list[dict[str, Any]], int]:
    saved_events = read_csv(saved_events_path)
    offline_rows = read_csv(offline_path)
    offline_times = [
        value for value in (safe_float(row.get("onset_ms_wav")) for row in offline_rows) if value is not None
    ]
    triggers: list[dict[str, Any]] = []
    for row in saved_events:
        event_index = str(row.get("event_index") or "")
        time_ms = safe_float(row.get("estimated_wav_ms"))
        if time_ms is None:
            rel_ms = safe_float(row.get("event_rel_ms_json_start"), 0.0) or 0.0
            time_ms = rel_ms / 1000.0
        delta = safe_float(row.get("nearest_offline_trigger_delta_ms"))
        if delta is None:
            delta = nearest_trigger_delta(time_ms, offline_times)
        triggers.append(
            {
                "id": f"event_{event_index}",
                "event_index": event_index,
                "time_s": round(float(time_ms) / 1000.0, 6),
                "model_label": row.get("model_label") or "-",
                "model_confidence": row.get("model_confidence") or "",
                "reject_reason": row.get("reject_reason") or "",
                "native_rms": row.get("native_rms") or "",
                "native_background_rms": row.get("native_background_rms") or "",
                "prob_racket_bounce": row.get("prob_racket_bounce") or "",
                "prob_noise": row.get("prob_noise") or "",
                "nearest_offline_trigger_delta_ms": "" if delta is None else round(float(delta), 3),
                "received_minus_onset_ms": row.get("received_minus_onset_ms") or "",
            }
        )
    triggers.sort(key=lambda item: item["time_s"])
    return triggers, len(offline_rows)


def build_payload() -> dict[str, Any]:
    wav_path = RAW_DIR / f"{SESSION_ID}.wav"
    json_path = RAW_DIR / f"{SESSION_ID}.json"
    summary_path = EVAL_DIR / "t0057_summary.json"
    if not summary_path.exists():
        summary_path = EVAL_DIR / "t0052_summary.json"
    for path in (wav_path, json_path):
        if not path.exists():
            raise FileNotFoundError(path)

    waveform, duration_s, sample_rate = read_wav_peaks(wav_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    if TRIGGER_CSV is not None:
        if not TRIGGER_CSV.exists():
            raise FileNotFoundError(TRIGGER_CSV)
        triggers = build_replay_csv_triggers(TRIGGER_CSV)
        offline_trigger_count = len(triggers)
        trigger_source = str(TRIGGER_CSV)
    else:
        saved_events_path = EVAL_DIR / "t0052_saved_json_events.csv"
        offline_path = EVAL_DIR / "t0052_offline_full_wav_triggers.csv"
        for path in (saved_events_path, offline_path):
            if not path.exists():
                raise FileNotFoundError(path)
        triggers, offline_trigger_count = build_t0052_saved_event_triggers(saved_events_path, offline_path)
        trigger_source = str(saved_events_path)

    return {
        "session_id": SESSION_ID,
        "wav_file": str(wav_path),
        "json_file": str(json_path),
        "labels_file": str(LABELS_PATH),
        "trigger_source": trigger_source,
        "gate_note": f"{len(triggers)} trigger candidates loaded from {Path(trigger_source).name}. {GATE_NOTE}",
        "manual_only": MANUAL_ONLY,
        "expected_count": EXPECTED_COUNT,
        "reported_app_count": REPORTED_APP_COUNT,
        "duration_s": duration_s,
        "sample_rate_hz": sample_rate,
        "waveform": waveform,
        "triggers": triggers,
        "summary": summary,
        "offline_trigger_count": offline_trigger_count,
        "review": load_review(),
    }


def save_review(payload: dict[str, Any]) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    review = {
        "session_id": SESSION_ID,
        "source_wav": str(RAW_DIR / f"{SESSION_ID}.wav"),
        "source_json": str(RAW_DIR / f"{SESSION_ID}.json"),
        "expected_count": payload.get("expected_count", EXPECTED_COUNT),
        "reported_app_count": payload.get("reported_app_count", REPORTED_APP_COUNT),
        "manual_only": bool(payload.get("manual_only", MANUAL_ONLY)),
        "trigger_labels": payload.get("trigger_labels") or {},
        "manual_markers": payload.get("manual_markers") or [],
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    LABELS_PATH.write_text(json.dumps(review, indent=2), encoding="utf-8")
    return {"ok": True, "saved_at": review["saved_at"], "path": str(LABELS_PATH)}


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT_DIR / path


def configure(args: argparse.Namespace) -> None:
    global SESSION_ID, RAW_DIR, EVAL_DIR, OUT_DIR, LABELS_PATH, EXPECTED_COUNT, REPORTED_APP_COUNT, TRIGGER_CSV, MANUAL_ONLY, GATE_NOTE
    SESSION_ID = args.session_id
    RAW_DIR = project_path(args.raw_dir)
    EVAL_DIR = project_path(args.eval_dir)
    OUT_DIR = project_path(args.out_dir)
    LABELS_PATH = OUT_DIR / f"{SESSION_ID}_review_labels.json"
    EXPECTED_COUNT = args.expected_count
    REPORTED_APP_COUNT = args.reported_app_count
    TRIGGER_CSV = project_path(args.trigger_csv) if args.trigger_csv else None
    MANUAL_ONLY = bool(args.manual_only)
    GATE_NOTE = args.gate_note


class Handler(BaseHTTPRequestHandler):
    server_version = "T0053TriggerReview/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        print("%s - %s" % (self.address_string(), fmt % args))

    def send_bytes(self, data: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def send_audio_bytes(self, data: bytes, content_type: str) -> None:
        size = len(data)
        range_header = self.headers.get("Range")
        if not range_header or not range_header.startswith("bytes="):
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(size))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
            return

        raw_range = range_header.removeprefix("bytes=").split(",", 1)[0].strip()
        start_text, _, end_text = raw_range.partition("-")
        try:
            if start_text:
                start = int(start_text)
                end = int(end_text) if end_text else size - 1
            else:
                suffix_len = int(end_text)
                start = max(0, size - suffix_len)
                end = size - 1
            start = max(0, min(start, size - 1))
            end = max(start, min(end, size - 1))
        except Exception:
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            return

        chunk = data[start:end + 1]
        self.send_response(HTTPStatus.PARTIAL_CONTENT)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(chunk)))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(chunk)

    def send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_bytes(json.dumps(payload).encode("utf-8"), "application/json", status)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        try:
            if path in {"/", "/index.html"}:
                self.send_bytes(HTML.encode("utf-8"), "text/html; charset=utf-8")
                return
            if path == "/api/session":
                self.send_json(build_payload())
                return
            if path == "/audio.wav":
                wav_path = RAW_DIR / f"{SESSION_ID}.wav"
                data = wav_path.read_bytes()
                self.send_audio_bytes(data, mimetypes.types_map.get(".wav", "audio/wav"))
                return
            self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/api/labels":
            self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length") or "0")
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8"))
            self.send_json(save_review(payload))
        except Exception as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the T0053 Fable trigger review UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--session-id", default=SESSION_ID)
    parser.add_argument("--raw-dir", default=str(RAW_DIR))
    parser.add_argument("--eval-dir", default=str(EVAL_DIR))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--trigger-csv", default="")
    parser.add_argument("--manual-only", action="store_true")
    parser.add_argument("--expected-count", type=int, default=EXPECTED_COUNT)
    parser.add_argument("--reported-app-count", type=int, default=REPORTED_APP_COUNT)
    parser.add_argument(
        "--gate-note",
        default=GATE_NOTE,
        help="Short note shown above the waveform to explain which trigger source is being reviewed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure(args)
    # Build once at startup so missing data fails before opening the browser.
    payload = build_payload()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loaded {len(payload['triggers'])} trigger candidates from {payload['session_id']}")
    print(f"Labels will save to {LABELS_PATH}")
    print(f"Open http://{args.host}:{args.port}/")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
