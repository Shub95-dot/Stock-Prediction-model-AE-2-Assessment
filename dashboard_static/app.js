/* ===================================================================
   SOLiGence Barometer Dashboard — app.js
   Handles: ticker switching, signal fetching, Chart.js rendering,
            probability bars, stress tester, and toast notifications.
=================================================================== */

'use strict';

// ── State ──────────────────────────────────────────────────────────────────
const TICKERS = ['MSFT', 'AMZN', 'AMGN', 'NVDA'];
let activeTicker  = 'MSFT';
let mainChart     = null;
let chartMode     = 'price';  // 'price' | 'rsi' | 'macd'
let marketData    = {};        // cached market data per ticker
let lastSignalData = {};       // cached signal per ticker

// ── DOM refs ───────────────────────────────────────────────────────────────
const overlay      = document.getElementById('loading-overlay');
const mainContent  = document.getElementById('main-content');
const loadingTitle = document.getElementById('loading-title');
const loadingSub   = document.getElementById('loading-sub');
const tickerNav    = document.getElementById('ticker-nav');
const btnRefresh   = document.getElementById('btn-refresh');
const toast        = document.getElementById('toast');

// ── Init ───────────────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', async () => {
  buildTickerNav();
  setupChartToggles();
  setupSliders();
  document.getElementById('btn-stress').addEventListener('click', runStressTest);
  btnRefresh.addEventListener('click', () => loadSignal(activeTicker, true));

  await loadSignal(activeTicker, false);
  prefetchTickers();          // background load remaining tickers
});

// ── Ticker Nav ─────────────────────────────────────────────────────────────
function buildTickerNav() {
  tickerNav.innerHTML = '';
  TICKERS.forEach(t => {
    const btn = document.createElement('button');
    btn.className   = 'ticker-btn' + (t === activeTicker ? ' active' : '');
    btn.id          = `btn-ticker-${t}`;
    btn.textContent = t;
    btn.addEventListener('click', () => switchTicker(t));
    tickerNav.appendChild(btn);
  });
}

function updateTickerNav(loadedTickers) {
  TICKERS.forEach(t => {
    const btn = document.getElementById(`btn-ticker-${t}`);
    if (btn && loadedTickers.includes(t)) btn.classList.add('loaded');
  });
}

async function switchTicker(ticker) {
  if (ticker === activeTicker) return;
  activeTicker = ticker;
  document.querySelectorAll('.ticker-btn').forEach(b => b.classList.remove('active'));
  document.getElementById(`btn-ticker-${ticker}`)?.classList.add('active');
  await loadSignal(ticker, false);
}

// ── Load Signal ───────────────────────────────────────────────────────────
async function loadSignal(ticker, refresh) {
  showLoading(`Loading ${ticker}`, 'Fetching live data and running ensemble inference…');

  try {
    // Fetch signals
    const url = `/api/signal/${ticker}` + (refresh ? '?refresh=true' : '');
    const res  = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
    const data = await res.json();
    lastSignalData[ticker] = data;

    // Fetch market chart data
    const mRes = await fetch(`/api/market/${ticker}`);
    if (mRes.ok) {
      marketData[ticker] = await mRes.json();
    }

    hideLoading();
    mainContent.style.display = 'flex';
    mainContent.style.flexDirection = 'column';
    mainContent.style.gap = '20px';

    renderSignals(data);
    renderChart(ticker);
    updateTickerNav(Object.keys(lastSignalData));
    document.getElementById('chart-ticker').textContent = ticker;

    if (refresh) showToast(`${ticker} refreshed successfully`);

  } catch (err) {
    hideLoading();
    // Still show main content if something is cached
    if (Object.keys(lastSignalData).length > 0) {
      mainContent.style.display = 'flex';
    }
    showToast(`Error: ${err.message}`, true);
  }
}

// ── Render Signals ────────────────────────────────────────────────────────
function renderSignals(data) {
  // Last close
  animateText('val-close', `$${data.last_close.toFixed(2)}`);
  document.getElementById('val-date').textContent = data.last_date;

  const horizons = data.horizons;

  ['t1', 't5', 't21', 't63'].forEach(h => {
    const hd   = horizons[h];
    const sig  = hd.signal;  // BUY | HOLD | SELL
    const pred = `$${hd.price_pred.toFixed(2)}`;
    const pct  = hd.pct_change;
    const pctStr = (pct >= 0 ? '+' : '') + pct.toFixed(2) + '%';
    const conf = `${hd.confidence.toFixed(1)}% conf`;

    // Badge
    const badge = document.getElementById(`badge-${h}`);
    badge.textContent = sig;
    badge.className = `signal-badge ${sig}`;
    badge.classList.add('pulse');
    setTimeout(() => badge.classList.remove('pulse'), 600);

    // Meta
    document.getElementById(`pred-${h}`).textContent = `${pred} (${pctStr})`;
    const confEl = document.getElementById(`conf-${h}`);
    confEl.textContent = conf;

    // Probability bars
    setBar(`pbar-${h}`, hd.up_prob);
    document.getElementById(`prob-${h}`).textContent = `${hd.up_prob.toFixed(1)}%`;

    setBar(`cbar-${h}`, hd.confidence);
    document.getElementById(`conf-val-${h}`).textContent = `${hd.confidence.toFixed(1)}%`;
    document.getElementById(`conf-val-${h}`).nextElementSibling;  // update label too

    // Prediction table
    const ptRow = document.getElementById(`pt-${h}`);
    const cells = ptRow.querySelectorAll('span');
    cells[1].textContent = pred;
    cells[2].textContent = pctStr;
    cells[2].style.color = pct >= 0 ? 'var(--green)' : 'var(--red)';
  });
}

function setBar(id, pct) {
  const el = document.getElementById(id);
  if (el) el.style.width = Math.max(0, Math.min(100, pct)) + '%';
}

function animateText(id, text) {
  const el = document.getElementById(id);
  if (!el) return;
  el.classList.remove('pulse');
  void el.offsetWidth;
  el.textContent = text;
  el.classList.add('pulse');
}

// ── Chart ─────────────────────────────────────────────────────────────────
function setupChartToggles() {
  ['price', 'rsi', 'macd'].forEach(mode => {
    const btn = document.getElementById(`toggle-${mode}`);
    if (btn) btn.addEventListener('click', () => {
      chartMode = mode;
      document.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      renderChart(activeTicker);
    });
  });
}

function renderChart(ticker) {
  const data = marketData[ticker];
  if (!data) return;

  const ctx = document.getElementById('main-chart').getContext('2d');

  if (mainChart) { mainChart.destroy(); mainChart = null; }

  const labels = data.dates;
  let datasets, yLabel, yMin, yMax;

  if (chartMode === 'price') {
    const prices = data.close;
    const bbUp   = data.bb_upper;
    const bbLow  = data.bb_lower;
    const pMin   = Math.min(...prices);
    const pMax   = Math.max(...prices);
    const pad    = (pMax - pMin) * 0.1;
    yMin = pMin - pad; yMax = pMax + pad; yLabel = 'Price (USD)';
    datasets = [
      {
        label: 'BB Upper', data: bbUp, borderColor: 'rgba(124,107,255,0.3)',
        borderWidth: 1, borderDash: [4,4], fill: false, pointRadius: 0, tension: 0.3
      },
      {
        label: 'BB Lower', data: bbLow, borderColor: 'rgba(124,107,255,0.3)',
        borderWidth: 1, borderDash: [4,4], fill: '-1',
        backgroundColor: 'rgba(124,107,255,0.05)', pointRadius: 0, tension: 0.3
      },
      {
        label: `${ticker} Close`, data: prices,
        borderColor: '#00d4ff', borderWidth: 2.5,
        backgroundColor: 'rgba(0,212,255,0.08)',
        fill: false, pointRadius: 0, tension: 0.3,
        pointHoverRadius: 4, pointHoverBackgroundColor: '#00d4ff'
      }
    ];
  } else if (chartMode === 'rsi') {
    yMin = 0; yMax = 100; yLabel = 'RSI';
    datasets = [
      {
        label: 'RSI 14', data: data.rsi,
        borderColor: '#ffd32a', borderWidth: 2,
        fill: false, pointRadius: 0, tension: 0.3
      }
    ];
  } else {
    yMin = undefined; yMax = undefined; yLabel = 'MACD';
    datasets = [
      {
        label: 'MACD', data: data.macd,
        borderColor: '#00ff88', borderWidth: 2,
        fill: false, pointRadius: 0, tension: 0.3
      }
    ];
  }

  mainChart = new Chart(ctx, {
    type: 'line',
    data: { labels, datasets },
    options: {
      animation: { duration: 500 },
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: {
          labels: { color: '#8895b0', font: { family: 'Inter', size: 11 }, boxWidth: 24 }
        },
        tooltip: {
          backgroundColor: 'rgba(13,18,32,0.95)',
          borderColor: 'rgba(255,255,255,0.1)', borderWidth: 1,
          titleColor: '#e8edf5', bodyColor: '#8895b0',
          padding: 10,
          callbacks: {
            label: ctx => {
              const v = ctx.parsed.y;
              return ` ${ctx.dataset.label}: ${chartMode === 'price' ? '$' : ''}${v.toFixed(chartMode === 'macd' ? 4 : 2)}`;
            }
          }
        },
      },
      scales: {
        x: {
          ticks: {
            color: '#4e5c72', font: { size: 10 },
            maxTicksLimit: 10,
            maxRotation: 0
          },
          grid: { color: 'rgba(255,255,255,0.03)' }
        },
        y: {
          min: yMin, max: yMax,
          title: { display: false, text: yLabel, color: '#4e5c72' },
          ticks: {
            color: '#4e5c72', font: { size: 10 },
            callback: v => chartMode === 'price' ? `$${v.toFixed(0)}` : v.toFixed(1)
          },
          grid: { color: 'rgba(255,255,255,0.04)' }
        }
      }
    }
  });

  // Add reference lines for RSI
  if (chartMode === 'rsi' && mainChart) {
    mainChart.options.plugins.annotation = {};  // placeholder
  }
}

// ── Sliders ───────────────────────────────────────────────────────────────
function setupSliders() {
  const configs = [
    { id: 'sl-price',  val: 'sv-price',  fmt: v => (v >= 0 ? '+' : '') + v + '%',  min: -30, max: 30 },
    { id: 'sl-volume', val: 'sv-volume', fmt: v => (v >= 0 ? '+' : '') + v + '%',  min: -50, max: 100 },
    { id: 'sl-vix',    val: 'sv-vix',    fmt: v => (v >= 0 ? '+' : '') + v + ' pts', min: -10, max: 30 },
  ];

  configs.forEach(({ id, val, fmt, min, max }) => {
    const slider = document.getElementById(id);
    const display = document.getElementById(val);
    if (!slider || !display) return;

    const updateTrack = () => {
      const pct = ((slider.value - min) / (max - min)) * 100;
      slider.style.setProperty('--pct', pct + '%');
      display.textContent = fmt(Number(slider.value));
    };

    slider.addEventListener('input', updateTrack);
    updateTrack();  // init
  });
}

// ── Stress Test ───────────────────────────────────────────────────────────
async function runStressTest() {
  const btn = document.getElementById('btn-stress');
  const priceShock  = Number(document.getElementById('sl-price').value)  / 100;
  const volumeShock = Number(document.getElementById('sl-volume').value) / 100;
  const vixShock    = Number(document.getElementById('sl-vix').value);

  btn.disabled = true;
  btn.classList.add('loading');
  btn.textContent = 'Running…';

  try {
    const res = await fetch(`/api/whatif/${activeTicker}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ price_shock: priceShock, volume_shock: volumeShock, vix_shock: vixShock })
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    renderStressResults(data.results);

  } catch (err) {
    showToast(`Stress test failed: ${err.message}`, true);
  } finally {
    btn.disabled = false;
    btn.classList.remove('loading');
    btn.innerHTML = `
      <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16">
        <path d="M13 10l-8 5V5l8 5z" fill="currentColor" stroke="none"/>
      </svg>
      Run Stress Test`;
  }
}

function renderStressResults(results) {
  const container = document.getElementById('stress-results');
  container.style.display = 'block';

  ['t1', 't5', 't21', 't63'].forEach(h => {
    const r = results[h];
    if (!r) return;
    const row = document.getElementById(`sr-${h}`);
    const cells = row.querySelectorAll('span');
    const deltaClass = r.delta >= 0 ? 'delta-pos' : 'delta-neg';
    const deltaStr = (r.delta >= 0 ? '+' : '') + `$${r.delta.toFixed(2)} (${r.delta_pct.toFixed(2)}%)`;
    cells[1].textContent = `$${r.base_price.toFixed(2)}`;
    cells[2].textContent = `$${r.shocked_price.toFixed(2)}`;
    cells[3].innerHTML   = `<span class="${deltaClass}">${deltaStr}</span>`;
    cells[4].innerHTML   = `<span class="impact-badge ${r.impact}">${r.impact}</span>`;
  });
}

// ── Background prefetch remaining tickers ────────────────────────────────
async function prefetchTickers() {
  for (const t of TICKERS) {
    if (t === activeTicker) continue;
    try {
      // Just verify model exists; don't load yet (done lazily on click)
      const res = await fetch('/api/tickers');
      if (res.ok) {
        const info = await res.json();
        const loaded = Object.entries(info)
          .filter(([, v]) => v.loaded)
          .map(([k]) => k);
        updateTickerNav(loaded);
      }
    } catch {}
    break;  // only check once
  }
}

// ── Loading helpers ───────────────────────────────────────────────────────
function showLoading(title, sub) {
  loadingTitle.textContent = title || 'Loading…';
  loadingSub.textContent   = sub || '';
  overlay.style.display    = 'flex';
}

function hideLoading() {
  overlay.style.display = 'none';
}

// ── Toast ─────────────────────────────────────────────────────────────────
let toastTimer;
function showToast(msg, isError = false) {
  clearTimeout(toastTimer);
  toast.textContent = msg;
  toast.className   = 'toast show' + (isError ? ' error' : '');
  toastTimer = setTimeout(() => { toast.className = 'toast'; }, 3500);
}
