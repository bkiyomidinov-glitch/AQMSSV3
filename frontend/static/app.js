const API_URL = `${window.location.origin}/api`;
const SESSION_ID = `site_${Math.random().toString(36).slice(2, 9)}`;
let markets = [];
let current = null;
let chart;
let candleSeries;
let emaSeries;
let activeFilter = 'all';
let activeInterval = '15';
let showEma = true;
const $ = (id) => document.getElementById(id);
const price = (value) => Number(value || 0).toFixed(Number(value) > 100 ? 2 : 5);

function candlesForFallback(market) {
    const base = Number(market?.price || 1);
    const data = [];
    let close = base * 0.985;
    const step = base > 100 ? base * 0.001 : base * 0.0007;
    const now = Math.floor(Date.now() / 1000);
    for (let index = 0; index < 100; index += 1) {
        const wave = Math.sin(index * 0.28) * step * 1.6 + ((index % 9) - 4) * step * 0.08;
        const open = close;
        close = Math.max(0.0001, open + wave);
        data.push({ time: now - (100 - index) * 900, open, high: Math.max(open, close) + step * .5, low: Math.min(open, close) - step * .5, close });
    }
    const factor = base / data[data.length - 1].close;
    return data.map((candle) => ({ ...candle, open: candle.open * factor, high: candle.high * factor, low: candle.low * factor, close: candle.close * factor }));
}

function average(data, period = 20) {
    return data.map((item, index) => {
        const slice = data.slice(Math.max(0, index - period + 1), index + 1);
        return { time: item.time, value: slice.reduce((sum, value) => sum + value.close, 0) / slice.length };
    });
}

function drawChart(data) {
    const container = $('price-chart');
    if (!container || !window.LightweightCharts) return;
    if (chart) chart.remove();
    chart = LightweightCharts.createChart(container, { layout: { background: { color: '#101010' }, textColor: '#777', fontFamily: 'DM Mono' }, grid: { vertLines: { color: '#1b1b1b' }, horzLines: { color: '#1b1b1b' } }, rightPriceScale: { borderColor: '#333' }, timeScale: { borderColor: '#333', timeVisible: true }, crosshair: { mode: LightweightCharts.CrosshairMode.Normal } });
    candleSeries = chart.addCandlestickSeries({ upColor: '#fbfbf8', downColor: '#666', borderVisible: false, wickUpColor: '#fbfbf8', wickDownColor: '#777' });
    candleSeries.setData(data);
    emaSeries = chart.addLineSeries({ color: '#6eb9ff', lineWidth: 1, priceLineVisible: false, lastValueVisible: false });
    emaSeries.setData(average(data));
    chart.timeScale().fitContent();
}

function renderMarkets() {
    const filtered = markets.filter((market) => {
        const momentum = String(market.momentum || '').toLowerCase();
        return activeFilter === 'all' || (activeFilter === 'bullish' && momentum.includes('bull')) || (activeFilter === 'bearish' && momentum.includes('bear'));
    });
    $('terminal-markets').innerHTML = filtered.map((market) => `<div class="terminal-market ${current?.symbol === market.symbol ? 'selected' : ''}" data-symbol="${market.symbol}"><strong>${market.symbol}</strong><span class="${Number(market.change) >= 0 ? 'positive' : 'negative'}">${Number(market.change) >= 0 ? '+' : ''}${Number(market.change || 0).toFixed(2)}%</span><small>${market.name || 'Market'} · Q ${Number(market.quality_score || 0).toFixed(0)}</small></div>`).join('');
    document.querySelectorAll('.terminal-market').forEach((row) => row.addEventListener('click', () => chooseMarket(row.dataset.symbol)));
}

async function loadChart(symbol) {
    try {
        const response = await axios.get(`${API_URL}/chart/${symbol}?interval=${activeInterval}`);
        drawChart(response.data.data);
        $('chart-status').textContent = `Live · ${response.data.data.length} свечей`;
    } catch (error) {
        drawChart(candlesForFallback(current));
        $('chart-status').textContent = 'Fallback data · feed unavailable';
    }
}

function chooseMarket(symbol) {
    current = markets.find((market) => market.symbol === symbol) || markets[0];
    if (!current) return;
    $('terminal-symbol').textContent = current.symbol;
    $('terminal-price').textContent = price(current.price);
    const change = Number(current.change || 0);
    $('terminal-change').textContent = `${change >= 0 ? '+' : ''}${change.toFixed(2)}%`;
    $('terminal-change').className = change >= 0 ? 'positive' : 'negative';
    renderMarkets();
    loadChart(current.symbol);
}

async function loadData() {
    try {
        const [marketsResponse, aqmssResponse] = await Promise.all([axios.get(`${API_URL}/markets`), axios.get(`${API_URL}/aqmss`)]);
        markets = marketsResponse.data.data || [];
        const score = aqmssResponse.data.data?.current;
        if (score) $('ticker-core').textContent = `${Number(score.total_score || 0).toFixed(2)}/10`;
        const eur = markets.find((market) => market.symbol === 'EURUSD');
        const btc = markets.find((market) => market.symbol === 'BTCUSD');
        if (eur) { $('ticker-eurusd').textContent = price(eur.price); $('ticker-eurusd-change').textContent = `${Number(eur.change) >= 0 ? '+' : ''}${Number(eur.change || 0).toFixed(2)}%`; }
        if (btc) { $('ticker-btc').textContent = price(btc.price); $('ticker-btc-change').textContent = `${Number(btc.change) >= 0 ? '+' : ''}${Number(btc.change || 0).toFixed(2)}%`; }
        renderMarkets();
        chooseMarket('EURUSD');
    } catch (error) { $('terminal-markets').innerHTML = '<p>Market data unavailable.</p>'; }
}

async function sendCopilot() {
    const input = $('mini-input');
    const message = input.value.trim();
    if (!message) return;
    input.value = '';
    $('mini-chat').insertAdjacentHTML('beforeend', `<div class="mini-message user"><span class="message-avatar">AQ</span><span>${message}</span></div>`);
    $('typing-state').classList.add('visible');
    $('mini-chat').scrollTop = $('mini-chat').scrollHeight;
    try {
        const response = await axios.post(`${API_URL}/ai/chat`, { message, session_id: SESSION_ID });
        const reply = response.data.data?.ai_response || 'Нет ответа.';
        $('mini-chat').insertAdjacentHTML('beforeend', `<div class="mini-message ai"><span class="message-avatar">✦</span><span>${reply.replace(/\n/g, '<br>')}</span></div>`);
    } catch (error) { $('mini-chat').insertAdjacentHTML('beforeend', '<div class="mini-message ai"><span class="message-avatar">!</span><span>AI временно недоступен.</span></div>'); }
    $('typing-state').classList.remove('visible');
    $('mini-chat').scrollTop = $('mini-chat').scrollHeight;
}

function openChatWindow() {
    const popup = window.open('/copilot', 'aqmss-copilot', 'width=520,height=720,resizable=yes,scrollbars=yes');
    if (!popup) return;
    popup.focus();
}

function openTerminalWindow() {
    const popup = window.open(`${window.location.origin}/#terminal`, 'aqmss-terminal', 'width=1440,height=900,resizable=yes,scrollbars=yes');
    if (popup) popup.focus();
}

function showUploadedImage(event) {
    const file = event.target.files[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    $('upload-name').textContent = file.name;
    $('mini-chat').insertAdjacentHTML('beforeend', `<div class="mini-message user"><span class="message-avatar">AQ</span><span><img class="uploaded-chat-image" src="${url}" alt="Загруженный график">Изображение загружено. Опишите, что именно нужно проверить.</span></div>`);
    $('mini-chat').scrollTop = $('mini-chat').scrollHeight;
}

function clearCopilot() {
    $('mini-chat').innerHTML = '<div class="mini-message ai"><span class="message-avatar">✦</span><span>Диалог очищен. Я готов анализировать рынок.</span></div>';
    $('upload-name').textContent = '';
}

document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('[data-terminal-filter]').forEach((button) => button.addEventListener('click', () => { document.querySelectorAll('[data-terminal-filter]').forEach((item) => item.classList.remove('active')); button.classList.add('active'); activeFilter = button.dataset.terminalFilter; renderMarkets(); }));
    $('mini-send').addEventListener('click', sendCopilot);
    $('mini-input').addEventListener('keydown', (event) => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); sendCopilot(); } });
    document.querySelectorAll('[data-prompt]').forEach((button) => button.addEventListener('click', () => { $('mini-input').value = button.dataset.prompt; sendCopilot(); }));
    $('upload-chat').addEventListener('click', () => $('chat-image').click());
    $('chat-image').addEventListener('change', showUploadedImage);
    $('clear-chat').addEventListener('click', clearCopilot);
    $('popout-chat').addEventListener('click', openChatWindow);
    $('open-copilot-section').addEventListener('click', openChatWindow);
    document.querySelectorAll('[data-interval]').forEach((button) => button.addEventListener('click', () => { document.querySelectorAll('[data-interval]').forEach((item) => item.classList.remove('active')); button.classList.add('active'); activeInterval = button.dataset.interval; $('chart-caption').textContent = `${current?.symbol || 'EURUSD'} · ${button.textContent}`; loadChart(current?.symbol || 'EURUSD'); }));
    $('terminal-refresh').addEventListener('click', () => loadData());
    $('terminal-indicator').addEventListener('click', () => { showEma = !showEma; $('terminal-indicator').textContent = `${showEma ? '＋' : '−'} EMA 20`; if (emaSeries) emaSeries.applyOptions({ visible: showEma }); });
    $('terminal-fullscreen').addEventListener('click', () => $('live-chart-panel').requestFullscreen?.());
    loadData();
});
