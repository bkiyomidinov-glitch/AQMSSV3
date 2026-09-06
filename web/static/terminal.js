const API = '/api';
let markets = [];
let selected;
let interval = '15';
let chart;

const $ = (id) => document.getElementById(id);
const fmt = (value) => Number(value || 0).toFixed(Number(value) > 100 ? 2 : 5);

function ema(data, period = 20) {
    return data.map((item, index) => {
        const slice = data.slice(Math.max(0, index - period + 1), index + 1);
        return { time: item.time, value: slice.reduce((sum, point) => sum + point.close, 0) / slice.length };
    });
}

function draw(data) {
    if (chart) chart.remove();
    chart = LightweightCharts.createChart($('price-chart'), { layout: { background: { color: '#12161b' }, textColor: '#808a96', fontFamily: 'DM Mono' }, grid: { vertLines: { color: '#1b2229' }, horzLines: { color: '#1b2229' } }, rightPriceScale: { borderColor: '#2a3038' }, timeScale: { borderColor: '#2a3038', timeVisible: true } });
    const candles = chart.addCandlestickSeries({ upColor: '#edf0f3', downColor: '#68717c', borderVisible: false, wickUpColor: '#edf0f3', wickDownColor: '#68717c' });
    candles.setData(data);
    const line = chart.addLineSeries({ color: '#78baff', lineWidth: 1, priceLineVisible: false, lastValueVisible: false });
    line.setData(ema(data));
    chart.timeScale().fitContent();
}

async function loadChart() {
    const saved = JSON.parse(localStorage.getItem('aqmss_uploaded_csv') || 'null');
    if (saved?.candles?.length) {
        draw(saved.candles);
        $('chart-status').textContent = `TEST CSV · ${saved.candles.length} candles`;
        return;
    }
    try {
        const response = await axios.get(`${API}/chart/${selected.symbol}?interval=${interval}`);
        draw(response.data.data);
        $('chart-status').textContent = `LIVE · ${response.data.data.length} candles`;
    } catch (error) {
        $('chart-status').textContent = 'Live feed unavailable';
    }
}

function render() {
    const filter = document.querySelector('.terminal-filter .active').dataset.filter;
    const list = markets.filter((market) => filter === 'all' || String(market.momentum).toLowerCase().includes(filter.slice(0, -1)));
    $('terminal-markets').innerHTML = list.map((market) => `<div class="market-row ${selected?.symbol === market.symbol ? 'selected' : ''}" data-symbol="${market.symbol}"><strong>${market.symbol}</strong><span class="${Number(market.change) >= 0 ? 'positive' : 'negative'}">${Number(market.change) >= 0 ? '+' : ''}${Number(market.change || 0).toFixed(2)}%</span><small>${market.name || 'Market'} · Quality ${Number(market.quality_score || 0).toFixed(0)}</small></div>`).join('');
    $('market-count').textContent = list.length;
    document.querySelectorAll('.market-row').forEach((row) => { row.onclick = () => pick(row.dataset.symbol); });
}

function pick(symbol) {
    selected = markets.find((market) => market.symbol === symbol) || markets[0];
    if (!selected) return;
    $('active-symbol').textContent = selected.symbol;
    $('active-price').textContent = fmt(selected.price);
    $('active-change').textContent = `${Number(selected.change) >= 0 ? '+' : ''}${Number(selected.change || 0).toFixed(2)}%`;
    render();
    loadChart();
}

async function load() {
    const response = await axios.get(`${API}/markets`);
    markets = response.data.data || [];
    pick('EURUSD');
}

document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('[data-filter]').forEach((button) => { button.onclick = () => { document.querySelectorAll('[data-filter]').forEach((item) => item.classList.remove('active')); button.classList.add('active'); render(); }; });
    document.querySelectorAll('[data-interval]').forEach((button) => { button.onclick = () => { document.querySelectorAll('[data-interval]').forEach((item) => item.classList.remove('active')); button.classList.add('active'); interval = button.dataset.interval; loadChart(); }; });
    $('terminal-refresh').onclick = load;
    $('terminal-fullscreen').onclick = () => $('big-chart').requestFullscreen?.();
    $('market-search').oninput = () => { const query = $('market-search').value.toLowerCase(); document.querySelectorAll('.market-row').forEach((row) => { row.style.display = row.textContent.toLowerCase().includes(query) ? 'grid' : 'none'; }); };
    load();
});
