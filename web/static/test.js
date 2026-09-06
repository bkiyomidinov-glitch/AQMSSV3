let chart;
let candles = [];
let summary = '';
let aqmssSummary = '';

const $ = (id) => document.getElementById(id);

function parseCsv(text) {
    const lines = text.trim().split(/\r?\n/).filter(Boolean);
    if (!lines.length) throw new Error('CSV is empty');

    const first = lines[0].split(',').map((value) => value.trim());
    const headerless = Number.isFinite(Number(first[1])) && Number.isFinite(Number(first[4]));
    const headers = headerless
        ? ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'volume2', 'other']
        : first.map((value) => value.toLowerCase());
    const rows = headerless ? lines : lines.slice(1);
    const index = (names) => names.map((name) => headers.indexOf(name)).find((value) => value >= 0);
    const timeIndex = index(['timestamp', 'datetime', 'date', 'time']);
    const closeIndex = index(['close', 'price', 'last']);

    if (closeIndex === undefined) throw new Error('CSV needs a close or price column');

    return rows.map((line, rowIndex) => {
        const values = line.split(',').map((value) => value.trim());
        const close = Number(values[closeIndex]);
        if (!Number.isFinite(close)) return null;

        const readNumber = (names, fallback) => {
            const column = index(names);
            const value = column === undefined ? NaN : Number(values[column]);
            return Number.isFinite(value) ? value : fallback;
        };
        const parsedTime = timeIndex === undefined
            ? 0
            : new Date(values[timeIndex].replace(/\./g, '-')).getTime() / 1000;

        return {
            time: parsedTime || Math.floor(Date.now() / 1000) - (rows.length - rowIndex) * 3600,
            open: readNumber(['open'], close),
            high: readNumber(['high'], close),
            low: readNumber(['low'], close),
            close,
            volume: readNumber(['volume', 'vol'], 0)
        };
    }).filter(Boolean).sort((a, b) => a.time - b.time);
}

function draw() {
    if (chart) chart.remove();
    chart = LightweightCharts.createChart($('test-chart'), {
        layout: { background: { color: '#11110f' }, textColor: '#858580', fontFamily: 'DM Mono' },
        grid: { vertLines: { color: '#252522' }, horzLines: { color: '#252522' } },
        rightPriceScale: { borderColor: '#383836' },
        timeScale: { borderColor: '#383836', timeVisible: true }
    });

    const series = chart.addCandlestickSeries({
        upColor: '#f3f2ee',
        downColor: '#777',
        borderVisible: false,
        wickUpColor: '#f3f2ee',
        wickDownColor: '#777'
    });
    series.setData(candles);
    chart.timeScale().fitContent();
}

function csvAqmss(data) {
    const ranges = data.map((item) => item.high - item.low).filter((value) => value > 0);
    const atr = ranges.slice(-50);
    const meanAtr = ranges.reduce((sum, value) => sum + value, 0) / Math.max(1, ranges.length);
    const currentAtr = atr.reduce((sum, value) => sum + value, 0) / Math.max(1, atr.length);
    const atrStd = Math.sqrt(atr.reduce((sum, value) => sum + (value - currentAtr) ** 2, 0) / Math.max(1, atr.length));
    const recent = data.slice(-10).map((item) => item.volume).filter((value) => value > 0);
    const allVolume = data.map((item) => item.volume).filter((value) => value > 0);
    const avgVolume = allVolume.reduce((sum, value) => sum + value, 0) / Math.max(1, allVolume.length);
    const recentVolume = recent.reduce((sum, value) => sum + value, 0) / Math.max(1, recent.length);
    const volumeCv = recent.length ? Math.sqrt(recent.reduce((sum, value) => sum + (value - recentVolume) ** 2, 0) / recent.length) / recentVolume : 0;
    const liquidity = Math.max(0, Math.min(10, (recentVolume / Math.max(1, avgVolume)) * 6 + (10 - Math.min(10, volumeCv * 10)) * .4));
    const volatilityRegime = Math.max(0, Math.min(10, (currentAtr / Math.max(meanAtr, 1e-9)) * 10));
    const volatilityLevel = Math.max(0, Math.min(10, 10 - (atrStd / Math.max(currentAtr, 1e-9)) * 10));
    const window = data.slice(-20);
    const closeStart = window[0]?.close || data[0].close;
    const closeEnd = window[window.length - 1]?.close || data[data.length - 1].close;
    const momentum = Math.max(0, Math.min(10, ((closeEnd - closeStart) / Math.max(closeStart, 1e-9)) * 40 + 5));
    const mean = window.reduce((sum, item) => sum + item.close, 0) / Math.max(1, window.length);
    const structure = Math.max(0, Math.min(10, 10 - (window.reduce((sum, item) => sum + Math.abs(item.close - mean), 0) / Math.max(1, window.length)) / Math.max(mean * .01, 1e-9) * 2));
    const score = volatilityRegime * .25 + liquidity * .15 + structure * .3 + momentum * .2 + volatilityLevel * .1;
    const condition = score >= 7.5 ? 'HIGH QUALITY MARKET' : score >= 6 ? 'NORMAL MARKET' : score >= 4 ? 'WEAK CONDITIONS' : 'NO TRADE';
    return { score, condition, structure, momentum, liquidity, volatilityRegime, volatilityLevel, probability: Math.max(0, Math.min(1, score / 10)) };
}

function showCsvAqmss(result, asset) {
    $('aqmss-score').textContent = result.score.toFixed(2);
    $('aqmss-condition').textContent = `${asset} · ${result.condition}`;
    $('aqmss-probability').textContent = `${(result.probability * 100).toFixed(1)}%`;
    $('aqmss-analysis-title').textContent = `${asset} AQMSS analysis`;
    $('aqmss-analysis').textContent = `Calculated from uploaded CSV: structure ${result.structure.toFixed(2)}, momentum ${result.momentum.toFixed(2)}, liquidity ${result.liquidity.toFixed(2)}, volatility regime ${result.volatilityRegime.toFixed(2)}.`;
    setFactor('structure', result.structure); setFactor('momentum', result.momentum); setFactor('liquidity', result.liquidity);
    aqmssSummary = `${asset} CSV AQMSS score ${result.score.toFixed(2)}/10 (${result.condition}); structure ${result.structure.toFixed(2)}, momentum ${result.momentum.toFixed(2)}, liquidity ${result.liquidity.toFixed(2)}, volatility regime ${result.volatilityRegime.toFixed(2)}, volatility level ${result.volatilityLevel.toFixed(2)}.`;
}

function loadFile(file) {
    const reader = new FileReader();
    reader.onload = () => {
        try {
            candles = parseCsv(reader.result);
            if (candles.length < 2) throw new Error('Not enough valid rows');

            const first = candles[0];
            const last = candles[candles.length - 1];
            const move = ((last.close - first.close) / first.close) * 100;
            const high = Math.max(...candles.map((candle) => candle.high));
            const low = Math.min(...candles.map((candle) => candle.low));

            summary = `Rows: ${candles.length}; first close: ${first.close}; last close: ${last.close}; move: ${move.toFixed(2)}%; range: ${low} - ${high}.`;
            const asset = file.name.split('_')[0].split('.')[0].toUpperCase();
            const csvScore = csvAqmss(candles);
            showCsvAqmss(csvScore, asset);
            localStorage.setItem('aqmss_uploaded_csv', JSON.stringify({ file: file.name, candles, summary, aqmss: aqmssSummary, csvScore, asset }));
            $('file-name').textContent = file.name;
            $('row-count').textContent = candles.length;
            $('last-close').textContent = last.close;
            $('date-range').textContent = `${new Date(first.time * 1000).toLocaleDateString()} — ${new Date(last.time * 1000).toLocaleDateString()}`;
            $('data-insight').textContent = summary;
            $('empty-state').hidden = true;
            $('workspace').hidden = false;
            draw();
            loadAqmss();
        } catch (error) {
            alert(error.message);
        }
    };
    reader.readAsText(file);
}

async function loadAqmss() {
    try {
        const saved = JSON.parse(localStorage.getItem('aqmss_uploaded_csv') || 'null');
        if (saved?.csvScore) {
            showCsvAqmss(saved.csvScore, saved.asset || 'UPLOADED ASSET');
            return;
        }
        const response = await fetch('/api/aqmss');
        const payload = await response.json();
        const current = payload.data?.current;
        if (localStorage.getItem('aqmss_uploaded_csv')) return;
        if (!current) return;

        const score = Number(current.total_score || 0);
        const probability = current.ai_high_quality_prob == null ? null : Number(current.ai_high_quality_prob);
        $('aqmss-score').textContent = score.toFixed(2);
        $('aqmss-condition').textContent = current.market_condition || 'UNKNOWN';
        $('aqmss-probability').textContent = probability == null ? '--%' : `${(probability * 100).toFixed(1)}%`;
        $('aqmss-analysis-title').textContent = score >= 7.5 ? 'Высококачественный рынок' : score >= 6 ? 'Нормальные условия' : score >= 4 ? 'Слабые условия' : 'Рынок вне сделки';
        $('aqmss-analysis').textContent = `Structure ${Number(current.structure).toFixed(1)}/10, momentum ${Number(current.momentum).toFixed(1)}/10, liquidity ${Number(current.true_liquidity).toFixed(1)}/10. Volatility regime ${Number(current.volatility_regime).toFixed(1)}/10.`;
        aqmssSummary = `AQMSS score ${score.toFixed(2)}/10 (${current.market_condition}); structure ${Number(current.structure).toFixed(1)}, momentum ${Number(current.momentum).toFixed(1)}, liquidity ${Number(current.true_liquidity).toFixed(1)}, volatility regime ${Number(current.volatility_regime).toFixed(1)}.`;
        setFactor('structure', current.structure);
        setFactor('momentum', current.momentum);
        setFactor('liquidity', current.true_liquidity);
    } catch (error) {
        $('aqmss-analysis').textContent = 'AQMSS analysis is currently unavailable.';
    }
}

function setFactor(name, value) {
    const score = Math.max(0, Math.min(10, Number(value || 0)));
    $(`factor-${name}`).style.width = `${score * 10}%`;
    $(`factor-${name}-value`).textContent = score.toFixed(1);
}

async function ask() {
    const input = $('test-prompt');
    const text = input.value.trim();
    if (!text) return;

    input.value = '';
    $('test-messages').insertAdjacentHTML('beforeend', `<div class="test-message user">${text}</div>`);
    $('test-typing').classList.add('visible');

    if (/number|numbers|цифр|покажи|score|оценк/i.test(text) && aqmssSummary) {
        $('test-messages').insertAdjacentHTML('beforeend', `<div class="test-message ai">${summary}<br><br>${aqmssSummary}</div>`);
        $('test-typing').classList.remove('visible');
        return;
    }

    try {
        const response = await fetch('/api/ai/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: `Это анализ загруженного пользователем актива, а не EURUSD. Используй только эти данные и не говори, что у тебя нет метрик: CSV: ${summary} Авторитетная рассчитанная оценка AQMSS для загруженного актива: ${aqmssSummary || 'не загружена'}. Вопрос: ${text}`,
                session_id: 'test-csv'
            })
        });
        const data = await response.json();
        const reply = data.data?.ai_response || 'Нет ответа';
        $('test-messages').insertAdjacentHTML('beforeend', `<div class="test-message ai">${reply.replace(/\n/g, '<br>')}</div>`);
    } catch (error) {
        $('test-messages').insertAdjacentHTML('beforeend', '<div class="test-message ai">Assistant временно недоступен.</div>');
    }
    $('test-typing').classList.remove('visible');
}

function reset() {
    candles = [];
    localStorage.removeItem('aqmss_uploaded_csv');
    $('workspace').hidden = true;
    $('empty-state').hidden = false;
    $('csv-file').value = '';
    $('empty-csv-file').value = '';
}

document.addEventListener('DOMContentLoaded', () => {
    $('csv-file').onchange = (event) => event.target.files[0] && loadFile(event.target.files[0]);
    $('empty-csv-file').onchange = (event) => event.target.files[0] && loadFile(event.target.files[0]);
    $('test-send').onclick = ask;
    $('test-prompt').onkeydown = (event) => {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            ask();
        }
    };
    $('reset-data').onclick = reset;
    loadAqmss();
});
