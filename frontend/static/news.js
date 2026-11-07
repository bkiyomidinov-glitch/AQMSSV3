const API = '/api';
let markets = [];
let activeFilter = 'all';
const $ = (id) => document.getElementById(id);

function getCategory(symbol) {
    if (/BTC|ETH/.test(symbol)) return 'crypto';
    if (/AAPL|TSLA|NVDA|MSFT|GOOGL|AMZN/.test(symbol)) return 'stocks';
    return 'forex';
}

function createStory(market, index) {
    const change = Number(market.change || 0);
    const quality = Number(market.quality_score || 0);
    const direction = String(market.momentum || 'NEUTRAL');
    const tone = direction.includes('BULL') ? 'positive' : direction.includes('BEAR') ? 'negative' : '';
    const time = `${String(10 + index).padStart(2, '0')}:${String(20 + index * 3).padStart(2, '0')}`;
    return `<article class="news-item"><time>${time}</time><div><h3>${market.symbol}: ${market.market_condition || 'Market update'}</h3><p>Quality ${quality.toFixed(0)}/100 · liquidity ${Number(market.liquidity || 0).toFixed(0)} · volatility ${Number(market.volatility || 0).toFixed(1)}% · change ${change >= 0 ? '+' : ''}${change.toFixed(2)}%</p></div><span class="news-tag ${tone}">${direction}</span></article>`;
}

function renderNews() {
    const visible = markets.filter((market) => activeFilter === 'all' || getCategory(market.symbol) === activeFilter);
    $('news-list').innerHTML = visible.length ? visible.slice(0, 8).map(createStory).join('') : '<p>Нет событий для этого фильтра.</p>';
    $('news-updated').textContent = `Обновлено ${new Date().toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' })}`;
}

async function loadNews() {
    try {
        const responses = await Promise.all([fetch(`${API}/markets`), fetch(`${API}/aqmss`)]);
        const [marketsPayload, aqmssPayload] = await Promise.all(responses.map((response) => response.json()));
        markets = Array.isArray(marketsPayload?.data) ? marketsPayload.data : [];
        const current = aqmssPayload?.data?.current;
        const leader = [...markets].sort((a, b) => Number(b.quality_score || 0) - Number(a.quality_score || 0))[0];
        if (leader) {
            $('brief-title').textContent = `${leader.symbol} leads the scan`;
            $('brief-text').textContent = `Quality ${Number(leader.quality_score || 0).toFixed(0)}/100 · ${leader.market_condition || 'Market update'}.`;
        } else if (current) {
            $('brief-title').textContent = `Core score ${Number(current.total_score || 0).toFixed(2)}/10`;
            $('brief-text').textContent = current.market_condition || 'Live market conditions available.';
        }
        renderNews();
    } catch (error) {
        console.error('News load failed', error);
        $('news-list').innerHTML = '<p>Market news unavailable.</p>';
    }
}

document.querySelectorAll('[data-news-filter]').forEach((button) => button.addEventListener('click', () => {
    document.querySelectorAll('[data-news-filter]').forEach((item) => item.classList.remove('active'));
    button.classList.add('active');
    activeFilter = button.dataset.newsFilter;
    renderNews();
}));

loadNews();
setInterval(loadNews, 30000);
