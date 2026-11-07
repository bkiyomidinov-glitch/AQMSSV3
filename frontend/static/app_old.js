// AQMSS Dashboard JavaScript

const API_URL = 'http://localhost:5000/api';
let markets = [];
let filteredMarkets = [];
let qualityChart = null;
let momentumChart = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
});

function initializeApp() {
    // Setup tab navigation
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            switchTab(link.dataset.tab);
        });
    });

    // Setup sorting
    document.querySelectorAll('.sortable').forEach(header => {
        header.addEventListener('click', () => sortTable(header.dataset.sort));
    });

    // Setup filters
    document.getElementById('search-market').addEventListener('input', filterMarkets);
    document.getElementById('quality-filter').addEventListener('change', filterMarkets);

    // Setup chat
    document.getElementById('chat-input').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });

    // Update time
    updateTime();
    setInterval(updateTime, 1000);

    // Load initial data
    loadMarkets();
    loadDashboardStats();
    loadInsights();
}

function updateTime() {
    const now = new Date();
    document.getElementById('current-time').textContent = now.toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    });
}

async function loadMarkets() {
    try {
        const response = await axios.get(`${API_URL}/markets`);
        if (response.data.success) {
            markets = response.data.data;
            filteredMarkets = [...markets];
            renderMarkets();
        }
    } catch (error) {
        console.error('Error loading markets:', error);
        showError('Failed to load market data');
    }
}

function renderMarkets() {
    const tbody = document.getElementById('table-body');
    tbody.innerHTML = '';

    if (filteredMarkets.length === 0) {
        tbody.innerHTML = '<tr class="loading"><td colspan="10">No markets found</td></tr>';
        return;
    }

    filteredMarkets.forEach(market => {
        const row = document.createElement('tr');
        const changePercent = parseFloat(market.change);
        const changeClass = changePercent > 0 ? 'positive' : changePercent < 0 ? 'negative' : '';
        
        const qualityScore = parseFloat(market.quality_score);
        let qualityClass = 'quality-poor';
        if (qualityScore >= 75) qualityClass = 'quality-excellent';
        else if (qualityScore >= 60) qualityClass = 'quality-good';
        else if (qualityScore >= 45) qualityClass = 'quality-medium';

        const momentum = String(market.momentum).toUpperCase();
        const momentumClass = momentum.includes('BULL') ? 'bullish' : momentum.includes('BEAR') ? 'bearish' : 'neutral';

        row.innerHTML = `
            <td class="symbol-cell">${market.symbol}</td>
            <td class="price">${formatPrice(market.price)}</td>
            <td class="change ${changeClass}">${changePercent > 0 ? '+' : ''}${changePercent.toFixed(2)}%</td>
            <td>${formatVolume(market.volume)}</td>
            <td>${market.liquidity.toFixed(1)}</td>
            <td>${market.volatility.toFixed(2)}</td>
            <td><span class="momentum ${momentumClass}">${momentum}</span></td>
            <td><span class="quality-score ${qualityClass}">${qualityScore.toFixed(1)}</span></td>
            <td>${market.market_condition}</td>
            <td><button class="btn-detail" onclick="showMarketDetail('${market.symbol}')">View</button></td>
        `;
        tbody.appendChild(row);
    });
}

function filterMarkets() {
    const searchTerm = document.getElementById('search-market').value.toLowerCase();
    const qualityFilter = document.getElementById('quality-filter').value;

    filteredMarkets = markets.filter(market => {
        const matchesSearch = market.symbol.toLowerCase().includes(searchTerm) ||
                             market.name.toLowerCase().includes(searchTerm);
        
        let matchesQuality = true;
        if (qualityFilter) {
            const score = market.quality_score;
            if (qualityFilter === 'high' && score <= 75) matchesQuality = false;
            if (qualityFilter === 'medium' && (score > 75 || score < 50)) matchesQuality = false;
            if (qualityFilter === 'low' && score >= 50) matchesQuality = false;
        }

        return matchesSearch && matchesQuality;
    });

    renderMarkets();
}

function sortTable(field) {
    const isAscending = document.querySelector(`[data-sort="${field}"]`).classList.toggle('ascending');
    
    filteredMarkets.sort((a, b) => {
        let aVal = a[field];
        let bVal = b[field];

        if (typeof aVal === 'string') {
            aVal = aVal.toLowerCase();
            bVal = bVal.toLowerCase();
        }

        if (isAscending) {
            return aVal > bVal ? 1 : -1;
        } else {
            return aVal < bVal ? 1 : -1;
        }
    });

    renderMarkets();
}

async function showMarketDetail(symbol) {
    try {
        const response = await axios.get(`${API_URL}/market/${symbol}`);
        if (response.data.success) {
            const market = response.data.data;
            const modal = document.getElementById('marketModal');
            const body = document.getElementById('modal-body');

            const changePercent = parseFloat(market.change);
            const changeClass = changePercent > 0 ? 'positive' : changePercent < 0 ? 'negative' : '';

            body.innerHTML = `
                <h2>${market.symbol}</h2>
                <p>${market.name}</p>
                
                <div class="modal-detail-grid">
                    <div class="detail-item">
                        <div class="detail-label">Price</div>
                        <div class="detail-value">${formatPrice(market.price)}</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">24H Change</div>
                        <div class="detail-value ${changeClass}">${changePercent > 0 ? '+' : ''}${changePercent.toFixed(2)}%</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">Volume</div>
                        <div class="detail-value">${formatVolume(market.volume)}</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">High 24H</div>
                        <div class="detail-value">${formatPrice(market.high_24h)}</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">Low 24H</div>
                        <div class="detail-value">${formatPrice(market.low_24h)}</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">Quality Score</div>
                        <div class="detail-value">${market.quality_score.toFixed(1)}</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">Liquidity</div>
                        <div class="detail-value">${market.liquidity.toFixed(1)}</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">Volatility</div>
                        <div class="detail-value">${market.volatility.toFixed(2)}</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">Momentum</div>
                        <div class="detail-value">${market.momentum}</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">Trend</div>
                        <div class="detail-value">${market.trend}</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">ATR</div>
                        <div class="detail-value">${market.atr.toFixed(4)}</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">Condition</div>
                        <div class="detail-value">${market.market_condition}</div>
                    </div>
                </div>
            `;

            modal.style.display = 'block';
        }
    } catch (error) {
        console.error('Error loading market detail:', error);
        showError('Failed to load market details');
    }
}

function closeModal() {
    document.getElementById('marketModal').style.display = 'none';
}

window.onclick = (event) => {
    const modal = document.getElementById('marketModal');
    if (event.target === modal) {
        modal.style.display = 'none';
    }
};

async function loadDashboardStats() {
    try {
        const response = await axios.get(`${API_URL}/dashboard/stats`);
        if (response.data.success) {
            const stats = response.data.data;
            document.getElementById('stat-total').textContent = stats.total_markets;
            document.getElementById('stat-avg').textContent = stats.avg_quality.toFixed(1) + '%';
            document.getElementById('stat-high').textContent = stats.high_quality_count;
            document.getElementById('stat-bullish').textContent = stats.bullish_count;
            document.getElementById('stat-bearish').textContent = stats.bearish_count;
            document.getElementById('stat-volatile').textContent = stats.volatile_count;
        }
    } catch (error) {
        console.error('Error loading stats:', error);
    }
}

async function loadInsights() {
    try {
        const response = await axios.get(`${API_URL}/ai/insights`);
        if (response.data.success) {
            const insights = response.data.data;
            const container = document.getElementById('insights-container');
            container.innerHTML = '';

            if (insights.length === 0) {
                container.innerHTML = '<p class="loading">No insights available</p>';
                return;
            }

            insights.forEach(insight => {
                const card = document.createElement('div');
                card.className = `insight-card ${insight.type}`;
                card.innerHTML = `
                    <div class="insight-type">${insight.type} <span class="priority-${insight.priority.toLowerCase()}">[${insight.priority}]</span></div>
                    <div class="insight-title">${insight.title}</div>
                    <div class="insight-description">${insight.description}</div>
                `;
                container.appendChild(card);
            });
        }
    } catch (error) {
        console.error('Error loading insights:', error);
    }
}

function switchTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });

    // Show selected tab
    document.getElementById(`${tabName}-tab`).classList.add('active');

    // Update nav links
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.remove('active');
    });
    document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');

    // Initialize charts if analytics tab is opened
    if (tabName === 'charts') {
        setTimeout(() => {
            if (!qualityChart) initializeCharts();
        }, 100);
    }
}

function initializeCharts() {
    if (!markets.length) return;

    // Quality distribution chart
    const qualityCtx = document.getElementById('qualityChart');
    if (qualityCtx) {
        const excellent = markets.filter(m => m.quality_score >= 75).length;
        const good = markets.filter(m => m.quality_score >= 60 && m.quality_score < 75).length;
        const medium = markets.filter(m => m.quality_score >= 45 && m.quality_score < 60).length;
        const poor = markets.filter(m => m.quality_score < 45).length;

        qualityChart = new Chart(qualityCtx, {
            type: 'doughnut',
            data: {
                labels: ['Excellent (>75)', 'Good (60-75)', 'Medium (45-60)', 'Poor (<45)'],
                datasets: [{
                    data: [excellent, good, medium, poor],
                    backgroundColor: ['#26a69a', '#7cb342', '#ff7f0e', '#ef5350'],
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        labels: { color: '#f5f5f5' }
                    }
                }
            }
        });
    }

    // Momentum chart
    const momentumCtx = document.getElementById('momentumChart');
    if (momentumCtx) {
        const bullish = markets.filter(m => String(m.momentum).toUpperCase().includes('BULLISH')).length;
        const bearish = markets.filter(m => String(m.momentum).toUpperCase().includes('BEARISH')).length;
        const neutral = markets.filter(m => String(m.momentum).toUpperCase().includes('NEUTRAL')).length;

        momentumChart = new Chart(momentumCtx, {
            type: 'bar',
            data: {
                labels: ['Bullish', 'Bearish', 'Neutral'],
                datasets: [{
                    label: 'Market Count',
                    data: [bullish, bearish, neutral],
                    backgroundColor: ['#26a69a', '#ef5350', '#9e9e9e'],
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        labels: { color: '#f5f5f5' }
                    }
                },
                scales: {
                    y: {
                        ticks: { color: '#f5f5f5' },
                        grid: { color: '#293548' }
                    },
                    x: {
                        ticks: { color: '#f5f5f5' },
                        grid: { color: '#293548' }
                    }
                }
            }
        });
    }
}

async function sendMessage() {
    const input = document.getElementById('chat-input');
    const message = input.value.trim();

    if (!message) return;

    // Add user message
    const messagesDiv = document.getElementById('chat-messages');
    const userDiv = document.createElement('div');
    userDiv.className = 'chat-message user-message';
    userDiv.innerHTML = `<div class="message-content">${escapeHtml(message)}</div>`;
    messagesDiv.appendChild(userDiv);

    input.value = '';
    messagesDiv.scrollTop = messagesDiv.scrollHeight;

    try {
        const response = await axios.post(`${API_URL}/ai/chat`, { message });
        if (response.data.success) {
            const aiDiv = document.createElement('div');
            aiDiv.className = 'chat-message ai-message';
            aiDiv.innerHTML = `<div class="message-content">${escapeHtml(response.data.data.ai_response)}</div>`;
            messagesDiv.appendChild(aiDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
    } catch (error) {
        console.error('Error sending message:', error);
        const errorDiv = document.createElement('div');
        errorDiv.className = 'chat-message ai-message';
        errorDiv.innerHTML = `<div class="message-content">Sorry, I encountered an error. Please try again.</div>`;
        messagesDiv.appendChild(errorDiv);
    }
}

// Utility functions
function formatPrice(price) {
    if (price >= 1000000) return (price / 1000000).toFixed(2) + 'M';
    if (price >= 1000) return (price / 1000).toFixed(2) + 'K';
    return parseFloat(price).toFixed(2);
}

function formatVolume(volume) {
    if (volume >= 1000000000) return (volume / 1000000000).toFixed(2) + 'B';
    if (volume >= 1000000) return (volume / 1000000).toFixed(2) + 'M';
    if (volume >= 1000) return (volume / 1000).toFixed(2) + 'K';
    return volume.toFixed(0);
}

function escapeHtml(text) {
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return text.replace(/[&<>"']/g, m => map[m]);
}

function showError(message) {
    alert(message);
}
