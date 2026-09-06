const API = '/api';
const session = `assistant_${Math.random().toString(36).slice(2, 9)}`;
const $ = (id) => document.getElementById(id);

function csvContext() {
    const saved = JSON.parse(localStorage.getItem('aqmss_uploaded_csv') || 'null');
    if (!saved) return '';
    return ` Пользователь загрузил CSV ${saved.file}. Сводка: ${saved.summary || 'нет'}. AQMSS: ${saved.aqmss || 'нет'}.`;
}

function addMessage(text, user = false) {
    $('messages').insertAdjacentHTML('beforeend', `<div class="copilot-message ${user ? 'user' : ''}">${user ? '' : '<span class="message-avatar">✦</span>'}<div>${user ? '' : '<small>AQMSS ASSISTANT</small>'}<p>${text.replace(/\n/g, '<br>')}</p></div></div>`);
    $('messages').scrollTop = $('messages').scrollHeight;
}

async function send() {
    const input = $('prompt');
    const text = input.value.trim();
    if (!text) return;
    input.value = '';
    addMessage(text, true);
    $('status').classList.add('visible');
    try {
        const response = await fetch(`${API}/ai/chat`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ message: text + csvContext(), session_id: session }) });
        const data = await response.json();
        addMessage(data.data?.ai_response || 'Нет ответа.');
    } catch (error) {
        addMessage('Не удалось получить ответ от AI.');
    }
    $('status').classList.remove('visible');
}

document.querySelectorAll('[data-prompt]').forEach((button) => { button.onclick = () => { $('prompt').value = button.dataset.prompt; send(); }; });
$('send').onclick = send;
$('prompt').onkeydown = (event) => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); send(); } };
