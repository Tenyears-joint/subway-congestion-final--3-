// API 설정
const API_BASE_URL = 'http://localhost:8000';

// 전역 변수
let hourlyChart = null;
let stationList = [];

// DOM 요소
const stationSearch = document.getElementById('station-search');
const stationSuggestions = document.getElementById('station-suggestions');
const lineSelect = document.getElementById('line-select');
const hourSelect = document.getElementById('hour-select');
const predictBtn = document.getElementById('predict-btn');
const loadHourlyBtn = document.getElementById('load-hourly-btn');
const resultSection = document.getElementById('result-section');
const loading = document.getElementById('loading');
const errorMessage = document.getElementById('error-message');

// 초기화
document.addEventListener('DOMContentLoaded', () => {
    initHourSelect();
    loadStations();
    setupEventListeners();
});

// 시간 선택 초기화
function initHourSelect() {
    for (let hour = 5; hour <= 23; hour++) {
        const option = document.createElement('option');
        option.value = hour;
        option.textContent = `${hour}시`;
        hourSelect.appendChild(option);
    }
    
    // 현재 시간 선택
    const now = new Date();
    const currentHour = now.getHours();
    if (currentHour >= 5 && currentHour <= 23) {
        hourSelect.value = currentHour;
    } else {
        hourSelect.value = 8; // 기본값: 오전 8시
    }
}

// 역 목록 로드
async function loadStations() {
    try {
        const response = await fetch(`${API_BASE_URL}/stations`);
        if (!response.ok) throw new Error('역 목록을 불러올 수 없습니다');
        
        const data = await response.json();
        stationList = data;
    } catch (error) {
        console.error('역 목록 로드 실패:', error);
    }
}

// 이벤트 리스너 설정
function setupEventListeners() {
    // 역 검색 자동완성
    stationSearch.addEventListener('input', handleStationSearch);
    stationSearch.addEventListener('blur', () => {
        setTimeout(() => {
            stationSuggestions.classList.remove('active');
        }, 200);
    });
    
    // 예측 버튼
    predictBtn.addEventListener('click', predictCongestion);
    
    // 24시간 차트 버튼
    loadHourlyBtn.addEventListener('click', loadHourlyData);
    
    // Enter 키 이벤트
    stationSearch.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') predictCongestion();
    });
}

// 역 검색 자동완성 (개선 버전)
function handleStationSearch(e) {
    const query = e.target.value.trim();
    
    // 1글자 미만이면 숨김
    if (query.length < 1) {
        stationSuggestions.classList.remove('active');
        stationSuggestions.innerHTML = '';
        return;
    }
    
    // 역 이름에 검색어가 포함되는 것 필터링
    const filtered = stationList.filter(station => 
        station.station_name.includes(query)
    );
    
    // 중복 제거 (같은 역이 여러 호선에 있을 수 있음)
    const uniqueStations = [];
    const seen = new Set();
    
    for (const station of filtered) {
        const key = `${station.station_name}-${station.line_name}`;
        if (!seen.has(key)) {
            seen.add(key);
            uniqueStations.push(station);
        }
    }
    
    // 최대 15개까지만 표시
    const limitedStations = uniqueStations.slice(0, 15);
    
    if (limitedStations.length > 0) {
        stationSuggestions.innerHTML = limitedStations.map(station => `
            <div class="suggestion-item" 
                 data-station="${station.station_name}" 
                 data-line="${station.line_name}"
                 onmousedown="selectStation('${station.station_name}', '${station.line_name}')">
                <strong>${highlightMatch(station.station_name, query)}</strong> 
                <span class="line-badge">${station.line_name}</span>
            </div>
        `).join('');
        
        stationSuggestions.classList.add('active');
    } else {
        // 검색 결과 없음
        stationSuggestions.innerHTML = `
            <div class="suggestion-item no-result">
                검색 결과가 없습니다
            </div>
        `;
        stationSuggestions.classList.add('active');
    }
}

// 역 선택 함수
function selectStation(stationName, lineName) {
    stationSearch.value = stationName;
    lineSelect.value = lineName;
    stationSuggestions.classList.remove('active');
    stationSuggestions.innerHTML = '';
}

// 검색어 하이라이트
function highlightMatch(text, query) {
    if (!query) return text;
    
    const index = text.indexOf(query);
    if (index === -1) return text;
    
    return text.substring(0, index) + 
           '<mark>' + query + '</mark>' + 
           text.substring(index + query.length);
}

// 혼잡도 예측
async function predictCongestion() {
    const station = stationSearch.value.trim();
    const line = lineSelect.value;
    const hour = parseInt(hourSelect.value);
    
    // 유효성 검사
    if (!station) {
        showError('지하철역을 입력해주세요');
        return;
    }
    
    if (!line) {
        showError('호선을 선택해주세요');
        return;
    }
    
    if (!hour) {
        showError('시간을 선택해주세요');
        return;
    }
    
    // 로딩 표시
    showLoading();
    
    try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                station_name: station,
                line_name: line,
                hour: hour
            })
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || '예측에 실패했습니다');
        }
        
        const data = await response.json();
        displayResult(data);
        
    } catch (error) {
        console.error('예측 오류:', error);
        showError(error.message);
    } finally {
        hideLoading();
    }
}

// 결과 표시
function displayResult(data) {
    // 결과 섹션 표시
    resultSection.style.display = 'block';
    resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    
    // 역 정보
    document.getElementById('result-station').textContent = 
        `${data.station_name} ${data.line_name}`;
    
    // 시간 정보
    const date = new Date(data.date);
    const hourText = data.hour < 12 ? `오전 ${data.hour}시` : 
                     data.hour === 12 ? `오후 12시` : 
                     `오후 ${data.hour - 12}시`;
    document.getElementById('result-time').textContent = 
        `${date.getFullYear()}년 ${date.getMonth() + 1}월 ${date.getDate()}일 ${hourText}`;
    
    // 혼잡도 레벨
    const congestionLevel = document.getElementById('congestion-level');
    congestionLevel.className = `congestion-level level-${data.congestion_level}`;
    congestionLevel.querySelector('.level-text').textContent = data.congestion_label;
    
    // 아이콘 변경
    const icons = {
        0: '😊',
        1: '🙂',
        2: '😰',
        3: '🚫'
    };
    congestionLevel.querySelector('.level-icon').textContent = icons[data.congestion_level];
    
    // 예상 인원
    document.getElementById('predicted-passengers').textContent = 
        data.predicted_passengers.toLocaleString() + '명';
    
    // 신뢰도
    document.getElementById('confidence').textContent = 
        (data.confidence * 100).toFixed(0) + '%';
    
    // 추천 메시지
    document.getElementById('recommendation').innerHTML = 
        `<p>${data.recommendation}</p>`;
    
    // 차트 초기화
    if (hourlyChart) {
        hourlyChart.destroy();
        hourlyChart = null;
    }
}

// 24시간 혼잡도 로드
async function loadHourlyData() {
    const station = stationSearch.value.trim();
    const line = lineSelect.value;
    
    if (!station || !line) {
        showError('역과 호선을 먼저 선택해주세요');
        return;
    }
    
    showLoading();
    
    try {
        const response = await fetch(
            `${API_BASE_URL}/predict/hourly/${encodeURIComponent(station)}/${encodeURIComponent(line)}`
        );
        
        if (!response.ok) throw new Error('시간대별 데이터를 불러올 수 없습니다');
        
        const data = await response.json();
        displayHourlyChart(data.hourly_predictions);
        
    } catch (error) {
        console.error('시간대별 데이터 로드 실패:', error);
        showError(error.message);
    } finally {
        hideLoading();
    }
}

// 시간대별 차트 표시
function displayHourlyChart(predictions) {
    const ctx = document.getElementById('hourly-chart').getContext('2d');
    
    // 기존 차트 제거
    if (hourlyChart) {
        hourlyChart.destroy();
    }
    
    const hours = predictions.map(p => `${p.hour}시`);
    const levels = predictions.map(p => p.congestion_level);
    const colors = predictions.map(p => p.congestion_color);
    
    hourlyChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: hours,
            datasets: [{
                label: '혼잡도 레벨',
                data: levels,
                backgroundColor: colors,
                borderRadius: 8,
                maxBarThickness: 50
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const labels = ['여유', '보통', '혼잡', '매우혼잡'];
                            return labels[context.parsed.y];
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 3,
                    ticks: {
                        stepSize: 1,
                        callback: function(value) {
                            const labels = ['여유', '보통', '혼잡', '매우혼잡'];
                            return labels[value];
                        }
                    }
                }
            }
        }
    });
}

// 로딩 표시
function showLoading() {
    loading.style.display = 'block';
}

function hideLoading() {
    loading.style.display = 'none';
}

// 에러 메시지 표시
function showError(message) {
    errorMessage.querySelector('p').textContent = message;
    errorMessage.style.display = 'block';
    
    setTimeout(() => {
        errorMessage.style.display = 'none';
    }, 3000);
}
