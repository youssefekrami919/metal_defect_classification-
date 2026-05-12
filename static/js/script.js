let defectChart;
let trainingChart;
let classPerfChart;
let distChart;

const classColors = {
    'crazing': '#6366F1',
    'inclusion': '#34D399',
    'patches': '#F87171',
    'pitted_surface': '#FBBF24',
    'rolled-in_scale': '#A855F7',
    'scratches': '#EC4899',
    'Good': '#94A3B8'
};

// Tab Logic
function openTab(event, tabId) {
    if (event) event.preventDefault();
    
    document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    
    document.getElementById(tabId).classList.add('active');
    if (event) event.currentTarget.classList.add('active');

    if (tabId === 'dashboard-tab') {
        updateDashboard();
        updateTrainingInfo();
    }
}

// --- Dashboard Logic ---
async function updateDashboard() {
    try {
        const response = await fetch('/dashboard_stats');
        const stats = await response.json();
        
        document.getElementById('stat-total').innerText = stats.total_inspections;
        document.getElementById('stat-defects').innerText = stats.defects_found;
        document.getElementById('stat-accuracy').innerText = `${(stats.avg_confidence * 100).toFixed(1)}%`;

        renderChart(stats.defect_counts);
    } catch (error) {
        console.error('Error fetching stats:', error);
    }
}

async function updateTrainingInfo() {
    try {
        const response = await fetch('/training_info');
        const data = await response.json();
        
        // Update Methods UI
        document.getElementById('m-arch').innerText = data.methods.architecture;
        document.getElementById('m-opt').innerText = data.methods.optimizer;
        document.getElementById('m-loss').innerText = data.methods.loss_function;
        document.getElementById('m-dim').innerText = data.methods.dimensionality_reduction;

        // Update Hardware Performance UI
        document.getElementById('p-latency').innerText = data.hardware_metrics.inference_latency;
        document.getElementById('p-fps').innerText = data.hardware_metrics.throughput;
        document.getElementById('p-cpu').innerText = data.hardware_metrics.cpu_model;

        renderTrainingChart(data);
        renderClassMetrics(data);
    } catch (error) {
        console.error('Error fetching training info:', error);
    }
}

function renderClassMetrics(data) {
    const perfCtx = document.getElementById('classPerfChart').getContext('2d');
    const distCtx = document.getElementById('distChart').getContext('2d');

    if (classPerfChart) classPerfChart.destroy();
    if (distChart) distChart.destroy();

    const classes = Object.keys(data.class_performance);
    const colors = classes.map(cls => classColors[cls] || '#6366F1');

    // Class Performance Chart
    classPerfChart = new Chart(perfCtx, {
        type: 'bar',
        data: {
            labels: classes,
            datasets: [{
                label: 'Validation Accuracy (%)',
                data: Object.values(data.class_performance),
                backgroundColor: colors,
                borderRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'y',
            plugins: { legend: { display: false } },
            scales: {
                x: { max: 100, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94A3B8' } },
                y: { ticks: { color: '#94A3B8' } }
            }
        }
    });

    // Dataset Distribution Chart
    const trainData = classes.map(cls => data.dataset_distribution.train[cls]);
    const valData = classes.map(cls => data.dataset_distribution.val[cls]);

    distChart = new Chart(distCtx, {
        type: 'bar',
        data: {
            labels: classes,
            datasets: [
                { label: 'Training Set', data: trainData, backgroundColor: '#6366F1', borderRadius: 5 },
                { label: 'Validation Set', data: valData, backgroundColor: '#34D399', borderRadius: 5 }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { stacked: true, ticks: { color: '#94A3B8' } },
                y: { stacked: true, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94A3B8' } }
            },
            plugins: {
                legend: { labels: { color: '#E2E8F0' } }
            }
        }
    });
}

function renderTrainingChart(data) {
    const ctx = document.getElementById('trainingChart').getContext('2d');
    
    if (trainingChart) {
        trainingChart.destroy();
    }

    trainingChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.epochs.map(e => `Epoch ${e}`),
            datasets: [
                {
                    label: 'Accuracy (%)',
                    data: data.accuracy,
                    borderColor: '#34D399',
                    backgroundColor: 'rgba(52, 211, 153, 0.1)',
                    yAxisID: 'y',
                    tension: 0.3,
                    fill: true
                },
                {
                    label: 'Loss',
                    data: data.loss,
                    borderColor: '#F87171',
                    backgroundColor: 'transparent',
                    yAxisID: 'y1',
                    tension: 0.3,
                    borderDash: [5, 5]
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: { display: true, text: 'Accuracy (%)', color: '#94A3B8' },
                    ticks: { color: '#94A3B8' },
                    grid: { color: 'rgba(255,255,255,0.05)' }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: { display: true, text: 'Loss', color: '#94A3B8' },
                    ticks: { color: '#94A3B8' },
                    grid: { drawOnChartArea: false }
                },
                x: {
                    ticks: { color: '#94A3B8' },
                    grid: { color: 'rgba(255,255,255,0.05)' }
                }
            },
            plugins: {
                legend: {
                    labels: { color: '#E2E8F0', font: { family: 'Inter' } }
                }
            }
        }
    });
}

function renderChart(counts) {
    const ctx = document.getElementById('defectChart').getContext('2d');
    const labels = Object.keys(counts);
    const data = Object.values(counts);

    if (defectChart) {
        defectChart.data.datasets[0].data = data;
        defectChart.update();
    } else {
        defectChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: labels,
                datasets: [{
                    data: data,
                    backgroundColor: [
                        '#6366F1', '#34D399', '#F87171', '#FBBF24', '#A855F7', '#EC4899', '#94A3B8'
                    ],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: { color: '#E2E8F0', font: { family: 'Inter' } }
                    }
                }
            }
        });
    }
}

// Initial dashboard load
window.addEventListener('DOMContentLoaded', () => {
    updateDashboard();
    updateTrainingInfo();
});

// --- Image Upload Logic ---
const dropArea = document.getElementById('drop-area');
const fileInput = document.getElementById('file-input');
const imagePreview = document.getElementById('image-preview');
const resultBox = document.getElementById('result-box');
const resLabel = document.getElementById('res-label');
const resConf = document.getElementById('res-conf');
const spinner = document.getElementById('spinner');

dropArea.addEventListener('click', () => fileInput.click());

['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropArea.addEventListener(eventName, e => { e.preventDefault(); e.stopPropagation(); }, false);
});

dropArea.addEventListener('dragenter', () => dropArea.classList.add('dragover'));
dropArea.addEventListener('dragleave', () => dropArea.classList.remove('dragover'));
dropArea.addEventListener('drop', (e) => {
    dropArea.classList.remove('dragover');
    handleFiles(e.dataTransfer.files);
});

fileInput.addEventListener('change', (e) => handleFiles(e.target.files));

function handleFiles(files) {
    if (files.length > 0 && files[0].type.startsWith('image/')) {
        showPreview(files[0]);
        uploadFile(files[0]);
    }
}

function showPreview(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        imagePreview.style.display = 'block';
        resultBox.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

async function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    // UI Feedback
    spinner.style.display = 'block';
    const slab = document.getElementById('metal-slab');
    const slabStatus = document.getElementById('slab-status');
    
    // Trigger Scanning State in 3D Model
    slab.classList.add('scanning');
    slab.classList.remove('state-good', 'state-defect');
    slabStatus.innerText = "Scanning...";

    try {
        const response = await fetch('/predict', { method: 'POST', body: formData });
        const result = await response.json();
        
        spinner.style.display = 'none';
        slab.classList.remove('scanning');

        if (result.error) {
            alert(result.error);
            slabStatus.innerText = "ERROR";
        } else {
            showResult(result);
            updateDashboard(); // Update dashboard after prediction
            
            // Update 3D Slab State
            const isGood = result.class.includes('Good');
            slab.classList.add(isGood ? 'state-good' : 'state-defect');
            slabStatus.innerText = isGood ? "Optimal Quality" : result.class.toUpperCase();
        }
    } catch (error) {
        spinner.style.display = 'none';
        slab.classList.remove('scanning');
        alert('An error occurred.');
    }
}

function showResult(result) {
    resultBox.style.display = 'block';
    resLabel.innerText = result.class;
    resConf.innerText = `Confidence: ${(result.confidence * 100).toFixed(2)}%`;
    resLabel.className = result.class.includes('Good') ? 'result-label status-good' : 'result-label status-defect';
}

// --- Video Upload Logic ---
const videoDropArea = document.getElementById('video-drop-area');
const videoFileInput = document.getElementById('video-file-input');
const videoContainer = document.getElementById('video-container');
const videoStream = document.getElementById('video-stream');

videoDropArea.addEventListener('click', () => videoFileInput.click());

['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    videoDropArea.addEventListener(eventName, e => { e.preventDefault(); e.stopPropagation(); }, false);
});

videoDropArea.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].type.startsWith('video/')) {
        uploadVideo(files[0]);
    }
});

videoFileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) uploadVideo(e.target.files[0]);
});

async function uploadVideo(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    // Show some loading indicator
    videoDropArea.innerText = "Uploading video...";

    try {
        const response = await fetch('/upload_video', { method: 'POST', body: formData });
        const result = await response.json();
        
        if (result.filename) {
            startStream(result.filename);
        } else {
            alert('Upload failed');
            videoDropArea.innerHTML = '<span class="upload-icon">🎥</span><p class="upload-text">Upload inspection video to analyze in real-time</p>';
        }
    } catch (error) {
        alert('An error occurred.');
    }
}

function startStream(filename) {
    videoDropArea.style.display = 'none';
    videoContainer.style.display = 'block';
    videoStream.src = `/video_feed?filename=${filename}`;
}

function stopVideo() {
    videoStream.src = "";
    videoContainer.style.display = 'none';
    videoDropArea.style.display = 'block';
    videoDropArea.innerHTML = '<span class="upload-icon">🎥</span><p class="upload-text">Upload inspection video to analyze in real-time</p>';
}
