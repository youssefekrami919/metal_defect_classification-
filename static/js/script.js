// Tab Logic
function openTab(tabId) {
    document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    
    document.getElementById(tabId).classList.add('active');
    event.currentTarget.classList.add('active');
}

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
    spinner.style.display = 'block';
    try {
        const response = await fetch('/predict', { method: 'POST', body: formData });
        const result = await response.json();
        spinner.style.display = 'none';
        if (result.error) alert(result.error);
        else showResult(result);
    } catch (error) {
        spinner.style.display = 'none';
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
