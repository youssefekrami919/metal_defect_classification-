const dropArea = document.getElementById('drop-area');
const fileInput = document.getElementById('file-input');
const imagePreview = document.getElementById('image-preview');
const resultBox = document.getElementById('result-box');
const resLabel = document.getElementById('res-label');
const resConf = document.getElementById('res-conf');
const spinner = document.getElementById('spinner');

// Handle click to browse
dropArea.addEventListener('click', () => fileInput.click());

// Handle drag events
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropArea.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

['dragenter', 'dragover'].forEach(eventName => {
    dropArea.addEventListener(eventName, () => dropArea.classList.add('dragover'), false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropArea.addEventListener(eventName, () => dropArea.classList.remove('dragover'), false);
});

// Handle drop
dropArea.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    handleFiles(files);
});

// Handle file selection
fileInput.addEventListener('change', (e) => {
    handleFiles(e.target.files);
});

function handleFiles(files) {
    if (files.length > 0) {
        const file = files[0];
        if (file.type.startsWith('image/')) {
            showPreview(file);
            uploadFile(file);
        } else {
            alert('Please upload an image file.');
        }
    }
}

function showPreview(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        imagePreview.style.display = 'block';
        resultBox.style.display = 'none'; // Hide old results
    };
    reader.readAsDataURL(file);
}

async function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);

    // Show spinner
    spinner.style.display = 'block';

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        spinner.style.display = 'none';

        if (result.error) {
            alert('Error: ' + result.error);
        } else {
            showResult(result);
        }
    } catch (error) {
        spinner.style.display = 'none';
        console.error('Error:', error);
        alert('An error occurred while analyzing the image.');
    }
}

function showResult(result) {
    resultBox.style.display = 'block';
    
    const label = result.class;
    const confidence = (result.confidence * 100).toFixed(2);

    resLabel.innerText = label;
    resConf.innerText = `Confidence: ${confidence}%`;

    // Style based on result
    if (label.includes('Good')) {
        resLabel.className = 'result-label status-good';
    } else {
        resLabel.className = 'result-label status-defect';
    }
}
