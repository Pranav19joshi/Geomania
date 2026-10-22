const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const previewContainer = document.getElementById('preview-container');
const imagePreview = document.getElementById('image-preview');
const predictBtn = document.getElementById('predict-btn');
const resetBtn = document.getElementById('reset-btn');
const loadingIndicator = document.getElementById('loading-indicator');
const loadingText = document.getElementById('loading-text');
const errorMessage = document.getElementById('error-message');
const resultSection = document.getElementById('result-section');
const uploadSection = document.getElementById('upload-section');

let selectedFile = null;
let map = null;
let marker = null;

const loadingMessages = [
    "Locating...",
    "Searching...",
    "Finding matches..."
];

let messageInterval;

// Handle Drag & Drop
dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
        handleFile(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
        handleFile(e.target.files[0]);
    }
});

function handleFile(file) {
    if (!file.type.match('image.*')) {
        showError("Please select a valid image file.");
        return;
    }
    
    selectedFile = file;
    const url = URL.createObjectURL(file);
    imagePreview.src = url;
    
    dropZone.classList.add('hidden');
    previewContainer.classList.remove('hidden');
    errorMessage.classList.add('hidden');
    resultSection.classList.add('hidden');
}

resetBtn.addEventListener('click', () => {
    selectedFile = null;
    imagePreview.src = "";
    previewContainer.classList.add('hidden');
    dropZone.classList.remove('hidden');
    resultSection.classList.add('hidden');
    errorMessage.classList.add('hidden');
});

function showError(msg) {
    errorMessage.textContent = msg;
    errorMessage.classList.remove('hidden');
    loadingIndicator.classList.add('hidden');
    predictBtn.disabled = false;
    
    if(selectedFile) {
        previewContainer.classList.remove('hidden');
    } else {
        dropZone.classList.remove('hidden');
    }
}

predictBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    predictBtn.disabled = true;
    previewContainer.classList.add('hidden');
    resultSection.classList.add('hidden');
    errorMessage.classList.add('hidden');
    loadingIndicator.classList.remove('hidden');
    
    let msgIdx = 0;
    loadingText.textContent = loadingMessages[0];
    messageInterval = setInterval(() => {
        msgIdx = (msgIdx + 1) % loadingMessages.length;
        loadingText.textContent = loadingMessages[msgIdx];
    }, 2000);

    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            let errorMsg = `Server error: ${response.status}`;
            try {
                const errData = await response.json();
                if(errData.detail) errorMsg = errData.detail;
            } catch(e) {}
            throw new Error(errorMsg);
        }

        const data = await response.json();
        clearInterval(messageInterval);
        displayResult(data);
    } catch (err) {
        clearInterval(messageInterval);
        showError(err.message);
    }
});

function displayResult(data) {
    loadingIndicator.classList.add('hidden');
    resultSection.classList.remove('hidden');
    previewContainer.classList.remove('hidden');
    predictBtn.disabled = false;

    const lat = data.latitude;
    const lon = data.longitude;

    document.getElementById('res-lat').textContent = lat.toFixed(5);
    document.getElementById('res-lon').textContent = lon.toFixed(5);

    document.getElementById('osm-link').href = `https://www.openstreetmap.org/?mlat=${lat}&mlon=${lon}#map=12/${lat}/${lon}`;
    
    setupMap(lat, lon);
}

function setupMap(lat, lon) {
    if (!map) {
        map = L.map('map', {
            zoomControl: false // Disable default zoom control to keep UI clean
        }).setView([lat, lon], 10);
        
        // Add zoom control to bottom right
        L.control.zoom({
            position: 'bottomright'
        }).addTo(map);

        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors'
        }).addTo(map);
        
        // Custom sleek marker
        const customIcon = L.divIcon({
            className: 'custom-marker',
            html: `<div style="background-color: #6366f1; width: 16px; height: 16px; border-radius: 50%; border: 3px solid white; box-shadow: 0 0 10px rgba(0,0,0,0.5);"></div>`,
            iconSize: [16, 16],
            iconAnchor: [8, 8]
        });

        marker = L.marker([lat, lon], {icon: customIcon}).addTo(map);
    } else {
        map.setView([lat, lon], 10);
        marker.setLatLng([lat, lon]);
    }
    
    setTimeout(() => {
        map.invalidateSize();
    }, 100);
}

document.getElementById('copy-btn').addEventListener('click', (e) => {
    // Traverse DOM to find the actual span containing the text, since e.target might be the SVG icon
    const btn = e.currentTarget; 
    const lat = document.getElementById('res-lat').textContent;
    const lon = document.getElementById('res-lon').textContent;
    
    navigator.clipboard.writeText(`${lat}, ${lon}`).then(() => {
        const originalHTML = btn.innerHTML;
        btn.innerHTML = '✅ Copied!';
        setTimeout(() => {
            btn.innerHTML = originalHTML;
        }, 2000);
    });
});
