console.log("SnapFen App Initialized v1.4");

document.addEventListener('DOMContentLoaded', () => {
    // --- VARIABLES ---
    let cropper = null;
    let currentActiveBlob = null;
    let rawFenBase = ""; 
    let isManualCrop = false; // <--- NEW FLAG

    // 1. Elements
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.getElementById('preview-container');
    const mainPreview = document.getElementById('main-preview');

    // 2. Buttons
    const btnNewImage = document.getElementById('btn-new-image');
    const btnQuickScan = document.getElementById('btn-quick-scan');
    const btnCopy = document.getElementById('btn-copy');
    
    // Inline Cropper Controls
    const btnCropToggle = document.getElementById('btn-crop-toggle'); 
    const btnCancelCrop = document.getElementById('btn-cancel-crop');
    const btnConfirmCrop = document.getElementById('btn-confirm-crop');
    
    const actionButtons = document.getElementById('action-buttons');
    const cropButtons = document.getElementById('crop-buttons');

    // 3. Toggles & Results
    const povToggle = document.getElementById('pov-toggle');
    const turnToggle = document.getElementById('turn-toggle');
    const turnLabel = document.getElementById('turn-label');
    const resultContent = document.getElementById('result-content');
    const emptyState = document.getElementById('empty-result-state');
    const fenText = document.getElementById('fen-text');
    const analyzedPreview = document.getElementById('analyzed-preview');

    // 4. History
    const btnHistory = document.getElementById('btn-history');
    const historyDrawer = document.getElementById('history-drawer');
    const historyList = document.getElementById('history-list');
    const drawerBackdrop = document.getElementById('drawer-backdrop');
    const btnCloseHistory = document.getElementById('btn-close-history');

    // --- EVENT LISTENERS ---

    // File Upload
    if (fileInput) fileInput.addEventListener('change', handleFileSelect);
    if (dropZone && fileInput) {
        dropZone.addEventListener('click', (e) => {
            if (e.target !== fileInput) fileInput.click();
        });
    }

    // Toggles
    if (povToggle) povToggle.addEventListener('change', updateGlobalFen);
    if (turnToggle) {
        turnToggle.addEventListener('change', () => {
            updateTurnLabel();
            updateGlobalFen();
        });
    }

    // Main Buttons
    if (btnNewImage) btnNewImage.addEventListener('click', () => fileInput.click());
    
    if (btnQuickScan) {
        btnQuickScan.addEventListener('click', () => { 
            if(currentActiveBlob) sendToBackend(currentActiveBlob, "scan.png"); 
        });
    }
    
    if (btnCopy) btnCopy.addEventListener('click', copyToClipboard);

    // Cropper Listeners
    if (btnCropToggle) btnCropToggle.addEventListener('click', startCropping);
    if (btnCancelCrop) btnCancelCrop.addEventListener('click', cancelCropping);
    if (btnConfirmCrop) btnConfirmCrop.addEventListener('click', applyCrop);

    // History Listeners
    if (btnHistory) {
        btnHistory.addEventListener('click', () => {
            historyDrawer.classList.remove('closed');
            historyDrawer.classList.add('open');
            drawerBackdrop.classList.remove('hidden');
            fetchHistory();
        });
    }
    function closeHistory() {
        historyDrawer.classList.remove('open');
        historyDrawer.classList.add('closed');
        drawerBackdrop.classList.add('hidden');
    }
    if (btnCloseHistory) btnCloseHistory.addEventListener('click', closeHistory);
    if (drawerBackdrop) drawerBackdrop.addEventListener('click', closeHistory);

    // --- FUNCTIONS ---

    function handleFileSelect(e) {
        if (e.target.files && e.target.files[0]) {
            currentActiveBlob = e.target.files[0];
            const url = URL.createObjectURL(currentActiveBlob);
            mainPreview.src = url;
            
            // RESET STATE
            isManualCrop = false; // <--- RESET FLAG
            
            dropZone.classList.add('hidden');
            previewContainer.classList.remove('hidden');
            resetResults();
            fileInput.value = ''; 
        }
    }
    
    // --- CROPPER LOGIC ---
    function startCropping() {
        actionButtons.classList.add('hidden');
        cropButtons.classList.remove('hidden');

        if(cropper) cropper.destroy();
        cropper = new Cropper(mainPreview, {
            viewMode: 1, 
            dragMode: 'move',
            autoCropArea: 0.8,
            background: false, 
            guides: true,
            center: true,
            highlight: false,
            cropBoxMovable: true,
            cropBoxResizable: true,
            toggleDragModeOnDblclick: false,
        });
    }

    function cancelCropping() {
        if(cropper) cropper.destroy();
        cropper = null;
        cropButtons.classList.add('hidden');
        actionButtons.classList.remove('hidden');
    }

    function applyCrop() {
        if(!cropper) return;
        cropper.getCroppedCanvas({ width: 1024, height: 1024 }).toBlob((blob) => {
            currentActiveBlob = blob;
            isManualCrop = true; // <--- SET FLAG
            
            const newUrl = URL.createObjectURL(blob);
            mainPreview.src = newUrl;
            cancelCropping(); 
        }, 'image/png');
    }

    // --- BACKEND SEND ---
    function sendToBackend(fileBlob, fileName) {
        setLoading(true);
        const formData = new FormData();
        formData.append('file', fileBlob, fileName);
        formData.append('pov', 'w');
        
        // SEND FLAG TO SERVER
        formData.append('is_manual', isManualCrop); 

        fetch('/predict', { method: 'POST', body: formData })
        .then(res => res.json())
        .then(data => { 
            setLoading(false); 
            if(data.error) {
                alert(data.error); 
            } else { 
                rawFenBase = data.fen; 
                showResult(data.cropped_image); 
            }
        })
        .catch(err => { 
            setLoading(false); 
            console.error(err); 
            alert("Server Error."); 
        });
    }

    // --- HELPERS ---
    function updateGlobalFen() {
        if (!rawFenBase) return;
        let currentFen = rawFenBase;
        
        if (povToggle.checked) currentFen = flipFenBoard(currentFen);

        let parts = currentFen.split(' ');
        if (parts.length >= 2) {
            parts[1] = turnToggle.checked ? 'b' : 'w';
            currentFen = parts.join(' ');
        }

        fenText.value = currentFen;
        document.getElementById('link-lichess').href = `https://lichess.org/analysis/${currentFen}`;
        document.getElementById('link-chesscom').href = `https://www.chess.com/analysis?fen=${currentFen}`;
    }

    function flipFenBoard(fen) {
        let parts = fen.split(' ');
        let board = parts[0];
        let ranks = board.split('/').reverse();
        let flippedRanks = ranks.map(rank => rank.split('').reverse().join(''));
        parts[0] = flippedRanks.join('/');
        return parts.join(' ');
    }

    function updateTurnLabel() {
        if (turnToggle.checked) {
            turnLabel.innerText = "Black";
            turnLabel.classList.remove('text-white');
            turnLabel.classList.add('text-gray-400');
        } else {
            turnLabel.innerText = "White";
            turnLabel.classList.add('text-white');
            turnLabel.classList.remove('text-gray-400');
        }
    }

    function showResult(base64Image) {
        emptyState.classList.add('hidden');
        resultContent.classList.remove('hidden');
        resultContent.classList.add('animate-fade-in');
        if (base64Image) analyzedPreview.src = base64Image;
        updateGlobalFen();
        btnQuickScan.classList.add('btn-disabled', 'opacity-50');
    }

    function resetResults() {
        emptyState.classList.remove('hidden');
        resultContent.classList.add('hidden');
        btnQuickScan.classList.remove('btn-disabled', 'opacity-50');
        btnQuickScan.innerHTML = '<i class="fa-solid fa-bolt"></i> <span>Scan</span>';
    }

    function setLoading(isLoading) {
        if (isLoading) {
            btnQuickScan.innerHTML = '<div class="loader w-5 h-5 border-2"></div>';
            if(btnConfirmCrop) btnConfirmCrop.innerHTML = '<div class="loader w-5 h-5 border-2"></div>';
            btnQuickScan.disabled = true; 
            if(btnConfirmCrop) btnConfirmCrop.disabled = true; 
            resultContent.classList.add('opacity-50');
        } else {
            btnQuickScan.innerHTML = '<i class="fa-solid fa-bolt"></i> <span>Scan</span>';
            if(btnConfirmCrop) btnConfirmCrop.innerHTML = '<i class="fa-solid fa-check"></i><span>Apply Crop</span>';
            btnQuickScan.disabled = false; 
            if(btnConfirmCrop) btnConfirmCrop.disabled = false; 
            resultContent.classList.remove('opacity-50');
        }
    }
    
    function copyToClipboard() {
        fenText.select(); document.execCommand('copy');
        const icon = btnCopy.querySelector('i');
        icon.className = 'fa-solid fa-check text-green-400';
        setTimeout(() => icon.className = 'fa-regular fa-copy', 2000);
    }

    // --- FETCH HISTORY ---
    function fetchHistory() {
        historyList.innerHTML = '<div class="flex flex-col items-center justify-center h-32 text-gray-500 gap-2"><div class="loader w-6 h-6 border-2"></div><span class="text-xs">Loading...</span></div>';
        fetch('/api/history').then(res => res.json()).then(data => {
            historyList.innerHTML = '';
            if (data.length === 0) { historyList.innerHTML = '<p class="text-gray-500 text-center mt-10 text-sm">No scans yet.</p>'; return; }
            data.forEach(item => {
                const div = document.createElement('div');
                div.className = 'p-3 bg-white/5 rounded-xl border border-white/10 hover:bg-white/10 cursor-pointer transition group flex gap-3 items-center';
                div.innerHTML = `<img src="${item.image}" class="w-12 h-12 rounded-lg object-cover border border-white/20"><div class="overflow-hidden flex-1"><p class="text-[10px] text-gray-400 mb-1">${item.date}</p><p class="text-xs font-mono text-blue-300 truncate w-full bg-black/30 p-1 rounded">${item.fen}</p></div>`;
                div.addEventListener('click', () => { rawFenBase = item.fen; showResult(item.image); closeHistory(); });
                historyList.appendChild(div);
            });
        }).catch(err => { console.error(err); historyList.innerHTML = '<p class="text-red-400 text-center mt-4 text-xs">Failed to load history.</p>'; });
    }

    // --- MODAL LOGIC (Floating vs Footer) ---
    // 1. AI Feedback (Floating Button)
    const feedbackModal = document.getElementById('feedback-modal');
    const btnOpenFeedback = document.getElementById('btn-open-feedback');
    const btnCloseFeedback = document.getElementById('btn-close-feedback');
    const btnSubmitFeedback = document.getElementById('btn-submit-feedback');
    const feedbackTags = document.querySelectorAll('.feedback-tag');
    let selectedTag = "General Issue";

    if(btnOpenFeedback) btnOpenFeedback.addEventListener('click', () => feedbackModal.classList.remove('hidden'));
    if(btnCloseFeedback) btnCloseFeedback.addEventListener('click', () => feedbackModal.classList.add('hidden'));

    feedbackTags.forEach(tag => {
        tag.addEventListener('click', () => {
            feedbackTags.forEach(t => { t.classList.remove('bg-blue-600', 'text-white'); t.classList.add('bg-white/5', 'text-gray-300'); });
            tag.classList.remove('bg-white/5', 'text-gray-300'); tag.classList.add('bg-blue-600', 'text-white');
            selectedTag = tag.innerText;
        });
    });

    if(btnSubmitFeedback) {
        btnSubmitFeedback.addEventListener('click', () => {
            const text = document.getElementById('feedback-text').value;
            const originalHTML = btnSubmitFeedback.innerHTML;
            btnSubmitFeedback.innerText = "Sending...";
            btnSubmitFeedback.disabled = true;
            const formData = new FormData();
            formData.append('feedback', text);
            formData.append('tags', selectedTag);
            formData.append('fen', fenText.value);
            if (currentActiveBlob) formData.append('original_image', currentActiveBlob, 'original.png');
            fetch('/report_issue', { method: 'POST', body: formData }).then(res => res.json()).then(() => {
                btnSubmitFeedback.innerText = "Sent!";
                setTimeout(() => { feedbackModal.classList.add('hidden'); btnSubmitFeedback.innerHTML = originalHTML; btnSubmitFeedback.disabled = false; document.getElementById('feedback-text').value = ''; }, 1500);
            });
        });
    }

    // 2. Bug Report (Footer Link)
    const bugModal = document.getElementById('bug-modal');
    const footerReportBug = document.getElementById('footer-report-bug');
    const btnCloseBug = document.getElementById('btn-close-bug');
    const btnSubmitBug = document.getElementById('btn-submit-bug');

    if(footerReportBug) {
        footerReportBug.addEventListener('click', (e) => { e.preventDefault(); bugModal.classList.remove('hidden'); });
    }
    if(btnCloseBug) btnCloseBug.addEventListener('click', () => bugModal.classList.add('hidden'));

    if(btnSubmitBug) {
        btnSubmitBug.addEventListener('click', () => {
            const text = document.getElementById('bug-text').value;
            const file = document.getElementById('bug-file').files[0];
            const originalHTML = btnSubmitBug.innerHTML;
            btnSubmitBug.innerText = "Sending...";
            btnSubmitBug.disabled = true;
            const formData = new FormData();
            formData.append('feedback', text);
            formData.append('tags', "General Bug");
            if (file) formData.append('attachment', file);
            fetch('/report_issue', { method: 'POST', body: formData }).then(res => res.json()).then(() => {
                btnSubmitBug.innerText = "Sent!";
                setTimeout(() => { bugModal.classList.add('hidden'); btnSubmitBug.innerHTML = originalHTML; btnSubmitBug.disabled = false; document.getElementById('bug-text').value = ''; }, 1500);
            }).catch(err => { console.error(err); btnSubmitBug.innerText = "Error"; setTimeout(() => { btnSubmitBug.disabled = false; btnSubmitBug.innerHTML = originalHTML; }, 2000); });
        });
    }
    
    const footerUploadLink = document.getElementById('footer-upload-link');
    if (footerUploadLink && fileInput) {
        footerUploadLink.addEventListener('click', (e) => { e.preventDefault(); window.scrollTo({ top: 0, behavior: 'smooth' }); setTimeout(() => fileInput.click(), 500); });
    }
});
