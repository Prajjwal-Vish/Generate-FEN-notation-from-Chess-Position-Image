console.log("SnapFen App Initialized v1.3");

document.addEventListener('DOMContentLoaded', () => {
    // --- VARIABLES ---
    let cropper = null;
    let currentActiveBlob = null;
    let rawFenBase = ""; 

    // 1. Elements: Upload & Preview
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.getElementById('preview-container');
    const mainPreview = document.getElementById('main-preview');

    // 2. Elements: Buttons
    const btnCropModal = document.getElementById('btn-crop-modal');
    const btnNewImage = document.getElementById('btn-new-image');
    const btnQuickScan = document.getElementById('btn-quick-scan');
    const btnConfirmCrop = document.getElementById('btn-confirm-crop');
    const btnCopy = document.getElementById('btn-copy');

    // 3. Elements: Toggles & Results
    const povToggle = document.getElementById('pov-toggle');
    const turnToggle = document.getElementById('turn-toggle');
    const turnLabel = document.getElementById('turn-label');
    const resultContent = document.getElementById('result-content');
    const emptyState = document.getElementById('empty-result-state');
    const fenText = document.getElementById('fen-text');
    const analyzedPreview = document.getElementById('analyzed-preview');

    // 4. Elements: Modals
    const cropModal = document.getElementById('crop-modal');
    const cropImageTarget = document.getElementById('crop-image-target');
    const btnCloseModal = document.getElementById('btn-close-modal');
    
    // 5. Elements: History (RESTORED)
    const btnHistory = document.getElementById('btn-history');
    const historyDrawer = document.getElementById('history-drawer');
    const historyList = document.getElementById('history-list');
    const drawerBackdrop = document.getElementById('drawer-backdrop');
    const btnCloseHistory = document.getElementById('btn-close-history');

    // --- EVENT LISTENERS ---

    // File Upload Logic
    if (fileInput) {
        fileInput.addEventListener('change', handleFileSelect);
    }

    // Safe Click for Dropzone
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
    if (btnCropModal) btnCropModal.addEventListener('click', openCropper);
    if (btnCloseModal) btnCloseModal.addEventListener('click', () => cropModal.classList.add('hidden'));
    if (btnConfirmCrop) btnConfirmCrop.addEventListener('click', performCrop);
    if (btnQuickScan) {
        btnQuickScan.addEventListener('click', () => { 
            if(currentActiveBlob) sendToBackend(currentActiveBlob, "scan.png"); 
        });
    }
    if (btnCopy) btnCopy.addEventListener('click', copyToClipboard);

    // --- HISTORY LOGIC (RESTORED) ---
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

    // 1. History Fetcher
    function fetchHistory() {
        historyList.innerHTML = '<div class="flex flex-col items-center justify-center h-32 text-gray-500 gap-2"><div class="loader w-6 h-6 border-2"></div><span class="text-xs">Loading...</span></div>';
        
        fetch('/api/history')
        .then(res => res.json())
        .then(data => {
            historyList.innerHTML = '';
            if (data.length === 0) {
                historyList.innerHTML = '<p class="text-gray-500 text-center mt-10 text-sm">No scans yet.</p>';
                return;
            }
            data.forEach(item => {
                const div = document.createElement('div');
                div.className = 'p-3 bg-white/5 rounded-xl border border-white/10 hover:bg-white/10 cursor-pointer transition group flex gap-3 items-center';
                div.innerHTML = `
                    <img src="${item.image}" class="w-12 h-12 rounded-lg object-cover border border-white/20">
                    <div class="overflow-hidden flex-1">
                        <p class="text-[10px] text-gray-400 mb-1">${item.date}</p>
                        <p class="text-xs font-mono text-blue-300 truncate w-full bg-black/30 p-1 rounded">${item.fen}</p>
                    </div>
                `;
                // On click: Load result and close drawer
                div.addEventListener('click', () => { 
                    // Note: History items usually store the final FEN. 
                    // To support toggling, ideally we'd store rawFen, but for now we load the FEN as is.
                    rawFenBase = item.fen; 
                    showResult(item.image); 
                    closeHistory(); 
                });
                historyList.appendChild(div);
            });
        })
        .catch(err => {
            console.error(err);
            historyList.innerHTML = '<p class="text-red-400 text-center mt-4 text-xs">Failed to load history.</p>';
        });
    }

    // 2. File Handling
    function handleFileSelect(e) {
        if (e.target.files && e.target.files[0]) {
            currentActiveBlob = e.target.files[0];
            const url = URL.createObjectURL(currentActiveBlob);
            mainPreview.src = url;
            
            dropZone.classList.add('hidden');
            previewContainer.classList.remove('hidden');
            resetResults();
            fileInput.value = ''; 
        }
    }

    // 3. Cropper
    function openCropper() {
        cropModal.classList.remove('hidden');
        cropImageTarget.src = mainPreview.src;
        if(cropper) cropper.destroy();
        cropper = new Cropper(cropImageTarget, { aspectRatio: 1, viewMode: 1, dragMode: 'move', autoCropArea: 0.95, background: false });
    }

    function performCrop() {
        if(!cropper) return;
        cropper.getCroppedCanvas({ width: 512, height: 512 }).toBlob((blob) => {
            currentActiveBlob = blob;
            cropModal.classList.add('hidden');
            sendToBackend(currentActiveBlob, "cropped.png");
        }, 'image/png');
    }

    // 4. Backend Communication
    function sendToBackend(fileBlob, fileName) {
        setLoading(true);
        const formData = new FormData();
        formData.append('file', fileBlob, fileName);
        formData.append('pov', 'w'); // Always ask for White POV

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

    // 5. Global FEN Logic
    function updateGlobalFen() {
        if (!rawFenBase) return;

        let currentFen = rawFenBase;
        
        if (povToggle.checked) {
            currentFen = flipFenBoard(currentFen);
        }

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

    // 6. UI Helpers
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
            btnConfirmCrop.innerHTML = '<div class="loader w-5 h-5 border-2"></div>';
            btnQuickScan.disabled = true; 
            btnConfirmCrop.disabled = true; 
            resultContent.classList.add('opacity-50');
        } else {
            btnQuickScan.innerHTML = '<i class="fa-solid fa-bolt"></i> <span>Scan</span>';
            btnConfirmCrop.innerHTML = 'Confirm';
            btnQuickScan.disabled = false; 
            btnConfirmCrop.disabled = false; 
            resultContent.classList.remove('opacity-50');
        }
    }
    
    function copyToClipboard() {
        fenText.select(); document.execCommand('copy');
        const icon = btnCopy.querySelector('i');
        icon.className = 'fa-solid fa-check text-green-400';
        setTimeout(() => icon.className = 'fa-regular fa-copy', 2000);
    }

    // --- 1. AI FEEDBACK LOGIC (Existing Floating Button) ---
    const feedbackModal = document.getElementById('feedback-modal');
    const btnOpenFeedback = document.getElementById('btn-open-feedback');
    const btnCloseFeedback = document.getElementById('btn-close-feedback');
    const btnSubmitFeedback = document.getElementById('btn-submit-feedback');
    const feedbackTags = document.querySelectorAll('.feedback-tag');
    let selectedTag = "General Issue";

    // Toggle Modal
    if(btnOpenFeedback) btnOpenFeedback.addEventListener('click', () => feedbackModal.classList.remove('hidden'));
    if(btnCloseFeedback) btnCloseFeedback.addEventListener('click', () => feedbackModal.classList.add('hidden'));

    // Tag Selection
    feedbackTags.forEach(tag => {
        tag.addEventListener('click', () => {
            feedbackTags.forEach(t => { t.classList.remove('bg-blue-600', 'text-white'); t.classList.add('bg-white/5', 'text-gray-300'); });
            tag.classList.remove('bg-white/5', 'text-gray-300');
            tag.classList.add('bg-blue-600', 'text-white');
            selectedTag = tag.innerText;
        });
    });

    // Submit AI Feedback
    if(btnSubmitFeedback) {
        btnSubmitFeedback.addEventListener('click', () => {
            const text = document.getElementById('feedback-text').value;
            const originalHTML = btnSubmitFeedback.innerHTML;
            btnSubmitFeedback.innerText = "Sending...";
            btnSubmitFeedback.disabled = true;

            const formData = new FormData();
            formData.append('feedback', text);
            formData.append('tags', selectedTag); // e.g. "Wrong Piece"
            formData.append('fen', fenText.value);
            // Send the chess image blobs if they exist
            if (currentActiveBlob) formData.append('original_image', currentActiveBlob, 'original.png');

            fetch('/report_issue', { method: 'POST', body: formData })
            .then(res => res.json())
            .then(() => {
                btnSubmitFeedback.innerText = "Sent!";
                setTimeout(() => {
                    feedbackModal.classList.add('hidden');
                    btnSubmitFeedback.innerHTML = originalHTML;
                    btnSubmitFeedback.disabled = false;
                    document.getElementById('feedback-text').value = '';
                }, 1500);
            });
        });
    }

    // --- 2. BUG REPORT LOGIC (Footer Link) ---
    const bugModal = document.getElementById('bug-modal');
    const footerReportBug = document.getElementById('footer-report-bug'); // Ensure footer link has this ID
    const btnCloseBug = document.getElementById('btn-close-bug');
    const btnSubmitBug = document.getElementById('btn-submit-bug');

    // Open/Close Logic
    if(footerReportBug) {
        footerReportBug.addEventListener('click', (e) => {
            e.preventDefault();
            bugModal.classList.remove('hidden');
        });
    }
    if(btnCloseBug) btnCloseBug.addEventListener('click', () => bugModal.classList.add('hidden'));

    // Submit Bug Report
    if(btnSubmitBug) {
        btnSubmitBug.addEventListener('click', () => {
            const text = document.getElementById('bug-text').value;
            const fileInput = document.getElementById('bug-file');
            const file = fileInput.files[0];

            const originalHTML = btnSubmitBug.innerHTML;
            btnSubmitBug.innerText = "Sending...";
            btnSubmitBug.disabled = true;

            const formData = new FormData();
            formData.append('feedback', text);
            formData.append('tags', "General Bug"); // Fixed tag for footer reports
            if (file) formData.append('attachment', file); // The new attachment logic

            fetch('/report_issue', { method: 'POST', body: formData })
            .then(res => res.json())
            .then(() => {
                btnSubmitBug.innerText = "Report Sent!";
                setTimeout(() => {
                    bugModal.classList.add('hidden');
                    btnSubmitBug.innerHTML = originalHTML;
                    btnSubmitBug.disabled = false;
                    document.getElementById('bug-text').value = '';
                    fileInput.value = '';
                }, 1500);
            })
            .catch(err => {
                console.error(err);
                btnSubmitBug.innerText = "Error";
                setTimeout(() => { btnSubmitBug.disabled = false; btnSubmitBug.innerHTML = originalHTML; }, 2000);
            });
        });
    }
});