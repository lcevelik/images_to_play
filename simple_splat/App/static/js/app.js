let selectedFiles = [];
let currentJobId = null;
let processingStartTime = null;
let elapsedTimer = null;
let gpuSpeedMultiplier = 1.0;
let gpuName = null;
let currentMethod = 'traditional';
let mlsharpAvailable = false;

const fileInput = document.getElementById('fileInput');
const uploadSection = document.getElementById('uploadSection');
const fileList = document.getElementById('fileList');
const processBtn = document.getElementById('processBtn');
const progressSection = document.getElementById('progressSection');
const resultSection = document.getElementById('resultSection');
const progressFill = document.getElementById('progressFill');
const errorMsg = document.getElementById('errorMsg');

/* ── Tabs ───────────────────────────────────────── */
function switchTab(name) {
    document.querySelectorAll('.tab-view').forEach(v => v.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.getElementById('tab-' + name).classList.add('active');
    document.getElementById('tabBtn-' + name).classList.add('active');
}

function setTabDot(which, state) {
    const dot = document.getElementById(which + 'TabDot');
    if (!dot) return;
    dot.className = 'tab-dot' + (state ? ' ' + state : '');
}

/* ── Settings summary chips on Create tab ───────── */
function updateSettingsSummary() {
    const wrap = document.getElementById('settingsSummary');
    if (!wrap) return;   // summary row removed — Draft/Production is the visible control now
    const method = document.querySelector('input[name="method"]:checked').value;
    const chips = [];
    if (method === 'mlsharp') {
        chips.push('<b>ML-Sharp</b> single image');
        chips.push('Device: <b>' + document.getElementById('mlsharpDevice').value + '</b>');
    } else {
        const presetEl = document.querySelector('input[name="preset"]:checked');
        const trainerEl = document.querySelector('input[name="trainer"]:checked');
        const presetName = presetEl ? presetEl.parentElement.querySelector('.po-name').textContent.trim() : 'Balanced';
        const trainerName = trainerEl ? trainerEl.parentElement.querySelector('.po-name').textContent.trim() : 'Brush';
        const scale = document.getElementById('qualityScale').value;
        chips.push('Preset: <b>' + presetName + '</b>');
        chips.push('Trainer: <b>' + trainerName + '</b>');
        chips.push('Scale: <b>' + scale.charAt(0).toUpperCase() + scale.slice(1) + '</b>');
    }
    wrap.innerHTML = chips.map(c =>
        '<span class="summary-chip" onclick="switchTab(\'settings\')"><span class="dot"></span>' + c + '</span>'
    ).join('');
}

// Quality scale radio group drives the hidden #qualityScale select
document.querySelectorAll('input[name="qualityScaleRadio"]').forEach(r => {
    r.addEventListener('change', () => {
        document.getElementById('qualityScale').value = r.value;
        updateSettingsSummary();
    });
});
document.querySelectorAll('input[name="preset"], input[name="trainer"]').forEach(r => {
    r.addEventListener('change', updateSettingsSummary);
});
document.getElementById('mlsharpDevice').addEventListener('change', updateSettingsSummary);

fileInput.addEventListener('change', handleFiles);
document.querySelectorAll('input[name="quality"]').forEach(r => r.addEventListener('change', applyQuality));
document.getElementById('fbxMatchmove').addEventListener('change', applyFbxToggle);

// Header hamburger menu (Open Splat / Logs / Stop / Clean)
function toggleHeaderMenu(e) {
    e.stopPropagation();
    const open = document.getElementById('headerMenu').classList.toggle('open');
    document.getElementById('hamburgerBtn').setAttribute('aria-expanded', open);
}
function closeHeaderMenu() {
    document.getElementById('headerMenu').classList.remove('open');
    document.getElementById('hamburgerBtn').setAttribute('aria-expanded', 'false');
}
document.addEventListener('click', (e) => {
    const wrap = document.querySelector('.menu-wrap');
    if (wrap && !wrap.contains(e.target)) closeHeaderMenu();
});
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeHeaderMenu(); });

uploadSection.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadSection.classList.add('dragover');
});

uploadSection.addEventListener('dragleave', () => {
    uploadSection.classList.remove('dragover');
});

uploadSection.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadSection.classList.remove('dragover');
    fileInput.files = e.dataTransfer.files;
    handleFiles();
});

async function fetchGPUInfo() {
    try {
        const response = await fetch('/gpu-info');
        const data = await response.json();
        if (data.speed_multiplier) {
            gpuSpeedMultiplier = data.speed_multiplier;
            gpuName = data.gpu_name || 'CPU';
            console.log(`GPU detected: ${gpuName}, speed multiplier: ${gpuSpeedMultiplier}`);
        }
    } catch (error) {
        console.log('Could not fetch GPU info:', error);
        gpuSpeedMultiplier = 1.0;
    }
}

// Check ML-Sharp availability on page load
async function checkMLSharpAvailability() {
    try {
        const response = await fetch('/mlsharp-info');
        const info = await response.json();
        mlsharpAvailable = info.available;
        if (!mlsharpAvailable) {
            document.getElementById('methodMLSharp').disabled = true;
            document.getElementById('methodMLSharp').parentElement.style.opacity = '0.5';
            document.getElementById('methodMLSharp').parentElement.title = 'ML-Sharp not installed. See README for installation instructions';
        }
    } catch (error) {
        console.log('Could not fetch ML-Sharp info:', error);
        mlsharpAvailable = false;
    }
}

// Call on page load
fetchGPUInfo();
checkMLSharpAvailability();
updateSettingsSummary();

// Update UI based on selected method
function updateUIForMethod() {
    const method = document.querySelector('input[name="method"]:checked').value;
    currentMethod = method;

    if (method === 'mlsharp') {
        // ML-Sharp mode: single image only
        fileInput.removeAttribute('multiple');
        fileInput.accept = 'image/*';

        // Hide traditional settings
        document.getElementById('presetGroup').style.display = 'none';
        document.getElementById('videoControls').style.display = 'none';

        // Show ML-Sharp options
        document.getElementById('mlsharpOptions').style.display = 'block';
        document.getElementById('pipelineStages').style.display = 'none';
        document.getElementById('pipelineStagesMLSharp').style.display = 'block';

        processBtn.textContent = 'Process with ML-Sharp';

        // Clear existing files if multiple selected
        if (selectedFiles.length > 1) {
            clearFiles();
        }
    } else {
        // Traditional mode: restore defaults
        fileInput.setAttribute('multiple', 'multiple');
        fileInput.accept = 'image/*,.zip,video/*';

        document.getElementById('presetGroup').style.display = 'block';

        document.getElementById('mlsharpOptions').style.display = 'none';
        document.getElementById('pipelineStages').style.display = 'block';
        document.getElementById('pipelineStagesMLSharp').style.display = 'none';

        processBtn.textContent = 'Start Processing';
    }

    updateTimeEstimates();
    updateSettingsSummary();
}

// Add event listeners for method radio buttons
document.querySelectorAll('input[name="method"]').forEach(radio => {
    radio.addEventListener('change', updateUIForMethod);
});


function updateTimeEstimates() {
    const numImages = selectedFiles.length;

    // Base times per image (in minutes) — calibrated for RTX 8000
    const presetMeta = {
        low:     { steps: '5K',   min: 0.05, max: 0.10 },
        medium:  { steps: '15K',  min: 0.10, max: 0.25 },
        high:    { steps: '30K',  min: 0.25, max: 0.60 },
        quality: { steps: '60K',  min: 0.50, max: 1.20 },
        expert:  { steps: '100K', min: 0.80, max: 2.00 },
    };

    const defaultLabels = {
        low:     '5K steps, ~2-4 min',
        medium:  '15K steps, ~8-15 min',
        high:    '30K steps, ~15-30 min',
        quality: '60K steps, ~30-60 min',
        expert:  '100K steps, lens calibration',
    };

    for (const [preset, meta] of Object.entries(presetMeta)) {
        const el = document.querySelector(`input[name="preset"][value="${preset}"]`);
        if (!el) continue;
        const span = el.parentElement.querySelector('.po-desc');
        if (!span) continue;

        if (numImages === 0) {
            span.textContent = defaultLabels[preset];
            continue;
        }

        const minTime = Math.ceil(meta.min * numImages / gpuSpeedMultiplier);
        const maxTime = Math.ceil(meta.max * numImages / gpuSpeedMultiplier);
        let timeStr;
        if (maxTime < 60) {
            timeStr = `~${minTime}-${maxTime} min`;
        } else {
            const minH = Math.floor(minTime / 60), minM = minTime % 60;
            const maxH = Math.floor(maxTime / 60), maxM = maxTime % 60;
            timeStr = minH === 0 ? `~${minM}m - ${maxH}h ${maxM}m` : `~${minH}h ${minM}m - ${maxH}h ${maxM}m`;
        }
        span.textContent = `${meta.steps} steps, ${timeStr} (${numImages} images)`;
    }
}

function handleFiles() {
    selectedFiles = Array.from(fileInput.files);
    displayFileList();
    updateTimeEstimates();
    processBtn.disabled = selectedFiles.length === 0;
    applyInputRouting();
}

// Route the UI by what was dropped: single image -> ML-Sharp; sequence/ZIP -> Draft/Production; video -> + sampling/FBX.
function applyInputRouting() {
    const isVid = f => f.type.startsWith('video/') || /\.(mp4|avi|mov|mkv|webm)$/i.test(f.name);
    const isZip = f => /\.zip$/i.test(f.name);
    const isImg = f => f.type.startsWith('image/') || /\.(jpe?g|png|webp|heic|tiff?)$/i.test(f.name);
    const vids = selectedFiles.filter(isVid);
    const zips = selectedFiles.filter(isZip);
    const imgs = selectedFiles.filter(isImg);

    const banner = document.getElementById('modeBanner');
    const qcards = document.getElementById('qualityCards');
    document.getElementById('videoControls').style.display = vids.length > 0 ? 'block' : 'none';

    if (selectedFiles.length === 0) {
        banner.style.display = 'none'; qcards.style.display = 'none'; return;
    }

    const singleImage = imgs.length === 1 && vids.length === 0 && zips.length === 0;
    if (singleImage) {
        document.getElementById('methodMLSharp').checked = true;
        qcards.style.display = 'none';
        banner.style.display = 'block';
        banner.innerHTML = '<strong>Single image → ML-Sharp.</strong> Instant AI splat from one photo — no quality choice needed.';
        if (typeof updateSettingsSummary === 'function') updateSettingsSummary();
    } else {
        document.getElementById('methodTraditional').checked = true;
        qcards.style.display = 'block';
        banner.style.display = 'block';
        const what = vids.length ? 'Video' : (zips.length ? 'ZIP archive' : imgs.length + ' images');
        banner.innerHTML = `<strong>${what} → reconstruction.</strong> Pick Draft (fast) or Production (highest quality) below.`;
        applyQuality();
    }
}

// Draft/Production drive the existing (advanced) trainer + preset fields, so the backend contract is unchanged.
function applyQuality() {
    const prod = document.getElementById('qualityProduction').checked;
    // Draft = fast Brush; Production = high MCMC (interim until the MCMC+Brush combine pipeline is wired in).
    const trainer = document.querySelector(`input[name="trainer"][value="${prod ? 'mcmc' : 'brush'}"]`);
    if (trainer) trainer.checked = true;
    const preset = document.querySelector(`input[name="preset"][value="${prod ? 'high' : 'medium'}"]`);
    if (preset) preset.checked = true;
    if (typeof updateSettingsSummary === 'function') updateSettingsSummary();
}

// FBX matchmove camera -> solve every frame, capped for GPU.
function applyFbxToggle() {
    const on = document.getElementById('fbxMatchmove').checked;
    const interval = document.getElementById('interval');
    interval.value = on ? '1' : interval.value;
    interval.disabled = on;
    if (on) document.getElementById('maxFrames').value = '500';
}

function displayFileList() {
    fileList.innerHTML = '';
    if (selectedFiles.length === 0) return;

    const totalSize = selectedFiles.reduce((sum, f) => sum + f.size, 0);
    const totalSizeMB = (totalSize / 1024 / 1024).toFixed(2);

    const fileItem = document.createElement('div');
    fileItem.className = 'file-item';

    if (selectedFiles.length === 1) {
        fileItem.innerHTML = `
            <span>${selectedFiles[0].name} (${totalSizeMB} MB)</span>
            <button onclick="clearFiles()" class="small-btn">Remove</button>
        `;
    } else {
        fileItem.innerHTML = `
            <span>${selectedFiles.length} files (${totalSizeMB} MB)</span>
            <button onclick="clearFiles()" class="small-btn">Clear</button>
        `;
    }
    fileList.appendChild(fileItem);
}

function clearFiles() {
    selectedFiles = [];
    fileInput.value = '';
    displayFileList();
    updateTimeEstimates();
    processBtn.disabled = true;
}

async function startProcessing() {
    if (selectedFiles.length === 0) {
        showError('Please select files first');
        return;
    }

    processBtn.disabled = true;
    document.getElementById('idleState').style.display = 'none';
    progressSection.classList.add('active');
    resultSection.classList.remove('active');
    document.getElementById('resultsEmpty').style.display = 'flex';
    errorMsg.style.display = 'none';
    setTabDot('process', 'running');
    setTabDot('results', '');
    switchTab('process');

    resetPipelineStages();
    resetPreview();
    processingStartTime = Date.now();
    startElapsedTimer();

    // Ask for desktop-notification permission inside this click gesture so the
    // browser actually shows the prompt; we fire a notification when the job ends.
    if ('Notification' in window && Notification.permission === 'default') {
        Notification.requestPermission();
    }

    const formData = new FormData();
    selectedFiles.forEach(file => formData.append('files', file));

    const method = document.querySelector('input[name="method"]:checked').value;

    // ML-Sharp validation
    if (method === 'mlsharp' && selectedFiles.length !== 1) {
        showError('ML-Sharp requires exactly one image');
        processBtn.disabled = false;
        stopElapsedTimer();
        setTabDot('process', '');
        switchTab('create');
        return;
    }

    // Add method
    formData.append('method', method);

    if (method === 'mlsharp') {
        // ML-Sharp specific options
        formData.append('mlsharp_render', document.getElementById('mlsharpRender').checked ? 'true' : 'false');
        formData.append('mlsharp_device', document.getElementById('mlsharpDevice').value);
    } else {
        // Traditional pipeline options
        const presetRadio = document.querySelector('input[name="preset"]:checked');
        const preset = presetRadio ? presetRadio.value : 'medium';
        formData.append('preset', preset);

        formData.append('sharpness_boost', 'false');
        formData.append('quality_scale', document.getElementById('qualityScale').value);
        const trainerRadio = document.querySelector('input[name="trainer"]:checked');
        formData.append('trainer', trainerRadio ? trainerRadio.value : 'brush');
        const budgetEl = document.getElementById('gaussianBudget');
        formData.append('gaussian_budget', budgetEl ? budgetEl.value : 'auto');
        formData.append('matcher_type', 'auto');
        formData.append('enable_dense', 'false');
        formData.append('mvs_quality_mode', 'balanced');
        formData.append('interval', document.getElementById('interval').value);
        formData.append('max_frames', document.getElementById('maxFrames').value);
        formData.append('export_fbx', document.getElementById('fbxMatchmove').checked ? 'true' : 'false');
        formData.append('quality', document.querySelector('input[name="quality"]:checked')?.value || 'production');
    }

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        if (response.ok) {
            currentJobId = data.job_id;
            pollStatus();
            pollLogs();
        } else {
            showError(data.error || 'Upload failed');
            processBtn.disabled = false;
            stopElapsedTimer();
            setTabDot('process', 'err');
            switchTab('create');
        }
    } catch (error) {
        showError('Error: ' + error.message);
        processBtn.disabled = false;
        stopElapsedTimer();
        setTabDot('process', 'err');
        switchTab('create');
    }
}

function resetPipelineStages() {
    for (let i = 1; i <= 5; i++) {
        const stage = document.getElementById('stage' + i);
        if (stage) {
            stage.className = 'pipeline-stage';
            stage.querySelector('.stage-icon').textContent = '•';
        }
    }
    document.getElementById('currentActivity').textContent = 'Initializing...';
    document.getElementById('liveLogPreview').innerHTML = '';
}

function updatePipelineStage(stageNum, status) {
    const stage = document.getElementById('stage' + stageNum);
    if (!stage) return;

    const icon = stage.querySelector('.stage-icon');

    if (status === 'running') {
        stage.className = 'pipeline-stage running';
        icon.textContent = '◌';
    } else if (status === 'complete') {
        stage.className = 'pipeline-stage complete';
        icon.textContent = '✓';
    } else if (status === 'error') {
        stage.className = 'pipeline-stage';
        icon.textContent = '✕';
    }
}

function startElapsedTimer() {
    stopElapsedTimer();
    elapsedTimer = setInterval(() => {
        if (processingStartTime) {
            const elapsed = Math.floor((Date.now() - processingStartTime) / 1000);
            const mins = Math.floor(elapsed / 60);
            const secs = elapsed % 60;
            document.getElementById('elapsedTime').textContent =
                `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
        }
        renderEta();  // smooth countdown of the remaining-time estimate
    }, 1000);
}

function stopElapsedTimer() {
    if (elapsedTimer) {
        clearInterval(elapsedTimer);
        elapsedTimer = null;
    }
    _etaSeconds = null;
    const etaEl = document.getElementById('etaRemaining');
    if (etaEl) etaEl.textContent = '';
}

let _lastProgress = -1;
let _pollInterval = 1500;  // Start at 1.5s, back off to 10s when idle

// ETA tracking: the backend sends eta_seconds at each poll; between polls
// we count it down locally (in the 1s elapsed timer) so it ticks smoothly.
let _etaSeconds = null;       // latest ETA from backend (seconds), or null
let _etaUpdatedAt = 0;        // Date.now() when _etaSeconds was set

function formatEta(secs) {
    if (secs == null || secs < 0) return '';
    if (secs < 60) return `~${secs}s left`;
    const m = Math.floor(secs / 60), s = secs % 60;
    if (m < 60) return `~${m}m ${s}s left`;
    const h = Math.floor(m / 60), mm = m % 60;
    return `~${h}h ${mm}m left`;
}

function renderEta() {
    const el = document.getElementById('etaRemaining');
    if (!el) return;
    if (_etaSeconds == null) { el.textContent = ''; return; }
    // Subtract time elapsed since the last backend update for a smooth countdown
    const drift = Math.floor((Date.now() - _etaUpdatedAt) / 1000);
    el.textContent = formatEta(Math.max(0, _etaSeconds - drift));
}

// Fire a desktop notification + a short beep when a job ends. Guarded so it
// only fires once per job, and degrades silently if permission was denied.
let _notifiedJob = null;
function notifyJobDone(success, message) {
    if (_notifiedJob === currentJobId) return;
    _notifiedJob = currentJobId;
    const title = success ? 'FonixFlow Splat — Done ✓' : 'FonixFlow Splat — Failed ✗';
    try {
        if ('Notification' in window && Notification.permission === 'granted') {
            const n = new Notification(title, { body: message || '', tag: 'fonixflow-job' });
            n.onclick = () => { window.focus(); n.close(); };
        }
    } catch (e) { /* notifications unsupported — ignore */ }
    // Audible cue (works even when the tab is backgrounded / permission denied)
    try {
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const osc = ctx.createOscillator(), gain = ctx.createGain();
        osc.connect(gain); gain.connect(ctx.destination);
        osc.frequency.value = success ? 880 : 300;
        gain.gain.setValueAtTime(0.0001, ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.2, ctx.currentTime + 0.02);
        gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.5);
        osc.start(); osc.stop(ctx.currentTime + 0.5);
    } catch (e) { /* audio blocked — ignore */ }
}

// 3D alignment preview: once COLMAP finishes, load the sparse points +
// camera frustums into the embedded SuperSplat viewer (replaces the log).
let _previewLoaded = false;
function maybeLoadPreview(data) {
    if (_previewLoaded || !data || !data.preview_ready || !currentJobId) return;
    const frame = document.getElementById('alignPreview');
    const hint = document.getElementById('alignPreviewHint');
    if (!frame) return;
    frame.src = `/static/align/viewer.html?job=${currentJobId}`;
    frame.style.display = 'block';
    if (hint) hint.style.display = 'none';
    _previewLoaded = true;
}
function resetPreview() {
    _previewLoaded = false;
    const frame = document.getElementById('alignPreview');
    const hint = document.getElementById('alignPreviewHint');
    if (frame) { frame.src = 'about:blank'; frame.style.display = 'none'; }
    if (hint) hint.style.display = 'flex';
}

async function pollStatus() {
    if (!currentJobId) return;

    try {
        const response = await fetch(`/status/${currentJobId}`);
        const data = await response.json();

        if (!response.ok) {
            if (response.status === 404) {
                setTimeout(pollStatus, 1000);
                return;
            }
            showError(data.error || 'Failed to get status');
            processBtn.disabled = false;
            stopElapsedTimer();
            setTabDot('process', 'err');
            return;
        }

        // Load the 3D alignment view as soon as COLMAP has produced it
        maybeLoadPreview(data);

        if (data.status === 'completed') {
            progressFill.style.width = '100%';
            _etaSeconds = null;
            document.getElementById('etaRemaining').textContent = '';
            document.getElementById('currentActivity').textContent = 'Complete';
            for (let i = 1; i <= 5; i++) updatePipelineStage(i, 'complete');
            updateStageTimings(data.stages);   // hold the final per-stage times (don't reset to 0)
            stopElapsedTimer();
            processBtn.disabled = selectedFiles.length === 0;
            setTabDot('process', '');
            setTabDot('results', 'done');
            notifyJobDone(true, 'Your Gaussian splat is ready to view.');
            showResults(data);
            switchTab('results');
        } else if (data.status === 'error') {
            showError(data.error || 'Processing failed');
            document.getElementById('currentActivity').textContent = 'Error';
            processBtn.disabled = false;
            stopElapsedTimer();
            setTabDot('process', 'err');
            notifyJobDone(false, data.error || 'Processing failed.');
            switchTab('create');
        } else if (data.status === 'cancelled') {
            showError('Processing was cancelled');
            document.getElementById('currentActivity').textContent = 'Cancelled';
            processBtn.disabled = false;
            stopElapsedTimer();
            setTabDot('process', '');
            switchTab('create');
        } else {
            const step = data.step || 'Processing...';
            progressFill.style.width = (data.progress || 0) + '%';
            document.getElementById('currentActivity').textContent = step;

            // Capture the backend ETA; renderEta() counts it down between polls
            if (typeof data.eta_seconds === 'number') {
                _etaSeconds = data.eta_seconds;
                _etaUpdatedAt = Date.now();
            } else {
                _etaSeconds = null;
            }
            renderEta();

            // Use stage field from backend for more accurate tracking
            if (data.stage) {
                updateStagesFromStage(data.stage, data.stages);
            } else {
                updateStagesFromStep(step, data.progress || 0);
            }

            // FIX #10: Adaptive polling — slow down when progress stalls
            const currentProgress = data.progress || 0;
            if (currentProgress !== _lastProgress) {
                _pollInterval = 1500;  // Progress changed: poll fast
            } else {
                _pollInterval = Math.min(_pollInterval * 1.5, 10000);  // No change: back off to 10s max
            }
            _lastProgress = currentProgress;
            setTimeout(pollStatus, _pollInterval);
        }
    } catch (error) {
        _pollInterval = Math.min(_pollInterval * 2, 15000);  // Error: back off more
        showError('Error checking status: ' + error.message);
        processBtn.disabled = false;
        stopElapsedTimer();
        setTabDot('process', 'err');
    }
}

function formatElapsed(seconds) {
    if (!seconds && seconds !== 0) return '';
    if (seconds < 60) return seconds.toFixed(1) + 's';
    const m = Math.floor(seconds / 60);
    const s = Math.round(seconds % 60);
    return m + 'm ' + s + 's';
}

function updateStageTimings(stages) {
    if (!stages) return;
    const stageKeys = ['feature_extraction', 'feature_matching', 'mapping', 'dense_mvs', 'training'];
    stageKeys.forEach((key, i) => {
        const el = document.getElementById('stage' + (i+1) + '-time');
        const s = stages[key];
        // The server keeps `elapsed` current for the running stage and frozen
        // for finished ones, so any started stage holds a real time (never 0).
        if (el && s && s.elapsed != null) {
            el.textContent = formatElapsed(s.elapsed);
        }
    });
}

function updateStagesFromStage(stage, stages) {
    // Update timing display
    updateStageTimings(stages);

    // Map backend stage names to pipeline stage numbers
    const stageMap = {
        'initialization': 0,
        'feature_extraction': 1,
        'feature_matching': 2,
        'mapping': 3,
        'dense_mvs': 4,
        'training': 5
    };
    const currentStageNum = stageMap[stage] || 0;
    for (let i = 1; i <= 5; i++) {
        if (i < currentStageNum) {
            updatePipelineStage(i, 'complete');
        } else if (i === currentStageNum) {
            updatePipelineStage(i, 'running');
        }
    }
}

function updateStagesFromStep(step, progress) {
    const stepLower = step.toLowerCase();

    if (currentMethod === 'mlsharp') {
        // ML-Sharp pipeline stages
        if (stepLower.includes('loading') || stepLower.includes('initializing') || stepLower.includes('model')) {
            updateMLSharpStage(1, 'running');
        } else if (stepLower.includes('predict') || stepLower.includes('inference') || stepLower.includes('gaussians')) {
            updateMLSharpStage(1, 'complete');
            updateMLSharpStage(2, 'running');
        } else if (stepLower.includes('saving') || stepLower.includes('writing') || stepLower.includes('complete')) {
            updateMLSharpStage(1, 'complete');
            updateMLSharpStage(2, 'complete');
            updateMLSharpStage(3, 'running');
        }
    } else {
        // Traditional pipeline stages
        if (stepLower.includes('feature extraction') || stepLower.includes('extracting features')) {
            updatePipelineStage(1, 'running');
        } else if (stepLower.includes('feature matching') || stepLower.includes('matching')) {
            updatePipelineStage(1, 'complete');
            updatePipelineStage(2, 'running');
        } else if (stepLower.includes('3d mapping') || stepLower.includes('mapper') || stepLower.includes('registered')) {
            updatePipelineStage(1, 'complete');
            updatePipelineStage(2, 'complete');
            updatePipelineStage(3, 'running');
        } else if (stepLower.includes('undistort')) {
            updatePipelineStage(1, 'complete');
            updatePipelineStage(2, 'complete');
            updatePipelineStage(3, 'complete');
            updatePipelineStage(4, 'running');
        } else if (stepLower.includes('gaussian') || stepLower.includes('brush') || stepLower.includes('training')) {
            updatePipelineStage(1, 'complete');
            updatePipelineStage(2, 'complete');
            updatePipelineStage(3, 'complete');
            updatePipelineStage(4, 'complete');
            updatePipelineStage(5, 'running');
        }
    }
}

function updateMLSharpStage(stageNum, status) {
    const stage = document.getElementById('mlsharp-stage' + stageNum);
    if (!stage) return;
    const icon = stage.querySelector('.stage-icon');
    if (status === 'running') {
        stage.className = 'pipeline-stage running';
        icon.textContent = '◌';
    } else if (status === 'complete') {
        stage.className = 'pipeline-stage complete';
        icon.textContent = '✓';
    }
}

async function pollLogs() {
    if (!currentJobId) return;

    try {
        const response = await fetch(`/logs/json/${currentJobId}`);
        const logs = await response.json();

        const recentLogs = logs.slice(-30);

        const logPreview = document.getElementById('liveLogPreview');
        if (logPreview && recentLogs.length > 0) {
            logPreview.innerHTML = '';

            recentLogs.forEach(log => {
                let color = '#6b6b76';
                if (log.includes('[ERROR]')) color = '#f87171';
                else if (log.includes('[WARNING]')) color = '#fbbf24';
                else if (log.includes('complete')) color = '#34d399';
                else if (log.includes('Step')) color = '#818cf8';

                const div = document.createElement('div');
                div.style.color = color;
                div.style.margin = '2px 0';
                div.textContent = log;
                logPreview.appendChild(div);
            });
            logPreview.scrollTop = logPreview.scrollHeight;
        }

        // Continue polling (pollStatus will stop log polling when done)
        setTimeout(pollLogs, 2000);
    } catch (error) {
        console.log('Log polling error:', error);
    }
}

function showResults(data) {
    document.getElementById('resultsEmpty').style.display = 'none';
    resultSection.classList.add('active');
    document.getElementById('viewBtn').href = `/static/supersplat/index.html?load=/ply/${currentJobId}.ply`;
    // FBX matchmove camera (auto-exported by the recipe when the toggle was on)
    const fbxBtn = document.getElementById('fbxBtn');
    if (data && data.fbx) {
        fbxBtn.href = `/download/${currentJobId}/tracking/cameras.fbx`;
        fbxBtn.style.display = 'inline-block';
    } else {
        fbxBtn.style.display = 'none';
    }

    // "Send to Production" — re-runs this job at production quality on the camera solve
    // it already has. The server decides eligibility (it depends on what's on disk), so
    // just honour can_promote rather than re-deriving it here.
    const promoteBtn = document.getElementById('promoteBtn');
    const promoteNote = document.getElementById('promoteNote');
    const show = !!(data && data.can_promote);
    promoteBtn.style.display = show ? 'inline-block' : 'none';
    promoteNote.style.display = show ? 'block' : 'none';
    promoteBtn.disabled = false;
    promoteBtn.textContent = '⬆ Send to Production';

    // Once a production run has replaced the result, the draft stays reachable for an A/B.
    const draftBtn = document.getElementById('viewDraftBtn');
    if (data && data.has_draft) {
        draftBtn.href = `/static/supersplat/index.html?load=/ply/${currentJobId}/draft.ply`;
        draftBtn.style.display = 'inline-block';
    } else {
        draftBtn.style.display = 'none';
    }
}

/* ── Jobs tab ───────────────────────────────────── */
function _agoLabel(epochSeconds) {
    const mins = Math.max(0, (Date.now() / 1000 - epochSeconds) / 60);
    if (mins < 1) return 'just now';
    if (mins < 60) return Math.round(mins) + ' min ago';
    const hrs = mins / 60;
    if (hrs < 24) return Math.round(hrs) + (Math.round(hrs) === 1 ? ' hour ago' : ' hours ago');
    const days = Math.round(hrs / 24);
    return days + (days === 1 ? ' day ago' : ' days ago');
}

async function loadJobs() {
    const grid = document.getElementById('jobsGrid');
    const empty = document.getElementById('jobsEmpty');
    try {
        const response = await fetch('/jobs');
        const data = await response.json();
        const jobs = (data && data.jobs) || [];
        grid.innerHTML = '';
        if (!jobs.length) {
            empty.textContent = 'No jobs yet — create one from the Create tab.';
            empty.style.display = 'block';
            return;
        }
        empty.style.display = 'none';
        jobs.forEach(job => grid.appendChild(buildJobCard(job)));
    } catch (e) {
        empty.textContent = 'Could not load jobs: ' + e;
        empty.style.display = 'block';
    }
}

function buildJobCard(job) {
    const card = document.createElement('div');
    card.className = 'job-card' + (job.job_id === currentJobId ? ' is-current' : '');

    const running = job.status === 'processing';
    const badge = running ? ['run', 'Running']
                : job.status === 'completed' ? ['ok', job.quality || 'done']
                : job.status === 'error' ? ['err', 'Failed']
                : ['', job.status || 'unknown'];

    const thumb = document.createElement(job.has_thumb ? 'img' : 'div');
    thumb.className = 'job-thumb' + (job.has_thumb ? '' : ' placeholder');
    if (job.has_thumb) {
        thumb.src = `/jobs/${job.job_id}/thumb.jpg`;
        thumb.alt = '';
        thumb.loading = 'lazy';
        // A job whose images were cleaned up still lists — don't leave a broken icon.
        thumb.onerror = () => { thumb.replaceWith(Object.assign(document.createElement('div'),
            { className: 'job-thumb placeholder', textContent: 'no preview' })); };
    } else {
        thumb.textContent = 'no preview';
    }
    thumb.onclick = () => selectJob(job.job_id);
    card.appendChild(thumb);

    const body = document.createElement('div');
    body.className = 'job-body';
    body.innerHTML = `
        <div class="job-title">${job.job_id.slice(0, 8)}</div>
        <div class="job-badges">
            <span class="job-badge ${badge[0]}">${badge[1]}</span>
            ${job.has_draft ? '<span class="job-badge">draft kept</span>' : ''}
        </div>
        <div class="job-meta">
            ${_agoLabel(job.created)}${job.images ? ' · ' + job.images + ' images' : ''}
            ${job.result_mb ? '<br>' + job.result_mb + ' MB result' : ''}
            ${running && job.step ? '<br>' + job.step + ' (' + (job.progress || 0) + '%)' : ''}
        </div>`;

    const actions = document.createElement('div');
    actions.className = 'job-actions';

    const open = document.createElement('button');
    open.className = 'job-btn' + (job.has_result || running ? ' primary' : '');
    open.textContent = running ? 'Watch' : 'Open';
    open.disabled = !job.has_result && !running;
    open.onclick = () => selectJob(job.job_id);
    actions.appendChild(open);

    if (job.can_promote) {
        const promote = document.createElement('button');
        promote.className = 'job-btn';
        promote.textContent = '⬆ Production';
        // Point at the job directly rather than via selectJob(): sendToProduction() only
        // needs currentJobId, and selectJob's async tail would race it re-rendering the
        // result buttons this click is about to change.
        promote.onclick = () => { currentJobId = job.job_id; _notifiedJob = null; sendToProduction(); };
        actions.appendChild(promote);
    }

    body.appendChild(actions);
    card.appendChild(body);
    return card;
}

/* Attach the UI to an existing job — the way back in after a page reload, since the
   browser otherwise only knows the job it uploaded itself this session. */
async function selectJob(jobId, opts) {
    const silent = opts && opts.silent;
    currentJobId = jobId;
    _notifiedJob = null;      // this run hasn't been announced to THIS page yet
    resetPreview();
    try {
        const response = await fetch(`/status/${jobId}`);
        const data = await response.json();
        if (!response.ok) {
            showError(data.error || 'Could not open that job');
            return;
        }
        if (data.status === 'processing') {
            // Re-attach to a run already in flight and resume the live view.
            document.getElementById('idleState').style.display = 'none';
            progressSection.classList.add('active');
            resultSection.classList.remove('active');
            errorMsg.style.display = 'none';
            setTabDot('process', 'running');
            processingStartTime = (data.started_at ? data.started_at * 1000 : Date.now());
            startElapsedTimer();
            if (!silent) switchTab('process');
            pollStatus();
            pollLogs();
        } else {
            updateStageTimings(data.stages);
            showResults(data);
            setTabDot('results', 'done');
            if (!silent) switchTab('results');
        }
    } catch (e) {
        showError('Could not open that job: ' + e);
    }
}

async function sendToProduction() {
    const btn = document.getElementById('promoteBtn');
    if (!currentJobId || btn.disabled) return;
    if (!confirm('Re-run this scene at production quality?\n\n'
               + 'It reuses the camera solve you already have and retrains with MCMC + Brush '
               + '(~2.5 hr). Your draft is kept and stays viewable.')) return;

    btn.disabled = true;
    btn.textContent = 'Starting...';
    try {
        const response = await fetch(`/promote/${currentJobId}`, { method: 'POST' });
        const data = await response.json();
        if (!response.ok) {
            showError(data.error || 'Could not start the production run');
            btn.disabled = false;
            btn.textContent = '⬆ Send to Production';
            return;
        }
        // Same job id, so the existing pollers just pick the new run up. Mirror
        // startProcessing()'s reset of the Process tab, then hand back to pollStatus().
        document.getElementById('idleState').style.display = 'none';
        progressSection.classList.add('active');
        resultSection.classList.remove('active');
        document.getElementById('resultsEmpty').style.display = 'flex';
        errorMsg.style.display = 'none';
        setTabDot('process', 'running');
        setTabDot('results', '');
        switchTab('process');

        resetPipelineStages();
        progressFill.style.width = '0%';
        document.getElementById('currentActivity').textContent = 'Production pipeline starting...';
        _notifiedJob = null;          // re-arm the completion notification for this run
        processingStartTime = Date.now();
        startElapsedTimer();
        pollStatus();
        pollLogs();
    } catch (e) {
        showError('Could not start the production run: ' + e);
        btn.disabled = false;
        btn.textContent = '⬆ Send to Production';
    }
}

async function exportCameraTracking() {
    const btn = document.getElementById('trackingExportBtn');
    const status = document.getElementById('trackingStatus');
    const downloads = document.getElementById('trackingDownloads');
    const format = document.getElementById('trackingFormat').value;
    const fps = document.getElementById('trackingFps').value;
    const includePc = document.getElementById('trackingPc').checked;

    btn.disabled = true;
    btn.textContent = 'Exporting...';
    status.textContent = 'Extracting camera poses and exporting...';
    status.className = 'tracking-status';
    downloads.style.display = 'none';
    downloads.innerHTML = '';

    try {
        const params = new URLSearchParams({
            job_id: currentJobId,
            format: format,
            fps: fps,
            pointcloud: includePc
        });

        const response = await fetch(`/api/camera-tracking?${params}`);
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Export failed');
        }

        status.textContent = `✓ Exported ${data.poses_count} camera poses at ${data.fps} FPS`;
        status.className = 'tracking-status';

        // Build download links
        const fileMap = {
            'json': 'cameras.json',
            'gltf': 'cameras.gltf',
            'fbx': 'cameras.fbx',
            'blender': 'blender_camera.zip',
            'ply': 'sparse_cloud.ply'
        };

        data.files.forEach(file => {
            const fileName = fileMap[file.format] || file.format;
            const link = document.createElement('a');
            link.href = `/download/${currentJobId}/tracking/${fileName}`;
            link.download = fileName;
            link.textContent = `⬇ ${file.format.toUpperCase()}`;
            downloads.appendChild(link);
        });

        downloads.style.display = 'flex';
    } catch (error) {
        status.textContent = `✗ ${error.message}`;
        status.className = 'tracking-status error';
    } finally {
        btn.disabled = false;
        btn.textContent = 'Export Camera Tracking';
    }
}

function showError(message) {
    errorMsg.textContent = message;
    errorMsg.style.display = 'block';
}

async function killAllProcesses() {
    if (!confirm('Stop all running processes?')) return;
    try {
        const response = await fetch('/kill', { method: 'POST' });
        const data = await response.json();
        alert(data.message);
        progressSection.classList.remove('active');
        document.getElementById('idleState').style.display = 'flex';
        setTabDot('process', '');
        processBtn.disabled = selectedFiles.length === 0;
    } catch (error) {
        alert('Error: ' + error.message);
    }
}

async function cleanupJobs() {
    if (!confirm('Delete all previous jobs?')) return;
    try {
        const response = await fetch('/cleanup', { method: 'POST' });
        const data = await response.json();
        if (response.ok) {
            alert('Cleanup complete!');
            location.reload();
        } else {
            alert('Cleanup failed: ' + (data.error || 'Unknown error'));
        }
    } catch (error) {
        alert('Cleanup failed: ' + error.message);
    }
}

async function openSplatFile(input) {
    const file = input.files[0];
    if (!file) return;
    input.value = '';

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/upload-for-view', { method: 'POST', body: formData });
        const data = await response.json();
        if (data.token) {
            window.open(`/static/supersplat/index.html?load=/ply/${data.token}.ply`, '_blank');
        } else {
            alert('Error: ' + (data.error || 'Upload failed'));
        }
    } catch (err) {
        alert('Error uploading file: ' + err.message);
    }
}

/* ── Startup: re-attach to a run already in flight ──
   A reload wipes currentJobId, which used to strand a running job with no way back to
   its progress. Ask the server what's running and silently re-attach (no tab switch —
   the user may have reloaded intending to start something new). */
(async function reattachRunningJob() {
    try {
        const response = await fetch('/jobs');
        const data = await response.json();
        const running = ((data && data.jobs) || []).find(j => j.status === 'processing');
        if (running) {
            await selectJob(running.job_id, { silent: true });
            setTabDot('process', 'running');
        }
    } catch (e) {
        /* listing is best-effort — never block the page on it */
    }
})();
