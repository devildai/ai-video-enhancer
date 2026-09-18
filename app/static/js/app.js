/**
 * AI Video Enhancer — Frontend Application Logic
 * Pure Vanilla JavaScript (Zero External Dependencies)
 */

(function () {
  'use strict';

  // =========================================================================
  // 1. Application State
  // =========================================================================
  const state = {
    taskId: null,
    filename: '',
    fileSize: 0,
    metadata: {
      width: 0,
      height: 0,
      fps: 0,
      duration: 0,
      totalFrames: 0,
      codec: '',
      hasAudio: false,
      audioChannels: 0,
      audioCodec: ''
    },
    originalPreviewUrl: '',
    enhancedPreviewUrl: '',
    config: {
      resolution: '1080p',
      fps: '60fps',
      mode: 'quality'
    },
    processing: {
      active: false,
      progress: 0,
      stage: '',
      fps: 0,
      etaSec: 0,
      startTime: null,
      timerInterval: null
    },
    stream: {
      eventSource: null,
      webSocket: null,
      pollInterval: null
    },
    slider: {
      position: 50, // 0 to 100%
      isDragging: false,
      zoom: 1
    }
  };

  // =========================================================================
  // 2. DOM Elements Cache
  // =========================================================================
  const elements = {};

  function cacheDomElements() {
    // Header & Alert
    elements.systemStatusPill = document.getElementById('system-status-pill');
    elements.globalAlert = document.getElementById('global-alert');
    elements.alertTitle = document.getElementById('alert-title');
    elements.alertMessage = document.getElementById('alert-message');
    elements.alertCloseBtn = document.getElementById('alert-close-btn');
    elements.toastContainer = document.getElementById('toast-container');

    // Sections
    elements.uploadSection = document.getElementById('upload-section');
    elements.configSection = document.getElementById('config-section');
    elements.processingSection = document.getElementById('processing-section');
    elements.completedSection = document.getElementById('completed-section');

    // Upload Elements
    elements.dropzone = document.getElementById('dropzone');
    elements.fileInput = document.getElementById('file-input');
    elements.browseBtn = document.getElementById('browse-btn');
    elements.uploadProgressOverlay = document.getElementById('upload-progress-overlay');
    elements.uploadStatusText = document.getElementById('upload-status-text');
    elements.uploadProgressFill = document.getElementById('upload-progress-fill');

    // Metadata Elements
    elements.loadedFilename = document.getElementById('loaded-filename');
    elements.btnChangeVideo = document.getElementById('btn-change-video');
    elements.inputVideoPlayer = document.getElementById('input-video-player');
    elements.metaDurationDisplay = document.getElementById('meta-duration-display');
    elements.metaResolution = document.getElementById('meta-resolution');
    elements.metaAspect = document.getElementById('meta-aspect');
    elements.metaFps = document.getElementById('meta-fps');
    elements.metaTotalFrames = document.getElementById('meta-total-frames');
    elements.metaDuration = document.getElementById('meta-duration');
    elements.metaFilesize = document.getElementById('meta-filesize');
    elements.metaCodec = document.getElementById('meta-codec');
    elements.metaAudio = document.getElementById('meta-audio');
    elements.metaAudioChannels = document.getElementById('meta-audio-channels');
    elements.metaBudget = document.getElementById('meta-budget');

    // Split Comparison Slider Elements
    elements.sliderContainer = document.getElementById('slider-container');
    elements.originalLayer = document.getElementById('original-layer');
    elements.sliderDivider = document.getElementById('slider-divider');
    elements.enhancedSampleImg = document.getElementById('enhanced-sample-img');
    elements.originalSampleImg = document.getElementById('original-sample-img');
    elements.sampleLoadingOverlay = document.getElementById('sample-loading-overlay');
    elements.sampleEmptyOverlay = document.getElementById('sample-empty-overlay');
    elements.btnPreviewSample = document.getElementById('btn-preview-sample');
    elements.zoom1x = document.getElementById('zoom-1x');
    elements.zoom2x = document.getElementById('zoom-2x');

    // Config Elements
    elements.targetResCalc = document.getElementById('target-res-calc');
    elements.targetFpsCalc = document.getElementById('target-fps-calc');
    elements.pipelinePlanSummary = document.getElementById('pipeline-plan-summary');
    elements.btnStartProcess = document.getElementById('btn-start-process');

    // Processing Elements
    elements.btnCancelTask = document.getElementById('btn-cancel-task');
    elements.stageDescription = document.getElementById('stage-description');
    elements.progressPercentageNum = document.getElementById('progress-percentage-num');
    elements.progressFill = document.getElementById('progress-fill');
    elements.metricSpeed = document.getElementById('metric-speed');
    elements.metricEta = document.getElementById('metric-eta');
    elements.metricElapsed = document.getElementById('metric-elapsed');
    elements.metricFrameCounter = document.getElementById('metric-frame-counter');
    elements.metricFrameRatePct = document.getElementById('metric-frame-rate-pct');

    // Stepper Nodes
    elements.stepDemux = document.getElementById('step-demux');
    elements.stepDecode = document.getElementById('step-decode');
    elements.stepEnhance = document.getElementById('step-enhance');
    elements.stepInterpolate = document.getElementById('step-interpolate');
    elements.stepEncode = document.getElementById('step-encode');
    elements.conn1 = document.getElementById('conn-1');
    elements.conn2 = document.getElementById('conn-2');
    elements.conn3 = document.getElementById('conn-3');
    elements.conn4 = document.getElementById('conn-4');

    // Completed Elements
    elements.completeSrcVideo = document.getElementById('complete-src-video');
    elements.completeEnhVideo = document.getElementById('complete-enh-video');
    elements.completeSrcStats = document.getElementById('complete-src-stats');
    elements.completeEnhStats = document.getElementById('complete-enh-stats');
    elements.btnSyncPlay = document.getElementById('btn-sync-play');
    elements.resOutputBadge = document.getElementById('res-output-badge');
    elements.fpsOutputBadge = document.getElementById('fps-output-badge');
    elements.timeOutputBadge = document.getElementById('time-output-badge');
    elements.btnDownloadVideo = document.getElementById('btn-download-video');
    elements.btnEnhanceAnother = document.getElementById('btn-enhance-another');
  }

  // =========================================================================
  // 3. UI Helper Utilities
  // =========================================================================
  function formatSeconds(seconds) {
    if (!seconds || isNaN(seconds) || seconds < 0) return '00:00';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  }

  function formatFileSize(bytes) {
    if (!bytes || isNaN(bytes)) return '0 MB';
    const mb = bytes / (1024 * 1024);
    if (mb >= 1000) {
      return `${(mb / 1024).toFixed(2)} GB`;
    }
    return `${mb.toFixed(1)} MB`;
  }

  function calculateAspectRatio(width, height) {
    if (!width || !height) return '16:9';
    function gcd(a, b) {
      return b === 0 ? a : gcd(b, a % b);
    }
    const divisor = gcd(width, height);
    const rW = width / divisor;
    const rH = height / divisor;
    if ((rW === 16 && rH === 9) || (rW === 64 && rH === 36)) return '16:9';
    if (rW === 4 && rH === 3) return '4:3';
    if (rW === 9 && rH === 16) return '9:16 (Vertical)';
    if (rW === 1 && rH === 1) return '1:1 (Square)';
    return `${rW}:${rH}`;
  }

  function showToast(title, message, type = 'info') {
    if (!elements.toastContainer) return;
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    
    let iconSvg = '';
    if (type === 'error') {
      iconSvg = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="toast-icon"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>';
    } else if (type === 'success') {
      iconSvg = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="toast-icon"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>';
    } else {
      iconSvg = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="toast-icon"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>';
    }

    toast.innerHTML = `
      ${iconSvg}
      <div class="toast-body">
        <div class="toast-title">${escapeHtml(title)}</div>
        <div class="toast-msg">${escapeHtml(message)}</div>
      </div>
      <button type="button" class="toast-dismiss" aria-label="Close notification">&times;</button>
    `;

    elements.toastContainer.appendChild(toast);

    const dismissBtn = toast.querySelector('.toast-dismiss');
    const removeToast = () => {
      toast.classList.add('toast-exit');
      setTimeout(() => {
        if (toast.parentNode) toast.parentNode.removeChild(toast);
      }, 300);
    };

    dismissBtn.addEventListener('click', removeToast);
    setTimeout(removeToast, 6000);
  }

  function showAlert(title, message, type = 'error') {
    if (!elements.globalAlert) return;
    elements.globalAlert.className = `alert-banner alert-${type}`;
    elements.alertTitle.textContent = title;
    elements.alertMessage.textContent = message;
    elements.globalAlert.classList.remove('hidden');
  }

  function hideAlert() {
    if (elements.globalAlert) {
      elements.globalAlert.classList.add('hidden');
    }
  }

  function escapeHtml(str) {
    if (!str) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  // =========================================================================
  // 4. Preset Calculation & UI Update
  // =========================================================================
  function updatePresetCalculations() {
    const srcW = state.metadata.width || 1280;
    const srcH = state.metadata.height || 720;
    const srcFps = state.metadata.fps || 30;

    // 1. Resolution Calculation
    let targetW = srcW;
    let targetH = srcH;
    let resLabel = '';

    const aspect = srcW / srcH;
    if (state.config.resolution === 'Original') {
      targetW = srcW;
      targetH = srcH;
      resLabel = `${targetW} × ${targetH} (Original)`;
    } else if (state.config.resolution === '1080p') {
      targetH = 1080;
      targetW = Math.round((1080 * aspect) / 2) * 2;
      resLabel = `${targetW} × 1080 (Full HD)`;
    } else if (state.config.resolution === '2K') {
      targetH = 1440;
      targetW = Math.round((1440 * aspect) / 2) * 2;
      resLabel = `${targetW} × 1440 (2K QHD)`;
    } else if (state.config.resolution === '4K') {
      targetH = 2160;
      targetW = Math.round((2160 * aspect) / 2) * 2;
      resLabel = `${targetW} × 2160 (4K UHD)`;
    }

    elements.targetResCalc.textContent = resLabel;

    // 2. FPS Calculation
    let targetFps = srcFps;
    let fpsMultiplierText = '';

    if (state.config.fps === 'Original') {
      targetFps = srcFps;
      fpsMultiplierText = `${targetFps.toFixed(1)} FPS (1.0× Preserved)`;
    } else if (state.config.fps === '30fps') {
      targetFps = 30;
      const mult = (30 / srcFps).toFixed(1);
      fpsMultiplierText = `30 FPS (${mult}×)`;
    } else if (state.config.fps === '60fps') {
      targetFps = 60;
      const mult = (60 / srcFps).toFixed(1);
      fpsMultiplierText = `60 FPS (${mult}× Smooth)`;
    } else if (state.config.fps === '120fps') {
      targetFps = 120;
      const mult = (120 / srcFps).toFixed(1);
      fpsMultiplierText = `120 FPS (${mult}× Ultra)`;
    }

    elements.targetFpsCalc.textContent = fpsMultiplierText;

    // 3. Plan Summary Text
    const modeName = state.config.mode === 'quality' ? 'Quality Mode' : 'Speed Mode';
    elements.pipelinePlanSummary.textContent = `${state.config.resolution} • ${state.config.fps} • ${modeName}`;
  }

  // =========================================================================
  // 5. Upload & File Selection Handlers
  // =========================================================================
  function initUploadHandlers() {
    const dropzone = elements.dropzone;
    const fileInput = elements.fileInput;
    const browseBtn = elements.browseBtn;

    browseBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      fileInput.click();
    });

    dropzone.addEventListener('click', () => {
      fileInput.click();
    });

    // Drag & Drop Events
    ['dragenter', 'dragover'].forEach((eventName) => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.add('drag-over');
      });
    });

    ['dragleave', 'drop'].forEach((eventName) => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.remove('drag-over');
      });
    });

    dropzone.addEventListener('drop', (e) => {
      const dt = e.dataTransfer;
      if (dt && dt.files && dt.files.length > 0) {
        handleFileSelection(dt.files[0]);
      }
    });

    fileInput.addEventListener('change', (e) => {
      if (fileInput.files && fileInput.files.length > 0) {
        handleFileSelection(fileInput.files[0]);
      }
    });

    elements.btnChangeVideo.addEventListener('click', () => {
      elements.fileInput.value = '';
      elements.configSection.classList.add('hidden');
      elements.completedSection.classList.add('hidden');
      elements.processingSection.classList.add('hidden');
      elements.uploadSection.classList.remove('hidden');
      window.scrollTo({ top: 0, behavior: 'smooth' });
    });
  }

  function handleFileSelection(file) {
    if (!file) return;

    // Validate video file extension
    const validExtensions = ['.mp4', '.mkv', '.avi', '.mov', '.webm'];
    const fileName = file.name.toLowerCase();
    const hasValidExt = validExtensions.some(ext => fileName.endsWith(ext));

    if (!hasValidExt && !file.type.startsWith('video/')) {
      showToast('Unsupported Format', 'Please upload a video file (.mp4, .mkv, .avi, .mov, .webm)', 'error');
      return;
    }

    if (file.size === 0) {
      showToast('Empty File', 'The selected file is empty (0 bytes).', 'error');
      return;
    }

    uploadVideoFile(file);
  }

  async function uploadVideoFile(file) {
    hideAlert();
    elements.uploadProgressOverlay.classList.remove('hidden');
    elements.uploadProgressFill.style.width = '20%';
    elements.uploadStatusText.textContent = `Uploading "${file.name}"...`;

    const formData = new FormData();
    formData.append('file', file);

    try {
      elements.uploadProgressFill.style.width = '60%';
      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData
      });

      elements.uploadProgressFill.style.width = '90%';

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Upload failed with HTTP ${response.status}`);
      }

      const data = await response.json();
      elements.uploadProgressFill.style.width = '100%';
      elements.uploadStatusText.textContent = 'Probing complete!';

      // Record state
      state.taskId = data.task_id || data.file_id;
      state.filename = data.filename || file.name;
      state.fileSize = data.file_size || file.size;

      const meta = data.metadata || {};
      state.metadata = {
        width: meta.width || 1280,
        height: meta.height || 720,
        fps: meta.fps ? Number(meta.fps) : 30.0,
        duration: meta.duration ? Number(meta.duration) : 0,
        totalFrames: meta.total_frames || meta.nb_frames || 0,
        codec: meta.video_codec || meta.codec_name || 'H.264',
        hasAudio: meta.has_audio !== undefined ? meta.has_audio : true,
        audioChannels: meta.audio_channels || 2,
        audioCodec: meta.audio_codec || 'aac'
      };

      state.originalPreviewUrl = data.preview_url || `/api/preview/${state.taskId}/original`;

      // Render metadata
      renderLoadedVideoMetadata(file);

      // Transition views
      setTimeout(() => {
        elements.uploadProgressOverlay.classList.add('hidden');
        elements.uploadSection.classList.add('hidden');
        elements.configSection.classList.remove('hidden');
        window.scrollTo({ top: 0, behavior: 'smooth' });
        showToast('Video Loaded', `Successfully probed ${state.filename}`, 'success');
      }, 400);

    } catch (err) {
      elements.uploadProgressOverlay.classList.add('hidden');
      showAlert('Upload Error', err.message, 'error');
      showToast('Upload Failed', err.message, 'error');
    }
  }

  function renderLoadedVideoMetadata(file) {
    elements.loadedFilename.textContent = state.filename;
    elements.metaResolution.textContent = `${state.metadata.width} × ${state.metadata.height}`;
    elements.metaAspect.textContent = calculateAspectRatio(state.metadata.width, state.metadata.height);
    elements.metaFps.textContent = `${state.metadata.fps.toFixed(2)} FPS`;
    elements.metaTotalFrames.textContent = `${state.metadata.totalFrames || Math.round(state.metadata.duration * state.metadata.fps)} frames`;
    elements.metaDuration.textContent = `${state.metadata.duration.toFixed(1)}s`;
    elements.metaDurationDisplay.textContent = formatSeconds(state.metadata.duration);
    elements.metaFilesize.textContent = formatFileSize(state.fileSize);
    elements.metaCodec.textContent = state.metadata.codec.toUpperCase();
    
    if (state.metadata.hasAudio) {
      elements.metaAudio.textContent = 'Preserved Sync';
      elements.metaAudioChannels.textContent = `${state.metadata.audioCodec.toUpperCase()} (${state.metadata.audioChannels} ch)`;
    } else {
      elements.metaAudio.textContent = 'None (Silent)';
      elements.metaAudioChannels.textContent = 'No Audio Track';
    }

    // Set preview player source
    if (file) {
      const objectUrl = URL.createObjectURL(file);
      elements.inputVideoPlayer.src = objectUrl;
      elements.completeSrcVideo.src = objectUrl;
    } else if (state.originalPreviewUrl) {
      elements.inputVideoPlayer.poster = state.originalPreviewUrl;
    }

    // Set original sample frame in split slider
    if (state.originalPreviewUrl) {
      elements.originalSampleImg.src = state.originalPreviewUrl;
    }

    // Duration limit warning
    if (state.metadata.duration > 60) {
      showAlert('Notice: Extended Duration', `This clip is ${state.metadata.duration.toFixed(1)}s long. Processing videos longer than 60s on CPU will take extra time.`, 'warning');
    }

    updatePresetCalculations();
  }

  // =========================================================================
  // 6. Split-View Comparison Slider
  // =========================================================================
  function initSplitSlider() {
    const container = elements.sliderContainer;
    const divider = elements.dividerHandle || elements.sliderDivider;
    const originalLayer = elements.originalLayer;

    function setDividerPosition(percentage) {
      percentage = Math.max(0, Math.min(100, percentage));
      state.slider.position = percentage;
      divider.style.left = `${percentage}%`;
      originalLayer.style.clipPath = `polygon(0 0, ${percentage}% 0, ${percentage}% 100%, 0 100%)`;
      container.setAttribute('aria-valuenow', Math.round(percentage));
    }

    function updateFromPointer(clientX) {
      const rect = container.getBoundingClientRect();
      const x = clientX - rect.left;
      const pct = (x / rect.width) * 100;
      setDividerPosition(pct);
    }

    // Pointer / Touch / Mouse dragging
    container.addEventListener('pointerdown', (e) => {
      state.slider.isDragging = true;
      container.setPointerCapture(e.pointerId);
      updateFromPointer(e.clientX);
    });

    container.addEventListener('pointermove', (e) => {
      if (!state.slider.isDragging) return;
      updateFromPointer(e.clientX);
    });

    container.addEventListener('pointerup', (e) => {
      state.slider.isDragging = false;
      try {
        container.releasePointerCapture(e.pointerId);
      } catch (_) {}
    });

    container.addEventListener('pointercancel', () => {
      state.slider.isDragging = false;
    });

    // Keyboard accessibility
    container.addEventListener('keydown', (e) => {
      const step = e.shiftKey ? 10 : 2;
      if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') {
        e.preventDefault();
        setDividerPosition(state.slider.position - step);
      } else if (e.key === 'ArrowRight' || e.key === 'ArrowUp') {
        e.preventDefault();
        setDividerPosition(state.slider.position + step);
      } else if (e.key === 'Home') {
        e.preventDefault();
        setDividerPosition(0);
      } else if (e.key === 'End') {
        e.preventDefault();
        setDividerPosition(100);
      }
    });

    // Zoom Controls
    elements.zoom1x.addEventListener('click', () => {
      elements.zoom1x.classList.add('active');
      elements.zoom2x.classList.remove('active');
      container.classList.remove('zoom-2x');
      state.slider.zoom = 1;
    });

    elements.zoom2x.addEventListener('click', () => {
      elements.zoom2x.classList.add('active');
      elements.zoom1x.classList.remove('active');
      container.classList.add('zoom-2x');
      state.slider.zoom = 2;
    });

    // Generate Enhanced Sample Frame Button
    elements.btnPreviewSample.addEventListener('click', async () => {
      if (!state.taskId) return;
      elements.sampleLoadingOverlay.classList.remove('hidden');
      elements.sampleEmptyOverlay.classList.add('hidden');

      try {
        const resp = await fetch(`/api/compare-frame/${state.taskId}?mode=${state.config.mode}&resolution=${state.config.resolution}`);
        
        if (resp.status === 200) {
          const contentType = resp.headers.get('content-type') || '';
          if (contentType.includes('application/json')) {
            const data = await resp.json();
            if (data.original_url) elements.originalSampleImg.src = data.original_url;
            if (data.enhanced_url) elements.enhancedSampleImg.src = data.enhanced_url;
          } else {
            // Direct binary image
            const blob = await resp.blob();
            elements.enhancedSampleImg.src = URL.createObjectURL(blob);
          }
          elements.sampleLoadingOverlay.classList.add('hidden');
          setDividerPosition(50);
          showToast('Sample Ready', 'Enhanced neural preview generated successfully', 'success');
        } else if (resp.status === 202) {
          showToast('Rendering Sample', 'Sample frame is currently rendering on CPU...', 'info');
          // Poll after 2s
          setTimeout(() => elements.btnPreviewSample.click(), 2000);
        } else {
          throw new Error(`Server returned HTTP ${resp.status}`);
        }
      } catch (err) {
        elements.sampleLoadingOverlay.classList.add('hidden');
        elements.sampleEmptyOverlay.classList.remove('hidden');
        showToast('Preview Unavailable', `Could not generate preview: ${err.message}`, 'error');
      }
    });

    // Default position
    setDividerPosition(50);
  }

  // =========================================================================
  // 7. Preset Configuration Buttons
  // =========================================================================
  function initPresetButtons() {
    // Resolution Chips
    const resChips = document.querySelectorAll('[data-res]');
    resChips.forEach((chip) => {
      chip.addEventListener('click', () => {
        resChips.forEach(c => {
          c.classList.remove('active');
          c.setAttribute('aria-checked', 'false');
        });
        chip.classList.add('active');
        chip.setAttribute('aria-checked', 'true');
        state.config.resolution = chip.getAttribute('data-res');
        updatePresetCalculations();
      });
    });

    // FPS Chips
    const fpsChips = document.querySelectorAll('[data-fps]');
    fpsChips.forEach((chip) => {
      chip.addEventListener('click', () => {
        fpsChips.forEach(c => {
          c.classList.remove('active');
          c.setAttribute('aria-checked', 'false');
        });
        chip.classList.add('active');
        chip.setAttribute('aria-checked', 'true');
        state.config.fps = chip.getAttribute('data-fps');
        updatePresetCalculations();
      });
    });

    // Mode Cards
    const modeCards = document.querySelectorAll('[data-mode]');
    modeCards.forEach((card) => {
      card.addEventListener('click', () => {
        modeCards.forEach(c => {
          c.classList.remove('active');
          c.setAttribute('aria-checked', 'false');
        });
        card.classList.add('active');
        card.setAttribute('aria-checked', 'true');
        state.config.mode = card.getAttribute('data-mode');
        updatePresetCalculations();
      });

      card.addEventListener('keydown', (e) => {
        if (e.key === ' ' || e.key === 'Enter') {
          e.preventDefault();
          card.click();
        }
      });
    });
  }

  // =========================================================================
  // 8. Video Enhancement Launch & Progress Tracking
  // =========================================================================
  function initProcessHandlers() {
    elements.btnStartProcess.addEventListener('click', startEnhancementJob);
    elements.btnCancelTask.addEventListener('click', cancelEnhancementJob);
    elements.btnEnhanceAnother.addEventListener('click', resetToUploadView);

    if (elements.alertCloseBtn) {
      elements.alertCloseBtn.addEventListener('click', hideAlert);
    }
  }

  async function startEnhancementJob() {
    if (!state.taskId) {
      showToast('Error', 'Please upload a video file first.', 'error');
      return;
    }

    elements.btnStartProcess.disabled = true;
    hideAlert();

    const payload = {
      task_id: state.taskId,
      file_id: state.taskId,
      resolution: state.config.resolution,
      target_resolution: state.config.resolution,
      fps: state.config.fps,
      target_fps: state.config.fps,
      mode: state.config.mode,
      enhancement_mode: state.config.mode
    };

    try {
      const response = await fetch('/api/process', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Processing start failed (${response.status})`);
      }

      const resData = await response.json();
      if (resData.task_id) {
        state.taskId = resData.task_id;
      }

      // Transition to Processing View
      elements.configSection.classList.add('hidden');
      elements.completedSection.classList.add('hidden');
      elements.processingSection.classList.remove('hidden');
      window.scrollTo({ top: 0, behavior: 'smooth' });

      // Start timers and connection
      state.processing.active = true;
      state.processing.startTime = Date.now();
      startElapsedTimer();
      connectProgressStream(state.taskId);

      showToast('Enhancement Started', 'Job dispatched to AI streaming pipeline', 'info');

    } catch (err) {
      elements.btnStartProcess.disabled = false;
      showAlert('Pipeline Launch Failed', err.message, 'error');
      showToast('Launch Error', err.message, 'error');
    }
  }

  function startElapsedTimer() {
    if (state.processing.timerInterval) clearInterval(state.processing.timerInterval);
    state.processing.timerInterval = setInterval(() => {
      if (!state.processing.active) return;
      const elapsedSec = Math.floor((Date.now() - state.processing.startTime) / 1000);
      elements.metricElapsed.textContent = `Elapsed: ${formatSeconds(elapsedSec)}`;
    }, 1000);
  }

  function updateStepperStages(stageName, percentage) {
    const s = (stageName || '').toLowerCase();
    
    // Reset classes
    [elements.stepDemux, elements.stepDecode, elements.stepEnhance, elements.stepInterpolate, elements.stepEncode].forEach(node => {
      if (node) node.className = 'step-node';
    });
    [elements.conn1, elements.conn2, elements.conn3, elements.conn4].forEach(conn => {
      if (conn) conn.className = 'step-connector';
    });

    if (s.includes('demux') || percentage < 10) {
      elements.stepDemux.classList.add('active');
    } else if (s.includes('decode') || percentage < 25) {
      elements.stepDemux.classList.add('complete');
      elements.conn1.classList.add('active');
      elements.stepDecode.classList.add('active');
    } else if (s.includes('enhanc') || percentage < 60) {
      elements.stepDemux.classList.add('complete');
      elements.conn1.classList.add('active');
      elements.stepDecode.classList.add('complete');
      elements.conn2.classList.add('active');
      elements.stepEnhance.classList.add('active');
    } else if (s.includes('interp') || percentage < 85) {
      elements.stepDemux.classList.add('complete');
      elements.conn1.classList.add('active');
      elements.stepDecode.classList.add('complete');
      elements.conn2.classList.add('active');
      elements.stepEnhance.classList.add('complete');
      elements.conn3.classList.add('active');
      elements.stepInterpolate.classList.add('active');
    } else {
      elements.stepDemux.classList.add('complete');
      elements.conn1.classList.add('active');
      elements.stepDecode.classList.add('complete');
      elements.conn2.classList.add('active');
      elements.stepEnhance.classList.add('complete');
      elements.conn3.classList.add('active');
      elements.stepInterpolate.classList.add('complete');
      elements.conn4.classList.add('active');
      elements.stepEncode.classList.add('active');
    }
  }

  function handleProgressUpdate(data) {
    if (!data) return;

    const pct = typeof data.progress === 'number' ? data.progress : (data.percentage || 0);
    const clampedPct = Math.max(0, Math.min(100, pct));
    state.processing.progress = clampedPct;

    elements.progressFill.style.width = `${clampedPct.toFixed(1)}%`;
    elements.progressPercentageNum.textContent = `${Math.round(clampedPct)}%`;

    const stageDisplay = data.stage_display || data.stage || 'Processing frame stream...';
    elements.stageDescription.textContent = stageDisplay;
    updateStepperStages(stageDisplay, clampedPct);

    // Speed
    if (data.fps) {
      elements.metricSpeed.textContent = `${Number(data.fps).toFixed(1)} FPS`;
    }

    // ETA
    if (data.eta_sec !== undefined && data.eta_sec !== null) {
      elements.metricEta.textContent = `ETA: ${formatSeconds(data.eta_sec)}`;
    } else if (data.eta) {
      elements.metricEta.textContent = `ETA: ${data.eta}`;
    }

    // Frame counters
    if (data.current_frame && data.total_frames) {
      elements.metricFrameCounter.textContent = `${data.current_frame} / ${data.total_frames}`;
      elements.metricFrameRatePct.textContent = `${clampedPct.toFixed(0)}% rendered`;
    }

    // Comparison sample ready event
    if (data.enhanced_preview_url || data.preview_url) {
      const url = data.enhanced_preview_url || data.preview_url;
      elements.enhancedSampleImg.src = url;
      elements.sampleEmptyOverlay.classList.add('hidden');
    }

    // Check completion
    if (clampedPct >= 100 || data.status === 'completed' || data.status === 'finished') {
      onProcessingComplete(data);
    }
  }

  function connectProgressStream(taskId) {
    closeProgressConnections();

    // 1. Try Server-Sent Events (SSE)
    try {
      const sse = new EventSource(`/api/progress/${taskId}`);
      state.stream.eventSource = sse;

      sse.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          handleProgressUpdate(data);
        } catch (_) {}
      };

      sse.addEventListener('progress', (event) => {
        try {
          const data = JSON.parse(event.data);
          handleProgressUpdate(data);
        } catch (_) {}
      });

      sse.addEventListener('comparison_ready', (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.enhanced_preview_url) {
            elements.enhancedSampleImg.src = data.enhanced_preview_url;
            elements.sampleEmptyOverlay.classList.add('hidden');
          }
        } catch (_) {}
      });

      sse.addEventListener('complete', (event) => {
        try {
          const data = JSON.parse(event.data);
          onProcessingComplete(data);
        } catch (_) {}
      });

      sse.addEventListener('error', (event) => {
        // SSE disconnected or failed -> fallback to WebSocket or polling
        sse.close();
        state.stream.eventSource = null;
        connectWebSocketFallback(taskId);
      });

    } catch (e) {
      connectWebSocketFallback(taskId);
    }
  }

  function connectWebSocketFallback(taskId) {
    try {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${protocol}//${window.location.host}/ws/progress/${taskId}`;
      const ws = new WebSocket(wsUrl);
      state.stream.webSocket = ws;

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          handleProgressUpdate(data);
        } catch (_) {}
      };

      ws.onerror = () => {
        ws.close();
        startPollingFallback(taskId);
      };

      ws.onclose = () => {
        if (state.processing.active && state.processing.progress < 100) {
          startPollingFallback(taskId);
        }
      };

    } catch (_) {
      startPollingFallback(taskId);
    }
  }

  function startPollingFallback(taskId) {
    if (state.stream.pollInterval) return;
    state.stream.pollInterval = setInterval(async () => {
      if (!state.processing.active) {
        clearInterval(state.stream.pollInterval);
        state.stream.pollInterval = null;
        return;
      }
      try {
        const resp = await fetch(`/api/status/${taskId}`);
        if (resp.ok) {
          const data = await resp.json();
          handleProgressUpdate(data.progress ? { ...data.progress, status: data.status } : data);
          if (data.status === 'completed') {
            clearInterval(state.stream.pollInterval);
            onProcessingComplete(data);
          } else if (data.status === 'failed' || data.status === 'error') {
            clearInterval(state.stream.pollInterval);
            onProcessingFailed(data.error || 'Enhancement job failed on server.');
          }
        }
      } catch (_) {}
    }, 1500);
  }

  function closeProgressConnections() {
    if (state.stream.eventSource) {
      state.stream.eventSource.close();
      state.stream.eventSource = null;
    }
    if (state.stream.webSocket) {
      state.stream.webSocket.close();
      state.stream.webSocket = null;
    }
    if (state.stream.pollInterval) {
      clearInterval(state.stream.pollInterval);
      state.stream.pollInterval = null;
    }
  }

  async function cancelEnhancementJob() {
    if (!state.taskId) return;
    const confirmCancel = confirm('Are you sure you want to cancel the video enhancement?');
    if (!confirmCancel) return;

    try {
      await fetch(`/api/cancel/${state.taskId}`, { method: 'POST' });
    } catch (_) {}

    state.processing.active = false;
    closeProgressConnections();
    if (state.processing.timerInterval) clearInterval(state.processing.timerInterval);

    elements.processingSection.classList.add('hidden');
    elements.configSection.classList.remove('hidden');
    elements.btnStartProcess.disabled = false;
    showToast('Job Cancelled', 'Processing was aborted. Temporary files removed.', 'info');
  }

  function onProcessingFailed(errMsg) {
    state.processing.active = false;
    closeProgressConnections();
    if (state.processing.timerInterval) clearInterval(state.processing.timerInterval);

    elements.processingSection.classList.add('hidden');
    elements.configSection.classList.remove('hidden');
    elements.btnStartProcess.disabled = false;

    showAlert('Enhancement Error', errMsg, 'error');
    showToast('Processing Error', errMsg, 'error');
  }

  function onProcessingComplete(data) {
    state.processing.active = false;
    closeProgressConnections();
    if (state.processing.timerInterval) clearInterval(state.processing.timerInterval);

    // Calculate total time
    const elapsedSec = Math.floor((Date.now() - state.processing.startTime) / 1000);
    const downloadUrl = (data && data.download_url) || `/api/download/${state.taskId}`;

    // Update Completed View elements
    elements.resOutputBadge.textContent = elements.targetResCalc.textContent.split(' ')[0] || '1920 × 1080';
    elements.fpsOutputBadge.textContent = elements.targetFpsCalc.textContent.split(' ')[0] || '60 FPS';
    elements.timeOutputBadge.textContent = formatSeconds(elapsedSec);

    elements.completeSrcStats.textContent = `${state.metadata.width}×${state.metadata.height} @ ${state.metadata.fps.toFixed(0)} FPS`;
    elements.completeEnhStats.textContent = `${elements.resOutputBadge.textContent} @ ${elements.fpsOutputBadge.textContent}`;

    // Configure Download CTA button
    elements.btnDownloadVideo.href = downloadUrl;
    elements.btnDownloadVideo.setAttribute('download', `enhanced_${state.filename}`);

    // Set Enhanced Video player source
    elements.completeEnhVideo.src = downloadUrl;

    // Show completed section
    elements.processingSection.classList.add('hidden');
    elements.completedSection.classList.remove('hidden');
    window.scrollTo({ top: 0, behavior: 'smooth' });

    showToast('All Done!', 'Enhanced video is ready for download.', 'success');
  }

  // =========================================================================
  // 9. Sync Video Playback Feature
  // =========================================================================
  function initSyncPlayback() {
    const srcVid = elements.completeSrcVideo;
    const enhVid = elements.completeEnhVideo;
    const btnSync = elements.btnSyncPlay;

    let isSyncing = false;

    btnSync.addEventListener('click', () => {
      if (srcVid.paused) {
        enhVid.currentTime = srcVid.currentTime;
        srcVid.play();
        enhVid.play();
        btnSync.innerHTML = `
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="btn-icon">
            <rect x="6" y="4" width="4" height="16"></rect>
            <rect x="14" y="4" width="4" height="16"></rect>
          </svg>
          <span>Pause Dual Playback</span>
        `;
      } else {
        srcVid.pause();
        enhVid.pause();
        btnSync.innerHTML = `
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="btn-icon">
            <polygon points="5 3 19 12 5 21 5 3"></polygon>
          </svg>
          <span>Play Videos in Sync</span>
        `;
      }
    });

    srcVid.addEventListener('seeked', () => {
      if (!isSyncing) {
        isSyncing = true;
        enhVid.currentTime = srcVid.currentTime;
        setTimeout(() => { isSyncing = false; }, 50);
      }
    });

    enhVid.addEventListener('seeked', () => {
      if (!isSyncing) {
        isSyncing = true;
        srcVid.currentTime = enhVid.currentTime;
        setTimeout(() => { isSyncing = false; }, 50);
      }
    });
  }

  function resetToUploadView() {
    closeProgressConnections();
    if (state.processing.timerInterval) clearInterval(state.processing.timerInterval);

    state.taskId = null;
    state.filename = '';
    state.fileSize = 0;
    state.processing.active = false;

    elements.fileInput.value = '';
    elements.inputVideoPlayer.src = '';
    elements.completeSrcVideo.src = '';
    elements.completeEnhVideo.src = '';
    elements.enhancedSampleImg.src = '';
    elements.originalSampleImg.src = '';

    elements.sampleEmptyOverlay.classList.remove('hidden');
    elements.sampleLoadingOverlay.classList.add('hidden');
    elements.btnStartProcess.disabled = false;

    elements.completedSection.classList.add('hidden');
    elements.processingSection.classList.add('hidden');
    elements.configSection.classList.add('hidden');
    elements.uploadSection.classList.remove('hidden');

    hideAlert();
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  // =========================================================================
  // 10. Initialization Lifespan Hook
  // =========================================================================
  document.addEventListener('DOMContentLoaded', () => {
    cacheDomElements();
    initUploadHandlers();
    initSplitSlider();
    initPresetButtons();
    initProcessHandlers();
    initSyncPlayback();
  });

})();
