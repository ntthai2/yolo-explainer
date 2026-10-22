/**
 * Main Application Controller for YOLO Explainer.
 * Manages model switching, image loading/uploading, DAG rendering,
 * and Compound Scaling modal interactions.
 */

class AppController {
  constructor() {
    this.currentModel = "yolo11n";
    this.currentSampleId = "bus";
    this.customFile = null;
    this.modelsMap = {};
    this.currentInspectData = null;
    this.inspectCache = new Map();
    this.prefetchTimer = null;

    // Components
    this.inspector = new LayerInspector();
    this.dagRenderer = new DAGRenderer(
      "dag-container",
      (layerData, fullData, userInitiated) => {
        this.inspector.inspect(layerData, fullData);
        if (userInitiated) {
          this.openStudioDrawer();
        }
      },
      (fullData) => {
        this.inspector.inspectResult(fullData);
        this.openStudioDrawer();
      }
    );

    // Elements
    this.modelSwitcher = document.getElementById("model-switcher");
    this.sampleSelect = document.getElementById("sample-select");
    this.fileInput = document.getElementById("file-input");
    this.scalingBtn = document.getElementById("scaling-btn");
    this.scalingModal = document.getElementById("scaling-modal");
    this.modalClose = document.getElementById("modal-close");
    this.modalTableBody = document.getElementById("scaling-table-body");
    this.modalModelName = document.getElementById("modal-model-name");

    // Studio Drawer Elements
    this.studioCard = document.getElementById("studio-card");
    this.studioBackdrop = document.getElementById("studio-backdrop");
    this.openStudioBtn = document.getElementById("open-studio-btn");
    this.closeStudioBtn = document.getElementById("close-studio-btn");

    this.loadingOverlay = document.getElementById("loading-overlay");

    // Banner elements
    this.bannerModelTitle = document.getElementById("banner-model-title");
    this.bannerFeatures = document.getElementById("banner-features");
    this.bannerPerf = document.getElementById("banner-perf");

    this.init();
  }

  async init() {
    this.setupEvents();
    await this.fetchModels();
    await this.fetchSamples();
    await this.runInspection();
  }

  setupEvents() {
    // Model Switcher Click
    if (this.modelSwitcher) {
      this.modelSwitcher.addEventListener("click", (e) => {
        const btn = e.target.closest(".segment-btn");
        if (!btn) return;
        const modelId = btn.getAttribute("data-model");
        if (modelId && modelId !== this.currentModel) {
          this.switchModel(modelId);
        }
      });
    }

    // Sample Select Change
    if (this.sampleSelect) {
      this.sampleSelect.addEventListener("change", (e) => {
        this.currentSampleId = e.target.value;
        this.customFile = null;
        this.runInspection();
      });
    }

    // File Input Change (triggered from DAG Input card)
    if (this.fileInput) {
      this.fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files[0]) {
          this.customFile = e.target.files[0];
          this.currentSampleId = null;
          this.runInspection();
        }
      });
    }

    // Compound Scaling Modal
    if (this.scalingBtn && this.scalingModal) {
      this.scalingBtn.addEventListener("click", () => this.openScalingModal());
    }
    if (this.modalClose) {
      this.modalClose.addEventListener("click", () => this.closeScalingModal());
    }
    if (this.scalingModal) {
      this.scalingModal.addEventListener("click", (e) => {
        if (e.target === this.scalingModal) this.closeScalingModal();
      });
    }

    // Inspector Studio Drawer Toggle & Close
    if (this.openStudioBtn) {
      this.openStudioBtn.addEventListener("click", () => this.toggleStudioDrawer());
    }
    if (this.closeStudioBtn) {
      this.closeStudioBtn.addEventListener("click", () => this.closeStudioDrawer());
    }
    if (this.studioBackdrop) {
      this.studioBackdrop.addEventListener("click", () => this.closeStudioDrawer());
    }
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && this.studioCard && this.studioCard.classList.contains("open")) {
        this.closeStudioDrawer();
      }
    });
  }

  openStudioDrawer() {
    if (this.studioCard) this.studioCard.classList.add("open");
    if (this.studioBackdrop) this.studioBackdrop.classList.add("open");
    if (this.openStudioBtn) this.openStudioBtn.classList.add("active");
  }

  closeStudioDrawer() {
    if (this.studioCard) this.studioCard.classList.remove("open");
    if (this.studioBackdrop) this.studioBackdrop.classList.remove("open");
    if (this.openStudioBtn) this.openStudioBtn.classList.remove("active");
    if (this.dagRenderer) this.dagRenderer.resetSelection();
  }

  toggleStudioDrawer() {
    if (this.studioCard && this.studioCard.classList.contains("open")) {
      this.closeStudioDrawer();
    } else {
      this.openStudioDrawer();
    }
  }

  async fetchModels() {
    try {
      const res = await fetch("/api/models");
      const data = await res.json();
      (data.models || []).forEach(m => {
        this.modelsMap[m.id] = m;
      });
    } catch (err) {
      console.error("Failed to load models list:", err);
    }
  }

  async fetchSamples() {
    try {
      const res = await fetch("/api/samples");
      const data = await res.json();
      if (this.sampleSelect && data.samples) {
        this.sampleSelect.innerHTML = "";
        data.samples.forEach(s => {
          const opt = document.createElement("option");
          opt.value = s.id;
          opt.textContent = `Ảnh mẫu: ${s.title}`;
          if (s.id === this.currentSampleId) opt.selected = true;
          this.sampleSelect.appendChild(opt);
        });
      }
    } catch (err) {
      console.error("Failed to load samples list:", err);
    }
  }

  getCacheKey(modelId) {
    if (this.customFile) {
      return `${modelId}:file:${this.customFile.name}:${this.customFile.size}:${this.customFile.lastModified || 0}`;
    }
    return `${modelId}:sample:${this.currentSampleId || "bus"}`;
  }

  switchModel(modelId) {
    if (modelId === this.currentModel && this.currentInspectData) return;
    this.currentModel = modelId;

    // Update buttons
    document.querySelectorAll(".segment-btn").forEach(b => {
      if (b.getAttribute("data-model") === modelId) {
        b.classList.add("active");
      } else {
        b.classList.remove("active");
      }
    });

    // Check client-side memory cache for instantaneous 0ms, 0-byte tab switch
    const cacheKey = this.getCacheKey(modelId);
    if (this.inspectCache.has(cacheKey)) {
      const cachedData = this.inspectCache.get(cacheKey);
      this.currentInspectData = cachedData;
      this.updateBanner(cachedData);
      const meta = this.modelsMap[this.currentModel] || {};
      this.dagRenderer.render(cachedData, meta);
      return;
    }

    this.runInspection();
  }

  async runInspection() {
    const cacheKey = this.getCacheKey(this.currentModel);
    if (this.inspectCache.has(cacheKey)) {
      const cachedData = this.inspectCache.get(cacheKey);
      this.currentInspectData = cachedData;
      this.updateBanner(cachedData);
      const meta = this.modelsMap[this.currentModel] || {};
      this.dagRenderer.render(cachedData, meta);
      this.schedulePrefetch();
      return;
    }

    this.showLoading(true);

    const formData = new FormData();
    formData.append("model_name", this.currentModel);

    if (this.customFile) {
      formData.append("file", this.customFile);
    } else if (this.currentSampleId) {
      formData.append("sample_id", this.currentSampleId);
    }

    try {
      const res = await fetch("/api/inspect", {
        method: "POST",
        body: formData
      });

      if (!res.ok) {
        throw new Error(`Server returned status ${res.status}`);
      }

      const inspectData = await res.json();
      this.inspectCache.set(cacheKey, inspectData);
      this.currentInspectData = inspectData;

      // Update Header banner
      this.updateBanner(inspectData);

      // Render DAG
      const meta = this.modelsMap[this.currentModel] || {};
      this.dagRenderer.render(inspectData, meta);

      // Trigger background prefetching for the other models
      this.schedulePrefetch();

    } catch (err) {
      console.error("Inspection error:", err);
      alert("Không thể trích xuất đặc trưng: " + err.message);
    } finally {
      this.showLoading(false);
    }
  }

  schedulePrefetch() {
    if (this.prefetchTimer) clearTimeout(this.prefetchTimer);
    // Wait briefly after rendering to avoid blocking UI interactions
    this.prefetchTimer = setTimeout(() => {
      this.prefetchOtherModels();
    }, 200);
  }

  async prefetchOtherModels() {
    const allModels = ["yolov8n", "yolo11n", "yolo26n"];
    for (const mId of allModels) {
      if (mId === this.currentModel) continue;
      const key = this.getCacheKey(mId);
      if (this.inspectCache.has(key)) continue;

      try {
        const formData = new FormData();
        formData.append("model_name", mId);
        if (this.customFile) {
          formData.append("file", this.customFile);
        } else if (this.currentSampleId) {
          formData.append("sample_id", this.currentSampleId);
        }

        const res = await fetch("/api/inspect", {
          method: "POST",
          body: formData
        });

        if (res.ok) {
          const data = await res.json();
          this.inspectCache.set(key, data);
        }
      } catch (e) {
        // Silently ignore background prefetch errors
      }
    }
  }

  updateBanner(inspectData) {
    const meta = this.modelsMap[this.currentModel] || {};
    if (this.bannerModelTitle) {
      this.bannerModelTitle.textContent = meta.title || this.currentModel;
    }

    if (this.bannerFeatures) {
      this.bannerFeatures.innerHTML = "";
      (meta.key_features || []).slice(0, 3).forEach(f => {
        const pill = document.createElement("span");
        pill.className = "feature-pill";
        pill.textContent = f;
        this.bannerFeatures.appendChild(pill);
      });
    }

    if (this.bannerPerf) {
      this.bannerPerf.textContent = `${inspectData.inference_time_ms} ms trên ${inspectData.device || "GPU"}`;
    }
  }

  openScalingModal() {
    const meta = this.modelsMap[this.currentModel];
    if (!meta) return;

    if (this.modalModelName) {
      this.modalModelName.textContent = meta.name;
    }

    const scaling = meta.compound_scaling || {};
    this.modalTableBody.innerHTML = "";

    Object.entries(scaling).forEach(([variant, cfg]) => {
      const tr = document.createElement("tr");
      const isCurrent = variant === "n";
      tr.innerHTML = `
        <td><span class="variant-pill" style="${isCurrent ? 'background:var(--primary); color:#fff;' : ''}">${variant.toUpperCase()} ${isCurrent ? 'Hiện tại' : ''}</span></td>
        <td><code>depth: ${cfg.depth}</code></td>
        <td><code>width: ${cfg.width}</code></td>
        <td><strong>${cfg.params}</strong></td>
        <td><strong>${cfg.flops}</strong></td>
        <td style="font-size:0.8rem; color:var(--text-secondary);">${cfg.desc}</td>
      `;
      this.modalTableBody.appendChild(tr);
    });

    this.scalingModal.classList.add("open");
  }

  closeScalingModal() {
    this.scalingModal.classList.remove("open");
  }

  showLoading(show) {
    if (this.loadingOverlay) {
      if (show) this.loadingOverlay.classList.add("active");
      else this.loadingOverlay.classList.remove("active");
    }
  }
}

// Bootstrap on DOM ready
document.addEventListener("DOMContentLoaded", () => {
  window.app = new AppController();
});
