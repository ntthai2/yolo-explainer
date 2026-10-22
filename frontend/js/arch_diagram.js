/**
 * Interactive RangeKing-Style Architectural Diagram for YOLO Explainer.
 * Renders complete Macro Pipeline (Backbone, Neck, Head) and Micro Block Details
 * (C2f, C3k2, C2PSA Attention, SPPF, ConvModule, Detect Head) with live morphing
 * across YOLOv8, YOLO11, and YOLO26.
 */

class ArchDiagramViewer {
  constructor() {
    this.modal = document.getElementById("arch-compare-modal");
    this.closeBtn = document.getElementById("arch-modal-close");
    this.openBtn = document.getElementById("arch-compare-btn");
    this.svgContainer = document.getElementById("arch-svg-container");
    this.activeModel = "yolo11n";
    this.highlightDiff = true;
    this.zoomLevel = 1.0;

    this.init();
  }

  init() {
    if (this.openBtn) {
      this.openBtn.addEventListener("click", () => {
        const currentAppModel = window.app ? window.app.currentModel : "yolo11n";
        this.open(currentAppModel);
      });
    }

    if (this.closeBtn) {
      this.closeBtn.addEventListener("click", () => this.close());
    }

    if (this.modal) {
      this.modal.addEventListener("click", (e) => {
        if (e.target === this.modal) this.close();
      });
    }

    // Model switch buttons inside diagram
    const modelBtns = document.querySelectorAll(".arch-model-pill");
    modelBtns.forEach(btn => {
      btn.addEventListener("click", () => {
        modelBtns.forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        this.activeModel = btn.getAttribute("data-model");
        this.render();
      });
    });

    // Diff toggle
    const diffToggle = document.getElementById("arch-diff-toggle");
    if (diffToggle) {
      diffToggle.addEventListener("change", (e) => {
        this.highlightDiff = e.target.checked;
        this.render();
      });
    }

    // Micro Tab buttons
    const microTabs = document.querySelectorAll(".micro-tab-btn");
    microTabs.forEach(tab => {
      tab.addEventListener("click", () => {
        microTabs.forEach(t => t.classList.remove("active"));
        tab.classList.add("active");
        const targetBlock = tab.getAttribute("data-micro");
        this.showMicroDetail(targetBlock);
      });
    });

    // Zoom controls
    const zoomInBtn = document.getElementById("arch-zoom-in");
    const zoomOutBtn = document.getElementById("arch-zoom-out");
    const zoomResetBtn = document.getElementById("arch-zoom-reset");
    if (zoomInBtn) zoomInBtn.addEventListener("click", () => this.setZoom(this.zoomLevel + 0.15));
    if (zoomOutBtn) zoomOutBtn.addEventListener("click", () => this.setZoom(this.zoomLevel - 0.15));
    if (zoomResetBtn) zoomResetBtn.addEventListener("click", () => this.setZoom(1.0));
  }

  open(modelId = "yolo11n") {
    this.activeModel = modelId || "yolo11n";
    // Sync button state
    document.querySelectorAll(".arch-model-pill").forEach(btn => {
      if (btn.getAttribute("data-model") === this.activeModel) {
        btn.classList.add("active");
      } else {
        btn.classList.remove("active");
      }
    });

    if (this.modal) {
      this.modal.classList.add("open");
    }
    this.render();
    this.showMicroDetail(this.activeModel === "yolov8n" ? "c2f" : "c3k2");
  }

  close() {
    if (this.modal) {
      this.modal.classList.remove("open");
    }
  }

  setZoom(val) {
    this.zoomLevel = Math.max(0.7, Math.min(1.6, val));
    const wrapper = document.getElementById("arch-svg-wrapper");
    if (wrapper) {
      wrapper.style.transform = `scale(${this.zoomLevel})`;
      wrapper.style.transformOrigin = "top left";
    }
    const zoomLabel = document.getElementById("arch-zoom-level");
    if (zoomLabel) {
      zoomLabel.textContent = `${Math.round(this.zoomLevel * 100)}%`;
    }
  }

  jumpToApp(layerId) {
    this.close();
    if (window.app) {
      window.app.navigateToArchBlock(this.activeModel, layerId);
    }
  }

  render() {
    if (!this.svgContainer) return;
    const isV8 = this.activeModel === "yolov8n";
    const isV11 = this.activeModel === "yolo11n";
    const isV26 = this.activeModel === "yolo26n";
    const diff = this.highlightDiff;

    const blockType = isV8 ? "C2f" : "C3k2";
    const blockBadge = isV8 ? "C2f" : "C3k2";
    const blockClass = isV8 ? "node-c2f" : (diff ? "node-c3k2 diff-glow" : "node-c3k2");

    // Colors
    const c2fColor = "#0284c7";
    const c3k2Color = "#2563eb";
    const mainBlockColor = isV8 ? c2fColor : c3k2Color;

    // SVG string
    this.svgContainer.innerHTML = `
      <svg id="arch-main-svg" viewBox="0 0 1180 840" width="1180" height="840" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <linearGradient id="grad-stem" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#38bdf8" />
            <stop offset="100%" stop-color="#0284c7" />
          </linearGradient>
          <linearGradient id="grad-c2f" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#38bdf8" />
            <stop offset="100%" stop-color="#0284c7" />
          </linearGradient>
          <linearGradient id="grad-c3k2" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#60a5fa" />
            <stop offset="100%" stop-color="#2563eb" />
          </linearGradient>
          <linearGradient id="grad-sppf" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#f472b6" />
            <stop offset="100%" stop-color="#db2777" />
          </linearGradient>
          <linearGradient id="grad-c2psa" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#a78bfa" />
            <stop offset="100%" stop-color="#7c3aed" />
          </linearGradient>
          <linearGradient id="grad-upsample" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#4ade80" />
            <stop offset="100%" stop-color="#16a34a" />
          </linearGradient>
          <linearGradient id="grad-concat" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#fbbf24" />
            <stop offset="100%" stop-color="#d97706" />
          </linearGradient>
          <linearGradient id="grad-head-v8" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#f87171" />
            <stop offset="100%" stop-color="#dc2626" />
          </linearGradient>
          <linearGradient id="grad-head-v26" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#fb7185" />
            <stop offset="100%" stop-color="#e11d48" />
          </linearGradient>

          <!-- Drop shadows -->
          <filter id="shadow-card" x="-6%" y="-6%" width="112%" height="112%">
            <feDropShadow dx="0" dy="2" stdDeviation="3" flood-opacity="0.12" />
          </filter>
          <filter id="glow-purple" x="-20%" y="-20%" width="140%" height="140%">
            <feDropShadow dx="0" dy="0" stdDeviation="6" flood-color="#8b5cf6" flood-opacity="0.5" />
          </filter>
          <filter id="glow-red" x="-20%" y="-20%" width="140%" height="140%">
            <feDropShadow dx="0" dy="0" stdDeviation="6" flood-color="#ef4444" flood-opacity="0.5" />
          </filter>

          <!-- Marker Arrow -->
          <marker id="arrow" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 1 L 8 5 L 0 9 z" fill="#64748b" />
          </marker>
          <marker id="arrow-blue" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 1 L 8 5 L 0 9 z" fill="#0284c7" />
          </marker>
        </defs>

        <!-- ==================== 1. BACKBONE ZONE ==================== -->
        <rect x="20" y="20" width="310" height="795" rx="16" fill="#fffbeb" stroke="#fef08a" stroke-width="2" />
        <g transform="translate(35, 45)">
          <rect x="0" y="0" width="110" height="26" rx="6" fill="#fef08a" />
          <text x="10" y="18" font-size="12" font-weight="700" fill="#854d0e">BACKBONE</text>
          <text x="120" y="18" font-size="11" fill="#a16207">Trích xuất đặc trưng đa thang đo</text>
        </g>

        <!-- Input Image Box -->
        <g class="arch-interactive-node" onclick="window.archViewer.jumpToApp('layer_0_stem')" transform="translate(75, 85)" filter="url(#shadow-card)">
          <rect width="200" height="42" rx="8" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" />
          <text x="12" y="22" font-size="12" font-weight="700" fill="#1e293b">Ảnh Đầu Vào (Input)</text>
          <text x="12" y="36" font-size="10" font-family="monospace" fill="#64748b">640 × 640 × 3 (RGB)</text>
        </g>
        <line x1="175" y1="127" x2="175" y2="150" stroke="#64748b" stroke-width="2" marker-end="url(#arrow)" />

        <!-- Stage 0: Stem Conv -->
        <g class="arch-interactive-node" onclick="window.archViewer.jumpToApp('layer_0_stem')" transform="translate(75, 150)" filter="url(#shadow-card)">
          <rect width="200" height="46" rx="8" fill="url(#grad-stem)" />
          <text x="12" y="22" font-size="12" font-weight="700" fill="#ffffff">Stage 0: Stem Conv</text>
          <text x="12" y="38" font-size="10" fill="#e0f2fe">Conv 3×3, Stride 2 ➔ 320×320×16</text>
        </g>
        <line x1="175" y1="196" x2="175" y2="218" stroke="#64748b" stroke-width="2" marker-end="url(#arrow)" />

        <!-- Stage 1 + 2: P2 -->
        <g class="arch-interactive-node" onclick="window.archViewer.jumpToApp('layer_2_p2')" transform="translate(75, 218)" filter="url(#shadow-card)">
          <rect width="200" height="52" rx="8" fill="${isV8 ? 'url(#grad-c2f)' : 'url(#grad-c3k2)'}" />
          <text x="12" y="22" font-size="12" font-weight="700" fill="#ffffff">Stage 2: ${blockType} P2</text>
          <text x="12" y="37" font-size="10" fill="#e0f2fe">Stride 4 ➔ 160×160×32</text>
          <rect x="145" y="8" width="45" height="18" rx="4" fill="#ffffff" fill-opacity="0.25" />
          <text x="151" y="21" font-size="9.5" font-weight="700" fill="#ffffff">P2 / s4</text>
        </g>
        <line x1="175" y1="270" x2="175" y2="295" stroke="#64748b" stroke-width="2" marker-end="url(#arrow)" />

        <!-- Stage 3 + 4: P3 -->
        <g class="arch-interactive-node" onclick="window.archViewer.jumpToApp('layer_4_p3')" transform="translate(75, 295)" filter="url(#shadow-card)">
          <rect width="200" height="54" rx="8" fill="${isV8 ? 'url(#grad-c2f)' : 'url(#grad-c3k2)'}" />
          <text x="12" y="22" font-size="12" font-weight="700" fill="#ffffff">Stage 4: ${blockType} P3</text>
          <text x="12" y="38" font-size="10" fill="#e0f2fe">Stride 8 ➔ 80×80×64</text>
          <text x="12" y="49" font-size="9" fill="#bae6fd">Vật thể nhỏ (Small)</text>
          <rect x="145" y="8" width="45" height="18" rx="4" fill="#ffffff" fill-opacity="0.25" />
          <text x="151" y="21" font-size="9.5" font-weight="700" fill="#ffffff">P3 / s8</text>
        </g>
        <line x1="175" y1="349" x2="175" y2="378" stroke="#64748b" stroke-width="2" marker-end="url(#arrow)" />

        <!-- Stage 5 + 6: P4 -->
        <g class="arch-interactive-node" onclick="window.archViewer.jumpToApp('layer_6_p4')" transform="translate(75, 378)" filter="url(#shadow-card)">
          <rect width="200" height="54" rx="8" fill="${isV8 ? 'url(#grad-c2f)' : 'url(#grad-c3k2)'}" />
          <text x="12" y="22" font-size="12" font-weight="700" fill="#ffffff">Stage 6: ${blockType} P4</text>
          <text x="12" y="38" font-size="10" fill="#e0f2fe">Stride 16 ➔ 40×40×128</text>
          <text x="12" y="49" font-size="9" fill="#bae6fd">Vật thể vừa (Medium)</text>
          <rect x="145" y="8" width="45" height="18" rx="4" fill="#ffffff" fill-opacity="0.25" />
          <text x="151" y="21" font-size="9.5" font-weight="700" fill="#ffffff">P4 / s16</text>
        </g>
        <line x1="175" y1="432" x2="175" y2="460" stroke="#64748b" stroke-width="2" marker-end="url(#arrow)" />

        <!-- Stage 7 + 8: P5 -->
        <g class="arch-interactive-node" onclick="window.archViewer.jumpToApp('layer_8_p5')" transform="translate(75, 460)" filter="url(#shadow-card)">
          <rect width="200" height="54" rx="8" fill="${isV8 ? 'url(#grad-c2f)' : 'url(#grad-c3k2)'}" />
          <text x="12" y="22" font-size="12" font-weight="700" fill="#ffffff">Stage 8: ${blockType} P5</text>
          <text x="12" y="38" font-size="10" fill="#e0f2fe">Stride 32 ➔ 20×20×256</text>
          <text x="12" y="49" font-size="9" fill="#bae6fd">Vật thể lớn (Large)</text>
          <rect x="145" y="8" width="45" height="18" rx="4" fill="#ffffff" fill-opacity="0.25" />
          <text x="151" y="21" font-size="9.5" font-weight="700" fill="#ffffff">P5 / s32</text>
        </g>
        <line x1="175" y1="514" x2="175" y2="540" stroke="#64748b" stroke-width="2" marker-end="url(#arrow)" />

        <!-- Stage 9: SPPF -->
        <g class="arch-interactive-node" onclick="window.archViewer.jumpToApp('layer_9_sppf')" transform="translate(75, 540)" filter="url(#shadow-card)">
          <rect width="200" height="48" rx="8" fill="url(#grad-sppf)" />
          <text x="12" y="22" font-size="12" font-weight="700" fill="#ffffff">Stage 9: SPPF Pooling</text>
          <text x="12" y="38" font-size="10" fill="#fdf2f8">Spatial Pyramid 5×5 ➔ 20×20×256</text>
        </g>
        <line x1="175" y1="588" x2="175" y2="615" stroke="#64748b" stroke-width="2" marker-end="url(#arrow)" />

        <!-- Stage 10: C2PSA Attention (or Absent in v8) -->
        ${isV8 ? `
          <!-- v8 Missing Attention -->
          <g transform="translate(75, 615)">
            <rect width="200" height="54" rx="8" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5" stroke-dasharray="4 3" />
            <text x="12" y="24" font-size="11" font-weight="600" fill="#94a3b8">Stage 10: (Chưa có Attention)</text>
            <text x="12" y="40" font-size="9.5" fill="#94a3b8">YOLOv8 chuyển thẳng sang Neck</text>
          </g>
          <line x1="175" y1="669" x2="175" y2="720" stroke="#64748b" stroke-width="2" />
          <path d="M 175 720 L 400 720" fill="none" stroke="#64748b" stroke-width="2" marker-end="url(#arrow)" />
        ` : `
          <!-- 11 & 26 C2PSA Attention -->
          <g class="arch-interactive-node" onclick="window.archViewer.jumpToApp('layer_10_c2psa')" transform="translate(75, 615)" filter="${diff ? 'url(#glow-purple)' : 'url(#shadow-card)'}">
            <rect width="200" height="54" rx="8" fill="url(#grad-c2psa)" stroke="${diff ? '#c084fc' : 'none'}" stroke-width="${diff ? '2' : '0'}" />
            <text x="12" y="22" font-size="12" font-weight="700" fill="#ffffff">Stage 10: C2PSA Attention</text>
            <text x="12" y="38" font-size="10" fill="#f5f3ff">Spatial Self-Attention 20×20</text>
            <rect x="135" y="8" width="55" height="18" rx="4" fill="#ffffff" fill-opacity="0.25" />
            <text x="142" y="21" font-size="9" font-weight="700" fill="#ffffff">Attention</text>
          </g>
          <line x1="175" y1="669" x2="175" y2="720" stroke="#7c3aed" stroke-width="2" />
          <path d="M 175 720 L 400 720" fill="none" stroke="#7c3aed" stroke-width="2" marker-end="url(#arrow)" />
        `}

        <!-- ==================== 2. NECK ZONE (PAN-FPN) ==================== -->
        <rect x="350" y="20" width="460" height="795" rx="16" fill="#f0f9ff" stroke="#bae6fd" stroke-width="2" />
        <g transform="translate(365, 45)">
          <rect x="0" y="0" width="130" height="26" rx="6" fill="#bae6fd" />
          <text x="10" y="18" font-size="12" font-weight="700" fill="#0369a1">NECK (PAN-FPN)</text>
          <text x="140" y="18" font-size="11" fill="#0284c7">Hòa trộn Top-Down (FPN) & Bottom-Up (PAN)</text>
        </g>

        <!-- Top-Down Path (FPN) -->
        <!-- Neck FPN P5 Upsample -->
        <g transform="translate(400, 698)" filter="url(#shadow-card)">
          <rect width="120" height="38" rx="6" fill="url(#grad-upsample)" />
          <text x="10" y="23" font-size="11" font-weight="700" fill="#ffffff">Upsample 2× (➔ 40×40)</text>
        </g>
        <line x1="460" y1="698" x2="460" y2="665" stroke="#16a34a" stroke-width="2" marker-end="url(#arrow)" />

        <!-- Concat P4 -->
        <g transform="translate(400, 625)" filter="url(#shadow-card)">
          <rect width="120" height="38" rx="6" fill="url(#grad-concat)" />
          <text x="14" y="23" font-size="11" font-weight="700" fill="#ffffff">Concat P4</text>
        </g>
        <!-- Horizontal skip connection from Backbone P4 to Concat P4 -->
        <path d="M 275 405 L 370 405 L 370 644 L 400 644" fill="none" stroke="#0284c7" stroke-width="2" stroke-dasharray="4 2" marker-end="url(#arrow-blue)" />
        <text x="290" y="398" font-size="9.5" fill="#0284c7" font-weight="600">Skip P4</text>
        <line x1="460" y1="625" x2="460" y2="590" stroke="#d97706" stroke-width="2" marker-end="url(#arrow)" />

        <!-- FPN Block P4 -->
        <g class="arch-interactive-node" onclick="window.archViewer.jumpToApp('${isV8 ? "layer_12_neck_p4" : "layer_13_neck_p4"}')" transform="translate(385, 536)" filter="url(#shadow-card)">
          <rect width="150" height="52" rx="8" fill="${isV8 ? 'url(#grad-c2f)' : 'url(#grad-c3k2)'}" />
          <text x="12" y="22" font-size="11.5" font-weight="700" fill="#ffffff">FPN: ${blockType} P4</text>
          <text x="12" y="38" font-size="9.5" fill="#e0f2fe">40×40 · Ngữ nghĩa P4</text>
        </g>
        <line x1="460" y1="536" x2="460" y2="500" stroke="#0284c7" stroke-width="2" marker-end="url(#arrow)" />

        <!-- Neck FPN P4 Upsample -->
        <g transform="translate(400, 460)" filter="url(#shadow-card)">
          <rect width="120" height="38" rx="6" fill="url(#grad-upsample)" />
          <text x="10" y="23" font-size="11" font-weight="700" fill="#ffffff">Upsample 2× (➔ 80×80)</text>
        </g>
        <line x1="460" y1="460" x2="460" y2="425" stroke="#16a34a" stroke-width="2" marker-end="url(#arrow)" />

        <!-- Concat P3 -->
        <g transform="translate(400, 385)" filter="url(#shadow-card)">
          <rect width="120" height="38" rx="6" fill="url(#grad-concat)" />
          <text x="14" y="23" font-size="11" font-weight="700" fill="#ffffff">Concat P3</text>
        </g>
        <!-- Skip connection from Backbone P3 to Concat P3 -->
        <path d="M 275 322 L 360 322 L 360 404 L 400 404" fill="none" stroke="#0284c7" stroke-width="2" stroke-dasharray="4 2" marker-end="url(#arrow-blue)" />
        <text x="290" y="315" font-size="9.5" fill="#0284c7" font-weight="600">Skip P3</text>
        <line x1="460" y1="385" x2="460" y2="350" stroke="#d97706" stroke-width="2" marker-end="url(#arrow)" />

        <!-- FPN Block P3 -->
        <g class="arch-interactive-node" onclick="window.archViewer.jumpToApp('${isV8 ? "layer_15_neck_p3" : "layer_16_neck_p3"}')" transform="translate(385, 298)" filter="url(#shadow-card)">
          <rect width="150" height="52" rx="8" fill="${isV8 ? 'url(#grad-c2f)' : 'url(#grad-c3k2)'}" />
          <text x="12" y="22" font-size="11.5" font-weight="700" fill="#ffffff">FPN: ${blockType} P3</text>
          <text x="12" y="38" font-size="9.5" fill="#e0f2fe">80×80 · Đặc trưng P3</text>
        </g>

        <!-- P3 Detection Head Out Line -->
        <path d="M 535 324 L 840 324" fill="none" stroke="#dc2626" stroke-width="2.5" marker-end="url(#arrow)" />
        <text x="560" y="316" font-size="10" font-weight="700" fill="#dc2626">Đầu ra P3 (Vật thể nhỏ)</text>

        <!-- Bottom-Up Path (PAN) -->
        <path d="M 460 298 L 460 250 L 650 250 L 650 285" fill="none" stroke="#64748b" stroke-width="2" marker-end="url(#arrow)" />
        <text x="510" y="242" font-size="9.5" font-weight="600" fill="#64748b">PAN Bottom-Up ➔</text>

        <!-- Downsample Conv P3->P4 -->
        <g transform="translate(590, 285)" filter="url(#shadow-card)">
          <rect width="120" height="38" rx="6" fill="url(#grad-stem)" />
          <text x="10" y="23" font-size="10.5" font-weight="700" fill="#ffffff">Conv 3×3 s2 (➔ 40×40)</text>
        </g>
        <line x1="650" y1="323" x2="650" y2="355" stroke="#0284c7" stroke-width="2" marker-end="url(#arrow)" />

        <!-- Concat PAN P4 -->
        <g transform="translate(590, 355)" filter="url(#shadow-card)">
          <rect width="120" height="38" rx="6" fill="url(#grad-concat)" />
          <text x="14" y="23" font-size="11" font-weight="700" fill="#ffffff">Concat PAN P4</text>
        </g>
        <line x1="650" y1="393" x2="650" y2="425" stroke="#d97706" stroke-width="2" marker-end="url(#arrow)" />

        <!-- PAN Block P4 -->
        <g class="arch-interactive-node" onclick="window.archViewer.jumpToApp('${isV8 ? "layer_18_neck_pan_p4" : "layer_19_neck_pan_p4"}')" transform="translate(575, 425)" filter="url(#shadow-card)">
          <rect width="150" height="52" rx="8" fill="${isV8 ? 'url(#grad-c2f)' : 'url(#grad-c3k2)'}" />
          <text x="12" y="22" font-size="11.5" font-weight="700" fill="#ffffff">PAN: ${blockType} P4</text>
          <text x="12" y="38" font-size="9.5" fill="#e0f2fe">40×40 · Vị trí + Ngữ nghĩa</text>
        </g>

        <!-- P4 Detection Head Out Line -->
        <path d="M 725 451 L 840 451" fill="none" stroke="#dc2626" stroke-width="2.5" marker-end="url(#arrow)" />
        <text x="735" y="443" font-size="10" font-weight="700" fill="#dc2626">Đầu ra P4 (Vật thể vừa)</text>

        <!-- Downsample Conv P4->P5 -->
        <line x1="650" y1="477" x2="650" y2="510" stroke="#64748b" stroke-width="2" marker-end="url(#arrow)" />
        <g transform="translate(590, 510)" filter="url(#shadow-card)">
          <rect width="120" height="38" rx="6" fill="url(#grad-stem)" />
          <text x="10" y="23" font-size="10.5" font-weight="700" fill="#ffffff">Conv 3×3 s2 (➔ 20×20)</text>
        </g>
        <line x1="650" y1="548" x2="650" y2="580" stroke="#0284c7" stroke-width="2" marker-end="url(#arrow)" />

        <!-- Concat PAN P5 -->
        <g transform="translate(590, 580)" filter="url(#shadow-card)">
          <rect width="120" height="38" rx="6" fill="url(#grad-concat)" />
          <text x="14" y="23" font-size="11" font-weight="700" fill="#ffffff">Concat PAN P5</text>
        </g>
        <line x1="650" y1="618" x2="650" y2="650" stroke="#d97706" stroke-width="2" marker-end="url(#arrow)" />

        <!-- PAN Block P5 -->
        <g class="arch-interactive-node" onclick="window.archViewer.jumpToApp('${isV8 ? "layer_21_neck_pan_p5" : "layer_22_neck_pan_p5"}')" transform="translate(575, 650)" filter="url(#shadow-card)">
          <rect width="150" height="52" rx="8" fill="${isV8 ? 'url(#grad-c2f)' : 'url(#grad-c3k2)'}" />
          <text x="12" y="22" font-size="11.5" font-weight="700" fill="#ffffff">PAN: ${blockType} P5</text>
          <text x="12" y="38" font-size="9.5" fill="#e0f2fe">20×20 · Ngữ nghĩa sâu P5</text>
        </g>

        <!-- P5 Detection Head Out Line -->
        <path d="M 725 676 L 840 676" fill="none" stroke="#dc2626" stroke-width="2.5" marker-end="url(#arrow)" />
        <text x="735" y="668" font-size="10" font-weight="700" fill="#dc2626">Đầu ra P5 (Vật thể lớn)</text>


        <!-- ==================== 3. HEAD ZONE ==================== -->
        <rect x="830" y="20" width="330" height="795" rx="16" fill="#fff1f2" stroke="#fecdd3" stroke-width="2" />
        <g transform="translate(845, 45)">
          <rect x="0" y="0" width="130" height="26" rx="6" fill="#fecdd3" />
          <text x="10" y="18" font-size="12" font-weight="700" fill="#9f1239">DETECT HEAD</text>
          <text x="140" y="18" font-size="11" fill="#be123c">${isV26 ? 'NMS-Free End2End' : 'Decoupled + NMS'}</text>
        </g>

        <!-- Head P3 Node -->
        <g class="arch-interactive-node" onclick="window.archViewer.jumpToApp('${isV8 ? "layer_22_head" : "layer_23_head"}')" transform="translate(850, 300)" filter="${isV26 && diff ? 'url(#glow-red)' : 'url(#shadow-card)'}">
          <rect width="290" height="60" rx="8" fill="${isV26 ? 'url(#grad-head-v26)' : 'url(#grad-head-v8)'}" stroke="${isV26 && diff ? '#fda4af' : 'none'}" stroke-width="${isV26 && diff ? '2' : '0'}" />
          <text x="14" y="24" font-size="12.5" font-weight="700" fill="#ffffff">P3 Detect Head (Small · 80×80)</text>
          <text x="14" y="44" font-size="10" fill="#ffe4e6">${isV26 ? 'Hồi quy trực tiếp (reg_max=1) · Bỏ qua NMS' : 'Nhánh Cls + Bbox (DFL reg_max=16) ➔ Lọc NMS'}</text>
        </g>

        <!-- Head P4 Node -->
        <g class="arch-interactive-node" onclick="window.archViewer.jumpToApp('${isV8 ? "layer_22_head" : "layer_23_head"}')" transform="translate(850, 427)" filter="${isV26 && diff ? 'url(#glow-red)' : 'url(#shadow-card)'}">
          <rect width="290" height="60" rx="8" fill="${isV26 ? 'url(#grad-head-v26)' : 'url(#grad-head-v8)'}" stroke="${isV26 && diff ? '#fda4af' : 'none'}" stroke-width="${isV26 && diff ? '2' : '0'}" />
          <text x="14" y="24" font-size="12.5" font-weight="700" fill="#ffffff">P4 Detect Head (Medium · 40×40)</text>
          <text x="14" y="44" font-size="10" fill="#ffe4e6">${isV26 ? 'Hồi quy trực tiếp (reg_max=1) · Bỏ qua NMS' : 'Nhánh Cls + Bbox (DFL reg_max=16) ➔ Lọc NMS'}</text>
        </g>

        <!-- Head P5 Node -->
        <g class="arch-interactive-node" onclick="window.archViewer.jumpToApp('${isV8 ? "layer_22_head" : "layer_23_head"}')" transform="translate(850, 652)" filter="${isV26 && diff ? 'url(#glow-red)' : 'url(#shadow-card)'}">
          <rect width="290" height="60" rx="8" fill="${isV26 ? 'url(#grad-head-v26)' : 'url(#grad-head-v8)'}" stroke="${isV26 && diff ? '#fda4af' : 'none'}" stroke-width="${isV26 && diff ? '2' : '0'}" />
          <text x="14" y="24" font-size="12.5" font-weight="700" fill="#ffffff">P5 Detect Head (Large · 20×20)</text>
          <text x="14" y="44" font-size="10" fill="#ffe4e6">${isV26 ? 'Hồi quy trực tiếp (reg_max=1) · Bỏ qua NMS' : 'Nhánh Cls + Bbox (DFL reg_max=16) ➔ Lọc NMS'}</text>
        </g>

        <!-- Final Box Outputs -->
        <g transform="translate(850, 725)" filter="url(#shadow-card)">
          <rect width="290" height="75" rx="8" fill="#ffffff" stroke="#e2e8f0" stroke-width="1.5" />
          <text x="14" y="24" font-size="11.5" font-weight="700" fill="#0f172a">Hộp Bounding Boxes & Nhãn Lớp</text>
          <text x="14" y="43" font-size="10" fill="#475569">
            ${isV26 ? 'YOLO26 One-to-One Matching: Không sử dụng NMS' : 'Thuật toán NMS (Non-Maximum Suppression) IoU = 0.7'}
          </text>
          <text x="14" y="60" font-size="9.5" fill="${isV26 ? '#16a34a' : '#64748b'}" font-weight="600">
            ${isV26 ? 'Độ trễ suy luận ổn định và độc lập với số lượng đối tượng' : 'Hậu xử lý lọc trùng lặp bounding box'}
          </text>
        </g>
      </svg>
    `;
  }

  showMicroDetail(blockName) {
    const detailBox = document.getElementById("arch-micro-content");
    if (!detailBox) return;

    if (blockName === "c2f") {
      detailBox.innerHTML = `
        <div class="micro-header">
          <div style="display:flex; align-items:center; gap:8px;">
            <span class="micro-badge badge-c2f">Module C2f</span>
            <strong style="color:var(--text-primary);">YOLOv8 Core Bottleneck</strong>
          </div>
          <span style="font-size:0.75rem; color:var(--text-muted);">Cross Stage Partial với 2 tích chập</span>
        </div>
        <p style="font-size:0.8rem; color:var(--text-secondary); line-height:1.5; margin:8px 0;">
          C2f tách nhánh đầu ra sau Conv 1×1 thành 2 nửa: một nửa đi tắt (shortcut skip), nửa còn lại đi qua chuỗi các khối Bottleneck. Cuối cùng tất cả được nối lại (Concat) giúp bảo toàn dòng gradient phong phú.
        </p>
        <div class="micro-schema-flow">
          <div class="flow-pill">Input</div> ➔
          <div class="flow-pill primary">Conv 1×1</div> ➔
          <div class="flow-pill warning">Split (0.5×c)</div> ➔
          <div class="flow-box-sub">
            <div class="flow-pill info">Bottleneck 1</div>
            <div class="flow-pill info">Bottleneck 2</div>
            <div style="font-size:0.7rem; color:#64748b;">... nx</div>
          </div> ➔
          <div class="flow-pill warning">Concat</div> ➔
          <div class="flow-pill primary">Conv 1×1</div> ➔
          <div class="flow-pill">Output</div>
        </div>
      `;
    } else if (blockName === "c3k2") {
      detailBox.innerHTML = `
        <div class="micro-header">
          <div style="display:flex; align-items:center; gap:8px;">
            <span class="micro-badge badge-c3k2">Module C3k2</span>
            <strong style="color:var(--text-primary);">Cải tiến cốt lõi YOLO11 & YOLO26</strong>
          </div>
          <span style="font-size:0.75rem; color:#0284c7; font-weight:600;">Nhẹ hơn C2f ~15% FLOPs</span>
        </div>
        <p style="font-size:0.8rem; color:var(--text-secondary); line-height:1.5; margin:8px 0;">
          Thay vì sử dụng các bottleneck đơn lẻ nặng nề như C2f, C3k2 sử dụng các bộ tích chập kernel nhỏ (C3k = 2 convs) có khả năng tùy biến độ sâu và kích thước kernel linh hoạt, giảm áp lực bộ nhớ và tăng tốc độ xử lý trên GPU.
        </p>
        <div class="micro-schema-flow">
          <div class="flow-pill">Input</div> ➔
          <div class="flow-pill primary">Conv 1×1</div> ➔
          <div class="flow-pill warning">Split</div> ➔
          <div class="flow-box-sub">
            <div class="flow-pill success">C3k (2× Conv nhỏ)</div>
            <div class="flow-pill success">C3k (2× Conv nhỏ)</div>
          </div> ➔
          <div class="flow-pill warning">Concat</div> ➔
          <div class="flow-pill primary">Conv 1×1</div> ➔
          <div class="flow-pill">Output</div>
        </div>
      `;
    } else if (blockName === "c2psa") {
      detailBox.innerHTML = `
        <div class="micro-header">
          <div style="display:flex; align-items:center; gap:8px;">
            <span class="micro-badge badge-c2psa">C2PSA Attention</span>
            <strong style="color:var(--text-primary);">Pointwise Spatial Attention (YOLO11 & 26)</strong>
          </div>
          <span style="font-size:0.75rem; color:#7c3aed; font-weight:600;">Vị trí: Ngay sau Stage 9 SPPF</span>
        </div>
        <p style="font-size:0.8rem; color:var(--text-secondary); line-height:1.5; margin:8px 0;">
          C2PSA kết hợp cấu trúc phân nhánh C2 với khối Attention PSA (Polarized Self-Attention). Khối này tính toán ma trận tương quan không gian giữa các pixel ở tầng P5, giúp mô hình tập trung tối đa vào vật thể chính và bỏ qua nhiễu nền phức tạp.
        </p>
        <div class="micro-schema-flow">
          <div class="flow-pill">Input P5</div> ➔
          <div class="flow-pill primary">Conv 1×1</div> ➔
          <div class="flow-pill warning">Split</div> ➔
          <div class="flow-box-sub purple">
            <div class="flow-pill purple">PSA Self-Attention (Query · Key · Value)</div>
            <div class="flow-pill purple">Spatial Softmax Weighting</div>
          </div> ➔
          <div class="flow-pill warning">Concat</div> ➔
          <div class="flow-pill">Output (Vào Neck)</div>
        </div>
      `;
    } else if (blockName === "sppf") {
      detailBox.innerHTML = `
        <div class="micro-header">
          <div style="display:flex; align-items:center; gap:8px;">
            <span class="micro-badge badge-sppf">SPPF Module</span>
            <strong style="color:var(--text-primary);">Spatial Pyramid Pooling Fast</strong>
          </div>
          <span style="font-size:0.75rem; color:var(--text-muted);">Tất cả phiên bản YOLOv8, 11, 26</span>
        </div>
        <p style="font-size:0.8rem; color:var(--text-secondary); line-height:1.5; margin:8px 0;">
          SPPF đưa tensor qua 3 lớp MaxPool 5×5 nối tiếp nhau (tương đương với việc dùng kernel 5×5, 9×9, 13×13 nhưng nhanh hơn gấp đôi nhờ tái sử dụng kết quả). Cuối cùng ghép cả 4 nhánh lại để thu nhận thông tin đa trường nhìn (Multi-receptive Field).
        </p>
        <div class="micro-schema-flow">
          <div class="flow-pill">Input</div> ➔
          <div class="flow-pill primary">Conv 1×1</div> ➔
          <div class="flow-box-sub" style="flex-direction:row;">
            <div class="flow-pill info">MaxPool 5×5</div> ➔
            <div class="flow-pill info">MaxPool 5×5</div> ➔
            <div class="flow-pill info">MaxPool 5×5</div>
          </div> ➔
          <div class="flow-pill warning">Concat (4 Nhánh)</div> ➔
          <div class="flow-pill primary">Conv 1×1</div> ➔
          <div class="flow-pill">Output</div>
        </div>
      `;
    } else if (blockName === "head") {
      detailBox.innerHTML = `
        <div class="micro-header">
          <div style="display:flex; align-items:center; gap:8px;">
            <span class="micro-badge badge-head">Detect Head</span>
            <strong style="color:var(--text-primary);">So sánh Decoupled Head vs NMS-Free End-to-End</strong>
          </div>
          <span style="font-size:0.75rem; color:#dc2626; font-weight:600;">Đặc điểm kiến trúc YOLO26</span>
        </div>
        <div style="display:grid; grid-template-columns: 1fr 1fr; gap:12px; margin-top:8px;">
          <div style="background:#fff; border:1px solid #e2e8f0; border-radius:6px; padding:8px 10px;">
            <strong style="color:#0284c7; font-size:0.8rem;">YOLOv8 & YOLO11: Decoupled + NMS</strong>
            <p style="font-size:0.75rem; color:#64748b; margin-top:4px;">
              • Tách riêng hai nhánh phân loại (Classification) và hồi quy vị trí (Bounding Box).<br>
              • DFL dự đoán phân bố xác suất trên 16 mốc cho từng cạnh hộp.<br>
              • Sử dụng thuật toán NMS ở pha hậu xử lý để loại bỏ các hộp trùng lặp.
            </p>
          </div>
          <div style="background:#fff5f5; border:1px solid #fecaca; border-radius:6px; padding:8px 10px;">
            <strong style="color:#dc2626; font-size:0.8rem;">YOLO26: End-to-End NMS-Free</strong>
            <p style="font-size:0.75rem; color:#64748b; margin-top:4px;">
              • Thiết lập reg_max=1, hồi quy trực tiếp tọa độ hộp.<br>
              • Cơ chế gán nhãn One-to-One Matching trong huấn luyện (mỗi đối tượng tương ứng duy nhất một dự đoán).<br>
              • Không yêu cầu NMS: mô hình xuất trực tiếp kết quả, giữ thời gian trễ ổn định.
            </p>
          </div>
        </div>
      `;
    }
  }
}

// Instantiate on load
document.addEventListener("DOMContentLoaded", () => {
  window.archViewer = new ArchDiagramViewer();
});
