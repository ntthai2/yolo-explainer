/**
 * Pipeline DAG Renderer for YOLO Explainer.
 * Implements:
 * 1. Interactive Highlighting (Đề xuất 1): Default calm/subtle wire opacity (0.3);
 *    dynamic focus on hover or node selection.
 * 2. Floating Tooltip (Đề xuất 3): Eliminated mid-wire floating text rectangles;
 *    rich contextual tooltip on wire/port hover.
 * 3. Orthogonal & Channel Gutter Routing (Đề xuất 4): Wires follow layout gutters
 *    and bottom bypass channels with zero cross-card collisions.
 */

class DAGRenderer {
  constructor(containerId, onSelectLayer, onSelectResult = null) {
    this.container = document.getElementById(containerId);
    this.onSelectLayer = onSelectLayer;
    this.onSelectResult = onSelectResult;
    this.selectedLayerId = null;
    this.currentData = null;
    this.currentModelMeta = null;
    this.svgOverlay = null;
    this.wrapper = null;
    this.tooltipEl = null;
    this.drawRafId = null;
    this.resizeObserver = null;
    this.connections = [];
    this.showAllWires = true;

    window.addEventListener("resize", () => this.scheduleDrawConnections());
  }

  render(inspectData, modelMeta) {
    this.currentData = inspectData;
    this.currentModelMeta = modelMeta;
    const layers = inspectData.layers || {};
    const modelName = inspectData.model_name || "yolo11n";
    const isV8 = modelName.startsWith("yolov8");
    const isV11 = modelName.startsWith("yolo11");
    const isV26 = modelName.startsWith("yolo26");
    const blockType = isV8 ? "C2f" : "C3k2";

    this.container.innerHTML = "";

    const wrapper = document.createElement("div");
    wrapper.className = "pipeline-canvas-container";
    this.wrapper = wrapper;

    // Stage indexing based on architecture
    const S = {
      stem: 0,
      p2: 2,
      p3: 4,
      p4: 6,
      p5: 8,
      sppf: 9,
      c2psa: isV8 ? null : 10,

      // Neck FPN
      fpn_up1: isV8 ? 10 : 11,
      fpn_concat1: isV8 ? 11 : 12,
      fpn_block1: isV8 ? 12 : 13,
      fpn_up2: isV8 ? 13 : 14,
      fpn_concat2: isV8 ? 14 : 15,
      fpn_block2: isV8 ? 15 : 16,

      // Neck PAN
      pan_conv1: isV8 ? 16 : 17,
      pan_concat1: isV8 ? 17 : 18,
      pan_block1: isV8 ? 18 : 19,
      pan_conv2: isV8 ? 19 : 20,
      pan_concat2: isV8 ? 20 : 21,
      pan_block2: isV8 ? 21 : 22,

      // Head
      head: isV8 ? 22 : 23
    };

    // -------------------------------------------------------------
    // 1. ZONE: BACKBONE
    // -------------------------------------------------------------
    const backboneZone = document.createElement("div");
    backboneZone.className = "pipeline-zone zone-backbone";
    backboneZone.innerHTML = `
      <div class="pipeline-zone-header">
        <div class="zone-badge badge-bb">1. BACKBONE</div>
        <div class="zone-info">
          <span class="zone-title">Trích xuất đặc trưng đa mức (P2 ➔ P5)</span>
          <span class="zone-desc">Từ viền cơ bản (Stride 2) đến ngữ nghĩa trừu tượng cao (Stride 32)</span>
        </div>
      </div>
    `;

    const bbFlow = document.createElement("div");
    bbFlow.className = "pipeline-flow-backbone";

    // Sub-column 1: Tầng nông (Input -> P3)
    const bbColShallow = document.createElement("div");
    bbColShallow.className = "bb-subcol bb-col-shallow";
    bbColShallow.id = "bb-col-shallow";
    bbColShallow.innerHTML = `
      <div class="subcol-header bb-subcol-header">Tầng nông (P1 ➔ P3)</div>
    `;

    // Input image card with integrated Upload button
    const inputCard = document.createElement("div");
    inputCard.className = "pipeline-node-card node-input";
    inputCard.id = "node-input";
    inputCard.innerHTML = `
      <div class="node-card-top">
        <span class="node-title">Ảnh Đầu Vào</span>
        <div style="display: flex; align-items: center; gap: 6px;">
          <span class="node-tag">RGB</span>
          <button class="btn-node-upload" id="node-upload-btn" type="button" title="Tải ảnh mới từ máy tính">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
              <polyline points="17 8 12 3 7 8"></polyline>
              <line x1="12" y1="3" x2="12" y2="15"></line>
            </svg>
            Tải ảnh
          </button>
        </div>
      </div>
      <div class="node-card-body">
        <div class="node-thumb-wrap input-thumb-wrap" title="Nhấp để tải ảnh mới">
          <img class="node-thumb-img" src="${inspectData.original_image || ''}" alt="Input Image" />
          <div class="thumb-hover-overlay">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
              <polyline points="17 8 12 3 7 8"></polyline>
              <line x1="12" y1="3" x2="12" y2="15"></line>
            </svg>
          </div>
        </div>
        <div class="node-card-meta">
          <span class="shape-badge">640 × 640 × 3</span>
          <span class="stride-badge">Stride 1 · Click để đổi ảnh</span>
        </div>
      </div>
    `;

    const triggerUpload = () => {
      const fi = document.getElementById("file-input");
      if (fi) fi.click();
    };
    const uploadBtn = inputCard.querySelector("#node-upload-btn");
    if (uploadBtn) {
      uploadBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        triggerUpload();
      });
    }
    const thumbWrap = inputCard.querySelector(".input-thumb-wrap");
    if (thumbWrap) {
      thumbWrap.addEventListener("click", () => triggerUpload());
    }

    bbColShallow.appendChild(inputCard);
    bbColShallow.appendChild(this.createArrow("down"));

    // Stage 0: Stem
    bbColShallow.appendChild(this.createInteractiveNode("layer_0_stem", "Stage 0: Stem Conv", "Conv", "Stride 2 · P1"));
    bbColShallow.appendChild(this.createArrow("down"));

    // Stage 1: Conv 3×3 s2 (Downsample sang P2)
    bbColShallow.appendChild(this.createHelperNode("Stage 1: Conv s2", "Conv", "Giảm mẫu 320² ➔ 160²", "node-layer_1_conv"));
    bbColShallow.appendChild(this.createArrow("down"));

    // Stage 2: P2
    bbColShallow.appendChild(this.createInteractiveNode("layer_2_p2", `Stage ${S.p2}: ${blockType} P2`, blockType, "Stride 4 · P2"));
    bbColShallow.appendChild(this.createArrow("down"));

    // Stage 3: Conv 3×3 s2 (Downsample sang P3)
    bbColShallow.appendChild(this.createHelperNode("Stage 3: Conv s2", "Conv", "Giảm mẫu 160² ➔ 80²", "node-layer_3_conv"));
    bbColShallow.appendChild(this.createArrow("down"));

    // Stage 4: P3 (Small objects)
    const p3Node = this.createInteractiveNode("layer_4_p3", `Stage ${S.p3}: ${blockType} P3`, blockType, "Stride 8 · P3");
    p3Node.appendChild(this.createPin("right", "p3"));
    bbColShallow.appendChild(p3Node);

    // Sub-column 2: Tầng sâu (P4 -> P5)
    const bbColDeep = document.createElement("div");
    bbColDeep.className = "bb-subcol bb-col-deep";
    bbColDeep.id = "bb-col-deep";
    bbColDeep.innerHTML = `
      <div class="subcol-header bb-subcol-header">Tầng sâu (P4 ➔ P5)</div>
      <div class="bb-entry-badge">Nhận P3 từ Stage ${S.p3}</div>
    `;

    // Stage 5: Conv 3×3 s2 (Downsample sang P4)
    const conv5Node = this.createHelperNode("Stage 5: Conv s2", "Conv", "Giảm mẫu 80² ➔ 40²", "node-layer_5_conv");
    conv5Node.appendChild(this.createPin("left", "p3"));
    bbColDeep.appendChild(conv5Node);
    bbColDeep.appendChild(this.createArrow("down"));

    // Stage 6: P4 (Medium objects)
    const p4Node = this.createInteractiveNode("layer_6_p4", `Stage ${S.p4}: ${blockType} P4`, blockType, "Stride 16 · P4");
    p4Node.appendChild(this.createPin("right", "p4"));
    bbColDeep.appendChild(p4Node);
    bbColDeep.appendChild(this.createArrow("down"));

    // Stage 7: Conv 3×3 s2 (Downsample sang P5)
    bbColDeep.appendChild(this.createHelperNode("Stage 7: Conv s2", "Conv", "Giảm mẫu 40² ➔ 20²", "node-layer_7_conv"));
    bbColDeep.appendChild(this.createArrow("down"));

    // Stage 8: P5 (Large objects)
    bbColDeep.appendChild(this.createInteractiveNode("layer_8_p5", `Stage ${S.p5}: ${blockType} P5`, blockType, "Stride 32 · P5"));
    bbColDeep.appendChild(this.createArrow("down"));

    // Stage 9: SPPF (cuối Backbone của YOLOv8, xuất cổng P5)
    const sppfNode = this.createInteractiveNode("layer_9_sppf", `Stage ${S.sppf}: SPPF`, "SPPF", "MaxPool 5×5");
    if (isV8) {
      sppfNode.appendChild(this.createPin("right", "p5"));
    }
    bbColDeep.appendChild(sppfNode);

    // Stage 10: C2PSA Attention (chỉ có trên YOLO11 & YOLO26)
    if (!isV8) {
      bbColDeep.appendChild(this.createArrow("down"));
      const c2psaNode = this.createInteractiveNode("layer_10_c2psa", `Stage ${S.c2psa}: C2PSA`, "C2PSA", "Self-Attention P5");
      c2psaNode.classList.add("node-attention-glow");
      c2psaNode.appendChild(this.createPin("right", "p5"));
      bbColDeep.appendChild(c2psaNode);
    }

    bbFlow.appendChild(bbColShallow);
    bbFlow.appendChild(bbColDeep);
    backboneZone.appendChild(bbFlow);
    wrapper.appendChild(backboneZone);

    // -------------------------------------------------------------
    // 2. ZONE: NECK (PAN-FPN FUSION)
    // -------------------------------------------------------------
    const neckZone = document.createElement("div");
    neckZone.className = "pipeline-zone zone-neck";
    neckZone.innerHTML = `
      <div class="pipeline-zone-header">
        <div class="zone-badge badge-neck">2. NECK (PAN-FPN)</div>
        <div class="zone-info">
          <span class="zone-title">Hòa trộn đặc trưng (Stage ${S.fpn_up1} ➔ Stage ${S.pan_block2})</span>
          <span class="zone-desc">FPN truyền ngữ nghĩa từ trên xuống & PAN truyền tọa độ từ dưới lên</span>
        </div>
      </div>
    `;

    const neckFlow = document.createElement("div");
    neckFlow.className = "pipeline-flow-neck";

    // Cột A: Top-Down FPN
    const fpnCol = document.createElement("div");
    fpnCol.className = "neck-subcol fpn-col";
    fpnCol.innerHTML = `
      <div class="subcol-header">Top-Down FPN (Ngữ nghĩa)</div>
      <div class="neck-entry-badge">Nhận P5 từ Backbone Stage ${isV8 ? S.sppf : S.c2psa} (20×20)</div>
    `;

    // Stage fpn_up1
    const fpnUp1 = this.createHelperNode(`Stage ${S.fpn_up1}: Upsample 2×`, "Upsample", "Phóng to 20² ➔ 40²", "node-fpn-up1");
    fpnUp1.appendChild(this.createPin("left", "p5"));
    fpnCol.appendChild(fpnUp1);
    fpnCol.appendChild(this.createArrow("down"));

    // Stage fpn_concat1
    const concatP4 = this.createHelperNode(`Stage ${S.fpn_concat1}: Concat P4`, "Concat", "Ghép kênh Backbone P4", "node-fpn-concat-p4");
    concatP4.classList.add("node-concat");
    concatP4.appendChild(this.createPin("left", "p4"));
    fpnCol.appendChild(concatP4);
    fpnCol.appendChild(this.createArrow("down"));

    // Stage fpn_block1
    const fpnP4Id = isV8 ? "layer_12_neck_p4" : "layer_13_neck_p4";
    const fpnP4Node = this.createInteractiveNode(fpnP4Id, `Stage ${S.fpn_block1}: FPN ${blockType} P4`, blockType, "Hòa trộn ngữ nghĩa P5 & P4");
    fpnP4Node.appendChild(this.createPin("right", "p4"));
    fpnCol.appendChild(fpnP4Node);
    fpnCol.appendChild(this.createArrow("down"));

    // Stage fpn_up2
    const fpnUp2 = this.createHelperNode(`Stage ${S.fpn_up2}: Upsample 2×`, "Upsample", "Phóng to 40² ➔ 80²", "node-fpn-up2");
    fpnCol.appendChild(fpnUp2);
    fpnCol.appendChild(this.createArrow("down"));

    // Stage fpn_concat2
    const concatP3 = this.createHelperNode(`Stage ${S.fpn_concat2}: Concat P3`, "Concat", "Ghép kênh Backbone P3", "node-fpn-concat-p3");
    concatP3.classList.add("node-concat");
    concatP3.appendChild(this.createPin("left", "p3"));
    fpnCol.appendChild(concatP3);
    fpnCol.appendChild(this.createArrow("down"));

    // Stage fpn_block2
    const fpnP3Id = isV8 ? "layer_15_neck_p3" : "layer_16_neck_p3";
    const fpnP3Node = this.createInteractiveNode(fpnP3Id, `Stage ${S.fpn_block2}: FPN ${blockType} P3`, blockType, "Đặc trưng vật thể nhỏ P3");
    fpnP3Node.appendChild(this.createPin("right", "p3"));
    fpnCol.appendChild(fpnP3Node);

    // Cột B: Bottom-Up PAN
    const panCol = document.createElement("div");
    panCol.className = "neck-subcol pan-col";
    panCol.innerHTML = `
      <div class="subcol-header">Bottom-Up PAN (Tọa độ)</div>
      <div class="neck-entry-badge">Nhận P3 từ FPN Stage ${S.fpn_block2}</div>
    `;

    // Stage pan_conv1
    const panConv1 = this.createHelperNode(`Stage ${S.pan_conv1}: Conv s2`, "Conv", "Giảm mẫu 80² ➔ 40²", "node-pan-conv1");
    panConv1.appendChild(this.createPin("left", "p3"));
    panCol.appendChild(panConv1);
    panCol.appendChild(this.createArrow("down"));

    // Stage pan_concat1
    const concatPanP4 = this.createHelperNode(`Stage ${S.pan_concat1}: Concat P4`, "Concat", "Ghép kênh FPN P4", "node-pan-concat-p4");
    concatPanP4.classList.add("node-concat");
    concatPanP4.appendChild(this.createPin("left", "p4"));
    panCol.appendChild(concatPanP4);
    panCol.appendChild(this.createArrow("down"));

    // Stage pan_block1
    const panP4Id = isV8 ? "layer_18_neck_pan_p4" : "layer_19_neck_pan_p4";
    const panP4Node = this.createInteractiveNode(panP4Id, `Stage ${S.pan_block1}: PAN ${blockType} P4`, blockType, "Tái định vị không gian P4");
    panP4Node.appendChild(this.createPin("right", "p4"));
    panCol.appendChild(panP4Node);
    panCol.appendChild(this.createArrow("down"));

    // Stage pan_conv2
    const panConv2 = this.createHelperNode(`Stage ${S.pan_conv2}: Conv s2`, "Conv", "Giảm mẫu 40² ➔ 20²", "node-pan-conv2");
    panCol.appendChild(panConv2);
    panCol.appendChild(this.createArrow("down"));

    // Stage pan_concat2
    const concatPanP5 = this.createHelperNode(`Stage ${S.pan_concat2}: Concat P5`, "Concat", "Ghép kênh Backbone P5", "node-pan-concat-p5");
    concatPanP5.classList.add("node-concat");
    concatPanP5.appendChild(this.createPin("left", "p5"));
    panCol.appendChild(concatPanP5);
    panCol.appendChild(this.createArrow("down"));

    // Stage pan_block2
    const panP5Id = isV8 ? "layer_21_neck_pan_p5" : "layer_22_neck_pan_p5";
    const panP5Node = this.createInteractiveNode(panP5Id, `Stage ${S.pan_block2}: PAN ${blockType} P5`, blockType, "Tái định vị không gian P5");
    panP5Node.appendChild(this.createPin("right", "pan-p5"));
    panCol.appendChild(panP5Node);

    neckFlow.appendChild(fpnCol);
    neckFlow.appendChild(panCol);
    neckZone.appendChild(neckFlow);
    wrapper.appendChild(neckZone);

    // -------------------------------------------------------------
    // 3. ZONE: HEAD (DETECTION OUTPUT)
    // -------------------------------------------------------------
    const headZone = document.createElement("div");
    headZone.className = "pipeline-zone zone-head";
    headZone.innerHTML = `
      <div class="pipeline-zone-header">
        <div class="zone-badge badge-head">3. HEAD</div>
        <div class="zone-info">
          <span class="zone-title">Đầu phát hiện</span>
          <span class="zone-desc">${isV26 ? 'NMS-Free Matching' : 'Decoupled + NMS'}</span>
        </div>
      </div>
    `;

    const headFlow = document.createElement("div");
    headFlow.className = "pipeline-flow-vertical";

    // 3 Cổng đầu vào nhận từ Neck
    const headP3 = document.createElement("div");
    headP3.className = "head-port-card";
    headP3.id = "node-head-port-p3";
    headP3.innerHTML = `
      <div class="port-tag tag-p3">Cổng P3 · 80×80 (Nhỏ)</div>
      <div class="port-desc">Nhận từ FPN Stage ${S.fpn_block2}</div>
    `;
    headP3.appendChild(this.createPin("left", "p3"));
    headFlow.appendChild(headP3);

    const headP4 = document.createElement("div");
    headP4.className = "head-port-card";
    headP4.id = "node-head-port-p4";
    headP4.innerHTML = `
      <div class="port-tag tag-p4">Cổng P4 · 40×40 (Vừa)</div>
      <div class="port-desc">Nhận từ PAN Stage ${S.pan_block1}</div>
    `;
    headP4.appendChild(this.createPin("left", "p4"));
    headFlow.appendChild(headP4);

    const headP5 = document.createElement("div");
    headP5.className = "head-port-card";
    headP5.id = "node-head-port-p5";
    headP5.innerHTML = `
      <div class="port-tag tag-p5">Cổng P5 · 20×20 (Lớn)</div>
      <div class="port-desc">Nhận từ PAN Stage ${S.pan_block2}</div>
    `;
    headP5.appendChild(this.createPin("left", "pan-p5"));
    headFlow.appendChild(headP5);

    headFlow.appendChild(this.createArrow("down"));

    // Nút Detect Head chính
    const headLayerId = isV8 ? "layer_22_head" : "layer_23_head";
    const masterHeadNode = this.createInteractiveNode(
      headLayerId,
      isV26 ? `Stage ${S.head}: NMS-Free Head` : `Stage ${S.head}: Detect Head`,
      isV26 ? "NMS-Free" : "Detect",
      "Dự đoán Boxes & Classes"
    );
    headFlow.appendChild(masterHeadNode);

    headFlow.appendChild(this.createArrow("down"));

    // Final Output Card (Block Kết Quả)
    const detections = inspectData.detections || [];
    const outCard = document.createElement("div");
    outCard.className = "pipeline-node-card node-result interactive";
    outCard.id = "node-result";
    outCard.innerHTML = `
      <div class="node-card-top">
        <span class="node-title">Kết quả phát hiện</span>
        <span class="node-tag tag-success">${detections.length} hộp</span>
      </div>
      <div class="node-card-body">
        <div class="node-thumb-wrap result-thumb">
          <img class="node-thumb-img result" src="${inspectData.detection_image || ''}" alt="Detection Output" />
        </div>
        <div class="node-card-meta">
          <span class="latency-badge">${inspectData.inference_time_ms || 0} ms (${inspectData.device || 'CPU'})</span>
          <span class="stride-badge" style="color: #16a34a; font-weight: 600;">Xem chi tiết ➔</span>
        </div>
      </div>
    `;

    outCard.addEventListener("mouseenter", () => {
      this.highlightForNode("node-result");
    });
    outCard.addEventListener("mouseleave", () => {
      this.resetSelection();
    });
    outCard.addEventListener("click", (e) => {
      e.stopPropagation();
      if (this.selectedLayerId === "result") {
        this.resetSelection();
        return;
      }
      this.selectResult(inspectData);
    });

    headFlow.appendChild(outCard);

    headZone.appendChild(headFlow);
    wrapper.appendChild(headZone);

    // -------------------------------------------------------------
    // 4. SVG CONNECTION OVERLAY (CALM DEFAULT, CLEAN CHANNELS)
    // -------------------------------------------------------------
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("class", "pipeline-svg-overlay");
    svg.innerHTML = `
      <defs>
        <!-- Arrowhead markers with fixed size (markerUnits="userSpaceOnUse") to avoid scaling distortion -->
        <marker id="arrow-p3" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="8" markerHeight="8" orient="auto" markerUnits="userSpaceOnUse">
          <path d="M 1 2 L 8 5 L 1 8 Z" fill="#0284c7" />
        </marker>
        <marker id="arrow-p4" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="8" markerHeight="8" orient="auto" markerUnits="userSpaceOnUse">
          <path d="M 1 2 L 8 5 L 1 8 Z" fill="#d97706" />
        </marker>
        <marker id="arrow-p5" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="8" markerHeight="8" orient="auto" markerUnits="userSpaceOnUse">
          <path d="M 1 2 L 8 5 L 1 8 Z" fill="#dc2626" />
        </marker>
        <marker id="arrow-pan-p5" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="8" markerHeight="8" orient="auto" markerUnits="userSpaceOnUse">
          <path d="M 1 2 L 8 5 L 1 8 Z" fill="#dc2626" />
        </marker>
        <marker id="arrow-red" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="8" markerHeight="8" orient="auto" markerUnits="userSpaceOnUse">
          <path d="M 1 2 L 8 5 L 1 8 Z" fill="#dc2626" />
        </marker>
      </defs>
      <g class="svg-wires-layer"></g>
    `;
    wrapper.appendChild(svg);
    this.svgOverlay = svg;

    // Floating Tooltip Element (Đề xuất 3)
    const tooltip = document.createElement("div");
    tooltip.className = "pipeline-wire-tooltip";
    tooltip.id = "pipeline-wire-tooltip";
    wrapper.appendChild(tooltip);
    this.tooltipEl = tooltip;

    this.container.appendChild(wrapper);

    // Set up ResizeObserver to continuously track layout changes
    if (window.ResizeObserver) {
      if (this.resizeObserver) this.resizeObserver.disconnect();
      this.resizeObserver = new ResizeObserver(() => {
        this.scheduleDrawConnections();
      });
      this.resizeObserver.observe(wrapper);
    }

    // Schedule connection rendering
    this.scheduleDrawConnections();

    // Clean initial state (no stuck clicked block or forced wires)
    this.resetSelection();

    // Click on canvas background / wrapper resets selection
    wrapper.addEventListener("click", (e) => {
      if (!e.target.closest(".pipeline-node-card") && !e.target.closest(".head-port-card") && !e.target.closest(".btn-node-upload")) {
        this.resetSelection();
      }
    });
  }

  scheduleDrawConnections() {
    if (this.drawRafId) cancelAnimationFrame(this.drawRafId);
    this.drawRafId = requestAnimationFrame(() => {
      this.drawConnections();
      setTimeout(() => this.drawConnections(), 100);
    });
  }

  drawConnections() {
    if (!this.svgOverlay || !this.wrapper) return;
    const wiresLayer = this.svgOverlay.querySelector(".svg-wires-layer");
    if (!wiresLayer) return;

    wiresLayer.innerHTML = "";

    const wrapRect = this.wrapper.getBoundingClientRect();
    if (wrapRect.width === 0 || wrapRect.height === 0) return;

    const scrollW = this.wrapper.scrollWidth || wrapRect.width;
    const scrollH = this.wrapper.scrollHeight || wrapRect.height;
    this.svgOverlay.setAttribute("width", scrollW);
    this.svgOverlay.setAttribute("height", scrollH);
    this.svgOverlay.style.width = `${scrollW}px`;
    this.svgOverlay.style.height = `${scrollH}px`;

    const isV8 = this.currentData && this.currentData.model_name && this.currentData.model_name.startsWith("yolov8");
    const bbP5Id = isV8 ? "node-layer_9_sppf" : "node-layer_10_c2psa";
    const fpnP4Id = isV8 ? "node-layer_12_neck_p4" : "node-layer_13_neck_p4";
    const fpnP3Id = isV8 ? "node-layer_15_neck_p3" : "node-layer_16_neck_p3";
    const panP4Id = isV8 ? "node-layer_18_neck_pan_p4" : "node-layer_19_neck_pan_p4";
    const panP5Id = isV8 ? "node-layer_21_neck_pan_p5" : "node-layer_22_neck_pan_p5";
    const headId = isV8 ? "node-layer_22_head" : "node-layer_23_head";

    this.connections = [
      // 0. Chuyển tiếp liên tầng Backbone P3 -> Stage 5 (Blue)
      {
        id: "wire-bb-col1-col2",
        from: "node-layer_4_p3",
        to: "node-layer_5_conv",
        color: "#0284c7",
        marker: "arrow-p3",
        scale: "p3",
        title: "Chuyển tiếp Backbone P3 ➔ P4 (80×80 ➔ 40×40)",
        flow: "Backbone Stage 4 (C3k2/C2f P3) ➔ Stage 5 (Conv 3×3 s2)",
        desc: "Dòng dữ liệu chính đi từ tầng nông sang tầng sâu của Backbone để tiếp tục giảm mẫu xuống P4.",
        routeType: "bb-col-transfer"
      },
      // 1. Backbone P5 -> Neck FPN Upsample 1 (Red)
      {
        id: "wire-bb-p5-fpn",
        from: bbP5Id,
        to: "node-fpn-up1",
        color: "#dc2626",
        marker: "arrow-red",
        scale: "p5",
        title: "Đầu ra Backbone P5 ➔ FPN (Stride 32 · 20×20)",
        flow: `Backbone Stage ${isV8 ? 9 : 10} ➔ Neck FPN Stage ${isV8 ? 10 : 11} (Upsample 2×)`,
        desc: "Truyền đặc trưng ngữ nghĩa mức cao sang tuyến FPN để phóng to và phân phối xuống các tầng dưới.",
        routeType: "gutter-1"
      },
      // 2. Backbone P4 -> Neck FPN Concat P4 (Amber)
      {
        id: "wire-bb-p4-fpn",
        from: "node-layer_6_p4",
        to: "node-fpn-concat-p4",
        color: "#d97706",
        marker: "arrow-p4",
        scale: "p4",
        title: "Nhánh tắt P4 (Stride 16 · 40×40)",
        flow: `Backbone Stage 6 ➔ Neck FPN Stage ${isV8 ? 11 : 12} (Concat P4)`,
        desc: "Truyền đặc trưng vật thể trung bình từ Backbone vào tầng ghép kênh FPN để hòa trộn cùng ngữ nghĩa P5.",
        routeType: "gutter-1"
      },
      // 3. Backbone P3 -> Neck FPN Concat P3 (Blue)
      {
        id: "wire-bb-p3-fpn",
        from: "node-layer_4_p3",
        to: "node-fpn-concat-p3",
        color: "#0284c7",
        marker: "arrow-p3",
        scale: "p3",
        title: "Nhánh tắt P3 (Stride 8 · 80×80)",
        flow: `Backbone Stage 4 ➔ Neck FPN Stage ${isV8 ? 14 : 15} (Concat P3)`,
        desc: "Truyền đặc trưng vật thể nhỏ độ phân giải cao từ Backbone vào tầng ghép kênh FPN.",
        routeType: "bottom-channel-p3"
      },
      // 4. Backbone P5 -> Neck PAN Concat P5 (Red, bypass qua kênh đáy)
      {
        id: "wire-bb-p5-pan",
        from: bbP5Id,
        to: "node-pan-concat-p5",
        color: "#dc2626",
        marker: "arrow-red",
        scale: "p5",
        title: "Nhánh tắt P5 sang PAN (20×20)",
        flow: `Backbone Stage ${isV8 ? 9 : 10} ➔ Neck PAN Stage ${isV8 ? 20 : 21} (Concat P5)`,
        desc: "Truyền đặc trưng P5 gốc vào tầng cuối của PAN để kết hợp với thông tin tọa độ từ dưới đi lên.",
        routeType: "bottom-channel"
      },
      // 5. Neck FPN P4 -> Neck PAN Concat P4 (Amber)
      {
        id: "wire-fpn-pan-p4",
        from: fpnP4Id,
        to: "node-pan-concat-p4",
        color: "#d97706",
        marker: "arrow-p4",
        scale: "p4",
        title: "Nhánh giao FPN P4 ➔ PAN P4 (40×40)",
        flow: `Neck FPN Stage ${isV8 ? 12 : 13} ➔ Neck PAN Stage ${isV8 ? 17 : 18} (Concat PAN P4)`,
        desc: "Chuyển đặc trưng đã giàu ngữ nghĩa từ FPN sang tuyến PAN để tái định vị không gian.",
        routeType: "gutter-neck"
      },
      // 6. Neck FPN P3 -> Neck PAN Conv 1 (Blue)
      {
        id: "wire-fpn-pan-p3",
        from: fpnP3Id,
        to: "node-pan-conv1",
        color: "#0284c7",
        marker: "arrow-p3",
        scale: "p3",
        title: "Chuyển tiếp FPN P3 ➔ PAN P3 (80×80)",
        flow: `Neck FPN Stage ${isV8 ? 15 : 16} ➔ Neck PAN Stage ${isV8 ? 16 : 17} (Conv 3×3 s2)`,
        desc: "Đầu ra FPN P3 chuyển sang khối tích chập giảm mẫu để bắt đầu tuyến Bottom-Up PAN.",
        routeType: "gutter-neck"
      },
      // 7. Neck FPN P3 -> Head Port P3 (Blue, bypass via bottom channel)
      {
        id: "wire-fpn-head-p3",
        from: fpnP3Id,
        to: "node-head-port-p3",
        color: "#0284c7",
        marker: "arrow-p3",
        scale: "p3",
        title: "Đầu ra Neck P3 ➔ Cổng Head P3 (80×80)",
        flow: `Neck FPN Stage ${isV8 ? 15 : 16} ➔ Detect Head Cổng P3`,
        desc: "Cung cấp tensor đặc trưng tối ưu cho việc dự đoán các vật thể kích thước nhỏ.",
        routeType: "bottom-to-head"
      },
      // 8. Neck PAN P4 -> Head Port P4 (Amber)
      {
        id: "wire-pan-head-p4",
        from: panP4Id,
        to: "node-head-port-p4",
        color: "#d97706",
        marker: "arrow-p4",
        scale: "p4",
        title: "Đầu ra Neck P4 ➔ Cổng Head P4 (40×40)",
        flow: `Neck PAN Stage ${isV8 ? 18 : 19} ➔ Detect Head Cổng P4`,
        desc: "Cung cấp tensor đặc trưng tối ưu cho việc dự đoán các vật thể kích thước trung bình.",
        routeType: "pan-to-head-p4"
      },
      // 9. Neck PAN P5 -> Head Port P5 (Red)
      {
        id: "wire-pan-head-p5",
        from: panP5Id,
        to: "node-head-port-p5",
        color: "#dc2626",
        marker: "arrow-red",
        scale: "p5",
        title: "Đầu ra Neck P5 ➔ Cổng Head P5 (20×20)",
        flow: `Neck PAN Stage ${isV8 ? 21 : 22} ➔ Detect Head Cổng P5`,
        desc: "Cung cấp tensor đặc trưng tối ưu cho việc dự đoán các vật thể kích thước lớn.",
        routeType: "pan-to-head-p5"
      },
      // 10. Head Port P3 -> Detect Head (Blue)
      {
        id: "wire-port-p3-head",
        from: "node-head-port-p3",
        to: headId,
        color: "#0284c7",
        marker: "arrow-p3",
        scale: "p3",
        title: "Cổng P3 ➔ Detect Head",
        flow: `Cổng P3 (80×80) ➔ Detect Head`,
        desc: "Nhánh xử lý vật thể nhỏ trong Detect Head.",
        routeType: "head-bus",
        busIndex: 0
      },
      // 11. Head Port P4 -> Detect Head (Amber)
      {
        id: "wire-port-p4-head",
        from: "node-head-port-p4",
        to: headId,
        color: "#d97706",
        marker: "arrow-p4",
        scale: "p4",
        title: "Cổng P4 ➔ Detect Head",
        flow: `Cổng P4 (40×40) ➔ Detect Head`,
        desc: "Nhánh xử lý vật thể vừa trong Detect Head.",
        routeType: "head-bus",
        busIndex: 1
      },
      // 12. Head Port P5 -> Detect Head (Red)
      {
        id: "wire-port-p5-head",
        from: "node-head-port-p5",
        to: headId,
        color: "#dc2626",
        marker: "arrow-red",
        scale: "p5",
        title: "Cổng P5 ➔ Detect Head",
        flow: `Cổng P5 (20×20) ➔ Detect Head`,
        desc: "Nhánh xử lý vật thể lớn trong Detect Head.",
        routeType: "head-bus",
        busIndex: 2
      }
    ];

    this.connections.forEach(conn => {
      const elFrom = document.getElementById(conn.from);
      const elTo = document.getElementById(conn.to);
      if (!elFrom || !elTo) return;

      const rFrom = elFrom.getBoundingClientRect();
      const rTo = elTo.getBoundingClientRect();

      let d = "";
      let dBg = "";

      if (conn.routeType === "head-bus") {
        // Bus arc around the right edge of Head column entering strictly horizontally into Detect Head body
        const x1 = rFrom.right - wrapRect.left;
        const y1 = rFrom.top + rFrom.height / 2 - wrapRect.top;
        const x2 = rTo.right - wrapRect.left;
        const y2 = rTo.top + 13 + conn.busIndex * 12 - wrapRect.top;
        const busX = Math.max(x1, x2) + 20 + conn.busIndex * 12;

        d = `M ${x1} ${y1} ` +
          `L ${busX - 8} ${y1} ` +
          `C ${busX} ${y1}, ${busX} ${y1 + 4}, ${busX} ${y1 + 10} ` +
          `L ${busX} ${y2 - 10} ` +
          `C ${busX} ${y2 - 4}, ${busX} ${y2}, ${busX - 8} ${y2} ` +
          `L ${x2} ${y2}`;
        dBg = `M ${x1} ${y1} ` +
          `L ${busX - 8} ${y1} ` +
          `C ${busX} ${y1}, ${busX} ${y1 + 4}, ${busX} ${y1 + 10} ` +
          `L ${busX} ${y2 - 10} ` +
          `C ${busX} ${y2 - 4}, ${busX} ${y2}, ${busX - 8} ${y2} ` +
          `L ${x2 + 7} ${y2}`;
      } else if (conn.routeType === "pan-to-head-p4") {
        // Middle lane in Gutter 2: Stage 19 up to Cổng P4, entering strictly horizontally
        const x1 = rFrom.right - wrapRect.left;
        const y1 = rFrom.top + rFrom.height / 2 - wrapRect.top;
        const x2 = rTo.left - wrapRect.left;
        const y2 = rTo.top + rTo.height / 2 - wrapRect.top;
        const laneX = x2 - 16;

        d = `M ${x1} ${y1} ` +
          `C ${laneX} ${y1}, ${laneX} ${y1 - 6}, ${laneX} ${y1 - 12} ` +
          `L ${laneX} ${y2 + 12} ` +
          `C ${laneX} ${y2 + 4}, ${laneX} ${y2}, ${laneX + 6} ${y2} ` +
          `L ${x2} ${y2}`;
        dBg = `M ${x1} ${y1} ` +
          `C ${laneX} ${y1}, ${laneX} ${y1 - 6}, ${laneX} ${y1 - 12} ` +
          `L ${laneX} ${y2 + 12} ` +
          `C ${laneX} ${y2 + 4}, ${laneX} ${y2}, ${laneX + 6} ${y2} ` +
          `L ${x2 - 7} ${y2}`;
      } else if (conn.routeType === "pan-to-head-p5") {
        // Inner lane in Gutter 2: Stage 22 up to Cổng P5, entering strictly horizontally
        const x1 = rFrom.right - wrapRect.left;
        const y1 = rFrom.top + rFrom.height / 2 - wrapRect.top;
        const x2 = rTo.left - wrapRect.left;
        const y2 = rTo.top + rTo.height / 2 - wrapRect.top;
        const laneX = x2 - 8;

        d = `M ${x1} ${y1} ` +
          `C ${laneX} ${y1}, ${laneX} ${y1 - 6}, ${laneX} ${y1 - 12} ` +
          `L ${laneX} ${y2 + 10} ` +
          `C ${laneX} ${y2 + 3}, ${laneX} ${y2}, ${laneX + 4} ${y2} ` +
          `L ${x2} ${y2}`;
        dBg = `M ${x1} ${y1} ` +
          `C ${laneX} ${y1}, ${laneX} ${y1 - 6}, ${laneX} ${y1 - 12} ` +
          `L ${laneX} ${y2 + 10} ` +
          `C ${laneX} ${y2 + 3}, ${laneX} ${y2}, ${laneX + 4} ${y2} ` +
          `L ${x2 - 7} ${y2}`;
      } else if (conn.routeType === "bottom-to-head") {
        // Outer lane in Gutter 2: FPN P3 via bottom channel snugly below Stage 21/22 PAN P5
        const x1 = rFrom.right - wrapRect.left;
        const y1 = rFrom.top + rFrom.height / 2 - wrapRect.top;
        const x2 = rTo.left - wrapRect.left;
        const y2 = rTo.top + rTo.height / 2 - wrapRect.top;

        const elPanP5 = document.getElementById(panP5Id);
        const rPanP5 = elPanP5 ? elPanP5.getBoundingClientRect() : rFrom;
        const bottomY = Math.max(rFrom.bottom, rPanP5.bottom) - wrapRect.top + 22;
        const gutterHeadX = x2 - 20;

        d = `M ${x1} ${y1} ` +
          `C ${x1 + 14} ${y1}, ${x1 + 14} ${bottomY - 10}, ${x1 + 24} ${bottomY} ` +
          `L ${gutterHeadX - 12} ${bottomY} ` +
          `C ${gutterHeadX} ${bottomY}, ${gutterHeadX} ${bottomY - 6}, ${gutterHeadX} ${bottomY - 12} ` +
          `L ${gutterHeadX} ${y2 + 10} ` +
          `C ${gutterHeadX} ${y2 + 3}, ${gutterHeadX} ${y2}, ${gutterHeadX + 6} ${y2} ` +
          `L ${x2} ${y2}`;
        dBg = `M ${x1} ${y1} ` +
          `C ${x1 + 14} ${y1}, ${x1 + 14} ${bottomY - 10}, ${x1 + 24} ${bottomY} ` +
          `L ${gutterHeadX - 12} ${bottomY} ` +
          `C ${gutterHeadX} ${bottomY}, ${gutterHeadX} ${bottomY - 6}, ${gutterHeadX} ${bottomY - 12} ` +
          `L ${gutterHeadX} ${y2 + 10} ` +
          `C ${gutterHeadX} ${y2 + 3}, ${gutterHeadX} ${y2}, ${gutterHeadX + 6} ${y2} ` +
          `L ${x2 - 7} ${y2}`;
      } else if (conn.routeType === "bb-col-transfer") {
        // Inter-column transfer between Sub-column 1 and Sub-column 2 inside Backbone
        const x1 = rFrom.right - wrapRect.left;
        const y1 = rFrom.top + rFrom.height / 2 - wrapRect.top;
        const x2 = rTo.left - wrapRect.left;
        const y2 = rTo.top + rTo.height / 2 - wrapRect.top;
        const midX = (x1 + x2) / 2;

        d = `M ${x1} ${y1} C ${midX} ${y1}, ${midX} ${y2}, ${x2 - 10} ${y2} L ${x2} ${y2}`;
        dBg = `M ${x1} ${y1} C ${midX} ${y1}, ${midX} ${y2}, ${x2 - 10} ${y2} L ${x2 - 7} ${y2}`;
      } else if (conn.routeType === "bottom-channel-p3") {
        // Lane 1: Backbone P3 to Neck FPN Concat P3 safely below Stage 10 C2PSA / Stage 9 SPPF
        const elBbBottom = document.getElementById(bbP5Id);
        const rBbBottom = elBbBottom ? elBbBottom.getBoundingClientRect() : rFrom;

        const x1 = rFrom.right - wrapRect.left;
        const y1 = rFrom.top + rFrom.height / 2 - wrapRect.top;
        const x2 = rTo.left - wrapRect.left;
        const y2 = rTo.top + rTo.height / 2 - wrapRect.top;
        const bottomY = Math.max(rFrom.bottom, rBbBottom.bottom, rTo.bottom) - wrapRect.top + 10;
        const gapX = x1 + 10;

        d = `M ${x1} ${y1} ` +
          `C ${gapX} ${y1}, ${gapX} ${bottomY - 8}, ${gapX + 10} ${bottomY} ` +
          `L ${x2 - 14} ${bottomY} ` +
          `C ${x2 - 8} ${bottomY}, ${x2 - 8} ${y2 + 5}, ${x2 - 8} ${y2} ` +
          `L ${x2} ${y2}`;
        dBg = `M ${x1} ${y1} ` +
          `C ${gapX} ${y1}, ${gapX} ${bottomY - 8}, ${gapX + 10} ${bottomY} ` +
          `L ${x2 - 14} ${bottomY} ` +
          `C ${x2 - 8} ${bottomY}, ${x2 - 8} ${y2 + 5}, ${x2 - 8} ${y2} ` +
          `L ${x2 - 7} ${y2}`;
      } else if (conn.routeType === "bottom-channel") {
        // Lane 2: Runs cleanly below FPN column to reach Neck PAN Concat P5
        const elFpnP3 = document.getElementById(fpnP3Id);
        const rFpnP3 = elFpnP3 ? elFpnP3.getBoundingClientRect() : rFrom;

        const x1 = rFrom.right - wrapRect.left;
        const y1 = rFrom.top + rFrom.height / 2 - wrapRect.top;
        const x2 = rTo.left - wrapRect.left;
        const y2 = rTo.top + rTo.height / 2 - wrapRect.top;
        const bottomY = Math.max(rFrom.bottom, rFpnP3.bottom, rTo.bottom) - wrapRect.top + 18;
        const gutter1X = x1 + 14;

        d = `M ${x1} ${y1} ` +
          `C ${gutter1X} ${y1}, ${gutter1X} ${bottomY - 8}, ${gutter1X + 12} ${bottomY} ` +
          `L ${x2 - 14} ${bottomY} ` +
          `C ${x2 - 8} ${bottomY}, ${x2 - 8} ${y2 + 5}, ${x2 - 8} ${y2} ` +
          `L ${x2} ${y2}`;
        dBg = `M ${x1} ${y1} ` +
          `C ${gutter1X} ${y1}, ${gutter1X} ${bottomY - 8}, ${gutter1X + 12} ${bottomY} ` +
          `L ${x2 - 14} ${bottomY} ` +
          `C ${x2 - 8} ${bottomY}, ${x2 - 8} ${y2 + 5}, ${x2 - 8} ${y2} ` +
          `L ${x2 - 7} ${y2}`;
      } else {
        // Gutter-1, Gutter-Neck: clean orthogonal S-curves ending with straight horizontal segment
        const x1 = rFrom.right - wrapRect.left;
        const y1 = rFrom.top + rFrom.height / 2 - wrapRect.top;
        const x2 = rTo.left - wrapRect.left;
        const y2 = rTo.top + rTo.height / 2 - wrapRect.top;
        const midX = (x1 + x2) / 2;

        d = `M ${x1} ${y1} C ${midX} ${y1}, ${midX} ${y2}, ${x2 - 10} ${y2} L ${x2} ${y2}`;
        dBg = `M ${x1} ${y1} C ${midX} ${y1}, ${midX} ${y2}, ${x2 - 10} ${y2} L ${x2 - 7} ${y2}`;
      }

      if (!dBg) {
        dBg = d;
      }

      // Halo path (background contrast, stops before arrowhead so it never bleeds into it)
      const bgPath = document.createElementNS("http://www.w3.org/2000/svg", "path");
      bgPath.setAttribute("d", dBg);
      bgPath.setAttribute("class", `pipeline-wire-bg bg-${conn.id}`);
      bgPath.setAttribute("data-conn-id", conn.id);
      wiresLayer.appendChild(bgPath);

      // Main vector wire
      const wirePath = document.createElementNS("http://www.w3.org/2000/svg", "path");
      wirePath.setAttribute("d", d);
      wirePath.setAttribute("class", `pipeline-wire wire-${conn.id}`);
      wirePath.setAttribute("stroke", conn.color);
      wirePath.setAttribute("marker-end", `url(#${conn.marker})`);
      wirePath.setAttribute("data-conn-id", conn.id);
      wirePath.setAttribute("data-from", conn.from);
      wirePath.setAttribute("data-to", conn.to);
      wirePath.setAttribute("data-scale", conn.scale);

      // Interactive hover on wire (Đề xuất 1 + 3)
      wirePath.addEventListener("mouseenter", (e) => {
        this.highlightSingleWire(conn.id);
        this.showTooltip(conn, e);
      });
      wirePath.addEventListener("mousemove", (e) => {
        this.positionTooltip(e);
      });
      wirePath.addEventListener("mouseleave", () => {
        this.hideTooltip();
        this.resetSelection();
      });

      wiresLayer.appendChild(wirePath);
    });

    // Re-apply wire state (preserves showAllWires toggle and selected layer)
    this.applyWireState();
  }

  toggleAllWires() {
    this.showAllWires = !this.showAllWires;
    this.applyWireState();
    return this.showAllWires;
  }

  setAllWiresMode(enabled) {
    this.showAllWires = !!enabled;
    this.applyWireState();
  }

  applyWireState() {
    if (!this.svgOverlay) return;
    const wires = this.svgOverlay.querySelectorAll(".pipeline-wire");
    const bgs = this.svgOverlay.querySelectorAll(".pipeline-wire-bg");
    const pins = this.wrapper ? this.wrapper.querySelectorAll(".card-pin") : [];

    if (this.showAllWires) {
      wires.forEach(w => {
        w.classList.add("wire-highlighted");
        w.classList.remove("wire-dimmed");
      });
      bgs.forEach(b => {
        b.classList.add("bg-highlighted");
        b.classList.remove("bg-dimmed");
      });
      pins.forEach(p => p.classList.add("pin-active"));
    } else {
      if (this.selectedLayerId) {
        this.highlightForNode(`node-${this.selectedLayerId}`);
      } else {
        wires.forEach(w => w.classList.remove("wire-highlighted", "wire-dimmed"));
        bgs.forEach(b => b.classList.remove("bg-highlighted", "bg-dimmed"));
        pins.forEach(p => p.classList.remove("pin-active"));
      }
    }
  }

  highlightSingleWire(connId) {
    if (!this.svgOverlay) return;
    const wires = this.svgOverlay.querySelectorAll(".pipeline-wire");
    const bgs = this.svgOverlay.querySelectorAll(".pipeline-wire-bg");
    const pins = this.wrapper ? this.wrapper.querySelectorAll(".card-pin") : [];

    const conn = this.connections.find(c => c.id === connId);
    if (!conn) return;

    if (this.showAllWires) {
      // Khi đang bật tất cả dây: giữ toàn bộ dây sáng, chỉ làm đậm hơn dây đang hover
      wires.forEach(w => {
        w.classList.add("wire-highlighted");
        w.classList.remove("wire-dimmed");
        if (w.getAttribute("data-conn-id") === connId) {
          w.classList.add("wire-focused");
        } else {
          w.classList.remove("wire-focused");
        }
      });

      bgs.forEach(b => {
        b.classList.add("bg-highlighted");
        b.classList.remove("bg-dimmed");
        if (b.getAttribute("data-conn-id") === connId) {
          b.classList.add("bg-focused");
        } else {
          b.classList.remove("bg-focused");
        }
      });

      pins.forEach(pin => pin.classList.add("pin-active"));
      return;
    }

    wires.forEach(w => {
      if (w.getAttribute("data-conn-id") === connId) {
        w.classList.add("wire-highlighted", "wire-focused");
        w.classList.remove("wire-dimmed");
      } else {
        w.classList.remove("wire-highlighted", "wire-focused");
        w.classList.add("wire-dimmed");
      }
    });

    bgs.forEach(b => {
      if (b.getAttribute("data-conn-id") === connId) {
        b.classList.add("bg-highlighted", "bg-focused");
        b.classList.remove("bg-dimmed");
      } else {
        b.classList.remove("bg-highlighted", "bg-focused");
        b.classList.add("bg-dimmed");
      }
    });

    pins.forEach(pin => {
      const parent = pin.closest(".pipeline-node-card, .head-port-card");
      if (parent && (parent.id === conn.from || parent.id === conn.to)) {
        pin.classList.add("pin-active");
      } else {
        pin.classList.remove("pin-active");
      }
    });
  }

  highlightForNode(nodeId) {
    if (!this.svgOverlay) return;
    const wires = this.svgOverlay.querySelectorAll(".pipeline-wire");
    const bgs = this.svgOverlay.querySelectorAll(".pipeline-wire-bg");
    const pins = this.wrapper ? this.wrapper.querySelectorAll(".card-pin") : [];

    if (!nodeId) {
      if (this.showAllWires) {
        this.applyWireState();
        return;
      }
      wires.forEach(w => w.classList.remove("wire-highlighted", "wire-dimmed", "wire-focused"));
      bgs.forEach(b => b.classList.remove("bg-highlighted", "bg-dimmed", "bg-focused"));
      pins.forEach(p => p.classList.remove("pin-active"));
      return;
    }

    const matchingConnIds = new Set();
    const activeNodes = new Set([nodeId]);

    this.connections.forEach(conn => {
      if (conn.from === nodeId || conn.to === nodeId) {
        matchingConnIds.add(conn.id);
        activeNodes.add(conn.from);
        activeNodes.add(conn.to);
      }
    });

    if (this.showAllWires) {
      // Khi đang bật tất cả dây: KHÔNG tắt/làm mờ các dây khác!
      // Giữ toàn bộ dây sáng (wire-highlighted), chỉ nhấn mạnh thêm (wire-focused) các dây kết nối vào khối đang tương tác
      wires.forEach(w => {
        w.classList.add("wire-highlighted");
        w.classList.remove("wire-dimmed");
        const cid = w.getAttribute("data-conn-id");
        if (matchingConnIds.has(cid)) {
          w.classList.add("wire-focused");
        } else {
          w.classList.remove("wire-focused");
        }
      });

      bgs.forEach(b => {
        b.classList.add("bg-highlighted");
        b.classList.remove("bg-dimmed");
        const cid = b.getAttribute("data-conn-id");
        if (matchingConnIds.has(cid)) {
          b.classList.add("bg-focused");
        } else {
          b.classList.remove("bg-focused");
        }
      });

      pins.forEach(pin => pin.classList.add("pin-active"));
      return;
    }

    if (matchingConnIds.size === 0) {
      if (this.showAllWires) {
        wires.forEach(w => {
          w.classList.add("wire-highlighted");
          w.classList.remove("wire-dimmed", "wire-focused");
        });
        bgs.forEach(b => {
          b.classList.add("bg-highlighted");
          b.classList.remove("bg-dimmed", "bg-focused");
        });
        pins.forEach(p => p.classList.add("pin-active"));
      } else {
        wires.forEach(w => w.classList.remove("wire-highlighted", "wire-dimmed", "wire-focused"));
        bgs.forEach(b => b.classList.remove("bg-highlighted", "bg-dimmed", "bg-focused"));
        pins.forEach(p => p.classList.remove("pin-active"));
      }
      return;
    }

    wires.forEach(w => {
      const cid = w.getAttribute("data-conn-id");
      if (matchingConnIds.has(cid)) {
        w.classList.add("wire-highlighted", "wire-focused");
        w.classList.remove("wire-dimmed");
      } else {
        w.classList.remove("wire-highlighted", "wire-focused");
        w.classList.add("wire-dimmed");
      }
    });

    bgs.forEach(b => {
      const cid = b.getAttribute("data-conn-id");
      if (matchingConnIds.has(cid)) {
        b.classList.add("bg-highlighted", "bg-focused");
        b.classList.remove("bg-dimmed");
      } else {
        b.classList.remove("bg-highlighted", "bg-focused");
        b.classList.add("bg-dimmed");
      }
    });

    pins.forEach(pin => {
      const parent = pin.closest(".pipeline-node-card, .head-port-card");
      if (parent && activeNodes.has(parent.id)) {
        pin.classList.add("pin-active");
      } else {
        pin.classList.remove("pin-active");
      }
    });
  }

  showTooltip(conn, event) {
    if (!this.tooltipEl) return;
    this.tooltipEl.innerHTML = `
      <div class="tooltip-title" style="color: ${conn.color};">
        <span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:${conn.color};"></span>
        ${conn.title}
      </div>
      <div class="tooltip-flow">${conn.flow}</div>
      <div class="tooltip-desc">${conn.desc}</div>
    `;
    this.tooltipEl.classList.add("visible");
    this.positionTooltip(event);
  }

  positionTooltip(event) {
    if (!this.tooltipEl || !this.wrapper) return;
    const wrapRect = this.wrapper.getBoundingClientRect();
    const x = event.clientX - wrapRect.left + 14;
    const y = event.clientY - wrapRect.top + 14;

    const maxLeft = wrapRect.width - 320;
    const clampedX = Math.max(10, Math.min(x, maxLeft));

    this.tooltipEl.style.left = `${clampedX}px`;
    this.tooltipEl.style.top = `${y}px`;
  }

  hideTooltip() {
    if (this.tooltipEl) {
      this.tooltipEl.classList.remove("visible");
    }
  }

  createInteractiveNode(layerId, name, type, roleSummary) {
    const layers = this.currentData ? this.currentData.layers || {} : {};
    const layerData = layers[layerId] || { id: layerId, name, type, shape: ["-", "-", "-", "-"] };

    const card = document.createElement("div");
    let classNames = "pipeline-node-card interactive";
    if (this.selectedLayerId === layerId) classNames += " selected";
    card.className = classNames;
    card.id = `node-${layerId}`;
    card.setAttribute("data-layer", layerId);

    // Top
    const topEl = document.createElement("div");
    topEl.className = "node-card-top";

    const nameEl = document.createElement("span");
    nameEl.className = "node-title";
    nameEl.textContent = name;
    topEl.appendChild(nameEl);

    const typeEl = document.createElement("span");
    typeEl.className = `node-tag tag-${type.toLowerCase()}`;
    typeEl.textContent = type;
    topEl.appendChild(typeEl);

    card.appendChild(topEl);

    // Body: Thumbnail + Shape
    const bodyEl = document.createElement("div");
    bodyEl.className = "node-card-body";

    const wrap = document.createElement("div");
    wrap.className = "node-thumb-wrap";
    if (layerData.mean_heatmap) {
      const img = document.createElement("img");
      img.className = "node-thumb-img";
      img.src = layerData.mean_heatmap;
      img.alt = name;
      img.loading = "lazy";
      img.addEventListener("load", () => this.scheduleDrawConnections());
      wrap.appendChild(img);
    } else if (layerId.includes("head")) {
      const placeholder = document.createElement("div");
      placeholder.className = "node-thumb-placeholder head-target-thumb";
      placeholder.title = "Detection Prediction Head";
      placeholder.innerHTML = `
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#dc2626" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
          <circle cx="12" cy="12" r="7"></circle>
          <line x1="12" y1="2" x2="12" y2="6"></line>
          <line x1="12" y1="18" x2="12" y2="22"></line>
          <line x1="2" y1="12" x2="6" y2="12"></line>
          <line x1="18" y1="12" x2="22" y2="12"></line>
        </svg>
      `;
      wrap.appendChild(placeholder);
    } else {
      const placeholder = document.createElement("div");
      placeholder.className = "node-thumb-placeholder";
      placeholder.innerHTML = `
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <rect x="3" y="3" width="7" height="7" rx="1"></rect>
          <rect x="14" y="3" width="7" height="7" rx="1"></rect>
          <rect x="14" y="14" width="7" height="7" rx="1"></rect>
          <rect x="3" y="14" width="7" height="7" rx="1"></rect>
        </svg>
      `;
      wrap.appendChild(placeholder);
    }
    bodyEl.appendChild(wrap);

    const metaCol = document.createElement("div");
    metaCol.className = "node-card-meta";

    if (layerData.shape && layerData.shape.length === 4) {
      const [b, c, h, w] = layerData.shape;
      const shapeSpan = document.createElement("span");
      shapeSpan.className = "shape-badge";
      shapeSpan.textContent = `${c}ch × ${w}×${h}`;
      metaCol.appendChild(shapeSpan);
    }

    if (roleSummary) {
      const descSpan = document.createElement("span");
      descSpan.className = "role-summary";
      descSpan.textContent = roleSummary;
      metaCol.appendChild(descSpan);
    }

    bodyEl.appendChild(metaCol);
    card.appendChild(bodyEl);

    // Hover interactive highlighting
    card.addEventListener("mouseenter", () => {
      this.highlightForNode(card.id);
    });
    card.addEventListener("mouseleave", () => {
      this.resetSelection();
    });

    // Click selection (toggle deselect if already selected)
    card.addEventListener("click", (e) => {
      e.stopPropagation();
      if (this.selectedLayerId === layerId) {
        this.resetSelection();
        return;
      }
      this.selectLayer(layerId, true);
    });

    return card;
  }

  createHelperNode(title, type, desc, id = null) {
    const card = document.createElement("div");
    card.className = "pipeline-node-card helper-node interactive";
    if (id) card.id = id;
    card.innerHTML = `
      <div class="node-card-top">
        <span class="node-title">${title}</span>
        <span class="node-tag tag-helper">${type}</span>
      </div>
      <div class="node-desc">${desc}</div>
    `;

    card.addEventListener("mouseenter", () => {
      if (card.id) this.highlightForNode(card.id);
    });
    card.addEventListener("mouseleave", () => {
      this.resetSelection();
    });

    if (id) {
      card.addEventListener("click", (e) => {
        e.stopPropagation();
        this.selectHelper(id, title, type, desc);
      });
    }

    return card;
  }

  createArrow(direction = "down") {
    const arrow = document.createElement("div");
    arrow.className = `pipeline-arrow ${direction}`;
    if (direction === "right") {
      arrow.innerHTML = `
        <svg width="18" height="14" viewBox="0 0 18 14" fill="none">
          <line x1="2" y1="7" x2="11" y2="7" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
          <polygon points="16,7 10,3 10,11" fill="currentColor"/>
        </svg>
      `;
    } else {
      arrow.innerHTML = `
        <svg width="14" height="18" viewBox="0 0 14 18" fill="none">
          <line x1="7" y1="2" x2="7" y2="11" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
          <polygon points="7,16 3,10 11,10" fill="currentColor"/>
        </svg>
      `;
    }
    return arrow;
  }

  createPin(side = "right", scaleType = "p3") {
    const pin = document.createElement("div");
    pin.className = `card-pin pin-${side} pin-${scaleType}`;
    return pin;
  }

  createSkipBadge(text) {
    const badge = document.createElement("div");
    badge.className = "skip-connection-badge";
    badge.innerHTML = `<span>${text}</span>`;
    return badge;
  }

  createOutBadge(text) {
    const badge = document.createElement("div");
    badge.className = "out-connection-badge";
    badge.innerHTML = `<span>${text}</span>`;
    return badge;
  }

  selectLayer(layerId, userInitiated = false) {
    this.selectedLayerId = layerId;

    // Update selected class on node cards
    document.querySelectorAll(".pipeline-node-card.interactive").forEach(node => {
      node.classList.remove("selected");
    });
    const activeNode = document.getElementById(`node-${layerId}`);
    if (activeNode) {
      activeNode.classList.add("selected");
    }

    // Highlight corresponding connections (Đề xuất 1)
    this.highlightForNode(`node-${layerId}`);

    // Trigger external callback with userInitiated flag
    if (this.currentData && this.currentData.layers && this.currentData.layers[layerId]) {
      this.onSelectLayer(this.currentData.layers[layerId], this.currentData, userInitiated);
    }
  }

  resetSelection() {
    this.selectedLayerId = null;
    if (this.wrapper) {
      this.wrapper.querySelectorAll(".pipeline-node-card.selected").forEach(node => {
        node.classList.remove("selected");
      });
    }
    this.highlightForNode(null);
  }

  selectResult(inspectData) {
    this.selectedLayerId = "result";

    if (this.wrapper) {
      this.wrapper.querySelectorAll(".pipeline-node-card.selected").forEach(node => {
        node.classList.remove("selected");
      });
    }
    const resultNode = document.getElementById("node-result");
    if (resultNode) {
      resultNode.classList.add("selected");
    }

    if (typeof this.onSelectResult === "function") {
      this.onSelectResult(inspectData);
    }
  }

  selectHelper(id, title, type, desc) {
    if (this.selectedLayerId === id) {
      this.resetSelection();
      return;
    }
    this.selectedLayerId = id;
    if (this.wrapper) {
      this.wrapper.querySelectorAll(".pipeline-node-card.selected").forEach(node => {
        node.classList.remove("selected");
      });
    }
    const card = document.getElementById(id);
    if (card) card.classList.add("selected");
    this.highlightForNode(id);

    const helperData = this.buildHelperLayerData(id, title, type, desc);
    if (typeof this.onSelectLayer === "function") {
      this.onSelectLayer(helperData, this.currentData, true);
    }
  }

  buildHelperLayerData(id, title, type, desc) {
    const layers = (this.currentData && this.currentData.layers) ? this.currentData.layers : {};
    const stemShape = layers["layer_0_stem"]?.shape || [1, 16, 320, 320];
    const p2Shape = layers["layer_2_p2"]?.shape || [1, 32, 160, 160];
    const p3Shape = layers["layer_4_p3"]?.shape || [1, 64, 80, 80];
    const p4Shape = layers["layer_6_p4"]?.shape || [1, 128, 40, 40];
    const p5Shape = layers["layer_8_p5"]?.shape || [1, 256, 20, 20];
    const fpnP4Shape = (layers["layer_12_neck_p4"] || layers["layer_13_neck_p4"])?.shape || [1, 128, 40, 40];
    const fpnP3Shape = (layers["layer_15_neck_p3"] || layers["layer_16_neck_p3"])?.shape || [1, 64, 80, 80];
    const panP4Shape = (layers["layer_18_neck_pan_p4"] || layers["layer_19_neck_pan_p4"])?.shape || [1, 128, 40, 40];
    const p5OutShape = (layers["layer_10_c2psa"] || layers["layer_9_sppf"])?.shape || [1, 256, 20, 20];

    let shape = [1, 32, 160, 160];
    let input_shape = [1, 16, 320, 320];
    let rf = "5×5";
    let badge = "Conv s=2";

    if (id === "node-layer_1_conv") {
      shape = [1, p2Shape[1], 160, 160];
      input_shape = [1, stemShape[1], 320, 320];
      rf = "5×5";
      badge = "Conv s=2";
    } else if (id === "node-layer_3_conv") {
      shape = [1, p3Shape[1], 80, 80];
      input_shape = [1, p2Shape[1], 160, 160];
      rf = "11×11";
      badge = "Conv s=2";
    } else if (id === "node-layer_5_conv") {
      shape = [1, p4Shape[1], 40, 40];
      input_shape = [1, p3Shape[1], 80, 80];
      rf = "23×23";
      badge = "Conv s=2";
    } else if (id === "node-layer_7_conv") {
      shape = [1, p5Shape[1], 20, 20];
      input_shape = [1, p4Shape[1], 40, 40];
      rf = "47×47";
      badge = "Conv s=2";
    } else if (id === "node-fpn-up1") {
      shape = [1, p5Shape[1], 40, 40];
      input_shape = [1, p5Shape[1], 20, 20];
      rf = "Multi-scale";
      badge = "Upsample 2×";
    } else if (id === "node-fpn-concat-p4") {
      shape = [1, p4Shape[1] + p5Shape[1], 40, 40];
      input_shape = [[1, p4Shape[1], 40, 40], [1, p5Shape[1], 40, 40]];
      rf = "Multi-scale";
      badge = "Concat P4";
    } else if (id === "node-fpn-up2") {
      shape = [1, fpnP4Shape[1], 80, 80];
      input_shape = [1, fpnP4Shape[1], 40, 40];
      rf = "Multi-scale";
      badge = "Upsample 2×";
    } else if (id === "node-fpn-concat-p3") {
      shape = [1, p3Shape[1] + fpnP4Shape[1], 80, 80];
      input_shape = [[1, p3Shape[1], 80, 80], [1, fpnP4Shape[1], 80, 80]];
      rf = "Multi-scale";
      badge = "Concat P3";
    } else if (id === "node-pan-conv1") {
      shape = [1, fpnP3Shape[1], 40, 40];
      input_shape = [1, fpnP3Shape[1], 80, 80];
      rf = "Multi-scale";
      badge = "Conv s=2";
    } else if (id === "node-pan-concat-p4") {
      shape = [1, fpnP4Shape[1] + fpnP3Shape[1], 40, 40];
      input_shape = [[1, fpnP4Shape[1], 40, 40], [1, fpnP3Shape[1], 40, 40]];
      rf = "Multi-scale";
      badge = "Concat PAN P4";
    } else if (id === "node-pan-conv2") {
      shape = [1, panP4Shape[1], 20, 20];
      input_shape = [1, panP4Shape[1], 40, 40];
      rf = "Multi-scale";
      badge = "Conv s=2";
    } else if (id === "node-pan-concat-p5") {
      shape = [1, p5OutShape[1] + panP4Shape[1], 20, 20];
      input_shape = [[1, p5OutShape[1], 20, 20], [1, panP4Shape[1], 20, 20]];
      rf = "Multi-scale";
      badge = "Concat PAN P5";
    }

    return {
      id: id,
      name: title,
      type: type,
      role: desc,
      stage: id.includes("layer_") ? "backbone" : "neck",
      shape: shape,
      input_shape: input_shape,
      receptive_field: rf,
      badge: badge,
      mean_heatmap: null,
      top_channels: []
    };
  }
}
