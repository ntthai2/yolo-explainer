/**
 * Deep-Dive Inspector Studio for YOLO Explainer.
 * Controls Before/After Slider, Filter Bank Top 16 Channels,
 * Architectural Theory, and Detection Output Tab.
 */

class LayerInspector {
  constructor() {
    this.titleEl = document.getElementById("studio-layer-title");
    this.descEl = document.getElementById("studio-desc");
    this.badgeEl = document.getElementById("studio-badge");

    this.sliderRange = document.getElementById("slider-range");
    this.sliderWrap = document.getElementById("slider-img-wrap");
    this.sliderHandle = document.getElementById("slider-handle");
    this.imgUnder = document.getElementById("slider-under-img");
    this.imgOver = document.getElementById("slider-over-img");

    this.filterGrid = document.getElementById("filter-grid");
    this.filterCount = document.getElementById("filter-count");

    this.theoryContainer = document.getElementById("theory-container");

    this.detImg = document.getElementById("detections-img");
    this.detList = document.getElementById("detections-list");
    this.detCount = document.getElementById("det-count");
    this.detCaption = document.getElementById("det-img-caption");
    this.detSegmentedToggle = document.getElementById("det-segmented-toggle");
    this.btnPostNms = document.getElementById("btn-mode-post-nms");
    this.btnPreNms = document.getElementById("btn-mode-pre-nms");
    this.countPostNms = document.getElementById("count-post-nms");
    this.countPreNms = document.getElementById("count-pre-nms");
    this.detMode = "post"; // "post" or "pre"

    if (this.btnPostNms) {
      this.btnPostNms.addEventListener("click", () => this.setDetMode("post"));
    }
    if (this.btnPreNms) {
      this.btnPreNms.addEventListener("click", () => this.setDetMode("pre"));
    }

    if (this.sliderRange) {
      this.sliderRange.addEventListener("input", (e) => {
        this.updateSlider(e.target.value);
      });
    }

    this.setupTabs();
  }

  setDetMode(mode) {
    this.detMode = mode;
    if (this.btnPostNms && this.btnPreNms) {
      if (mode === "post") {
        this.btnPostNms.classList.add("active");
        this.btnPreNms.classList.remove("active");
      } else {
        this.btnPostNms.classList.remove("active");
        this.btnPreNms.classList.add("active");
      }
    }
    if (this.currentInspectData) {
      this.renderDetections(this.currentInspectData);
    }
  }

  setupTabs() {
    const tabBtns = document.querySelectorAll(".tab-btn");
    tabBtns.forEach(btn => {
      btn.addEventListener("click", () => {
        tabBtns.forEach(b => b.classList.remove("active"));
        btn.classList.add("active");

        const targetTab = btn.getAttribute("data-tab");
        document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
        const activeTabEl = document.getElementById(`tab-${targetTab}`);
        if (activeTabEl) {
          activeTabEl.classList.add("active");
        }
      });
    });
  }

  updateSlider(val) {
    if (this.sliderWrap && this.sliderHandle) {
      const v = Math.max(0, Math.min(100, Number(val)));
      this.sliderWrap.style.width = "100%";
      this.sliderWrap.style.clipPath = `inset(0 ${100 - v}% 0 0)`;
      this.sliderWrap.style.webkitClipPath = `inset(0 ${100 - v}% 0 0)`;
      this.sliderHandle.style.left = `${v}%`;
    }
  }

  inspect(layerData, inspectData) {
    if (!layerData) return;

    this.titleEl.textContent = layerData.name || "Chi tiết khối";
    this.descEl.textContent = layerData.role || "";
    if (this.badgeEl) {
      this.badgeEl.textContent = layerData.badge || layerData.type || "";
      this.badgeEl.className = "studio-badge badge-c3k2";
    }

    // Configure tabs: Show Block tabs (Theory first, then Slider if has feature map), Hide Detection tab
    const tabTheory = document.getElementById("tab-btn-theory");
    const tabSlider = document.getElementById("tab-btn-slider");
    const tabDets = document.getElementById("tab-btn-detections");
    const hasFeatureMap = Boolean(layerData && layerData.mean_heatmap);

    if (tabTheory) tabTheory.style.display = "";
    if (tabSlider) tabSlider.style.display = hasFeatureMap ? "" : "none";
    if (tabDets) tabDets.style.display = "none";

    // Activate Sơ Đồ Cấu Tạo & Công Thức (theory) as first tab
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
    if (tabTheory) tabTheory.classList.add("active");
    const theoryContent = document.getElementById("tab-theory");
    if (theoryContent) theoryContent.classList.add("active");

    if (this.imgUnder && this.imgOver) {
      this.imgUnder.src = inspectData.original_image || "";
      this.imgOver.src = layerData.mean_heatmap || inspectData.original_image;
      if (this.sliderRange) {
        this.sliderRange.value = 50;
        this.updateSlider(50);
      }
    }

    this.renderTheory(layerData, inspectData.model_name);
  }

  inspectResult(inspectData) {
    if (!inspectData) return;

    const dets = inspectData.detections || [];
    this.titleEl.textContent = "Kết Quả Phát Hiện Đầu Ra (Detection Results)";
    this.descEl.textContent = `Mô hình phát hiện ${dets.length} đối tượng với thời gian suy luận ${inspectData.inference_time_ms || 0} ms trên thiết bị ${inspectData.device || 'CPU'}.`;
    if (this.badgeEl) {
      this.badgeEl.textContent = `${dets.length} Hộp`;
      this.badgeEl.className = "studio-badge badge-head";
    }

    // Configure tabs: Hide Block tabs, Show Detection tab
    const tabTheory = document.getElementById("tab-btn-theory");
    const tabSlider = document.getElementById("tab-btn-slider");
    const tabDets = document.getElementById("tab-btn-detections");
    if (tabTheory) tabTheory.style.display = "none";
    if (tabSlider) tabSlider.style.display = "none";
    if (tabDets) tabDets.style.display = "";

    // Activate Detections tab
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
    if (tabDets) tabDets.classList.add("active");
    const detContent = document.getElementById("tab-detections");
    if (detContent) detContent.classList.add("active");

    // Reset mode to Post-NMS upon new inspection
    this.detMode = "post";
    if (this.btnPostNms && this.btnPreNms) {
      this.btnPostNms.classList.add("active");
      this.btnPreNms.classList.remove("active");
    }

    this.renderDetections(inspectData);
  }

  renderFilterBank(channels) {
    this.filterGrid.innerHTML = "";
    if (this.filterCount) {
      this.filterCount.textContent = `${channels.length} kênh có chuẩn L2 lớn nhất`;
    }

    if (channels.length === 0) {
      this.filterGrid.innerHTML = "<div style='color:#94a3b8; font-size:0.8rem;'>Không có dữ liệu kênh.</div>";
      return;
    }

    channels.forEach(ch => {
      const card = document.createElement("div");
      card.className = "channel-card";

      const img = document.createElement("img");
      img.className = "channel-thumb";
      img.src = ch.image;
      img.alt = `Kênh ${ch.channel_index}`;
      img.loading = "lazy";
      card.appendChild(img);

      const label = document.createElement("div");
      label.className = "channel-label";
      label.textContent = `Kênh ${ch.channel_index}`;
      card.appendChild(label);

      const score = document.createElement("div");
      score.className = "channel-score";
      score.textContent = `L2: ${ch.energy}`;
      card.appendChild(score);

      card.title = `Kênh ${ch.channel_index} | Hạng ${ch.rank} | Nhỏ nhất ${ch.min_val} | Lớn nhất ${ch.max_val}`;
      this.filterGrid.appendChild(card);
    });
  }

  renderLatex(latex, displayMode = true) {
    if (window.katex && typeof window.katex.renderToString === "function") {
      try {
        return window.katex.renderToString(latex, {
          displayMode: displayMode,
          throwOnError: false
        });
      } catch (e) {
        console.warn("KaTeX render error:", e);
      }
    }
    return `<div class="formula-line">${latex}</div>`;
  }

  formatMathBlock(latexList) {
    if (!Array.isArray(latexList)) {
      latexList = [latexList];
    }
    return latexList.map(item => {
      if (typeof item === "string" && (item.includes("\\") || item.includes("=") || item.includes("\\times") || item.includes("\\in") || item.includes("\\to") || item.includes("\\rightarrow") || item.includes("\\left") || item.includes("\\lfloor"))) {
        return this.renderLatex(item, true);
      }
      return `<div class="formula-line">${item}</div>`;
    }).join("");
  }

  computeTransformationDetails(layerData, modelName) {
    const lType = layerData.type || "";
    const shape = layerData.shape || [1, 16, 320, 320];
    const b = shape[0] !== undefined ? shape[0] : 1;
    const c = shape[1] !== undefined ? shape[1] : 16;
    const h = shape[2] !== undefined ? shape[2] : 320;
    const w = shape[3] !== undefined ? shape[3] : 320;
    const isV8 = modelName === "yolov8n";
    const isV26 = modelName === "yolo26n";

    // 1. Stage 0: Stem Conv
    if (layerData.id === "layer_0_stem") {
      return {
        opBadge: "Conv 3×3 (s=2)",
        opShort: "ConvModule (s=2, p=1)",
        inputShapeDisplay: `1 × 3 × 640 × 640`,
        outputShapeDisplay: `1 × ${c} × ${h} × ${w}`,
        spatialMath: [
          "H_{out} = \\left\\lfloor \\frac{640 + 2(1) - 3}{2} \\right\\rfloor + 1 = 320, \\quad W_{out} = 320"
        ],
        notation: ["k = 3 (Kernel 3×3)", "s = 2 (Bước nhảy / Stride)", "p = 1 (Đệm viền / Padding)", `C_{out} = ${c} (Số filter)`],
        spatialNote: `Bước nhảy s=2 nén kích thước không gian một nửa (640² ➔ 320²), đồng thời mở rộng từ 3 kênh ảnh ban đầu sang ${c} kênh đặc trưng cơ sở.`,
        algebraicMath: [
          "Y = \\text{Conv2d}(X, W) + b",
          "\\hat{Y} = \\text{BatchNorm2d}(Y) = \\gamma \\cdot \\frac{Y - \\mu_B}{\\sqrt{\\sigma_B^2 + \\epsilon}} + \\beta",
          "X_{out} = \\text{SiLU}(\\hat{Y}) = \\hat{Y} \\cdot \\sigma(\\hat{Y}) = \\frac{\\hat{Y}}{1 + e^{-\\hat{Y}}}"
        ],
        algebraicNote: "Chuỗi chuẩn: Conv2d trích xuất biên ➔ BatchNorm ổn định phân phối ➔ SiLU (Swish) bảo toàn gradient hiệu quả.",
        channelMath: [
          `C_{in} = 3 \\text{ (RGB)} \\longrightarrow C_{out} = ${c} \\text{ kênh đặc trưng cơ sở}`
        ],
        channelNote: "",
        dlIntuitions: [
          "Dùng Conv stride=2 ngay đầu mạng để nén kích thước ảnh (640² ➔ 320²), giảm 4 lần số phép tính (FLOPs) cho toàn bộ các tầng phía sau.",
          "Conv2d đi kèm BatchNorm2d nên đặt bias=False vì hệ số dịch chuyển β của BatchNorm đã đảm nhiệm vai trò này."
        ]
      };
    }

    // 2. Convolution with Downsampling (Helper Conv stages or Conv type with s=2)
    if (lType === "Conv") {
      let inH = h * 2;
      let inW = w * 2;
      let inC = layerData.input_shape ? (Array.isArray(layerData.input_shape[0]) ? layerData.input_shape[0][1] : layerData.input_shape[1]) : Math.max(16, Math.round(c / 2));
      if (layerData.input_shape && layerData.input_shape.length === 4) {
        inC = layerData.input_shape[1];
        inH = layerData.input_shape[2];
        inW = layerData.input_shape[3];
      }

      return {
        opBadge: "Conv 3×3 (s=2)",
        opShort: `Downsample ${inH}×${inW} ➔ ${h}×${w}`,
        inputShapeDisplay: `1 × ${inC} × ${inH} × ${inW}`,
        outputShapeDisplay: `1 × ${c} × ${h} × ${w}`,
        spatialMath: [
          `H_{out} = \\left\\lfloor \\frac{${inH} + 2(1) - 3}{2} \\right\\rfloor + 1 = ${h}, \\quad W_{out} = ${w}`
        ],
        notation: ["k = 3", "s = 2 (Giảm mẫu / Downsample)", "p = 1 (Padding)", `C_{out} = ${c} (Số kênh ra)`],
        spatialNote: `Giảm mẫu không gian từ ${inH}×${inW} về ${h}×${w}, đồng thời tăng số kênh để duy trì dung lượng biểu diễn thông tin ngữ nghĩa.`,
        algebraicMath: [
          `\\hat{Y} = \\text{BatchNorm2d}\\left(\\text{Conv2d}_{3\\times 3, \\, s=2}(X)\\right)`,
          `X_{out} = \\text{SiLU}(\\hat{Y}) = \\frac{\\hat{Y}}{1 + e^{-\\hat{Y}}}`
        ],
        algebraicNote: "Tích chập 3×3 stride=2 kết hợp Batch Normalization và kích hoạt phi tuyến SiLU.",
        channelMath: [
          `C_{in} = ${inC} \\longrightarrow C_{out} = ${c} \\text{ kênh (Tăng dung lượng đặc trưng)}`
        ],
        channelNote: "",
        dlIntuitions: [
          "Dùng Conv stride=2 thay cho MaxPool để mô hình tự học các trọng số lấy mẫu giảm (downsampling) tối ưu theo dữ liệu.",
          "Quy tắc thiết kế kinh nghiệm (Heuristic) trong CNN: Khi kích thước không gian giảm một nửa (H/2, W/2), số kênh thường được tăng gấp đôi (2×C) nhằm duy trì dung lượng biểu diễn thông tin."
        ]
      };
    }

    // 3. CSP Blocks: C3k2 or C2f
    if (lType === "C3k2" || lType === "C2f") {
      const isC3k2 = lType === "C3k2";
      const cMid = Math.round(c * 0.5);
      return {
        opBadge: lType,
        opShort: `${lType} (CSP)`,
        inputShapeDisplay: `1 × ${c} × ${h} × ${w}`,
        outputShapeDisplay: `1 × ${c} × ${h} × ${w}`,
        spatialMath: [
          `H_{out} = H_{in} = ${h}, \\quad W_{out} = W_{in} = ${w} \\quad (k=3, \\, s=1, \\, p=1)`
        ],
        notation: ["k = 3", "s = 1 (Bảo toàn không gian)", "p = 1", "CSP = Cross Stage Partial"],
        spatialNote: `Khối ${lType} bảo toàn nguyên vẹn kích thước không gian ${h}×${w} và số kênh ${c}, chỉ tinh lọc các biểu diễn đặc trưng.`,
        algebraicMath: isC3k2 ? [
          `X_{base} = \\text{Conv}_{1\\times 1}(X) \\in \\mathbb{R}^{${c} \\times ${h} \\times ${w}}`,
          `[X_1, X_2] = \\text{Split}(X_{base}) \\quad \\left(X_1, X_2 \\in \\mathbb{R}^{${cMid} \\times ${h} \\times ${w}}\\right)`,
          `Y_1 = \\text{C3k}(X_2), \\quad Y_2 = \\text{C3k}(Y_1)`,
          `X_{out} = \\text{Conv}_{1\\times 1}\\left(\\text{Concat}([X_1, X_2, Y_1, Y_2])\\right) \\in \\mathbb{R}^{${c} \\times ${h} \\times ${w}}`
        ] : [
          `X_{base} = \\text{Conv}_{1\\times 1}(X), \\quad [X_1, X_2] = \\text{Split}(X_{base})`,
          `B_1 = \\text{Bottleneck}(X_2), \\quad B_2 = \\text{Bottleneck}(B_1)`,
          `X_{out} = \\text{Conv}_{1\\times 1}\\left(\\text{Concat}([X_1, X_2, B_1, B_2])\\right)`
        ],
        algebraicNote: isC3k2 ?
          "C3k2 (trong YOLO11 & 26) dùng module con C3k tinh gọn, giảm chi phí FLOPs và tham số mà vẫn đạt độ chính xác cao." :
          "C2f (trong YOLOv8) tách luồng đặc trưng để gradient lan truyền trực tiếp qua nhánh tắt, giúp huấn luyện ổn định.",
        channelMath: [
          `C_{in} = ${c} \\xrightarrow{\\text{Split}} [${cMid}, \\, ${cMid}] \\xrightarrow{\\text{Concat}} ${cMid * 4} \\xrightarrow{\\text{Conv}_{1\\times 1}} C_{out} = ${c}`
        ],
        channelNote: "",
        dlIntuitions: [
          "Bản chất CSPNet: Tách đôi tensor để một nửa đi tắt qua shortcut, giảm khối lượng tính toán và giúp gradient lan truyền sâu mà không bị suy biến (Vanishing Gradient).",
          isC3k2 ?
            "C3k2 thay thế Bottleneck tiêu chuẩn của C2f bằng module con C3k có số tham số thấp hơn, giúp giảm FLOPs trong khi duy trì độ chính xác (mAP)." :
            "C2f trong YOLOv8 kết hợp nhiều bottleneck nhỏ để tăng khả năng học các mối quan hệ phi tuyến tính."
        ]
      };
    }

    // 4. SPPF
    if (lType === "SPPF") {
      const cMid = Math.round(c * 0.5);
      const cCat = cMid * 4;
      return {
        opBadge: "SPPF",
        opShort: "Spatial Pyramid Pooling",
        inputShapeDisplay: `1 × ${c} × ${h} × ${w}`,
        outputShapeDisplay: `1 × ${c} × ${h} × ${w}`,
        spatialMath: [
          `H_{out} = 20, \\quad W_{out} = 20 \\quad (k=5, \\, s=1, \\, p=2)`
        ],
        notation: ["k = 5 (Kernel Pooling)", "s = 1 (Bước nhảy 1)", "p = 2 (Đệm viền giữ 20×20)", "Hợp nhất 4 thang đo"],
        spatialNote: "Giữ nguyên kích thước 20×20 và số lượng kênh, hợp nhất 4 cấp độ trường tiếp nhận từ cục bộ đến toàn cảnh.",
        algebraicMath: [
          `X_0 = \\text{Conv}_{1\\times 1}(X) \\in \\mathbb{R}^{${cMid} \\times 20 \\times 20}`,
          `Y_1 = \\text{MaxPool}_{5\\times 5}(X_0) \\quad (\\text{Trường nhìn } 5\\times 5)`,
          `Y_2 = \\text{MaxPool}_{5\\times 5}(Y_1) \\quad (\\text{Trường nhìn } 9\\times 9)`,
          `Y_3 = \\text{MaxPool}_{5\\times 5}(Y_2) \\quad (\\text{Trường nhìn } 13\\times 13)`,
          `X_{out} = \\text{Conv}_{1\\times 1}\\left(\\text{Concat}([X_0, Y_1, Y_2, Y_3])\\right) \\in \\mathbb{R}^{${c} \\times 20 \\times 20}`
        ],
        algebraicNote: "Nối tiếp 3 lớp MaxPool 5×5 thay vì tính toán với các kernel lớn (5×5, 9×9, 13×13), giúp giảm đáng kể khối lượng tính toán.",
        channelMath: [
          `C_{in} = ${c} \\xrightarrow{\\text{Conv}_{1\\times 1}} ${cMid} \\xrightarrow{\\text{Concat 4 luồng}} ${cCat} \\xrightarrow{\\text{Conv}_{1\\times 1}} C_{out} = ${c}`
        ],
        channelNote: "",
        dlIntuitions: [
          "Nguyên lý tương đương: 3 lớp MaxPool 5×5 nối tiếp đạt cùng trường tiếp nhận 13×13 nhưng có chi phí tính toán thấp hơn so với một phép MaxPool 13×13 trực tiếp.",
          "Mở rộng trường tiếp nhận đa thang đo (Multi-scale Receptive Field) giúp mô hình thu thập ngữ cảnh cho các kích thước đối tượng khác nhau trước khi chuyển sang Neck."
        ]
      };
    }

    // 5. C2PSA Attention
    if (lType === "C2PSA") {
      const cMid = Math.round(c * 0.5);
      return {
        opBadge: "C2PSA Attention",
        opShort: "Polarized Self-Attention",
        inputShapeDisplay: `1 × ${c} × 20 × 20`,
        outputShapeDisplay: `1 × ${c} × 20 × 20`,
        spatialMath: [
          `N = H \\times W = 20 \\times 20 = 400 \\text{ token không gian}, \\quad H_{out} = 20, \\; W_{out} = 20`
        ],
        notation: ["Q: Query (Truy vấn)", "K: Key (Khóa)", "V: Value (Giá trị)", "400 token = 20×20"],
        spatialNote: "Self-Attention hoạt động trên 400 điểm đặc trưng không gian ở tầng P5, bảo toàn kích thước và số kênh nhưng tái phân bổ trọng số chú ý.",
        algebraicMath: [
          `Q = X W_q, \\quad K = X W_k, \\quad V = X W_v \\quad \\left(W_q, W_k, W_v \\in \\mathbb{R}^{d \\times d}, \\, d = ${cMid}\\right)`,
          `A = \\text{Softmax}\\left(\\frac{Q K^T}{\\sqrt{d}}\\right) \\in \\mathbb{R}^{400 \\times 400} \\quad (\\text{Bản đồ tương quan không gian})`,
          `Z = A \\cdot V + X \\quad (\\text{Polarized Residual Connection})`,
          `X_{out} = \\text{Conv}_{1\\times 1}\\left(\\text{Concat}([X_{skip}, Z])\\right) \\in \\mathbb{R}^{${c} \\times 20 \\times 20}`
        ],
        algebraicNote: "Polarized Self-Attention (PSA) tính toán trọng số quan trọng giữa các vùng không gian, tăng cường chú ý vào đối tượng.",
        channelMath: [
          `C_{in} = ${c} \\xrightarrow{\\text{Split 50\\%}} [${cMid}, \\, ${cMid}] \\xrightarrow{\\text{PSA}} ${cMid} \\xrightarrow{\\text{Concat}} C_{out} = ${c}`
        ],
        channelNote: "",
        dlIntuitions: [
          "Self-Attention tính toán trọng số tương quan giữa 400 vùng không gian, giúp mạng tập trung vào vật thể chính và triệt tiêu nhiễu nền.",
          "Chỉ áp dụng Attention tại tầng sâu nhất P5 (20×20) và trên 50% số kênh để giữ tốc độ suy luận thời gian thực (Real-time)."
        ]
      };
    }

    // 6. Upsample
    if (lType === "Upsample") {
      let inH = Math.round(h / 2);
      let inW = Math.round(w / 2);
      if (layerData.input_shape && layerData.input_shape.length === 4) {
        inH = layerData.input_shape[2];
        inW = layerData.input_shape[3];
      }
      return {
        opBadge: "Upsample 2×",
        opShort: `Nearest 2× (${inH}×${inW} ➔ ${h}×${w})`,
        inputShapeDisplay: `1 × ${c} × ${inH} × ${inW}`,
        outputShapeDisplay: `1 × ${c} × ${h} × ${w}`,
        spatialMath: [
          `H_{out} = H_{in} \\times 2 = ${inH} \\times 2 = ${h}, \\quad W_{out} = W_{in} \\times 2 = ${inW} \\times 2 = ${w}`
        ],
        notation: ["Nội suy: Láng giềng gần nhất (Nearest-Neighbor)", "Tham số học được (Params): 0"],
        spatialNote: `Nội suy nhân đôi kích thước không gian (${inH}×${inW} ➔ ${h}×${w}) và bảo toàn số kênh ${c} để sẵn sàng ghép kênh với nhánh FPN.`,
        algebraicMath: [
          `X_{out}(b, c, y, x) = X_{in}\\left(b, c, \\left\\lfloor \\frac{y}{2} \\right\\rfloor, \\left\\lfloor \\frac{x}{2} \\right\\rfloor\\right)`,
          `\\text{Sao chép trực tiếp giá trị pixel liền kề, không cần ma trận trọng số}`
        ],
        algebraicNote: "Nội suy láng giềng gần nhất: sao chép giá trị điểm ảnh lân cận, không yêu cầu trọng số học được.",
        channelMath: [
          `C_{out} = C_{in} = ${c} \\quad (\\text{Bảo toàn 100\\% số lượng kênh})`
        ],
        channelNote: "",
        dlIntuitions: [
          "Phóng to bản đồ đặc trưng để đưa thông tin ngữ nghĩa mức cao (từ tầng sâu) xuống hòa trộn với độ phân giải không gian của tầng nông (tuyến FPN).",
          "Phương pháp Nearest-Neighbor không sử dụng trọng số học được (0 tham số), giúp giảm tải tính toán cho phần cứng."
        ]
      };
    }

    // 7. Concat
    if (lType === "Concat") {
      let c1 = Math.round(c * 0.5);
      let c2 = c - c1;
      if (layerData.input_shape && Array.isArray(layerData.input_shape[0])) {
        c1 = layerData.input_shape[0][1] || c1;
        c2 = layerData.input_shape[1][1] || c2;
      }
      return {
        opBadge: "Concat",
        opShort: "torch.cat(dim=1)",
        inputShapeDisplay: `2 Tensor: [${c1}ch, ${h}×${w}] & [${c2}ch, ${h}×${w}]`,
        outputShapeDisplay: `1 × ${c} × ${h} × ${w}`,
        spatialMath: [
          `\\text{Điều kiện không gian: } H_1 = H_2 = ${h}, \\quad W_1 = W_2 = ${w}`
        ],
        notation: ["dim = 1 (Ghép theo trục kênh / Channels)", "C_{out} = C_1 + C_2 = " + c],
        spatialNote: "Hai luồng tensor có cùng kích thước không gian, được ghép nối theo chiều sâu kênh để kết hợp thông tin hình học và ngữ nghĩa.",
        algebraicMath: [
          `X_{out} = \\text{torch.cat}([X_1, X_2], \\, \\text{dim}=1)`,
          `X_{out}[:, :C_1, :, :] = X_1, \\quad X_{out}[:, C_1:, :, :] = X_2`
        ],
        algebraicNote: "Xếp chồng tensor dọc theo trục số kênh để kết hợp thông tin hình học và ngữ nghĩa.",
        channelMath: [
          `C_{out} = C_1 + C_2 = ${c1} + ${c2} = ${c} \\text{ kênh tổng hợp}`
        ],
        channelNote: "",
        dlIntuitions: [
          "Khác với phép cộng (Add trong ResNet), phép ghép (Concat) giữ nguyên bản chất của cả 2 nguồn đặc trưng thay vì gộp lẫn vào nhau.",
          "Hợp nhất đồng thời chi tiết hình học cục bộ (từ Backbone) và ngữ nghĩa toàn cục (từ FPN/PAN) trước khi đưa vào khối phân tích."
        ]
      };
    }

    // 8. Detect Head
    if (lType === "Detect") {
      return {
        opBadge: isV26 ? "NMS-Free End-to-End" : "Decoupled Head",
        opShort: isV26 ? "One-to-One Direct Reg" : "DFL + NMS (reg_max=16)",
        inputShapeDisplay: "3 Thang đo: P3 (80×80), P4 (40×40), P5 (20×20)",
        outputShapeDisplay: isV26 ? "Tensor: 8,400 hộp × (4 tọa độ + 80 lớp)" : "Tensor: 8,400 hộp × (64 DFL + 80 lớp)",
        spatialMath: [
          `P_3 (\\text{Stride } 8): 80 \\times 80 = 6{,}400 \\text{ điểm neo (Vật nhỏ)}, \\quad P_4: 40 \\times 40 = 1{,}600 \\text{ neo}, \\quad P_5: 20 \\times 20 = 400 \\text{ neo}`,
          `N_{anchors} = 6{,}400 + 1{,}600 + 400 = 8{,}400 \\text{ hộp ứng viên}`
        ],
        notation: ["P3: 80×80 (Vật nhỏ)", "P4: 40×40 (Vật vừa)", "P5: 20×20 (Vật lớn)", "Tổng: 8,400 điểm neo"],
        spatialNote: "8,400 điểm neo đa tỷ lệ phủ kín toàn bộ bức ảnh 640×640, đưa ra dự đoán độc lập về lớp và tọa độ hộp trên 3 thang đo.",
        algebraicMath: isV26 ? [
          `\\hat{P}_{cls} = \\text{Sigmoid}(\\text{Conv}_{cls}(P_i)) \\in [0, 1]^{80 \\times H_i \\times W_i}`,
          `\\hat{B}_{reg} = \\text{Conv}_{reg}(P_i) \\in \\mathbb{R}^{4 \\times H_i \\times W_i} \\quad (\\text{reg\\_max}=1, \\, \\text{Hồi quy trực tiếp})`,
          `\\text{One-to-One Hungarian Matching} \\longrightarrow \\text{Không sử dụng NMS}`,
          `\\text{Detections} = \\text{TopK}(\\hat{B}_{reg}, \\, \\hat{P}_{cls})`
        ] : [
          `\\hat{P}_{cls} = \\text{Sigmoid}(\\text{Conv}_{cls}(P_i)) \\in [0, 1]^{80 \\times H_i \\times W_i}`,
          `\\hat{S} = \\text{Softmax}(\\text{Conv}_{reg}(P_i)) \\in \\mathbb{R}^{4 \\times 16 \\times H_i \\times W_i}`,
          `\\text{dist}_j = \\sum_{k=0}^{15} k \\cdot \\hat{S}_{j, k} \\quad (\\text{Distribution Focal Loss - DFL})`,
          `\\text{Detections} = \\text{NMS}(\\hat{B}_{reg}, \\, \\hat{P}_{cls}, \\, \\text{IoU}=0.7)`
        ],
        algebraicNote: isV26 ?
          "YOLO26 thay thế Distribution Focal Loss (DFL) bằng hồi quy trực tiếp reg_max=1 kết hợp One-to-One matching, loại bỏ bước lọc Non-Maximum Suppression (NMS) ở giai đoạn hậu xử lý." :
          "Decoupled Head tách riêng hai nhánh: Phân loại lớp (BCE loss) và Hồi quy tọa độ (DFL + CIoU loss), sau đó dùng thuật toán NMS để lọc bỏ các hộp trùng lặp.",
        channelMath: [
          `C_{cls} = 80 \\text{ kênh (Xác suất 80 lớp COCO)}, \\quad ` + (isV26 ? `C_{reg} = 4 \\text{ kênh (Tọa độ trực tiếp)}` : `C_{reg} = 64 \\text{ kênh (4 tọa độ } \\times 16 \\text{ bins DFL)}`)
        ],
        channelNote: "",
        dlIntuitions: isV26 ? [
          "YOLO26 sử dụng cơ chế One-to-One Matching trong huấn luyện để hồi quy trực tiếp tọa độ (reg_max=1), loại bỏ bước lọc trùng NMS và giúp ổn định độ trễ suy luận.",
          "Mô hình xử lý theo kiến trúc End-to-End: đầu ra sau mô hình là danh sách bounding box hoàn chỉnh mà không cần bước lọc NMS độc lập."
        ] : [
          "Decoupled Head tách riêng 2 nhánh: Phân loại lớp (BCE Loss) và Hồi quy tọa độ (DFL + CIoU Loss) để tránh hiện tượng xung đột đặc trưng giữa 2 nhiệm vụ.",
          "Distribution Focal Loss (DFL) hồi quy tọa độ dưới dạng phân phối xác suất trên 16 bin, hỗ trợ xác định ranh giới đối tượng chính xác hơn."
        ]
      };
    }

    // Fallback for any other layer
    return {
      opBadge: lType || "Layer",
      opShort: `Biến đổi tensor`,
      inputShapeDisplay: `Tensor đầu vào`,
      outputShapeDisplay: `1 × ${c} × ${h} × ${w}`,
      spatialMath: [`H_{out} = ${h}, \\quad W_{out} = ${w}`],
      notation: ["Duy trì kích thước không gian"],
      spatialNote: "Duy trì kích thước không gian qua lớp biến đổi.",
      algebraicMath: [`X_{out} = \\mathcal{F}(X_{in}, W)`],
      algebraicNote: "Phép biến đổi đặc trưng của tầng mạng.",
      channelMath: [`C_{out} = ${c} \\text{ kênh}`],
      channelNote: "",
      dlIntuitions: ["Biến đổi đặc trưng và trích xuất ngữ nghĩa cho tầng mạng."]
    };
  }

  renderTheory(layerData, modelName) {
    const lType = layerData.type || "";
    const shapeStr = (layerData.shape || []).join(" × ");
    const transform = this.computeTransformationDetails(layerData, modelName);

    // 1. Sơ đồ cấu tạo khối (Đưa lên trên cùng để người học quan sát trực quan luồng lớp con)
    let schemaHtml = "";
    if (lType === "Conv") {
      schemaHtml = `
        <div class="theory-card micro-schema-card">
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <h4 style="margin:0; font-size:0.9rem; font-weight:700; color:var(--text-primary); display:flex; align-items:center; gap:6px;">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7"></rect><rect x="14" y="3" width="7" height="7"></rect><rect x="14" y="14" width="7" height="7"></rect><rect x="3" y="14" width="7" height="7"></rect></svg>
              Sơ đồ cấu tạo khối: ConvModule
            </h4>
            <span class="micro-badge badge-c3k2">Conv + BN + SiLU</span>
          </div>
          <p style="font-size:0.79rem; color:var(--text-secondary); margin:6px 0 10px 0;">Chuỗi lớp tích chập tiêu chuẩn gồm Conv2d, Batch Normalization và hàm kích hoạt SiLU.</p>
          <div class="micro-schema-flow">
            <div class="flow-pill">Tensor đầu vào</div> ➔
            <div class="flow-pill primary">Conv 3×3 (s=2)</div> ➔
            <div class="flow-pill success">BatchNorm2d</div> ➔
            <div class="flow-pill warning">Kích hoạt SiLU</div> ➔
            <div class="flow-pill">Tensor đầu ra</div>
          </div>
        </div>
      `;
    } else if (lType === "C2f") {
      schemaHtml = `
        <div class="theory-card micro-schema-card">
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <h4 style="margin:0; font-size:0.9rem; font-weight:700; color:var(--text-primary); display:flex; align-items:center; gap:6px;">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7"></rect><rect x="14" y="3" width="7" height="7"></rect><rect x="14" y="14" width="7" height="7"></rect><rect x="3" y="14" width="7" height="7"></rect></svg>
              Sơ đồ cấu tạo khối: Khối C2f (YOLOv8)
            </h4>
            <span class="micro-badge badge-c2f">YOLOv8</span>
          </div>
          <p style="font-size:0.79rem; color:var(--text-secondary); margin:6px 0 10px 0;">Tách luồng đặc trưng sau Conv 1×1 thành hai nhánh: một nhánh đi tắt và một nhánh qua chuỗi DarknetBottleneck.</p>
          <div class="micro-schema-flow">
            <div class="flow-pill">Đầu vào</div> ➔
            <div class="flow-pill primary">Conv 1×1</div> ➔
            <div class="flow-pill warning">Chia nhánh (0.5×C)</div> ➔
            <div class="flow-box-sub">
              <div class="flow-pill info">Bottleneck 1 (3×3 + 3×3)</div>
              <div class="flow-pill info">Bottleneck 2 (3×3 + 3×3)</div>
              <span style="font-size:0.7rem; color:#64748b;">... nx Bottlenecks</span>
            </div> ➔
            <div class="flow-pill warning">Ghép kênh (Concat)</div> ➔
            <div class="flow-pill primary">Conv 1×1</div> ➔
            <div class="flow-pill">Đầu ra</div>
          </div>
        </div>
      `;
    } else if (lType === "C3k2") {
      schemaHtml = `
        <div class="theory-card micro-schema-card">
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <h4 style="margin:0; font-size:0.9rem; font-weight:700; color:var(--text-primary); display:flex; align-items:center; gap:6px;">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7"></rect><rect x="14" y="3" width="7" height="7"></rect><rect x="14" y="14" width="7" height="7"></rect><rect x="3" y="14" width="7" height="7"></rect></svg>
              Sơ đồ cấu tạo khối: Khối C3k2 (YOLO11 & YOLO26)
            </h4>
            <span class="micro-badge badge-c3k2">C3k2</span>
          </div>
          <p style="font-size:0.79rem; color:var(--text-secondary); margin:6px 0 10px 0;">Thay thế các khối bottleneck bằng module con C3k với cấu trúc tích chập nhỏ hơn nhằm giảm chi phí tính toán FLOPs và số lượng tham số.</p>
          <div class="micro-schema-flow">
            <div class="flow-pill">Đầu vào</div> ➔
            <div class="flow-pill primary">Conv 1×1</div> ➔
            <div class="flow-pill warning">Chia nhánh</div> ➔
            <div class="flow-box-sub">
              <div class="flow-pill success">Module con C3k</div>
              <div class="flow-pill success">Module con C3k</div>
            </div> ➔
            <div class="flow-pill warning">Ghép kênh (Concat)</div> ➔
            <div class="flow-pill primary">Conv 1×1</div> ➔
            <div class="flow-pill">Đầu ra</div>
          </div>
        </div>
      `;
    } else if (lType === "C2PSA") {
      schemaHtml = `
        <div class="theory-card micro-schema-card" style="border-color:#ddd6fe; background:#faf5ff;">
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <h4 style="margin:0; font-size:0.9rem; font-weight:700; color:#6b21a8; display:flex; align-items:center; gap:6px;">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7"></rect><rect x="14" y="3" width="7" height="7"></rect><rect x="14" y="14" width="7" height="7"></rect><rect x="3" y="14" width="7" height="7"></rect></svg>
              Sơ đồ cấu tạo khối: C2PSA (Polarized Self-Attention)
            </h4>
            <span class="micro-badge badge-c2psa">C2PSA</span>
          </div>
          <p style="font-size:0.79rem; color:var(--text-secondary); margin:6px 0 10px 0;">Tích hợp cơ chế Polarized Self-Attention ở tầng đặc trưng P5 nhằm tính toán trọng số tương quan không gian giữa các vùng đặc trưng.</p>
          <div class="micro-schema-flow">
            <div class="flow-pill">Đặc trưng P5 (20×20)</div> ➔
            <div class="flow-pill primary">Conv 1×1</div> ➔
            <div class="flow-pill warning">Chia nhánh</div> ➔
            <div class="flow-box-sub purple">
              <div class="flow-pill purple">PSA Query · Key · Value</div>
              <div class="flow-pill purple">Chuẩn hóa Softmax không gian</div>
            </div> ➔
            <div class="flow-pill warning">Ghép kênh (Concat)</div> ➔
            <div class="flow-pill">Đầu ra sang Neck</div>
          </div>
        </div>
      `;
    } else if (lType === "SPPF") {
      schemaHtml = `
        <div class="theory-card micro-schema-card">
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <h4 style="margin:0; font-size:0.9rem; font-weight:700; color:var(--text-primary); display:flex; align-items:center; gap:6px;">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7"></rect><rect x="14" y="3" width="7" height="7"></rect><rect x="14" y="14" width="7" height="7"></rect><rect x="3" y="14" width="7" height="7"></rect></svg>
              Sơ đồ cấu tạo khối: SPPF (Spatial Pyramid Pooling - Fast)
            </h4>
            <span class="micro-badge badge-sppf">SPPF</span>
          </div>
          <p style="font-size:0.79rem; color:var(--text-secondary); margin:6px 0 10px 0;">Mở rộng trường tiếp nhận bằng cách nối tiếp ba lớp MaxPool2d 5×5 và tái sử dụng đầu ra trung gian trước khi ghép kênh.</p>
          <div class="micro-schema-flow">
            <div class="flow-pill">Đặc trưng P5</div> ➔
            <div class="flow-pill primary">Conv 1×1</div> ➔
            <div class="flow-box-sub" style="flex-direction:row; align-items:center;">
              <div class="flow-pill info">MaxPool 5×5</div> ➔
              <div class="flow-pill info">MaxPool 5×5</div> ➔
              <div class="flow-pill info">MaxPool 5×5</div>
            </div> ➔
            <div class="flow-pill warning">Ghép 4 nhánh (Concat)</div> ➔
            <div class="flow-pill primary">Conv 1×1</div> ➔
            <div class="flow-pill">Đầu ra</div>
          </div>
        </div>
      `;
    } else if (lType === "Upsample") {
      schemaHtml = `
        <div class="theory-card micro-schema-card">
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <h4 style="margin:0; font-size:0.9rem; font-weight:700; color:var(--text-primary); display:flex; align-items:center; gap:6px;">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7"></rect><rect x="14" y="3" width="7" height="7"></rect><rect x="14" y="14" width="7" height="7"></rect><rect x="3" y="14" width="7" height="7"></rect></svg>
              Sơ đồ cấu tạo khối: FPN Upsample 2×
            </h4>
            <span class="micro-badge badge-c3k2">Nearest-Neighbor</span>
          </div>
          <p style="font-size:0.79rem; color:var(--text-secondary); margin:6px 0 10px 0;">Phóng to kích thước không gian để truyền thông tin ngữ nghĩa mức cao xuống tầng đặc trưng có độ phân giải mịn hơn.</p>
          <div class="micro-schema-flow">
            <div class="flow-pill">Tensor tầng trên</div> ➔
            <div class="flow-pill primary">Upsample 2×</div> ➔
            <div class="flow-pill success">Nhân đôi H & W</div> ➔
            <div class="flow-pill">Sang ghép kênh Concat</div>
          </div>
        </div>
      `;
    } else if (lType === "Concat") {
      schemaHtml = `
        <div class="theory-card micro-schema-card">
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <h4 style="margin:0; font-size:0.9rem; font-weight:700; color:var(--text-primary); display:flex; align-items:center; gap:6px;">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7"></rect><rect x="14" y="3" width="7" height="7"></rect><rect x="14" y="14" width="7" height="7"></rect><rect x="3" y="14" width="7" height="7"></rect></svg>
              Sơ đồ cấu tạo khối: Ghép kênh đặc trưng (Channel Concat)
            </h4>
            <span class="micro-badge badge-c2f">torch.cat</span>
          </div>
          <p style="font-size:0.79rem; color:var(--text-secondary); margin:6px 0 10px 0;">Xếp chồng hai bản đồ đặc trưng cùng độ phân giải theo chiều sâu kênh nhằm kết hợp chi tiết hình học cục bộ và thông tin ngữ nghĩa toàn cục.</p>
          <div class="micro-schema-flow">
            <div class="flow-pill">Nhánh FPN / PAN</div> +
            <div class="flow-pill">Nhánh Skip Backbone</div> ➔
            <div class="flow-pill warning">torch.cat(dim=1)</div> ➔
            <div class="flow-pill success">Hợp nhất kênh</div> ➔
            <div class="flow-pill primary">Vào khối CSP</div>
          </div>
        </div>
      `;
    } else if (lType === "Detect") {
      if (modelName === "yolo26n") {
        schemaHtml = `
          <div class="theory-card micro-schema-card" style="border-color:#fecaca; background:#fff5f5;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
              <h4 style="margin:0; font-size:0.9rem; font-weight:700; color:#b91c1c; display:flex; align-items:center; gap:6px;">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7"></rect><rect x="14" y="3" width="7" height="7"></rect><rect x="14" y="14" width="7" height="7"></rect><rect x="3" y="14" width="7" height="7"></rect></svg>
                Sơ đồ cấu tạo khối: Đầu phát hiện NMS-Free (YOLO26)
              </h4>
              <span class="micro-badge badge-head">NMS-Free</span>
            </div>
            <p style="font-size:0.79rem; color:var(--text-secondary); margin:6px 0 10px 0;">Sử dụng cơ chế gán nhãn One-to-One Matching để hồi quy trực tiếp tọa độ (reg_max=1), không yêu cầu bước lọc Non-Maximum Suppression (NMS) ở giai đoạn hậu xử lý.</p>
            <div class="micro-schema-flow">
              <div class="flow-pill">Đặc trưng P3, P4, P5</div> ➔
              <div class="flow-pill info">Nhánh One-to-One Matching</div> ➔
              <div class="flow-pill success">Hồi quy trực tiếp (reg_max=1)</div> ➔
              <div class="flow-pill warning">Không cần NMS</div> ➔
              <div class="flow-pill" style="background:#fee2e2; border-color:#fca5a5; font-weight:700;">Kết quả phát hiện</div>
            </div>
          </div>
        `;
      } else {
        schemaHtml = `
          <div class="theory-card micro-schema-card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
              <h4 style="margin:0; font-size:0.9rem; font-weight:700; color:var(--text-primary); display:flex; align-items:center; gap:6px;">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7"></rect><rect x="14" y="3" width="7" height="7"></rect><rect x="14" y="14" width="7" height="7"></rect><rect x="3" y="14" width="7" height="7"></rect></svg>
                Sơ đồ cấu tạo khối: Đầu phát hiện Decoupled Head (v8 & 11)
              </h4>
              <span class="micro-badge badge-c2f">DFL + NMS</span>
            </div>
            <p style="font-size:0.79rem; color:var(--text-secondary); margin:6px 0 10px 0;">Tách biệt hai nhánh: phân loại đối tượng (Classification) và hồi quy vị trí với hàm mất mát Distribution Focal Loss (DFL, reg_max=16), sau đó lọc bỏ các hộp trùng lặp bằng thuật toán Non-Maximum Suppression (NMS).</p>
            <div class="micro-schema-flow">
              <div class="flow-pill">Đặc trưng P3, P4, P5</div> ➔
              <div class="flow-box-sub">
                <div class="flow-pill info">Nhánh phân loại (BCE Loss)</div>
                <div class="flow-pill info">Nhánh tọa độ (DFL reg_max=16)</div>
              </div> ➔
              <div class="flow-pill warning">Lọc trùng lặp NMS (IoU=0.7)</div> ➔
              <div class="flow-pill">Kết quả phát hiện</div>
            </div>
          </div>
        `;
      }
    }

    let theoryHtml = `
      ${schemaHtml}

      <div class="theory-card role-card">
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <h4 style="margin:0; font-size:0.92rem; color:var(--text-primary); font-weight:700;">Chức năng kiến trúc</h4>
          <span class="micro-badge badge-c3k2">${layerData.type || 'Layer'}</span>
        </div>
        <p style="margin-top:6px; font-size:0.83rem; line-height:1.5;">${layerData.role || "Trích xuất và biến đổi biểu diễn đặc trưng."}</p>
        <div style="margin-top:10px; display:flex; gap:12px; font-family:var(--font-mono); font-size:0.75rem; color:var(--text-muted); flex-wrap:wrap;">
          <span>Trường tiếp nhận (Receptive Field): <strong>${layerData.receptive_field || "N/A"}</strong></span>
          <span>Kích thước đầu ra: <strong>${transform.outputShapeDisplay}</strong></span>
        </div>

        ${transform.dlIntuitions && transform.dlIntuitions.length > 0 ? `
          <div class="dl-takeaway-box">
            <div class="dl-takeaway-title">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>
              Trực giác học sâu (Deep Learning Intuition)
            </div>
            <ul class="dl-takeaway-list">
              ${transform.dlIntuitions.map(item => `<li>${item}</li>`).join("")}
            </ul>
          </div>
        ` : ""}
      </div>

      <div class="theory-card formula-card">
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <h4 style="margin:0; font-size:0.92rem; color:var(--text-primary); font-weight:700;">
            Biến đổi Tensor & Công thức toán học
          </h4>
          <span class="micro-badge" style="background:#eff6ff; color:#1d4ed8; border:1px solid #bfdbfe;">
            ${transform.opBadge}
          </span>
        </div>

        <!-- Banner theo dõi biến đổi đầu vào -> đầu ra -->
        <div class="tensor-transform-banner">
          <div class="tensor-box input">
            <span class="tensor-title">Tensor Đầu Vào (X_in)</span>
            <span class="tensor-shape">${transform.inputShapeDisplay}</span>
          </div>
          <div class="tensor-transform-arrow">
            <span style="font-size:1.1rem; line-height:1;">➔</span>
            <span>${transform.opShort}</span>
          </div>
          <div class="tensor-box output">
            <span class="tensor-title">Tensor Đầu Ra (X_out)</span>
            <span class="tensor-shape">${transform.outputShapeDisplay}</span>
          </div>
        </div>

        <!-- Bước 1: Biến đổi kích thước Tensor (Không gian & Số kênh) -->
        <div class="formula-step">
          <div class="formula-step-title">
            <span class="step-badge">Bước 1</span>
            <span class="step-name">Biến đổi kích thước Tensor <span style="white-space:nowrap;">(C × H × W)</span></span>
          </div>
          <div class="formula-math-block">${this.formatMathBlock([...transform.spatialMath, ...(transform.channelMath || [])])}</div>
          ${transform.notation && transform.notation.length > 0 ? `
            <div class="formula-notation-bar">
              <span style="font-weight:700; color:#1e293b;">Ký hiệu:</span>
              ${transform.notation.map(not => `<span class="notation-tag">${not}</span>`).join(" · ")}
            </div>
          ` : ""}
          <div class="formula-note">${transform.spatialNote}${transform.channelNote ? ` ${transform.channelNote}` : ""}</div>
        </div>

        <!-- Bước 2: Công thức đại số Forward Pass -->
        <div class="formula-step">
          <div class="formula-step-title">
            <span class="step-badge">Bước 2</span>
            <span class="step-name">Hàm biến đổi đại số <span style="white-space:nowrap;">(Forward Pass)</span></span>
          </div>
          <div class="formula-math-block">${this.formatMathBlock(transform.algebraicMath)}</div>
          <div class="formula-note">${transform.algebraicNote}</div>
        </div>
      </div>
    `;

    this.theoryContainer.innerHTML = theoryHtml;
  }

  renderDetections(inspectData) {
    if (!inspectData) return;
    this.currentInspectData = inspectData;

    const dets = inspectData.detections || [];
    const candidates = inspectData.candidates || [];
    const nmsStats = inspectData.nms_stats || {
      is_nms_free: inspectData.model_name === "yolo26n",
      candidate_count: candidates.length,
      final_count: dets.length,
      suppressed_count: Math.max(0, candidates.length - dets.length),
      suppression_rate: candidates.length > 0 ? `${Math.round((1 - dets.length / candidates.length) * 100)}%` : "0%"
    };

    if (this.countPostNms) {
      this.countPostNms.textContent = dets.length;
    }
    if (this.countPreNms) {
      this.countPreNms.textContent = nmsStats.is_nms_free ? dets.length : candidates.length.toLocaleString('vi-VN');
    }

    const isNmsFree = Boolean(nmsStats.is_nms_free || (inspectData.model_name && inspectData.model_name.startsWith("yolo26")));

    // For models without NMS (YOLO26), hide toggle and default to direct detections
    if (this.detSegmentedToggle) {
      this.detSegmentedToggle.style.display = isNmsFree ? "none" : "inline-flex";
    }
    if (isNmsFree) {
      this.detMode = "post";
    }

    const isPre = !isNmsFree && this.detMode === "pre";

    // Update Image & Caption
    if (this.detImg) {
      if (isPre) {
        this.detImg.src = inspectData.candidates_image || inspectData.detection_image || inspectData.original_image || "";
      } else {
        this.detImg.src = inspectData.detection_image || inspectData.original_image || "";
      }
    }

    if (this.detCaption) {
      if (isNmsFree) {
        this.detCaption.textContent = "Đang xem: Bounding box dự đoán trực tiếp (End-to-End, không dùng NMS)";
      } else if (isPre) {
        this.detCaption.textContent = `Đang xem: Các hộp ứng viên trùng lặp trước NMS (hiển thị ${Math.min(100, candidates.length)} hộp có điểm số cao nhất từ các thang đo P3, P4, P5)`;
      } else {
        this.detCaption.textContent = `Đang xem: ${dets.length} bounding box giữ lại sau bước lọc trùng lặp NMS`;
      }
    }

    // Update List header & count (Bỏ 'sau NMS' khi là kiến trúc NMS-Free như YOLO26)
    if (this.detCount) {
      if (isNmsFree) {
        this.detCount.textContent = `Hộp phát hiện (${dets.length})`;
      } else {
        this.detCount.textContent = isPre
          ? `Hộp ứng viên trước NMS (${candidates.length.toLocaleString('vi-VN')})`
          : `Hộp phát hiện sau NMS (${dets.length})`;
      }
    }

    const filterHint = document.getElementById("det-filter-hint");
    if (filterHint) {
      filterHint.textContent = `${inspectData.inference_time_ms || 0} ms (${inspectData.device || 'GPU'})`;
    }

    if (!this.detList) return;
    this.detList.innerHTML = "";

    if (isPre) {
      // Render Pre-NMS candidates
      if (candidates.length === 0) {
        this.detList.innerHTML = "<div style='color:#94a3b8; font-size:0.8rem; padding:8px;'>Không có hộp ứng viên nào.</div>";
        return;
      }

      // Identify survivor candidates (final detections)
      const survivorIndices = new Set(dets.map(d => d.anchor_idx).filter(idx => idx !== undefined && idx >= 0));

      const renderLimit = 100;
      const displayCandidates = candidates.slice(0, renderLimit);

      displayCandidates.forEach((c, idx) => {
        const item = document.createElement("div");
        const isSurvivor = survivorIndices.has(c.anchor_idx);
        item.className = `det-item ${isSurvivor ? "is-survivor" : "is-suppressed"}`;

        const left = document.createElement("div");
        left.className = "det-label";
        const boxCoords = c.box ? `[${c.box.map(Math.round).join(", ")}]` : "";
        const scaleBadge = c.scale ? `<span class="badge-scale badge-${c.scale.toLowerCase()}">${c.scale}</span>` : "";
        const statusBadge = isSurvivor
          ? `<span class="badge-status survivor">Được giữ lại</span>`
          : (nmsStats.is_nms_free ? "" : `<span class="badge-status suppressed">Loại bởi NMS (IoU)</span>`);

        left.innerHTML = `
          <span class="det-name">${idx + 1}. ${c.class_name}</span>
          ${scaleBadge}
          ${statusBadge}
          ${boxCoords ? `<span class="det-box-coord" style="font-size:0.7rem; color:var(--text-muted); font-family:var(--font-mono); margin-left:6px;">${boxCoords}</span>` : ""}
        `;
        item.appendChild(left);

        const right = document.createElement("div");
        right.className = "det-conf";
        right.textContent = `${Math.round(c.confidence * 100)}%`;
        item.appendChild(right);

        this.detList.appendChild(item);
      });

      if (candidates.length > renderLimit) {
        const moreInfo = document.createElement("div");
        moreInfo.style.cssText = "padding: 10px; font-size: 0.78rem; color: var(--text-muted); text-align: center; border-top: 1px dashed #e2e8f0; margin-top: 6px;";
        moreInfo.textContent = `Đang hiển thị ${renderLimit} hộp có độ tin cậy cao nhất trong tổng số ${candidates.length.toLocaleString('vi-VN')} hộp ứng viên trước NMS.`;
        this.detList.appendChild(moreInfo);
      }
    } else {
      // Render Post-NMS clean detections
      if (dets.length === 0) {
        this.detList.innerHTML = "<div style='color:#94a3b8; font-size:0.8rem; padding:8px;'>Không có đối tượng nào vượt ngưỡng tin cậy.</div>";
        return;
      }

      dets.forEach((d, idx) => {
        const item = document.createElement("div");
        item.className = "det-item is-survivor";

        const left = document.createElement("div");
        left.className = "det-label";
        const boxCoords = d.box ? `[${d.box.map(Math.round).join(", ")}]` : "";
        const scaleBadge = d.scale ? `<span class="badge-scale badge-${d.scale.toLowerCase()}">${d.scale} · Stride ${d.stride || (d.scale === 'P3' ? 8 : d.scale === 'P4' ? 16 : 32)}</span>` : "";

        left.innerHTML = `
          <span class="det-name">${idx + 1}. ${d.class_name}</span>
          ${scaleBadge}
          ${boxCoords ? `<span class="det-box-coord" style="font-size:0.7rem; color:var(--text-muted); font-family:var(--font-mono); margin-left:6px;">${boxCoords}</span>` : ""}
        `;
        item.appendChild(left);

        const right = document.createElement("div");
        right.className = "det-conf";
        right.textContent = `${Math.round(d.confidence * 100)}%`;
        item.appendChild(right);

        this.detList.appendChild(item);
      });
    }
  }
}
