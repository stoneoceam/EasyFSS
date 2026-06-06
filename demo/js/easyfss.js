const API_URL = window.EASYFSS_API_URL || "/api/easyfss/predict";

let supportShots = [
  { imageFile: null, maskFile: null, imageUrl: "", maskUrl: "" }
];

let queryItems = [
  { imageFile: null, imageUrl: "", maskFile: null, maskUrl: "", predMaskUrl: "", overlayEnabled: false }
];

let supportCollapsed = false;

/* =========================
 * Online annotation state
 * ========================= */
let maskEditorState = {
  mode: "brush",              // brush | erase
  brushSize: 6,
  drawing: false,
  targetType: null,           // "query" | "support"
  targetIndex: null,
  imageObj: null,
  scale: 1,
  minScale: 0.5,
  maxScale: 4,
  overlayCanvas: null,        // Actual binary mask data layer
  overlayCtx: null,
  displayCanvas: null,
  displayCtx: null,
};

function makeRequestId() {
  const now = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  const time =
    now.getFullYear() +
    pad(now.getMonth() + 1) +
    pad(now.getDate()) + "-" +
    pad(now.getHours()) +
    pad(now.getMinutes()) +
    pad(now.getSeconds());
  return `easyfss-${time}`;
}

function setStatus(text, type = "") {
  const statusEl = document.getElementById("statusText");
  if (!statusEl) return;
  statusEl.textContent = text;
  statusEl.className = `status action-status ${type}`;
  statusEl.classList.toggle("hidden", !text);
}

function showInlineLoading() {
  const loading = document.getElementById("inlineLoading");
  if (loading) loading.classList.remove("hidden");
}

function hideInlineLoading() {
  const loading = document.getElementById("inlineLoading");
  if (loading) loading.classList.add("hidden");
}

function createPreviewURL(file) {
  if (!file) return "";
  return URL.createObjectURL(file);
}

function updateShotCount() {
  const el = document.getElementById("shotCount");
  if (el) el.textContent = supportShots.length;
}

function updateQueryCount() {
  const el = document.getElementById("queryCount");
  if (el) el.textContent = queryItems.length;
}

function applySupportCollapseState() {
  const cards = document.querySelectorAll("#supportShotsContainer .support-shot-card");
  const btn = document.getElementById("toggleSupportBtn");
  const hiddenCount = Math.max(cards.length - 1, 0);

  if (hiddenCount === 0) {
    supportCollapsed = false;
  }

  cards.forEach((card, index) => {
    if (!supportCollapsed) {
      card.style.display = "";
    } else {
      card.style.display = index === 0 ? "" : "none";
    }
  });

  if (btn) {
    const label = hiddenCount === 0
      ? "Nothing to Collapse"
      : supportCollapsed
        ? "Show All Shots"
        : "Hide Extra Shots";
    const badgeText = supportCollapsed
      ? `${hiddenCount} hidden`
      : `${hiddenCount} extra`;

    btn.disabled = hiddenCount === 0;
    btn.classList.toggle("is-collapsed", supportCollapsed);
    btn.classList.toggle("has-hidden", hiddenCount > 0);
    btn.setAttribute("aria-expanded", String(!supportCollapsed));
    btn.setAttribute(
      "title",
      hiddenCount === 0 ? "Only one support shot is available." : `${label}`
    );
    btn.innerHTML = `
      <span class="support-toggle-icon" aria-hidden="true"></span>
      <span class="support-toggle-label">${label}</span>
      <span class="support-toggle-badge">${badgeText}</span>
    `;
  }
}

function getSelectedRadioValue(name, fallback = "base") {
  const checked = document.querySelector(`input[name="${name}"]:checked`);
  return checked ? checked.value : fallback;
}

function setSelectedRadioValue(name, value) {
  const target = document.querySelector(`input[name="${name}"][value="${value}"]`);
  if (target) target.checked = true;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function withCacheBust(url) {
  if (!url || url.startsWith("data:")) return url || "";
  return `${url}?t=${Date.now()}`;
}

/* =========================
 * Online annotation helpers
 * ========================= */
function getEditorCanvas() {
  return document.getElementById("editorCanvas");
}

function getPreviewCanvas() {
  return document.getElementById("maskPreviewCanvas");
}

function getEditorCtx() {
  const canvas = getEditorCanvas();
  return canvas ? canvas.getContext("2d") : null;
}

function getPreviewCtx() {
  const canvas = getPreviewCanvas();
  return canvas ? canvas.getContext("2d") : null;
}

function ensureOverlayCanvas(width, height) {
  const overlayCanvas = document.createElement("canvas");
  overlayCanvas.width = width;
  overlayCanvas.height = height;
  const overlayCtx = overlayCanvas.getContext("2d");

  overlayCtx.clearRect(0, 0, width, height);
  overlayCtx.fillStyle = "#000000";
  overlayCtx.fillRect(0, 0, width, height);

  maskEditorState.overlayCanvas = overlayCanvas;
  maskEditorState.overlayCtx = overlayCtx;
}

function ensureDisplayCanvas(width, height) {
  const displayCanvas = document.createElement("canvas");
  displayCanvas.width = width;
  displayCanvas.height = height;
  const displayCtx = displayCanvas.getContext("2d");

  maskEditorState.displayCanvas = displayCanvas;
  maskEditorState.displayCtx = displayCtx;
}

function renderEditorDisplay() {
  const canvas = getEditorCanvas();
  const ctx = getEditorCtx();
  const img = maskEditorState.imageObj;
  const overlayCanvas = maskEditorState.overlayCanvas;
  const scale = maskEditorState.scale;

  if (!canvas || !ctx || !img || !overlayCanvas) return;

  const w = img.width;
  const h = img.height;

  canvas.width = w;
  canvas.height = h;

  ctx.clearRect(0, 0, w, h);
  ctx.globalCompositeOperation = "source-over";
  ctx.drawImage(img, 0, 0, w, h);

  ctx.save();
  ctx.globalAlpha = 0.5;
  ctx.drawImage(overlayCanvas, 0, 0, w, h);
  ctx.restore();

  canvas.style.width = `${w * scale}px`;
  canvas.style.height = `${h * scale}px`;
}

function renderPreviewMask() {
  const previewCanvas = getPreviewCanvas();
  const previewCtx = getPreviewCtx();
  const overlayCanvas = maskEditorState.overlayCanvas;
  const img = maskEditorState.imageObj;
  const scale = maskEditorState.scale;

  if (!previewCanvas || !previewCtx || !overlayCanvas || !img) return;

  previewCanvas.width = img.width;
  previewCanvas.height = img.height;
  previewCtx.clearRect(0, 0, img.width, img.height);

  const src = maskEditorState.overlayCtx.getImageData(0, 0, img.width, img.height);
  const dst = previewCtx.createImageData(img.width, img.height);

  for (let i = 0; i < src.data.length; i += 4) {
    const isForeground = src.data[i] > 127;
    const val = isForeground ? 255 : 0;
    dst.data[i] = val;
    dst.data[i + 1] = val;
    dst.data[i + 2] = val;
    dst.data[i + 3] = 255;
  }

  previewCtx.putImageData(dst, 0, 0);
  previewCanvas.style.width = `${img.width * scale}px`;
  previewCanvas.style.height = `${img.height * scale}px`;
}

function syncEditorViews() {
  renderEditorDisplay();
  renderPreviewMask();
}

function openMaskEditor({ imageUrl, targetType, targetIndex = null }) {
  const modal = document.getElementById("maskEditorModal");
  const brushSizeInput = document.getElementById("brushSize");

  if (!modal) {
    alert("The annotation modal was not loaded correctly.");
    return;
  }

  if (!imageUrl) {
    alert("Please upload the corresponding image first.");
    return;
  }

  maskEditorState.targetType = targetType;
  maskEditorState.targetIndex = targetIndex;
  maskEditorState.mode = "brush";
  maskEditorState.brushSize = Number(brushSizeInput?.value || 6);
  maskEditorState.drawing = false;
  maskEditorState.scale = 1;

  const img = new Image();
  img.onload = () => {
    maskEditorState.imageObj = img;
    ensureOverlayCanvas(img.width, img.height);
    ensureDisplayCanvas(img.width, img.height);
    syncEditorViews();
    modal.classList.remove("hidden");
  };
  img.src = imageUrl;
}

function closeMaskEditor() {
  const modal = document.getElementById("maskEditorModal");
  if (modal) modal.classList.add("hidden");

  maskEditorState.targetType = null;
  maskEditorState.targetIndex = null;
  maskEditorState.imageObj = null;
  maskEditorState.drawing = false;
  maskEditorState.overlayCanvas = null;
  maskEditorState.overlayCtx = null;
  maskEditorState.displayCanvas = null;
  maskEditorState.displayCtx = null;
  maskEditorState.scale = 1;
}

function getCanvasPoint(e, canvas) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  return {
    x: (e.clientX - rect.left) * scaleX,
    y: (e.clientY - rect.top) * scaleY,
  };
}

function drawCircleOnOverlay(x, y, radius, isErase = false) {
  const overlayCtx = maskEditorState.overlayCtx;
  if (!overlayCtx) return;

  overlayCtx.beginPath();
  overlayCtx.arc(x, y, radius, 0, Math.PI * 2);
  overlayCtx.closePath();
  overlayCtx.fillStyle = isErase ? "#000000" : "#ff0000";
  overlayCtx.fill();
}

function drawLineOnOverlay(fromX, fromY, toX, toY, radius, isErase = false) {
  const overlayCtx = maskEditorState.overlayCtx;
  if (!overlayCtx) return;

  const dx = toX - fromX;
  const dy = toY - fromY;
  const dist = Math.max(Math.sqrt(dx * dx + dy * dy), 1);
  const step = Math.max(radius * 0.35, 1);
  const count = Math.ceil(dist / step);

  for (let i = 0; i <= count; i++) {
    const t = i / count;
    const x = fromX + dx * t;
    const y = fromY + dy * t;
    drawCircleOnOverlay(x, y, radius, isErase);
  }
}

function updateMaskPreview() {
  syncEditorViews();
}

function clearMaskEditor() {
  const img = maskEditorState.imageObj;
  if (!img) return;

  ensureOverlayCanvas(img.width, img.height);
  syncEditorViews();
}

function previewCanvasToBlob() {
  const previewCanvas = getPreviewCanvas();
  return new Promise((resolve, reject) => {
    if (!previewCanvas) {
      reject(new Error("maskPreviewCanvas does not exist."));
      return;
    }
    previewCanvas.toBlob((blob) => {
      if (!blob) {
        reject(new Error("Failed to export the mask."));
        return;
      }
      resolve(blob);
    }, "image/png");
  });
}

async function saveMaskFromEditor() {
  try {
    const blob = await previewCanvasToBlob();
    const file = new File([blob], "mask.png", { type: "image/png" });
    const url = URL.createObjectURL(blob);

    if (maskEditorState.targetType === "query") {
      const index = maskEditorState.targetIndex;
      if (index !== null && queryItems[index]) {
        queryItems[index].maskFile = file;
        queryItems[index].maskUrl = url;
      }
      renderQuerySection();
    } else if (maskEditorState.targetType === "support") {
      const index = maskEditorState.targetIndex;
      if (index !== null && supportShots[index]) {
        supportShots[index].maskFile = file;
        supportShots[index].maskUrl = url;
        renderSupportShots();
      }
    }

    closeMaskEditor();
  } catch (err) {
    alert(`Failed to save the mask: ${String(err)}`);
  }
}

function applyZoom(delta) {
  maskEditorState.scale += delta;
  if (maskEditorState.scale < maskEditorState.minScale) {
    maskEditorState.scale = maskEditorState.minScale;
  }
  if (maskEditorState.scale > maskEditorState.maxScale) {
    maskEditorState.scale = maskEditorState.maxScale;
  }
  syncEditorViews();
}

function floodFillMaskInterior() {
  const overlayCanvas = maskEditorState.overlayCanvas;
  const overlayCtx = maskEditorState.overlayCtx;
  const img = maskEditorState.imageObj;

  if (!overlayCanvas || !overlayCtx || !img) return;

  const w = overlayCanvas.width;
  const h = overlayCanvas.height;
  const imageData = overlayCtx.getImageData(0, 0, w, h);
  const data = imageData.data;
  const visited = new Uint8Array(w * h);
  const queue = new Int32Array(w * h);
  let head = 0;
  let tail = 0;

  function isBg(idx) {
    const i = idx * 4;
    return data[i] < 127;
  }

  function markVisited(idx) {
    visited[idx] = 1;
    queue[tail++] = idx;
  }

  for (let x = 0; x < w; x++) {
    const top = x;
    const bottom = (h - 1) * w + x;
    if (!visited[top] && isBg(top)) markVisited(top);
    if (!visited[bottom] && isBg(bottom)) markVisited(bottom);
  }

  for (let y = 0; y < h; y++) {
    const left = y * w;
    const right = y * w + (w - 1);
    if (!visited[left] && isBg(left)) markVisited(left);
    if (!visited[right] && isBg(right)) markVisited(right);
  }

  while (head < tail) {
    const idx = queue[head++];
    const x = idx % w;
    const y = (idx / w) | 0;

    if (x > 0) {
      const n = idx - 1;
      if (!visited[n] && isBg(n)) markVisited(n);
    }
    if (x < w - 1) {
      const n = idx + 1;
      if (!visited[n] && isBg(n)) markVisited(n);
    }
    if (y > 0) {
      const n = idx - w;
      if (!visited[n] && isBg(n)) markVisited(n);
    }
    if (y < h - 1) {
      const n = idx + w;
      if (!visited[n] && isBg(n)) markVisited(n);
    }
  }

  for (let idx = 0; idx < w * h; idx++) {
    const i = idx * 4;
    const red = data[i];
    const isBoundaryOrFg = red > 127;
    const isOutsideBg = visited[idx] === 1;

    if (!isBoundaryOrFg && !isOutsideBg) {
      data[i] = 255;
      data[i + 1] = 0;
      data[i + 2] = 0;
      data[i + 3] = 255;
    }
  }

  overlayCtx.putImageData(imageData, 0, 0);
  syncEditorViews();
}

function bindMaskEditorEvents() {
  const canvas = getEditorCanvas();
  const closeBtn = document.getElementById("closeMaskEditor");
  const brushBtn = document.getElementById("brushBtn");
  const eraserBtn = document.getElementById("eraserBtn");
  const brushSizeInput = document.getElementById("brushSize");
  const clearBtn = document.getElementById("clearCanvas");
  const saveBtn = document.getElementById("saveMask");
  const zoomInBtn = document.getElementById("zoomInBtn");
  const zoomOutBtn = document.getElementById("zoomOutBtn");
  const fillBtn = document.getElementById("fillMaskBtn");

  if (!canvas) return;

  let lastPoint = null;

  canvas.addEventListener("mousedown", (e) => {
    maskEditorState.drawing = true;
    const { x, y } = getCanvasPoint(e, canvas);
    const isErase = maskEditorState.mode === "erase";
    drawCircleOnOverlay(x, y, maskEditorState.brushSize, isErase);
    lastPoint = { x, y };
    updateMaskPreview();
  });

  canvas.addEventListener("mousemove", (e) => {
    if (!maskEditorState.drawing) return;
    const { x, y } = getCanvasPoint(e, canvas);
    const isErase = maskEditorState.mode === "erase";

    if (lastPoint) {
      drawLineOnOverlay(lastPoint.x, lastPoint.y, x, y, maskEditorState.brushSize, isErase);
    } else {
      drawCircleOnOverlay(x, y, maskEditorState.brushSize, isErase);
    }

    lastPoint = { x, y };
    updateMaskPreview();
  });

  window.addEventListener("mouseup", () => {
    maskEditorState.drawing = false;
    lastPoint = null;
  });

  canvas.addEventListener("mouseleave", () => {
    maskEditorState.drawing = false;
    lastPoint = null;
  });

  if (closeBtn) {
    closeBtn.addEventListener("click", closeMaskEditor);
  }

  if (brushBtn) {
    brushBtn.addEventListener("click", () => {
      maskEditorState.mode = "brush";
    });
  }

  if (eraserBtn) {
    eraserBtn.addEventListener("click", () => {
      maskEditorState.mode = "erase";
    });
  }

  if (brushSizeInput) {
    brushSizeInput.addEventListener("input", (e) => {
      maskEditorState.brushSize = Number(e.target.value || 6);
    });
  }

  if (clearBtn) {
    clearBtn.addEventListener("click", clearMaskEditor);
  }

  if (saveBtn) {
    saveBtn.addEventListener("click", saveMaskFromEditor);
  }

  if (zoomInBtn) {
    zoomInBtn.addEventListener("click", () => applyZoom(0.25));
  }

  if (zoomOutBtn) {
    zoomOutBtn.addEventListener("click", () => applyZoom(-0.25));
  }

  if (fillBtn) {
    fillBtn.addEventListener("click", floodFillMaskInterior);
  }
}

/* =========================
 * Upload cards
 * ========================= */
function createUploadSlot({
  title = "",
  file = null,
  previewUrl = "",
  inputId,
  onFileChange,
  removable = false,
  onRemove = null
}) {
  const wrapper = document.createElement("div");
  wrapper.className = "upload-slot-card";
  const uploadLabel = title.toLowerCase().includes("mask") ? "Upload Mask" : "Upload Image";
  const uploadHint = title || (uploadLabel === "Upload Mask" ? "Select a mask file" : "Select an image file");

  if (!file) {
    wrapper.innerHTML = `
      <div class="upload-dropzone">
        <label class="upload-overlay-label" for="${inputId}">
          <div class="upload-placeholder-content">
            <div class="upload-icon">+</div>
            <div class="upload-main-text">${uploadLabel}</div>
            <div class="upload-sub-text">${uploadHint}</div>
          </div>
        </label>
        <input class="hidden-file-input" id="${inputId}" type="file" accept="image/*" />
      </div>
    `;
  } else {
    wrapper.innerHTML = `
      <div class="upload-preview-container">
        <input class="hidden-file-input" id="${inputId}" type="file" accept="image/*" />
        <img class="upload-preview-image" src="${previewUrl}" alt="${title}" />
        <div class="upload-preview-toolbar">
          <div class="upload-preview-name" title="${file.name}">${file.name}</div>
          <div class="upload-preview-actions">
            <button type="button" class="mini-btn reselect-btn">Select Another</button>
            ${removable ? `<button type="button" class="mini-btn danger-btn remove-btn">Remove</button>` : ""}
          </div>
        </div>
      </div>
    `;
  }

  const input = wrapper.querySelector(`#${CSS.escape(inputId)}`);
  if (input) {
    input.addEventListener("change", () => {
      const selected = input.files[0] || null;
      onFileChange(selected);
    });
  }

  const reselectBtn = wrapper.querySelector(".reselect-btn");
  if (reselectBtn && input) {
    reselectBtn.addEventListener("click", () => {
      input.click();
    });
  }

  if (removable && onRemove) {
    const removeBtn = wrapper.querySelector(".remove-btn");
    if (removeBtn) {
      removeBtn.addEventListener("click", onRemove);
    }
  }

  return wrapper;
}

function createCardTitle(title, annotateHandler = null) {
  const row = document.createElement("div");
  row.className = "upload-card-title-row";

  const h3 = document.createElement("h3");
  h3.textContent = title;

  row.appendChild(h3);

  if (annotateHandler) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "mini-btn annotate-title-btn";
    btn.textContent = "Draw Mask";
    btn.addEventListener("click", annotateHandler);
    row.appendChild(btn);
  }

  return row;
}

/* =========================
 * Query area
 * ========================= */
function renderQuerySection() {
  const queryList = document.getElementById("queryList");
  if (!queryList) return;

  queryList.innerHTML = "";

  queryItems.forEach((query, index) => {
    const row = document.createElement("div");
    row.className = "query-row";

    const imageCard = document.createElement("div");
    imageCard.className = "upload-card";
    const imageSlot = document.createElement("div");

    const imageTitle = createCardTitle(`Query ${index + 1}`);
    const deleteBtn = document.createElement("button");
    deleteBtn.type = "button";
    deleteBtn.className = "mini-btn danger-btn query-delete-btn";
    deleteBtn.textContent = "Delete";
    deleteBtn.addEventListener("click", () => {
      if (queryItems.length === 1) {
        alert("At least 1 query image must be kept.");
        return;
      }
      queryItems.splice(index, 1);
      updateQueryCount();
      renderQuerySection();
    });
    imageTitle.appendChild(deleteBtn);

    imageCard.appendChild(imageTitle);
    imageCard.appendChild(imageSlot);
    imageSlot.appendChild(createQuerySlot(query, index));
    row.appendChild(imageCard);
    queryList.appendChild(row);
  });
}

function createQuerySlot(query, index) {
  const wrapper = document.createElement("div");
  wrapper.className = "query-result-card";

  const inputId = `query-image-file-${index}`;
  if (!query.imageFile) {
    const grid = document.createElement("div");
    grid.className = "query-result-grid";

    const uploadCell = document.createElement("div");
    uploadCell.className = "query-upload-cell";
    uploadCell.appendChild(
      createUploadSlot({
        title: `Query ${index + 1} Image`,
        file: null,
        previewUrl: "",
        inputId,
        onFileChange: (file) => {
          queryItems[index].imageFile = file;
          queryItems[index].imageUrl = file ? createPreviewURL(file) : "";
          queryItems[index].maskFile = null;
          queryItems[index].maskUrl = "";
          queryItems[index].predMaskUrl = "";
          queryItems[index].overlayEnabled = false;
          renderQuerySection();
        }
      })
    );

    const maskCell = document.createElement("div");
    maskCell.className = "query-mask-preview is-empty";
    maskCell.innerHTML = `
      <div class="query-mask-placeholder">
        <span>Predicted Mask</span>
      </div>
    `;

    grid.appendChild(uploadCell);
    grid.appendChild(maskCell);
    wrapper.appendChild(grid);
    return wrapper;
  }

  const safeName = escapeHtml(query.imageFile.name);
  const overlayClass = query.overlayEnabled ? "is-visible" : "";
  const maskUrl = query.predMaskUrl ? withCacheBust(query.predMaskUrl) : "";

  wrapper.innerHTML = `
    <input class="hidden-file-input" id="${inputId}" type="file" accept="image/*" />
    <div class="query-result-grid">
      <div class="query-image-stack">
        <img src="${query.imageUrl}" alt="${safeName}" />
        ${query.predMaskUrl ? `<canvas class="query-mask-overlay ${overlayClass}" data-mask-url="${maskUrl}"></canvas>` : ""}
      </div>
      <div class="query-mask-preview ${query.predMaskUrl ? "" : "is-empty"}">
        ${query.predMaskUrl ? `<img src="${maskUrl}" alt="${safeName} predicted mask" />` : `
          <div class="query-mask-placeholder">
            <span>Predicted Mask</span>
          </div>
        `}
      </div>
    </div>
    <div class="upload-preview-toolbar query-toolbar">
      <div class="upload-preview-name" title="${safeName}">${safeName}</div>
      <div class="upload-preview-actions">
        ${query.predMaskUrl ? `
          <label class="overlay-switch">
            <input type="checkbox" ${query.overlayEnabled ? "checked" : ""} />
            <span class="overlay-switch-track"></span>
            <span class="overlay-switch-text">Mask Overlay</span>
          </label>
        ` : ""}
        <button type="button" class="mini-btn reselect-btn">Select Another</button>
        <button type="button" class="mini-btn danger-btn remove-btn">Remove</button>
      </div>
    </div>
  `;

  const input = wrapper.querySelector(`#${CSS.escape(inputId)}`);
  input.addEventListener("change", () => {
    const file = input.files[0] || null;
    queryItems[index].imageFile = file;
    queryItems[index].imageUrl = file ? createPreviewURL(file) : "";
    queryItems[index].maskFile = null;
    queryItems[index].maskUrl = "";
    queryItems[index].predMaskUrl = "";
    queryItems[index].overlayEnabled = false;
    renderQuerySection();
  });

  wrapper.querySelector(".reselect-btn").addEventListener("click", () => input.click());
  wrapper.querySelector(".remove-btn").addEventListener("click", () => {
    queryItems[index].imageFile = null;
    queryItems[index].imageUrl = "";
    queryItems[index].maskFile = null;
    queryItems[index].maskUrl = "";
    queryItems[index].predMaskUrl = "";
    queryItems[index].overlayEnabled = false;
    renderQuerySection();
  });

  const overlaySwitch = wrapper.querySelector(".overlay-switch input");
  if (overlaySwitch) {
    overlaySwitch.addEventListener("change", () => {
      queryItems[index].overlayEnabled = overlaySwitch.checked;
      const overlay = wrapper.querySelector(".query-mask-overlay");
      if (overlay) overlay.classList.toggle("is-visible", overlaySwitch.checked);
    });
  }

  renderMaskOverlayCanvas(wrapper);

  return wrapper;
}

function renderMaskOverlayCanvas(wrapper) {
  const canvas = wrapper.querySelector(".query-mask-overlay");
  if (!canvas) return;

  const maskUrl = canvas.dataset.maskUrl;
  if (!maskUrl) return;

  const img = new Image();
  img.onload = () => {
    canvas.width = img.naturalWidth || img.width;
    canvas.height = img.naturalHeight || img.height;

    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;
    for (let i = 0; i < data.length; i += 4) {
      const alpha = Math.max(data[i], data[i + 1], data[i + 2]);
      data[i] = 239;
      data[i + 1] = 68;
      data[i + 2] = 68;
      data[i + 3] = alpha > 16 ? Math.round(alpha * 0.72) : 0;
    }
    ctx.putImageData(imageData, 0, 0);
  };
  img.src = maskUrl;
}

function addQuery() {
  queryItems.push({
    imageFile: null,
    imageUrl: "",
    maskFile: null,
    maskUrl: "",
    predMaskUrl: "",
    overlayEnabled: false
  });
  updateQueryCount();
  renderQuerySection();
}

/* =========================
 * Support area
 * ========================= */
function renderSupportShots() {
  const container = document.getElementById("supportShotsContainer");
  if (!container) return;

  if (supportShots.length <= 1) {
    supportCollapsed = false;
  }

  container.innerHTML = "";

  supportShots.forEach((shot, index) => {
    const shotCard = document.createElement("div");
    shotCard.className = "support-shot-card";

    const header = document.createElement("div");
    header.className = "support-shot-header";
    header.innerHTML = `
      <h3>Shot ${index + 1}</h3>
      <button type="button" class="danger short-btn">Delete</button>
    `;

    const removeBtn = header.querySelector("button");
    removeBtn.addEventListener("click", () => {
      if (supportShots.length === 1) {
        alert("At least 1 shot must be kept.");
        return;
      }
      supportShots.splice(index, 1);
      updateShotCount();
      renderSupportShots();
    });

    const grid = document.createElement("div");
    grid.className = "upload-grid-2";

    const imageCard = document.createElement("div");
    imageCard.className = "upload-card";
    const imageSlot = document.createElement("div");

    const maskCard = document.createElement("div");
    maskCard.className = "upload-card";
    const maskSlot = document.createElement("div");

    imageCard.appendChild(createCardTitle("Support Image"));
    imageCard.appendChild(imageSlot);

    maskCard.appendChild(
      createCardTitle("Support Mask", () => {
        if (!shot.imageUrl) {
        alert("Please upload a Support Image first.");
          return;
        }
        openMaskEditor({
          imageUrl: shot.imageUrl,
          targetType: "support",
          targetIndex: index
        });
      })
    );
    maskCard.appendChild(maskSlot);

    imageSlot.appendChild(
      createUploadSlot({
        title: `Shot ${index + 1} Image`,
        file: shot.imageFile,
        previewUrl: shot.imageUrl,
        inputId: `support-image-${index}`,
        onFileChange: (file) => {
          supportShots[index].imageFile = file;
          supportShots[index].imageUrl = file ? createPreviewURL(file) : "";

          if (!file) {
            supportShots[index].maskFile = null;
            supportShots[index].maskUrl = "";
          }

          renderSupportShots();
        },
        removable: !!shot.imageFile,
        onRemove: () => {
          supportShots[index].imageFile = null;
          supportShots[index].imageUrl = "";
          supportShots[index].maskFile = null;
          supportShots[index].maskUrl = "";
          renderSupportShots();
        }
      })
    );

    maskSlot.appendChild(
      createUploadSlot({
        title: `Shot ${index + 1} Mask`,
        file: shot.maskFile,
        previewUrl: shot.maskUrl,
        inputId: `support-mask-${index}`,
        onFileChange: (file) => {
          supportShots[index].maskFile = file;
          supportShots[index].maskUrl = file ? createPreviewURL(file) : "";
          renderSupportShots();
        },
        removable: !!shot.maskFile,
        onRemove: () => {
          supportShots[index].maskFile = null;
          supportShots[index].maskUrl = "";
          renderSupportShots();
        }
      })
    );

    grid.appendChild(imageCard);
    grid.appendChild(maskCard);

    shotCard.appendChild(header);
    shotCard.appendChild(grid);
    container.appendChild(shotCard);
  });

  applySupportCollapseState();
}

function addShot() {
  supportShots.push({
    imageFile: null,
    maskFile: null,
    imageUrl: "",
    maskUrl: ""
  });
  updateShotCount();
  renderSupportShots();
}

function toggleSupportSection() {
  if (supportShots.length <= 1) {
    supportCollapsed = false;
    applySupportCollapseState();
    return;
  }

  supportCollapsed = !supportCollapsed;
  applySupportCollapseState();
}

function clearForm() {
  const requestIdInput = document.getElementById("requestId");
  const rawResponse = document.getElementById("rawResponse");
  const timeCostText = document.getElementById("timeCostText");

  if (requestIdInput) requestIdInput.value = makeRequestId();

  setSelectedRadioValue("dinoSize", "base");
  setSelectedRadioValue("sam2Size", "base");

  queryItems = [
    { imageFile: null, imageUrl: "", maskFile: null, maskUrl: "", predMaskUrl: "", overlayEnabled: false }
  ];

  supportShots = [
    { imageFile: null, maskFile: null, imageUrl: "", maskUrl: "" }
  ];

  supportCollapsed = false;

  updateShotCount();
  updateQueryCount();
  renderQuerySection();
  renderSupportShots();

  if (rawResponse) rawResponse.textContent = "";
  if (timeCostText) timeCostText.textContent = "-";
  setStatus("");
  hideInlineLoading();
  closeMaskEditor();
}

/* =========================
 * Submit inference
 * ========================= */
async function submitInference() {
  const requestIdInput = document.getElementById("requestId");
  const rawResponse = document.getElementById("rawResponse");
  const timeCostText = document.getElementById("timeCostText");

  const dinoSize = getSelectedRadioValue("dinoSize", "base");
  const sam2Size = getSelectedRadioValue("sam2Size", "base");

  const requestId = makeRequestId();
  if (requestIdInput) requestIdInput.value = requestId;

  for (let i = 0; i < queryItems.length; i++) {
    if (!queryItems[i].imageFile) {
      alert(`Please upload Query Image ${i + 1}.`);
      return;
    }
  }

  for (let i = 0; i < supportShots.length; i++) {
    if (!supportShots[i].imageFile || !supportShots[i].maskFile) {
      alert(`Please add both a Support Image and a Support Mask for Shot ${i + 1}.`);
      return;
    }
  }

  const formData = new FormData();
  formData.append("request_id", requestId);
  formData.append("dinov2_size", dinoSize);
  formData.append("sam2_size", sam2Size);

  queryItems.forEach((query, index) => {
    formData.append("query_images", query.imageFile);
  });

  supportShots.forEach((shot) => {
    formData.append("support_images", shot.imageFile);
    formData.append("support_masks", shot.maskFile);
  });

  setStatus("Running inference...", "pending");
  showInlineLoading();

  if (rawResponse) rawResponse.textContent = "";
  queryItems.forEach((query) => {
    query.predMaskUrl = "";
    query.overlayEnabled = false;
  });
  renderQuerySection();
  if (timeCostText) timeCostText.textContent = "Running...";

  try {
    const resp = await fetch(API_URL, {
      method: "POST",
      body: formData
    });

    const data = await resp.json();

    if (!resp.ok || data.status === "failed") {
      const message = data?.error?.message || data?.detail || "Request failed";
      setStatus(`Request failed: ${message}`, "error");
      if (rawResponse) rawResponse.textContent = JSON.stringify(data, null, 2);
      return;
    }

    setStatus("Inference complete", "success");

    const outputs = data.outputs || {};
    const meta = data.meta || {};
    if (timeCostText) {
      timeCostText.textContent = meta.time_cost === undefined ? "-" : `${meta.time_cost}s`;
    }
    const items = outputs.items || [];
    if (items.length > 0) {
      items.forEach((item, index) => {
        if (queryItems[index]) {
          queryItems[index].predMaskUrl = item.pred_mask_url || "";
          queryItems[index].overlayEnabled = false;
        }
      });
      renderQuerySection();
    }
  } catch (err) {
    setStatus(`Request error: ${String(err)}`, "error");
    if (rawResponse) rawResponse.textContent = String(err);
    if (timeCostText) timeCostText.textContent = "-";
  } finally {
    hideInlineLoading();
  }
}

/* =========================
 * Initialization
 * ========================= */
window.addEventListener("DOMContentLoaded", () => {
  const requestIdInput = document.getElementById("requestId");
  const clearBtn = document.getElementById("clearBtn");
  const submitBtn = document.getElementById("submitBtn");
  const addQueryBtn = document.getElementById("addQueryBtn");
  const addShotBtn = document.getElementById("addShotBtn");
  const toggleSupportBtn = document.getElementById("toggleSupportBtn");

  if (requestIdInput) {
    requestIdInput.value = makeRequestId();
  }

  hideInlineLoading();
  updateQueryCount();
  updateShotCount();
  renderQuerySection();
  renderSupportShots();
  bindMaskEditorEvents();

  if (clearBtn) clearBtn.addEventListener("click", clearForm);
  if (submitBtn) submitBtn.addEventListener("click", submitInference);
  if (addQueryBtn) addQueryBtn.addEventListener("click", addQuery);
  if (addShotBtn) addShotBtn.addEventListener("click", addShot);
  if (toggleSupportBtn) toggleSupportBtn.addEventListener("click", toggleSupportSection);
});
