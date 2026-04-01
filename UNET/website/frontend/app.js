const API_URL = "http://127.0.0.1:8000/predict";

const imageInput = document.getElementById("imageInput");
const runBtn = document.getElementById("runBtn");
const statusEl = document.getElementById("status");
const originalPreview = document.getElementById("originalPreview");
const maskPreview = document.getElementById("maskPreview");
const metaInfo = document.getElementById("metaInfo");

let selectedFile = null;
let currentOriginalUrl = null;

function setStatus(message, type = "info") {
  statusEl.textContent = message;
  statusEl.className = `status status-${type}`;
}

function clearMaskOutput() {
  maskPreview.removeAttribute("src");
  metaInfo.textContent = "Inference metadata will appear here.";
}

imageInput.addEventListener("change", () => {
  const file =
    imageInput.files && imageInput.files[0] ? imageInput.files[0] : null;
  selectedFile = file;

  clearMaskOutput();

  if (!selectedFile) {
    originalPreview.removeAttribute("src");
    setStatus("Select a file, then run inference.", "info");
    return;
  }

  if (currentOriginalUrl) {
    URL.revokeObjectURL(currentOriginalUrl);
  }

  currentOriginalUrl = URL.createObjectURL(selectedFile);
  originalPreview.src = currentOriginalUrl;
  setStatus("Image selected. Click Run Inference.", "ok");
});

runBtn.addEventListener("click", async () => {
  if (!selectedFile) {
    setStatus("Please select an image first.", "error");
    return;
  }

  runBtn.disabled = true;
  setStatus("Running model inference...", "info");

  try {
    const formData = new FormData();
    formData.append("file", selectedFile);

    const response = await fetch(API_URL, {
      method: "POST",
      body: formData,
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Request failed.");
    }

    maskPreview.src = `data:image/png;base64,${payload.mask_base64}`;
    metaInfo.textContent = `Threshold: ${payload.threshold} | Inference: ${payload.inference_ms} ms | Output: ${payload.width}x${payload.height}`;
    setStatus("Inference done.", "ok");
  } catch (error) {
    clearMaskOutput();
    setStatus(`Error: ${error.message}`, "error");
  } finally {
    runBtn.disabled = false;
  }
});
