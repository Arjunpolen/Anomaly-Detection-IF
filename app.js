const primaryButton = document.querySelector(".primary");
const ghostButton = document.querySelector(".ghost");
const uploader = document.getElementById("uploader");
const csvInput = document.getElementById("csvInput");
const uploadBtn = document.getElementById("uploadBtn");
const sampleBtn = document.getElementById("sampleBtn");
const showResultsBtn = document.getElementById("showResultsBtn");
const rowsValue = document.getElementById("rowsValue");
const colsValue = document.getElementById("colsValue");
const missingValue = document.getElementById("missingValue");
const numericValue = document.getElementById("numericValue");
const normalValue = document.getElementById("normalValue");
const anomalyValue = document.getElementById("anomalyValue");
const accuracyValue = document.getElementById("accuracyValue");
const uploadStatusText = document.getElementById("uploadStatusText");
const dashboardImage = document.getElementById("dashboardImage");
const gallery = document.getElementById("gallery");
const labelNote = document.getElementById("labelNote");
const lightbox = document.getElementById("lightbox");
const lightboxImage = document.getElementById("lightboxImage");
const lightboxCaption = document.getElementById("lightboxCaption");
const lightboxClose = document.getElementById("lightboxClose");
const lightboxBackdrop = document.getElementById("lightboxBackdrop");

const apiBase =
  (window.API_BASE_URL || document.body?.dataset?.apiBase || "").trim();

const buildApiUrl = (path) => {
  if (!apiBase) {
    return path;
  }
  const base = apiBase.endsWith("/") ? apiBase.slice(0, -1) : apiBase;
  const suffix = path.startsWith("/") ? path : `/${path}`;
  return `${base}${suffix}`;
};

if (primaryButton) {
  primaryButton.addEventListener("click", () => {
    document.querySelector("#results").scrollIntoView({ behavior: "smooth" });
  });
}

if (ghostButton) {
  ghostButton.addEventListener("click", () => {
    window.print();
  });
}

const parseCsv = (text) => {
  const rows = [];
  let current = "";
  let row = [];
  let inQuotes = false;

  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    const next = text[i + 1];

    if (char === '"') {
      if (inQuotes && next === '"') {
        current += '"';
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }

    if (char === "," && !inQuotes) {
      row.push(current.trim());
      current = "";
      continue;
    }

    if ((char === "\n" || char === "\r") && !inQuotes) {
      if (char === "\r" && next === "\n") {
        i += 1;
      }
      row.push(current.trim());
      if (row.length > 1 || row[0] !== "") {
        rows.push(row);
      }
      row = [];
      current = "";
      continue;
    }

    current += char;
  }

  if (current.length || row.length) {
    row.push(current.trim());
    rows.push(row);
  }

  return rows;
};

const summarize = (headers, data) => {
  let missing = 0;
  let numericCount = 0;
  const columnStats = headers.map(() => ({ numeric: true }));

  data.forEach((row) => {
    headers.forEach((_, idx) => {
      const value = row[idx];
      if (value === undefined || value === "") {
        missing += 1;
        return;
      }
      if (columnStats[idx].numeric) {
        const num = Number(value);
        if (Number.isNaN(num)) {
          columnStats[idx].numeric = false;
        }
      }
    });
  });

  columnStats.forEach((col) => {
    if (col.numeric) {
      numericCount += 1;
    }
  });

  rowsValue.textContent = data.length.toString();
  colsValue.textContent = headers.length.toString();
  missingValue.textContent = missing.toString();
  numericValue.textContent = numericCount.toString();
};

const setUploadButtonLabel = (name) => {
  if (!uploadBtn) {
    return;
  }
  if (!name) {
    uploadBtn.textContent = "Upload CSV";
    return;
  }
  const trimmed =
    name.length > 22 ? `${name.slice(0, 10)}...${name.slice(-8)}` : name;
  uploadBtn.textContent = `Uploaded: ${trimmed}`;
};

const setUploadStatus = (message) => {
  if (!uploadStatusText) {
    return;
  }
  uploadStatusText.textContent = message;
};

const startProcessing = () => {
  setUploadStatus("Processing in background. Updating results...");
};

const finishProcessing = (success) => {
  setUploadStatus(
    success
      ? "Processing complete. Results updated below."
      : "Processing failed. Check the server and try again."
  );
};

const updateResultMetrics = (summary) => {
  if (!summary) {
    return;
  }
  if (normalValue) {
    normalValue.textContent =
      summary.normal_count !== undefined ? summary.normal_count : "—";
  }
  if (anomalyValue) {
    anomalyValue.textContent =
      summary.anomaly_count !== undefined ? summary.anomaly_count : "—";
  }
  if (accuracyValue) {
    accuracyValue.textContent =
      summary.metrics && summary.metrics.accuracy !== undefined
        ? summary.metrics.accuracy.toFixed(2)
        : "N/A";
  }
  if (labelNote) {
    labelNote.hidden = summary.has_labels !== false;
  }
};

const updatePlots = (plots) => {
  if (!plots) {
    return;
  }
  Object.entries(plots).forEach(([key, src]) => {
    const figure = document.querySelector(`[data-plot=\"${key}\"]`);
    if (figure) {
      const img = figure.querySelector("img");
      if (img) {
        img.src = src;
      }
    }
  });
  if (dashboardImage && plots.dashboard) {
    dashboardImage.src = plots.dashboard;
  }
};

const runPipeline = async (file) => {
  if (!file) {
    return;
  }
  startProcessing();
  if (showResultsBtn) {
    showResultsBtn.disabled = true;
  }
  const payload = new FormData();
  payload.append("file", file);

  try {
    const response = await fetch(buildApiUrl("/api/run"), {
      method: "POST",
      body: payload,
    });
    if (!response.ok) {
      throw new Error("Server error");
    }
    const data = await response.json();
    if (data.status !== "ok") {
      throw new Error(data.message || "Processing failed");
    }
    updatePlots(data.plots);
    updateResultMetrics(data.summary);
    finishProcessing(true);
    if (showResultsBtn) {
      showResultsBtn.disabled = false;
    }
  } catch (error) {
    finishProcessing(false);
  }
};

const handleCsv = (text, name = "Uploaded CSV", file = null) => {
  const rows = parseCsv(text);
  if (!rows.length) {
    return;
  }
  const headers = rows[0];
  const data = rows.slice(1);

  setUploadButtonLabel(name);
  setUploadStatus("Upload complete. Preparing results...");
  summarize(headers, data);
  if (file) {
    runPipeline(file);
  }

  if (sampleBtn && name === "sample.csv") {
    sampleBtn.textContent = "Sample Loaded";
    sampleBtn.disabled = true;
  }
};

const loadSample = () => {
  const sample = `flow_id,src_bytes,dst_bytes,protocol,flag,anomaly_score\n1,120,340,tcp,SF,0.04\n2,0,0,icmp,S0,0.95\n3,560,20,udp,SF,0.12\n4,80,120,tcp,REJ,0.87\n5,30,15,udp,SF,0.08`;
  const blob = new Blob([sample], { type: "text/csv" });
  const file = new File([blob], "sample.csv", { type: "text/csv" });
  handleCsv(sample, "sample.csv", file);
};

if (csvInput) {
  csvInput.addEventListener("change", (event) => {
    const file = event.target.files[0];
    if (!file) {
      return;
    }
    const reader = new FileReader();
    reader.onload = (e) => handleCsv(e.target.result, file.name, file);
    reader.readAsText(file);
  });
}

if (uploadBtn && csvInput) {
  uploadBtn.addEventListener("click", () => {
    csvInput.click();
  });
}

if (sampleBtn) {
  sampleBtn.addEventListener("click", (event) => {
    event.preventDefault();
    loadSample();
  });
}

if (showResultsBtn) {
  showResultsBtn.disabled = true;
  showResultsBtn.addEventListener("click", () => {
    document.querySelector("#results").scrollIntoView({ behavior: "smooth" });
  });
}

if (uploader) {
  uploader.addEventListener("click", (event) => {
    if (!csvInput) {
      return;
    }
    const isButton = event.target.closest("button");
    if (!isButton) {
      csvInput.click();
    }
  });

  ["dragenter", "dragover"].forEach((type) => {
    uploader.addEventListener(type, (event) => {
      event.preventDefault();
      uploader.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((type) => {
    uploader.addEventListener(type, (event) => {
      event.preventDefault();
      uploader.classList.remove("dragover");
    });
  });

  uploader.addEventListener("drop", (event) => {
    const file = event.dataTransfer.files[0];
    if (!file) {
      return;
    }
    const reader = new FileReader();
    reader.onload = (e) => handleCsv(e.target.result, file.name, file);
    reader.readAsText(file);
  });
}

if (gallery) {
  gallery.addEventListener("click", (event) => {
    const figure = event.target.closest("figure");
    if (!figure) {
      return;
    }
    const img = figure.querySelector("img");
    const caption = figure.querySelector("figcaption");
    lightboxImage.src = img.src;
    lightboxCaption.textContent = caption ? caption.textContent : "";
    lightbox.classList.add("show");
    lightbox.setAttribute("aria-hidden", "false");
  });
}

const closeLightbox = () => {
  lightbox.classList.remove("show");
  lightbox.setAttribute("aria-hidden", "true");
  lightboxImage.src = "";
};

if (lightboxClose) {
  lightboxClose.addEventListener("click", closeLightbox);
}

if (lightboxBackdrop) {
  lightboxBackdrop.addEventListener("click", closeLightbox);
}

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && lightbox.classList.contains("show")) {
    closeLightbox();
  }
});
