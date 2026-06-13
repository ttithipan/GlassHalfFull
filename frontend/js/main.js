/**
 * The Existential Glass Analyzer — Main Thread
 */

const panelUpload = document.getElementById("panel-upload");
const panelLoading = document.getElementById("panel-loading");
const panelResult = document.getElementById("panel-result");
const panelError = document.getElementById("panel-error");
const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const modelStatus = document.getElementById("model-status");
const splashText = document.getElementById("splash-text");
const resultImage = document.getElementById("result-image");
const bboxCanvas = document.getElementById("bbox-canvas");
const verdict = document.getElementById("verdict");
const errorText = document.getElementById("error-text");
const btnReset = document.getElementById("btn-reset");
const btnErrorReset = document.getElementById("btn-error-reset");

let modelReady = false;
let worker = null;
let queuedSample = null;

const SPLASH = [
  "Figuring out the existential questions...",
  "Consulting with Descartes...",
  "Measuring fluid dynamics and human sorrow...",
  "Determining if the GPU needs therapy...",
  "Counting water molecules...",
  "Asking the glass what it identifies as...",
  "Debating with Nietzsche...",
  "Calibrating the philosophy engine...",
  "Running a background check on your glass...",
  "Cross-referencing with ancient Greek philosophy...",
  "Asking the model to ponder its existence...",
  "Checking if the glass passed the vibe check...",
  "Consulting the oracle of hydration...",
  "Performing advanced tensor analysis...",
  "The model is thinking. Give it a moment.",
  "Reticulating splines...",
  "Summoning the spirit of a bartender...",
  "Evaluating the glass's life choices...",
  "Comparing against historical pour data...",
  "The GPU is in deep contemplation.",
];

const VERDICTS = {
  void: {
    label: "The Abyss",
    lines: [
      "There's more water in a cactus.",
      "The glass identifies as a paperweight.",
      "Congratulations, you've photographed air.",
      "Even the GPU is disappointed.",
      "This glass is on a hydration strike.",
      "Somewhere, a fish is laughing at you.",
      "The model checked twice. Still empty.",
    ],
  },
  pessim: {
    label: "A Reluctant Splash",
    lines: [
      "Someone poured this apologetically.",
      "The glass is questioning its purpose.",
      "Enough to be sad about, not enough to enjoy.",
      "This is the 4 PM of beverages.",
      "A tragedy in transparent form.",
      "The liquid is giving minimal effort.",
      "Even the ice cubes have given up.",
    ],
  },
  halfFull: {
    label: "The Optimist's Delusion",
    lines: [
      "The glass chooses hope today.",
      "Somewhere, Descartes is smiling.",
      "Your optimism has been noted by the algorithm.",
      "Half full — just like your browser tabs.",
      "The model sees potential in this glass.",
      "A win for positive thinking. Barely.",
      "The GPU flipped a coin. It landed on hope.",
    ],
  },
  halfEmp: {
    label: "The Pessimist's Validation",
    lines: [
      "The glass chooses realism today.",
      "Half empty — mirroring your faith in humanity.",
      "Somewhere, Schopenhauer is nodding.",
      "The model has concerns about this glass.",
      "Technically correct. The worst kind of correct.",
      "The GPU flipped a coin. It landed on despair.",
      "Even the liquid looks disappointed.",
    ],
  },
  optim: {
    label: "Brimming with Enthusiasm",
    lines: [
      "Someone poured with confidence.",
      "This glass is living its best life.",
      "An above-average hydration situation.",
      "The glass is 70% water, 30% ambition.",
      "Now we're talking. Or drinking. Or both.",
      "Surface tension hasn't even broken a sweat.",
      "The model approves of this pour.",
    ],
  },
  hubris: {
    label: "The Flood of Arrogance",
    lines: [
      "This glass has no concept of restraint.",
      "Surface tension is doing heroic work here.",
      "One wrong move and this is a disaster film.",
      "The glass is overcompensating for something.",
      "Brimming with confidence. And liquid.",
      "The meniscus is clinging on for dear life.",
      "Someone has never heard of moderation.",
    ],
  },
};

const NO_GLASS_MSG = "Model exploded trying to find a glass";
const NO_GLASS_ALT = "Staring into the abyss (No glass detected)";

function show(panel) {
  [panelUpload, panelLoading, panelResult, panelError].forEach((p) =>
    p.classList.add("hidden"),
  );
  panel.classList.remove("hidden");
}

function classifyRatio(R) {
  if (R < 0.15) return "void";
  if (R < 0.4) return "pessim";
  if (R <= 0.6) return Math.random() < 0.5 ? "halfFull" : "halfEmp";
  if (R <= 0.85) return "optim";
  return "hubris";
}

function drawBboxes(imageEl, canvasEl, bboxes) {
  const nw = imageEl.naturalWidth,
    nh = imageEl.naturalHeight;
  if (nw === 0 || nh === 0) return;
  canvasEl.width = nw;
  canvasEl.height = nh;
  const ctx = canvasEl.getContext("2d");
  ctx.clearRect(0, 0, nw, nh);
  const colors = {
    0: { stroke: "#3B82F6", label: "glass" },
    1: { stroke: "#F59E0B", label: "liquid" },
  };
  for (const b of bboxes) {
    const c = colors[b.class_id] || { stroke: "#FFF", label: "?" };
    const x = (b.x_center - b.width / 2) * nw,
      y = (b.y_center - b.height / 2) * nh;
    ctx.strokeStyle = c.stroke;
    ctx.lineWidth = Math.max(2, nw / 128);
    ctx.strokeRect(x, y, b.width * nw, b.height * nh);
    ctx.fillStyle = c.stroke;
    const fontSize = Math.max(10, nw / 22);
    ctx.font = fontSize + "px -apple-system, sans-serif";
    ctx.fillText(
      c.label + " " + ((b.confidence || 0) * 100).toFixed(1) + "%",
      x + 4,
      y + fontSize + 2,
    );
  }
}

function initWorker() {
  worker = new Worker("js/worker.js");
  worker.onmessage = (e) => {
    const msg = e.data;
    switch (msg.type) {
      case "status":
        break;
      case "ready":
        modelReady = true;
        if (queuedSample) {
          handleSampleClick(queuedSample);
          queuedSample = null;
        }
        break;
      case "result":
        handleResult(msg);
        break;
      case "error":
        show(panelError);
        errorText.textContent = msg.message || "The oracle is unavailable.";
        break;
    }
  };
  worker.onerror = () => {
    modelReady = false;
  };
  worker.postMessage({ type: "load" });
}

function handleFile(file) {
  if (!file || !file.type.startsWith("image/")) return;
  if (!modelReady) {
    show(panelError);
    errorText.textContent = "Model not ready.";
    return;
  }
  const reader = new FileReader();
  reader.onload = () => {
    show(panelLoading);
    startSplashCycle();
    worker.postMessage({ type: "infer", image: reader.result }, [
      reader.result,
    ]);
  };
  reader.readAsArrayBuffer(file);
}

function startSplashCycle() {
  stopSplashCycle(); // clear any stale interval first
  let idx = Math.floor(Math.random() * SPLASH.length);
  splashText.textContent = SPLASH[idx];
  splashText._interval = setInterval(() => {
    idx = (idx + 1) % SPLASH.length;
    splashText.textContent = SPLASH[idx];
  }, 700);
}

function stopSplashCycle() {
  if (splashText._interval) {
    clearInterval(splashText._interval);
    splashText._interval = null;
  }
}

async function handleSampleClick(imgEl) {
  if (!modelReady) {
    show(panelLoading);
    startSplashCycle();
    queuedSample = imgEl;
    return;
  }
  try {
    show(panelLoading);
    startSplashCycle();
    const resp = await fetch(imgEl.src);
    if (!resp.ok) throw new Error("Failed to load sample");
    const buf = await resp.arrayBuffer();
    worker.postMessage({ type: "infer", image: buf }, [buf]);
  } catch (err) {
    show(panelError);
    errorText.textContent = "Failed to load sample image.";
  }
}

function handleResult(msg) {
  stopSplashCycle();

  const bboxes = msg.bboxes || [];
  if (bboxes.length === 0) {
    show(panelError);
    errorText.textContent = Math.random() < 0.5 ? NO_GLASS_MSG : NO_GLASS_ALT;
    return;
  }

  const glass = bboxes.find((b) => b.class_id === 0);
  const liquid = bboxes.find((b) => b.class_id === 1);

  // Display result — with or without image
  function showResult() {
    if (glass && liquid) {
      const R = liquid.height / glass.height;
      const stage = VERDICTS[classifyRatio(R)];
      verdict.innerHTML =
        "<span class='verdict-label'>" +
        stage.label +
        "</span>" +
        "<span class='verdict-line'>" +
        stage.lines[Math.floor(Math.random() * stage.lines.length)] +
        "</span>";
    } else if (glass) {
      verdict.textContent = VERDICTS.hubris.label;
    } else {
      verdict.textContent = "No glass found";
    }
    show(panelResult);
  }

  if (msg.image) {
    const blob = new Blob([msg.image], { type: "image/png" });
    const url = URL.createObjectURL(blob);
    resultImage.onload = () => {
      URL.revokeObjectURL(url);
      drawBboxes(resultImage, bboxCanvas, bboxes);
      showResult();
    };
    resultImage.src = url;
  }
}

dropZone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});
dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("drag-over");
});
dropZone.addEventListener("dragleave", () =>
  dropZone.classList.remove("drag-over"),
);
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
});
btnReset.addEventListener("click", () => show(panelUpload));
btnErrorReset.addEventListener("click", () => show(panelUpload));

// Clickable sample gallery
for (const img of document.querySelectorAll(".samples-grid img")) {
  img.addEventListener("click", () => handleSampleClick(img));
}

show(panelUpload);
initWorker();
