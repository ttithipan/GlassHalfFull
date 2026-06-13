/**
 * Web Worker — ONNX Runtime Web Inference Engine
 */

const MODEL_URL = "../model.onnx";
const INPUT_SIZE = 256;
const GRID_S16 = 16;
const GRID_S32 = 8;
const CONF_THRESHOLD = 0.05;
const MIN_BBOX_SIZE = 0.0; // Disabled — was filtering valid small detections
const TEMPERATURE = 3.0; // Softens overconfident logits from synthetic-trained model

let session = null;
let modelReady = false;

function loadOrt() {
  return new Promise((resolve, reject) => {
    if (typeof ort !== "undefined") {
      resolve(ort);
      return;
    }
    try {
      importScripts(
        "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.min.js",
      );
      ort.env.wasm.wasmPaths =
        "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/";
      resolve(ort);
    } catch (e) {
      reject(new Error("Failed to load ONNX Runtime: " + e.message));
    }
  });
}

async function preprocessImage(arrayBuffer) {
  const blob = new Blob([arrayBuffer]);
  // Decode image at full resolution (no resize in bitmap creation)
  const bitmap = await createImageBitmap(blob);

  // Draw to canvas at target 256x256 — browser handles resize
  const canvas = new OffscreenCanvas(INPUT_SIZE, INPUT_SIZE);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0, INPUT_SIZE, INPUT_SIZE);
  bitmap.close();

  const imageData = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
  const pixels = imageData.data;
  const chw = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
  const planeSize = INPUT_SIZE * INPUT_SIZE;
  for (let i = 0; i < planeSize; i++) {
    chw[i] = pixels[i * 4] / 255.0;
    chw[i + planeSize] = pixels[i * 4 + 1] / 255.0;
    chw[i + 2 * planeSize] = pixels[i * 4 + 2] / 255.0;
  }

  return chw;
}

function decodeGrid(output, gridSize, confThreshold) {
  // ONNX Runtime returns NCHW tensors: all values of channel 0, then
  // channel 1, etc.  Each grid cell has 6 channels: cx,cy,w,h,cls0,cls1.
  const planeSize = gridSize * gridSize;
  const detections = [];
  for (let gy = 0; gy < gridSize; gy++) {
    for (let gx = 0; gx < gridSize; gx++) {
      const p = gy * gridSize + gx;
      const cx = output[p],
        cy = output[p + planeSize],
        w = output[p + 2 * planeSize],
        h = output[p + 3 * planeSize];
      const cls0 = output[p + 4 * planeSize] / TEMPERATURE,
        cls1 = output[p + 5 * planeSize] / TEMPERATURE;
      const m = Math.max(cls0, cls1);
      const conf0 =
        Math.exp(cls0 - m) / (Math.exp(cls0 - m) + Math.exp(cls1 - m));
      const conf1 =
        Math.exp(cls1 - m) / (Math.exp(cls0 - m) + Math.exp(cls1 - m));
      if (conf0 > confThreshold && w > MIN_BBOX_SIZE && h > MIN_BBOX_SIZE)
        detections.push({
          class_id: 0,
          x_center: cx,
          y_center: cy,
          width: w,
          height: h,
          confidence: conf0,
        });
      if (conf1 > confThreshold && w > MIN_BBOX_SIZE && h > MIN_BBOX_SIZE)
        detections.push({
          class_id: 1,
          x_center: cx,
          y_center: cy,
          width: w,
          height: h,
          confidence: conf1,
        });
    }
  }
  return detections;
}

function nms(detections, iouThreshold = 0.5) {
  if (detections.length === 0) return [];
  detections.sort((a, b) => b.confidence - a.confidence);
  const keep = [];
  while (detections.length > 0) {
    const best = detections.shift();
    keep.push(best);
    detections = detections.filter((d) => {
      if (d.class_id !== best.class_id) return true;
      const ax1 = best.x_center - best.width / 2,
        ay1 = best.y_center - best.height / 2;
      const ax2 = best.x_center + best.width / 2,
        ay2 = best.y_center + best.height / 2;
      const bx1 = d.x_center - d.width / 2,
        by1 = d.y_center - d.height / 2;
      const bx2 = d.x_center + d.width / 2,
        by2 = d.y_center + d.height / 2;
      const ix1 = Math.max(ax1, bx1),
        iy1 = Math.max(ay1, by1);
      const ix2 = Math.min(ax2, bx2),
        iy2 = Math.min(ay2, by2);
      const iArea = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
      const uArea =
        (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - iArea;
      return uArea > 0 ? iArea / uArea < iouThreshold : true;
    });
  }
  return keep;
}

self.onmessage = async (e) => {
  const msg = e.data;
  switch (msg.type) {
    case "load": {
      try {
        const ort = await loadOrt();
        const resp = await fetch(MODEL_URL);
        const buf = await resp.arrayBuffer();
        try {
          session = await ort.InferenceSession.create(buf, {
            executionProviders: ["webgpu", "wasm"],
          });
        } catch (e) {
          session = await ort.InferenceSession.create(buf, {
            executionProviders: ["wasm"],
          });
        }
        modelReady = true;
        self.postMessage({ type: "ready" });
      } catch (err) {
        self.postMessage({
          type: "error",
          message: "Failed to load model: " + err.message,
        });
      }
      break;
    }

    case "infer": {
      if (!modelReady || !session) {
        self.postMessage({ type: "error", message: "Model not loaded yet." });
        return;
      }
      try {
        let inputTensor;

        // Special test mode: load preprocessed tensor directly
        if (msg.testTensor) {
          const resp = await fetch("/test_tensor.bin");
          const buf = await resp.arrayBuffer();
          inputTensor = new Float32Array(buf);
        } else {
          inputTensor = await preprocessImage(msg.image);
        }
        const feeds = {
          input: new ort.Tensor("float32", inputTensor, [
            1,
            3,
            INPUT_SIZE,
            INPUT_SIZE,
          ]),
        };
        const results = await session.run(feeds);
        const outS16 = results.out_s16.data;
        const outS32 = results.out_s32.data;

        // Max confidence — walk each grid cell's cls0/cls1 (NCHW layout)
        let maxConf = 0;
        const s16Plane = GRID_S16 * GRID_S16;
        for (let p = 0; p < s16Plane; p++) {
          const cls0 = outS16[p + 4 * s16Plane] / TEMPERATURE;
          const cls1 = outS16[p + 5 * s16Plane] / TEMPERATURE;
          const m = Math.max(cls0, cls1);
          const s =
            Math.exp(cls0 - m) / (Math.exp(cls0 - m) + Math.exp(cls1 - m));
          if (s > maxConf) maxConf = s;
        }
        const s32Plane = GRID_S32 * GRID_S32;
        for (let p = 0; p < s32Plane; p++) {
          const cls0 = outS32[p + 4 * s32Plane] / TEMPERATURE;
          const cls1 = outS32[p + 5 * s32Plane] / TEMPERATURE;
          const m = Math.max(cls0, cls1);
          const s =
            Math.exp(cls0 - m) / (Math.exp(cls0 - m) + Math.exp(cls1 - m));
          if (s > maxConf) maxConf = s;
        }

        let detections = [
          ...decodeGrid(outS16, GRID_S16, CONF_THRESHOLD),
          ...decodeGrid(outS32, GRID_S32, CONF_THRESHOLD),
        ];
        const rawCount = detections.length;
        detections = nms(detections, 0.5);

        let glass = null,
          liquid = null;
        for (const d of detections) {
          if (d.class_id === 0 && (!glass || d.confidence > glass.confidence))
            glass = d;
          if (d.class_id === 1 && (!liquid || d.confidence > liquid.confidence))
            liquid = d;
        }

        const bboxes = [];
        if (glass) bboxes.push(glass);
        if (liquid) bboxes.push(liquid);

        let fillRatio = null;
        if (glass && liquid)
          fillRatio = liquid.height / Math.max(glass.height, 0.001);

        const allConfs = detections.map(
          (d) => `cls${d.class_id}:${(d.confidence * 100).toFixed(1)}%`,
        );
        allConfs.sort(
          (a, b) => parseFloat(b.split(":")[1]) - parseFloat(a.split(":")[1]),
        );

        const msgPayload = {
          type: "result",
          bboxes,
          analytics: {
            maxConfidence: maxConf,
            rawDetections: rawCount,
            afterNMS: bboxes.length,
            topConfs: allConfs.slice(0, 8).join(" "),
            fillRatio: fillRatio,
            glass: glass
              ? {
                  cx: glass.x_center.toFixed(3),
                  cy: glass.y_center.toFixed(3),
                  w: glass.width.toFixed(3),
                  h: glass.height.toFixed(3),
                  conf: (glass.confidence * 100).toFixed(1),
                }
              : null,
            liquid: liquid
              ? {
                  cx: liquid.x_center.toFixed(3),
                  cy: liquid.y_center.toFixed(3),
                  w: liquid.width.toFixed(3),
                  h: liquid.height.toFixed(3),
                  conf: (liquid.confidence * 100).toFixed(1),
                }
              : null,
          },
        };
        if (msg.image) {
          msgPayload.image = msg.image;
          self.postMessage(msgPayload, [msg.image]);
        } else {
          self.postMessage(msgPayload);
        }
      } catch (err) {
        self.postMessage({
          type: "error",
          message: "Inference failed: " + err.message,
        });
      }
      break;
    }
  }
};
