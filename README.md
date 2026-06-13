# The Existential Glass Analyzer

**Is your glass half full or half empty?** A browser-based AI that answers
life's most pointless question with shocking confidence.

![Demo](demo.png)

Drop in a photo of a glass. A tiny neural network runs entirely in your browser
(via WebGPU) and renders a philosophical verdict. No servers, no API keys, no
privacy concerns — just a 6 MB model, a canvas overlay, and an opinion.

---

## Try It

Visit `https://ttithipan.github.io/glass-half-full/`

---

## How It Works

1. **Tiny detector** — MobileNetV3-Small backbone (1.66M params) trained on
   8,000 synthetic Blender images with perfect bounding-box labels.

2. **In-browser inference** — ONNX Runtime Web loads the model into a Web
   Worker. WebGPU-accelerated when available, WASM fallback.

3. **Philosophy engine** — Fill ratio = liquid height / glass height.
   Below 15%: void. Above 85%: hubris. The 40–60% zone flips a coin between
   "Half Full" and "Half Empty."

---

## Tech Stack

| Layer | Tech |
|---|---|
| Model | MobileNetV3-Small + 2-scale FPN (PyTorch → ONNX) |
| Runtime | ONNX Runtime Web (WebGPU / WASM) |
| Frontend | Plain HTML / CSS / JS, single Web Worker |
| Hosting | GitHub Pages |

---

## Credits

- Wood floor texture: [OnAirDesign Dark Wood Texture Board](https://www.onairdesign.com/products/dark-wood-texture-board-ww-wallpaper-d-oa-145)
- Sky background: [Pinterest](https://cl.pinterest.com/pin/387942955385519792/)

---

## Appreciation

- **[DeepSeek](https://www.deepseek.com/en/)** — for building genuinely excellent models at prices that make
  agent workflows accessible to individual developers.
- **[Zed](https://zed.dev/)** — for creating an editor where AI-assisted development feels native,
fast, and joyful. One of the best developer tooling in the game.

Thank you for pushing the frontier and keeping it open.
