# X-RayTriage AI

> **Explainable AI for intelligent chest X-ray triage**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![Tests Passing](https://img.shields.io/badge/pytest-48%20passed-brightgreen.svg)](https://docs.pytest.org/)
[![Repo Size](https://img.shields.io/badge/repo%20size-%3C%201%20MB-success.svg)]()
[![License](https://img.shields.io/badge/license-MIT-blue.svg)]()

---

## ⚠ Important Clinical & Safety Notice

> **AI decision support — not a diagnosis. All results require qualified clinical review.**
>
> X-RayTriage AI is a research and decision-support prototype designed for low-resource clinical environments to assist radiologists in prioritizing work queues. It **does not** provide automated diagnoses of pneumonia, tuberculosis, malignancy, or any other disease.

---

## 1. Problem Statement

In low-resource district hospitals, community health centers, and high-volume public screening programs, radiologist shortages create severe diagnostic delays. Hundreds of chest radiographs (CXRs) may queue up daily, with emergency patients and routine occupational screenings placed in the same FIFO (first-in, first-out) order. 

Critical findings like acute lobar pneumonia or active cavitary tuberculosis can wait hours or days for human review, leading to delayed interventions, increased morbidity, and transmission risks.

## 2. Solution Overview

**X-RayTriage AI** is an explainable decision-support platform that acts as an intelligent pre-read triage layer:

1. Ingests and rigorously validates chest radiographs.
2. Runs deep learning inference to quantify abnormality signals (DenseNet121).
3. Produces visual attention heatmaps via **Grad-CAM** for radiologist interpretability.
4. Incorporates basic, strictly **non-identifying workflow context** (referral urgency, vitals/oxygenation, symptom duration).
5. Computes a transparent, deterministic **Priority Score (0–100)** with an auditable **Decision Trace**.
6. Dynamically enqueues cases into a prioritized radiology queue, ensuring the most urgent patients are inspected first.

---

## 3. Key Features

- **Robust Medical File Ingestion:** Strict MIME validation, dimension bounds (32px to 8192px), file size limits (10 MB), and image corruption detection.
- **Radiograph Preprocessing Pipeline:** Contrast-Limited Adaptive Histogram Equalization (CLAHE) approximation, high-fidelity Lanczos resampling to 224×224, and ImageNet normalization. Separation of training augmentation from deterministic inference.
- **PyTorch DenseNet121 Backbone:** Transfer learning architecture supporting CPU fallback and GPU acceleration.
- **Grad-CAM Explainability:** Real hook-based Gradient-weighted Class Activation Mapping highlighting convolutional feature maps in the final layer, with colorized Jet overlays.
- **Zero-PII Context Ingestion:** Workflow-only non-identifying fields: Age Group, Reported Symptoms, Symptom Duration, Referral Type (Emergency, Inpatient, Outpatient, Screening), and Oxygen Status.
- **Deterministic Triage Decision Engine:** Transparent mathematical scoring with unit-tested boundary thresholds ($0.80 \rightarrow \text{HIGH}$, $0.50 \rightarrow \text{MEDIUM}$, $<0.50 \rightarrow \text{LOW}$).
- **Human-in-the-Loop Case Review:** Dedicated inspection interface with side-by-side radiograph comparisons and workflow status controls (`Awaiting Review`, `In Review`, `Reviewed`, `Flagged`).
- **Comprehensive Test Suite:** 48 passing unit tests covering preprocessing, validation, inference determinism, triage boundary conditions, queue sorting, and API endpoints.
- **Dual-Mode Execution (Real Model + Demo Mode):** Operates immediately with zero weight downloads using stable perceptual digest fixtures, while supporting custom trained PyTorch weights.

---

## 4. Architecture

```text
+-----------------------------------------------------------------------------------+
|                                 CLIENT TIER                                       |
|  - Modern Clinical Dashboard (Tailwind CSS, Responsive, Dark/Light, WCAG AA)      |
|  - Drag-and-Drop Radiograph Ingestion & 1-Click Synthetic Sample CXR Loader      |
|  - Side-by-Side Grad-CAM Attention Map & Radiograph Overlay Inspector             |
|  - Dynamic Priority Queue with Real-time Search, Multi-Filter, & Custom Sort     |
+------------------------------------------+----------------------------------------+
                                           | HTTP REST API
                                           v
+-----------------------------------------------------------------------------------+
|                            FASTAPI BACKEND TIER                                   |
|  POST /api/analyze  |  GET /api/cases  |  GET /api/queue  |  PATCH /api/cases/{id} |
+------------------------------------------+----------------------------------------+
                                           |
    +--------------------------------------+-----------------------------------+
    |                                      |                                   |
    v                                      v                                   v
+------------------------+  +-------------------------------+  +------------------------+
|   validation.py        |  |     preprocessing.py          |  |      inference.py      |
| - MIME & Ext Check     |  | - 3-Channel RGB Conversion    |  | - PyTorch DenseNet121  |
| - Header / Corruption  |  | - Contrast Enhancement (CLAHE)|  | - Transfer Learning    |
| - 10 MB Size Limit     |  | - 224x224 Lanczos Resampling  |  | - CPU Fallback / GPU   |
| - Filename Sanitize    |  | - ImageNet Mean/Std Norm      |  | - Deterministic Demo   |
+------------------------+  +-------------------------------+  +-----------+------------+
                                                                           |
    +----------------------------------------------------------------------+
    |
    v
+---------------------------------------+      +----------------------------------------+
|             gradcam.py                |      |               triage.py                |
| - Target Conv Feature Maps            | ---> | - Deterministic Scoring Formula        |
| - Channel-wise Gradient Pooling       |      | - Non-Identifying Context Modifiers    |
| - ReLU + Normalization                |      | - Strict Boundary Thresholds (0.8/0.5) |
| - Jet Colormap & Blended Overlay      |      | - Explainable Decision Trace Array     |
+---------------------------------------+      +-------------------+--------------------+
                                                                   |
                                                                   v
                                               +----------------------------------------+
                                               |               queue.py                 |
                                               | - Thread-safe SQLite / Memory Store   |
                                               | - Multi-tier Priority Queue Sort:      |
                                               |   1. Priority Level (HIGH > MED > LOW) |
                                               |   2. Priority Score (Descending)       |
                                               |   3. Waiting Time (Oldest First)       |
                                               +----------------------------------------+
```

---

## 5. Algorithmic Workflow & Decision Logic

### A. Preprocessing Pipeline (`backend/app/preprocessing.py`)
1. Ingests raw decoded pixels from PIL.
2. Converts arbitrary single-channel grayscale or RGBA radiographs to standardized 3-channel RGB.
3. Applies adaptive histogram contrast equalization across hemithoraces.
4. Resizes to $224 \times 224$ using high-order Lanczos interpolation.
5. Standardizes pixel intensities using standard ImageNet parameters:
   $$\text{Input}_{\text{norm}} = \frac{\frac{\text{Pixel}}{255.0} - \mu}{\sigma}, \quad \mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$
6. Transposes to channel-first tensor format $(1, 3, 224, 224)$. Random training augmentations (rotations, flips) are strictly isolated and never invoked during inference.

### B. Grad-CAM Calculation (`backend/app/gradcam.py`)
Grad-CAM computes the importance of each feature map in the final convolutional layer $A^k$ for target class $c$:
1. Forward pass computes class score $y^c$.
2. Gradients are computed via backpropagation: $\frac{\partial y^c}{\partial A^k_{i, j}}$.
3. Channel-wise importance weights $\alpha_k^c$ are computed via Global Average Pooling:
   $$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A^k_{i, j}}$$
4. Weighted combination followed by Rectified Linear Unit (ReLU) to isolate positive contributions:
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
5. Min-max normalization scales intensities to $[0, 1]$, mapped to a Jet pseudo-color spectrum, and blended over the radiograph at $\alpha = 0.45$.

### C. Triage Scoring Formulation (`backend/app/triage.py`)
The system computes an auditable Priority Score between 0 and 100:

$$\text{Priority Score} = \min\left(100, \text{Base Points} + \sum \text{Context Modifiers}\right)$$

Where:
$$\text{Base Points} = \text{round}(\max(\text{Abnormal Probabilities}) \times 100)$$

#### Context Modifiers Table

| Clinical Context Factor | Condition | Urgency Points | Rationale |
|:---|:---|:---:|:---|
| **Referral Pathway** | Emergency Department | **+15** | Acute presentations require accelerated review |
| **Referral Pathway** | Inpatient Hospital Ward | **+10** | Admitted patients awaiting management decisions |
| **Vitals / Oxygenation** | Concerning / Desaturating | **+15** | Hypoxemia signals physiological compromise |
| **Symptoms** | Persistent Cough > 2 weeks | **+10** | WHO clinical hallmark for pulmonary tuberculosis |
| **Symptoms** | Shortness of Breath | **+8** | Respiratory distress indicator |
| **Symptoms** | Chest Pain | **+6** | Potential pleuritic / cardiovascular urgency |
| **Symptoms** | Fever | **+4** | Active systemic inflammatory process |
| **Demographics** | Pediatric (<18) or Older Adult (65+) | **+4** | Vulnerable clinical populations |

#### Prototype Priority Thresholds
- **HIGH PRIORITY (🔴):** $\text{Highest Abnormal Model Score} \ge 0.80$ OR $\text{Priority Score} \ge 80$ OR Acute Emergency Escalation.
- **MEDIUM PRIORITY (🟠):** $0.50 \le \text{Score} < 0.80$ OR Inpatient Context Escalation.
- **LOW PRIORITY (🟢):** $\text{Score} < 0.50$ (Normal screening / baseline findings).

*Boundary conditions are verified in automated test suites: $0.91 \rightarrow \text{HIGH}$, $0.80 \rightarrow \text{HIGH}$, $0.79 \rightarrow \text{MEDIUM}$, $0.50 \rightarrow \text{MEDIUM}$, $0.49 \rightarrow \text{LOW}$.*

---

## 6. Tech Stack

- **Backend:** Python 3.11+, FastAPI 0.110+, Uvicorn, Pydantic v2
- **Image Processing & ML:** NumPy 2.x, Pillow (PIL) 12.x, PyTorch (DenseNet121 architecture)
- **Frontend:** HTML5, Tailwind CSS (Clinical Dark/Light Palette), Lucide Icons, Vanilla JavaScript SPA
- **Testing:** pytest 9.1+, HTTPX, Starlette TestClient
- **Storage:** Thread-safe in-memory SQLite repository

---

## 7. Installation & Setup

### Prerequisites
- Python 3.11+
- Git

### 1. Clone Repository
```bash
git clone <repository-url>
cd Triage
```

### 2. Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### 3. Environment Configuration
Copy the provided `.env.example` template:
```bash
copy .env.example .env   # On Windows
cp .env.example .env     # On Linux / macOS
```

---

## 8. Running the Application

Launch the unified server (serves REST API and clinical frontend on port 8000):

```bash
python run.py
```

Open your browser to:
- **Clinical Frontend:** [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check Endpoint:** [http://localhost:8000/api/health](http://localhost:8000/api/health)

---

## 9. Operating Modes

### Mode A: Real Model
To run real PyTorch weights:
1. Place trained PyTorch model state dictionary weights at `models/model_weights.pth`.
2. The weights file is automatically ignored by `.gitignore` to maintain the under-10MB repository limit.
3. The server automatically detects the weights, executes GPU/CPU inference, and switches the indicator to `● Live Model`.

### Mode B: Demo Mode (Default Fallback)
If weights are not present or `FORCE_DEMO_MODE=true` is set:
- Deterministic simulated outputs are derived using stable perceptual luminance digests and anatomical quadrant features.
- The **exact same radiograph always produces the exact same probabilities**.
- The interface prominently displays: `DEMO MODE — simulated output for interface demonstration`.
- Grad-CAM renders an `Explainability preview — demonstration mode`.

---

## 10. Automated Testing

Run the comprehensive 48-test verification suite:

```bash
python -m pytest
```

### Test Coverage Highlights
- `test_validation.py`: Formats (.png, .jpg, .jpeg), corrupted image rejection, 0-byte uploads, oversized payload rejection, path sanitization.
- `test_preprocessing.py`: RGB normalization, 224×224 shape, ImageNet bounds, finite tensor checks, isolated training augmentations.
- `test_triage.py`: Exact boundary checks ($0.91, 0.80, 0.79, 0.50, 0.49$), context modifiers (Emergency, Concerning Oxygen, Cough > 2 wks), disclaimer inclusion.
- `test_queue.py`: Multi-tier sorting (Priority $\rightarrow$ Score $\rightarrow$ Waiting time), search, filter, status lifecycle transitions.
- `test_inference.py`: Model loading, determinism, probability sums $\approx 1.0$, CPU execution.
- `test_gradcam.py`: Heatmap shape (224, 224), min/max normalization, Jet color mapping, overlay blending.
- `test_api.py`: Health check, model card, upload endpoint, 400 bad request handling, queue retrieval, patch status update, analytics.

---

## 11. Security & Privacy Hygiene

- **Zero Personally Identifiable Information (PII):** The application explicitly rejects collection of patient names, phone numbers, addresses, government IDs, or hospital record numbers.
- **Upload Hardening:** Uploads are strictly checked against extension whitelists, MIME integrity, memory decoding limits, and maximum file size (10 MB). Executable files are never saved or executed.
- **Path Sanitization:** Filenames are stripped of path traversal characters (`..`, `/`, `\`) and non-alphanumeric symbols.
- **Secret Isolation:** API keys, database credentials, and internal paths are externalized into `.env`. No secrets or passwords exist in the source code.
- **Temporary In-Memory Storage:** Radiographs are processed in-memory or in ephemeral storage during triage analysis.

---

## 12. Ethical AI & Clinical Limitations

1. **Decision Support Only:** This platform is an engineering and research prototype designed to optimize radiologist queue sorting. It **does not substitute** for human clinical evaluation or definitive diagnosis.
2. **Dataset Demographics & Domain Shift:** Model accuracy can degrade when evaluating radiographs from patient populations, age distributions, or imaging equipment (portable CR vs fixed DR) different from training sets.
3. **Heatmap Interpretation:** Grad-CAM visualizations indicate receptive field activations that contributed to the model's numerical output. They do **not** delineate pathology borders, surgical margins, or biological disease extent.
4. **No Fabricated Accuracy Metrics:** The Analytics view displays:
   > *"Evaluation metrics unavailable — no validated test dataset configured."*
   Fabricating diagnostic accuracy (e.g. "98% accurate") without a prospectively locked, peer-reviewed evaluation dataset is clinically unethical and strictly avoided.

---

## 13. OptiForge Hackathon Compliance

- **Repository Size:** Entire repository size is **< 1 MB** (comfortably under the strict **10 MB** limit).
- **Single Branch:** Development is confined exclusively to the `main` branch.
- **Public Accessibility:** Open source, fully runnable without proprietary tokens or external API keys.
- **Immediate Local Execution:** Installs and launches in seconds via `python run.py`.
