# AI Dataset Size Study: Effect on Object Detection Performance

## Overview

This research project investigates how dataset size affects the accuracy and efficiency of AI object detection models.
We tested three dataset sizes (small, medium, large) using TensorFlow/Keras CNNs and measured **accuracy, loss, and prediction speed**.

* Medium dataset achieved the best performance (91.9% accuracy, 0.09s prediction time).
* Large dataset had slightly lower accuracy due to overfitting.
* Small dataset was fast but inaccurate.

This repo contains:

* Report (PDF & DOCX)
* Google Colab notebooks (training, analysis)
* Raw results (CSV) + graphs

---

## Repository Structure

```
report/         → Written report (PDF, DOCX)
notebooks/      → Colab notebooks for training and analysis
results/        → CSV files + plots
```

---

## Installation

Clone the repo:

```bash
git clone https://github.com/YOUR-USERNAME/ai-dataset-size-study.git
cd ai-dataset-size-study
```

If you want to run locally, install requirements:

```bash
pip install -r notebooks/requirements.txt
```

Or open notebooks directly in **Google Colab**.

---

## Usage

1. **Open in Google Colab**

   * Upload the repo or specific notebook to Colab.
   * Or open directly from GitHub using:

     ```
     File → Open Notebook → GitHub tab → paste repo URL
     ```

2. **Run training & testing notebook**
   Saves CSVs + models into `results/raw_csv/`.

3. **Run analysis notebook**
   Generates plots in `results/plots/`.

---

## Final Average Results

* Medium dataset = **91.9% accuracy, 0.09s speed**
* Small dataset = **70.1% accuracy, 0.04s speed**
* Large dataset = **84.5% accuracy, 0.14s speed**

---

## License

MIT License – free to use with attribution.
