# Smart Business Analytics Dashboard

A local-first business analytics dashboard that automates data preparation, visualization, anomaly detection, and natural-language business insights through a multi-agent workflow.

> **Graduation project | Nile University | 2026**
>
> Developed as a collaborative team project. See the project report and team documentation for full project context and contribution details.

## Overview

The application lets users upload structured business datasets, preprocess and explore the data, generate interactive visualizations, detect historical and forecast anomalies, and receive AI-generated explanations and recommendations.

The system uses a FastAPI backend, a vanilla JavaScript frontend, a LangGraph-orchestrated agent workflow, and a locally hosted LLaMA model served through Ollama. Dataset analysis and model inference are designed to run locally rather than sending business data to a cloud LLM provider.

## Architecture

The main workflow has four stages:

1. **Data upload and interaction** — upload supported datasets, choose single- or multi-view analysis, and detect the business domain.
2. **Data processing and analysis** — parse and preprocess data with Python/Pandas; coordinate the analysis workflow with LangGraph; calculate descriptive statistics and forecast anomalies with Prophet.
3. **AI reasoning and insight generation** — use LLaMA 3.1 through Ollama to generate business narratives, explain predictive anomalies, and suggest next actions.
4. **Dashboard output** — present data previews, KPIs, visualizations, AI insights, anomaly analysis, recommendations, and PDF report export.

### Multi-agent workflow

The backend defines seven agent roles:

- **Retriever** — loads and profiles datasets.
- **Planner** — selects relevant columns and analytical priorities.
- **Stylist** — recommends chart types and visual styles.
- **Visualizer** — builds chart specifications.
- **Critic** — checks visualization quality and can route work back for revision.
- **Insight** — generates natural-language business analysis.
- **Recommender** — proposes actionable next steps.

## Features

- CSV, Excel, JSON, XML, Parquet, and TXT data ingestion (subject to installed parsing engines).
- Data preprocessing and descriptive statistics.
- Single-dataset and multi-view analysis.
- Interactive Plotly visualizations.
- IQR-based outlier analysis.
- Prophet-based time-series forecasting and anomaly detection.
- LLM-generated business insights and recommendations.
- PDF executive report generation.
- Local LLM integration through Ollama.

## Technology Stack

**Backend:** Python, FastAPI, Pandas, NumPy, Scikit-learn  
**Agent orchestration:** LangGraph  
**LLM runtime:** Ollama, LLaMA 3.1  
**Forecasting:** Prophet  
**Visualization:** Plotly  
**Frontend:** HTML, CSS, Vanilla JavaScript

## Repository Structure

```text
.
├── backend/
│   ├── agents_core.py
│   ├── config.py
│   ├── main.py
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── app.js
├── launch.py
├── requirements.txt
├── .env.example
└── .gitignore
```

## Local Setup

### 1. Prerequisites

- Python 3.10 or newer (use a version compatible with the pinned dependencies).
- [Ollama](https://ollama.com/) installed and running.
- The LLaMA model pulled locally.

### 2. Clone the repository

```bash
git clone https://github.com/faridasalahh/smart-business-analytics-dashboard.git
cd smart-business-analytics-dashboard
```

### 3. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure Ollama

```bash
ollama pull llama3.1:8b
ollama serve
```

If you use a different model, set `LLM_MODEL` in your `.env` file. You can start from `.env.example`.

### 6. Run the dashboard

```bash
python launch.py
```

Then open:

- Dashboard: http://localhost:8000
- FastAPI documentation: http://localhost:8000/docs
- Health endpoint: http://localhost:8000/api/health

Stop the server with `Ctrl+C`.

## Privacy and Data Handling

The application is designed for local LLM inference through Ollama. Uploaded datasets and generated reports are written to runtime directories (`backend/uploads/` and `backend/charts/`); these directories are excluded from Git by `.gitignore`. Do not commit private, confidential, or personally identifiable datasets.

## Notes

- LLaMA inference requires Ollama to be running and the configured model to be available.
- Forecasting requires a usable date/time column and sufficient historical observations.
- Parquet support may require an additional engine such as `pyarrow`, depending on the environment.
- This repository contains the application source code, not bundled datasets or model weights.

## Acknowledgements

This is a collaborative university graduation project. Credit all team members and supervisors in the repository before publishing, and clearly describe your own contributions in the project documentation.

