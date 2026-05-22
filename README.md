# Time-to-Next Damaging Strike

**Recurrent Event Survival Analysis of Wildlife Strike Sequences at US Commercial Airports**

This repository contains the complete replication code for the manuscript:

> *"Time-to-Next Damaging Strike: Recurrent Event Survival Analysis of Wildlife Strike Sequences at United States Commercial Airports Using Prentice–Williams–Peterson Gap-Time Models, Exogenous Airport Classification, and Proportional Hazards Diagnostics"*

**Authors:** Doruk Gürkan, Ozgur Yurtsever

---

## 📁 Repository Contents

| File | Description |
|------|-------------|
| `survival_gap_analysis.py` | Python pipeline for extracting damaging strikes from FAA NWSD, constructing the airport‑level gap‑time panel, estimating five Cox‑based recurrent event models (Cox PHM, AG, PWP‑TT, PWP‑GT, WLW), and generating all figures (KM curves, HR forest plot, seasonal hazard, Nelson–Aalen). |
| `frailty_analysis.R` | R script for REML gamma frailty estimation using `survival::coxph`, Grambsch–Therneau proportional hazards test, time‑varying coefficient models, and RMST calculation. |
| `gap_time_panel.csv` | Example dataset (first few rows) – the full panel can be regenerated from the FAA NWSD using the Python script. |
| `requirements.txt` | Python dependencies (optional, see below). |

---

## 📊 Data Source

The Federal Aviation Administration (FAA) **National Wildlife Strike Database (NWSD)** is publicly available at:  
[https://wildlife.faa.gov](https://wildlife.faa.gov)

Download `Public.xlsx` (April 2025 snapshot) and place it in the working directory before running the Python script.

---

## 🛠 Requirements

### Python (≥3.10)

Install dependencies:

```bash
pip install pandas openpyxl lifelines matplotlib scipy numpy
