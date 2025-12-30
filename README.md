# Insurance Claims Denial Intelligence System

Rule-Based Validation + ML Risk Scoring + Decision Support

## Overview

Healthcare claims processing is expensive, error-prone, and heavily manual. A large percentage of insurance claim denials are preventable, often caused by missing prior authorizations, invalid coding combinations, or abnormal charges.

This project builds an __end-to-end claims denial intelligence system__ that:

* Simulates realistic healthcare claims data
* Applies deterministic rule-based validation
* Uses machine learning to predict denial risk
* Provides explainability (SHAP) for transparency
* Powers a decision-support tool for hospital billing staff

The goal is to reduce preventable denials, prioritize human review, and improve first-pass claim acceptance rates.

## Key Design Philosophy

Instead of treating this as a pure ML problem, the system mirrors real-world healthcare operations:

* Rules first for deterministic failures
* ML second for probabilistic risk
* Human-in-the-loop for ambiguous cases

This reflects how modern healthcare revenue cycle systems are actually built.

## System Architecture

Synthetic Claims Data
        │
        ▼
Rule-Based Validation Engine
        │
        ├── Hard Rule Fail (Auto-Deny)
        │
        ▼
Feature Engineering
        │
        ▼
ML Denial Risk Model
        │
        ├── High Risk → Review
        ├── Medium Risk → Review
        └── Low Risk → Safe to Submit
        │
        ▼
Explainability (SHAP)
        │
        ▼
Streamlit Decision Support Tool

## Data Generation

60,000 synthetic healthcare claims

Realistic distributions for:
- CPT codes
- ICD codes
- Provider specialties
- Charge amounts
- Prior authorization requirements

Denial reasons include:
- Prior authorization missing
- Invalid CPT–ICD combinations
- Charge outliers
- Modifier issues

## Rule-Based Validation Engine

Implements deterministic business rules commonly used by payers:
- Prior authorization enforcement
- Provider scope checks
- Charge anomaly detection
- Coding consistency validation

Each claim receives:
* rule_failed flag
* Structured rule_failure_reason

## Feature Engineering

Key engineered features include:
- Provider-level denial rate
- Charge-to-peer-average ratio
- Number of rule failures
- Historical provider behavior

These features intentionally reflect operational risk signals, not just raw claim attributes.

## Machine Learning Model
- Binary classification: denied vs approved
- Trained only on claims that passed hard rule checks
- Optimized for rare-event detection using PR-AUC
- Outputs a **risk score**, not an automated decision

The model is used to **prioritize review**, not to auto-deny claims.

## Explainability
SHAP is used to:
- Explain individual claim risk scores
- Identify global denial drivers
- Support auditability and stakeholder trust

Explainability is treated as a first-class requirement.

## Decision Support Tool
A Streamlit-based interface allows billing staff to:
- Enter claim details
- View rule validation outcomes
- See ML-based denial risk
- Receive a recommended action:
  - Safe to submit
  - Review recommended
  - Hard denial

The tool is designed for **human-in-the-loop decision support**, not full automation.

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, SHAP, Streamlit  
(Tableau used optionally for executive visualization)

## Why This Project
This project demonstrates:
- Healthcare domain understanding
- Hybrid rule + ML system design
- Explainability-first modeling
- Production-oriented decision intelligence
- Practical tradeoffs between automation and human review

## Author
Priyal Shah  