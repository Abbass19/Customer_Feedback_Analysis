# Customer Feedback Analysis

A project that analyzes Arabic patient feedback collected from Saudi Arabian hospitals. This system performs **aspect-based sentiment analysis** and **multi-aspect classification**, supporting insights into hospital services, staff performance, and patient satisfaction.

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Dataset](#dataset)  
- [Database and API](#database-and-api)  
- [Models](#models)  
  - [Classification Models](#classification-models)  
  - [NER Model](#ner-model)  
  - [STT Model](#stt-model)  
- [Project Structure](#project-structure)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Technologies](#technologies)  
- [References](#references)  

---

## Project Overview

This project captures real-world patient experiences in Saudi hospitals. Using collected Arabic feedback, it provides simultaneous **aspect-level sentiment analysis** and **multi-aspect classification**. The main objectives include:

- Extracting sentiment for multiple hospital service aspects: Pricing, Appointments, Medical Staff, Customer Service, Emergency Services.  
- Building AI models for classification, sentiment analysis, and speech-to-text tasks.  
- Providing an API-driven backend for seamless integration with a UI.

---

## Dataset

**HEAR Dataset**: ~30,000 Arabic patient feedback records collected from surveys, online reviews, and hospital forms.  
- **Fields**: Feedback text, hospital, department, doctor, aspect-level sentiment scores (0: Positive, 1: Negative, 2: Neutral, 3: Not Mentioned).  
- **Tasks**: Aspect-based sentiment analysis and multi-aspect classification simultaneously.

**Example Aspect Sentiment Vector**: `[0, 1, 2, 0, 3]`  
Each element corresponds to a hospital service aspect.

---

## Database and API

The backend is modular:

- **db_layer**: Handles database operations (PostgreSQL).  
- **ai_api**: Manages AI models for classification, sentiment analysis, and NER.  
- **app_api**: Connects the UI with both db_layer and ai_api.  

**Visualizations**:  
- Modules Diagram: `Documentation/Modules.PNG`  
- Database Schema: `Documentation/Entity-Relationship Diagram.PNG`  

---

## Models

### Classification Models

1. **Multi-Head BERT-Based Classifier**  
   - Base Model: `aubmindlab/bert-base-arabertv2`  
   - Predicts sentiment for all five aspects simultaneously using separate linear heads.  
   - **Performance**:

| Aspect             | Accuracy | Precision | Recall | F1-score | mAP  |
|-------------------|---------|-----------|--------|----------|------|
| Pricing           | 0.96    | 0.82      | 0.58   | 0.65     | 0.68 |
| Appointments      | 0.94    | 0.77      | 0.73   | 0.75     | 0.73 |
| Medical Staff     | 0.84    | 0.77      | 0.74   | 0.75     | 0.82 |
| Customer Service  | 0.82    | 0.74      | 0.69   | 0.71     | 0.76 |
| Emergency Services| 0.97    | 0.66      | 0.63   | 0.64     | 0.67 |

*Other classification models:*  
- TF-IDF Classifier (`Documentation/TF-IDF_Function.PNG`)  
- Pretrained Embeddings ML (`Documentation/Pretrained_Embeddings_ML.PNG`)  

> BERT-based model chosen for best performance, despite longer training time.

### NER Model

- **Model**: GLiNER (`Documentation/Bert_Classifier_Model_Class.PNG`)  
- Used for extracting entities, but not fully tested yet.  

### STT Model

- **Model**: Whisper (`Documentation/Wisper_Model.PNG`)  
- Tested with 20 Arabic audio recordings.  
- Performs well on clear Arabic speech; accuracy decreases with noise or dialectal variation.

---

## Project Structure

- **Core modules**: `db_layer`, `ai_api`, `app_api`  
- **Scripts**: Training, testing, and evaluation scripts are included.  
- **UI**: User interface connects to API (`Documentation/UI.PNG`).  

**Visual Overview**:  
- `Documentation/Project_Structure.PNG`  

---

## Installation

```bash
# Clone repository
git clone https://github.com/Abbass19/Customer_Feedback_Analysis_2.git

# Navigate into project folder
cd Customer_Feedback_Analysis_2

# Install dependencies
pip install -r requirements.txt
