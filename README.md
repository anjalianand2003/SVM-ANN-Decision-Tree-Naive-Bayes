# SVM-ANN-Decision-Tree-Naive-Bayes
# Ride Cancellation Prediction using R

## 🚖 Overview

This project aims to predict **cab ride cancellations** on the YourCabs.com platform using classification algorithms in R. The goal is to reduce last-minute cancellations and improve customer experience by proactively identifying high-risk bookings.

## 📊 Dataset

- 10,000 bookings from Bangalore-based cab service.
- 17 features including:
  - Booking method (mobile/online)
  - Travel type
  - Ride cancellation status (`Car_Cancellation`)

## 🔍 Objective

Build multiple classification models to predict whether a ride will be canceled, using:

- Naive Bayes
- Decision Trees (C5.0)
- Neural Networks
- Support Vector Machines (SVM)

## 📈 Models and Highlights

| Model                | Key Features                         | Accuracy     |
|---------------------|--------------------------------------|--------------|
| Naive Bayes         | Basic and with `travel_type_id`      | Moderate     |
| Decision Tree       | C5.0, with and without boosting       | High         |
| ANN                 | 1-layer and 2-layer neural nets       | Improved     |
| SVM (Linear)        | Fast but less flexible                | Decent       |
| SVM (RBF Kernel)    | Best accuracy with tuning             | **92.6%**    |

⚠️ **Class Imbalance:** Majority of rides were not canceled. Although models had high sensitivity, specificity remained low (unable to detect cancellations well).

## 🧠 Final Verdict

- **Best Models:** SVM (RBF) & Decision Tree with Boosting.
- These models achieved up to **92.6% accuracy** and better handled complex patterns.

## 🧑‍💻 How to Run

```r
# Required packages
install.packages(c("caret", "klaR", "C50", "neuralnet", "fastDummies", "kernlab", "dplyr"))

# Run the script
source("yourcab_assignment2.R")
