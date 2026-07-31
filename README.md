# Hotel Cancellation Prediction System

**MIDS 207 — Summer 2026**

## Start here

**Canonical notebook:** [`notebooks/final_combined.ipynb`](https://github.com/eddy-cho26/hotel_cancellation_model_207/blob/main/notebooks/final_combined.ipynb)

This notebook runs the full pipeline end-to-end: cleaning, feature prep, EDA, models, and subgroup evaluation. Earlier per-model notebooks are kept under `notebooks/archive/` for reference.

## Problem

Hotels suffer significant revenue leakage from last-minute cancellations and no-shows. When a booking is cancelled too late to rebook the room, the room sits empty — a loss that cannot be recovered. Traditional overbooking strategies are blunt instruments: too conservative and rooms go empty; too aggressive and guests get walked, damaging brand reputation.

This project builds a machine learning system that assigns a **cancellation probability score** to every booking at the time it is made. By quantifying risk at the individual booking level, the hotel can make smarter, data-driven decisions about overbooking buffers, deposit policies, and outreach to at-risk guests.

## Goal

Flag high-risk bookings early enough to act on them — through targeted overbooking, early re-marketing of the room, or requiring a non-refundable deposit — without penalizing low-risk guests unnecessarily.

## Data

[Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) (Kaggle)

Key features:

| Feature | Description |
|---|---|
| `lead_time` | Days between booking and arrival — longer leads correlate with higher cancellation risk |
| `arrival_date` | Arrival date (year, month, week, day-of-month) |
| `hotel` | Resort Hotel vs. City Hotel |
| `adults` / `children` / `babies` | Guest composition |
| `distribution_channel` | Booking channel (TA/TO, Direct, Corporate, etc.) |
| `deposit_type` | No deposit, Non-refundable, or Refundable |
| `previous_cancellations` | Historical cancellation behavior of the guest |
| `booking_changes` | Number of modifications made to the booking |
| `adr` | Average daily rate |

**Target variable:** `is_canceled` (binary: 0 = stayed, 1 = cancelled)

## Technical Approach

- **Task:** Binary classification
- **Models:** Majority-class baseline → Logistic Regression (TensorFlow) → Decision Tree → Random Forest → 1D CNN (TensorFlow)
- **Evaluation:** ROC-AUC and PR-AUC (primary), plus precision / recall / F1; subgroup performance by hotel type, customer type, distribution channel, guest status, and lead-time bucket
- **Key concern:** Class imbalance handling; ranking metrics matter more than raw accuracy since the output drives a business decision (overbooking level), not a hard label

## Project Structure

```text
hotel_cancellation_model_207/
├── data/
│   ├── raw/                  # Original data — read-only
│   └── processed/            # Outputs produced by the notebook
├── notebooks/
│   ├── final_combined.ipynb  # Canonical end-to-end notebook
│   └── archive/              # Earlier per-model notebooks
└── README.md
```

## Workflow

- **Run `notebooks/final_combined.ipynb`.** It loads `data/raw/`, writes cleaned splits to `data/processed/`, then trains and evaluates all models.
- `data/raw/` — the original dataset. **Read-only: never edit it.**
- `data/processed/` — cleaned / feature-engineered train–test splits used by the modeling sections.

## Business Impact

A well-calibrated model allows hotels to set overbooking levels proportional to predicted aggregate cancellation probability for a given night, replacing fixed historical rules with dynamic, booking-level risk scores.
