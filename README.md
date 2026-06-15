# Hotel Cancellation Prediction System

**MIDS 207 — Summer 2026**

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
| `market_segment` | Distribution channel (Online TA, Direct, Corporate, etc.) |
| `deposit_type` | No deposit, Non-refundable, or Refundable |
| `previous_cancellations` | Historical cancellation behavior of the guest |
| `booking_changes` | Number of modifications made to the booking |
| `reserved_room_type` | Room category requested |
| `adr` | Average daily rate |

**Target variable:** `is_canceled` (binary: 0 = stayed, 1 = cancelled)

## Technical Approach

- **Task:** Binary classification
- **Models:** Logistic Regression (baseline) → Gradient Boosting (XGBoost / LightGBM)
- **Evaluation:** ROC-AUC, Precision-Recall AUC, calibrated probability scores
- **Key concern:** Class imbalance handling; model calibration matters more than raw accuracy since the output drives a business decision (overbooking level), not a hard label

## Project Structure

```
hotel_cancellation_model_207/
├── data/
│   ├── raw/            # Original data — read-only, do not edit
│   └── processed/      # Outputs produced by notebook jobs
├── notebooks/          # All work happens here (e.g. data-processing.ipynb)
├── src/                # Feature engineering and model pipeline
├── models/             # Saved model artifacts
└── README.md
```

## Workflow

- **Do the work in notebooks.** Each job is a notebook under `notebooks/*.ipynb`.
- **Save each job's result to `data/`.** A notebook reads its input and writes
  its output back to `data/` so the next job can pick it up.
  - `data/raw/` — the original dataset. **Read-only: never edit it.**
  - `data/processed/` — cleaned / feature-engineered outputs from notebook jobs.

## Business Impact

A well-calibrated model allows hotels to set overbooking levels proportional to predicted aggregate cancellation probability for a given night, replacing fixed historical rules with dynamic, booking-level risk scores.
