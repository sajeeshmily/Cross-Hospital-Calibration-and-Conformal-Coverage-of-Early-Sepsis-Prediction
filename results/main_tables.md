## Panel A. Discrimination, Utility and calibration

| Model | Evaluation | Test patients (septic) | AUROC | AUPRC | Utility (normalised) | Calibration slope | Calibration intercept | Brier score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GBM, values + mask (primary) | A→A (in-site) | 4,068 (351) | 0.781 (0.772–0.789) | 0.084 (0.079–0.091) | 0.335 (0.315–0.352) | 0.87 (0.82–0.91) | -0.44 (-0.60–-0.30) | 0.0204 (0.0196–0.0212) |
| GBM, values + mask (primary) | A→B (cross-site) | 12,000 (678) | 0.732 (0.725–0.740) | 0.047 (0.045–0.050) | 0.167 (0.155–0.180) | 0.55 (0.51–0.59) | -1.73 (-1.90–-1.56) | 0.0137 (0.0132–0.0141) |
| GBM, values + mask (primary) | B→B (in-site) | 4,000 (212) | 0.824 (0.813–0.836) | 0.080 (0.074–0.095) | 0.319 (0.292–0.339) | 0.92 (0.84–1.01) | -0.39 (-0.71–-0.11) | 0.0126 (0.0119–0.0133) |
| GBM, values + mask (primary) | B→A (cross-site) | 12,202 (1063) | 0.715 (0.709–0.721) | 0.053 (0.051–0.056) | 0.216 (0.204–0.227) | 0.61 (0.58–0.63) | -1.39 (-1.48–-1.31) | 0.0209 (0.0204–0.0213) |
| GRU-D (exploratory) | A→A (in-site) | 4,068 (351) | 0.730 (0.720–0.740) | 0.059 (0.055–0.065) | 0.234 (0.213–0.255) | 0.84 (0.77–0.89) | -0.59 (-0.81–-0.38) | 0.0206 (0.0198–0.0215) |
| GRU-D (exploratory) | A→B (cross-site) | 12,000 (678) | 0.701 (0.693–0.709) | 0.038 (0.036–0.041) | 0.093 (0.078–0.104) | 0.73 (0.67–0.79) | -1.17 (-1.40–-0.93) | 0.0137 (0.0132–0.0140) |
| GRU-D (exploratory) | B→B (in-site) | 4,000 (212) | 0.787 (0.776–0.799) | 0.060 (0.055–0.067) | 0.220 (0.195–0.240) | 0.96 (0.91–1.01) | -0.27 (-0.46–-0.08) | 0.0127 (0.0120–0.0134) |
| GRU-D (exploratory) | B→A (cross-site) | 12,202 (1063) | 0.673 (0.668–0.679) | 0.042 (0.041–0.044) | 0.130 (0.117–0.143) | 0.52 (0.50–0.54) | -1.76 (-1.85–-1.69) | 0.0213 (0.0208–0.0217) |

## Panel B. Conformal coverage and alert performance

| Model | Evaluation | Test patients (septic) | Coverage, septic (target 0.80) | Coverage, non-septic (target 0.95) | Sensitivity, % septic patients alerted | Alert-hours per non-septic patient-day | Non-septic patients alerted, % | Median lead time, h |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GBM, values + mask (primary) | A→A (in-site) | 4,068 (351) | 0.87 (0.84–0.90) | 0.95 (0.95–0.95) | 59 (56–61) | 1.01 (0.96–1.07) | 20 (19–20) | 30.7 (27.6–33.2) |
| GBM, values + mask (primary) | A→B (cross-site) | 12,000 (678) | 0.59 (0.55–0.62) | 0.99 (0.99–0.99) | 36 (35–38) | 0.43 (0.41–0.46) | 10 (9–10) | 35.3 (30.8–40.0) |
| GBM, values + mask (primary) | B→B (in-site) | 4,000 (212) | 0.87 (0.83–0.92) | 0.95 (0.95–0.96) | 65 (62–67) | 0.95 (0.89–1.01) | 17 (16–17) | 26.6 (22.0–31.6) |
| GBM, values + mask (primary) | B→A (cross-site) | 12,202 (1063) | 0.91 (0.90–0.93) | 0.87 (0.86–0.87) | 68 (67–69) | 2.58 (2.54–2.62) | 41 (41–42) | 34.9 (32.8–37.8) |
| GRU-D (exploratory) | A→A (in-site) | 4,068 (351) | 0.83 (0.80–0.87) | 0.95 (0.95–0.96) | 54 (52–56) | 1.01 (0.96–1.05) | 20 (19–21) | 30.2 (26.6–33.2) |
| GRU-D (exploratory) | A→B (cross-site) | 12,000 (678) | 0.59 (0.55–0.63) | 0.99 (0.99–0.99) | 25 (23–26) | 0.25 (0.24–0.27) | 7 (6–7) | 28.2 (23.6–32.8) |
| GRU-D (exploratory) | B→B (in-site) | 4,000 (212) | 0.86 (0.81–0.92) | 0.95 (0.95–0.96) | 65 (62–68) | 0.95 (0.91–1.00) | 21 (21–22) | 29.6 (23.0–35.8) |
| GRU-D (exploratory) | B→A (cross-site) | 12,202 (1063) | 0.95 (0.93–0.96) | 0.83 (0.82–0.83) | 74 (73–75) | 3.12 (3.08–3.16) | 52 (52–53) | 36.4 (34.0–38.6) |

Values are the mean of five seeds, each seed with its own patient-level split and retrained models, followed by the 95% interval (percentile range of the seed-averaged patient-level bootstrap replicates, 200 per seed). Intervals reflect resampling of test patients; the spread across seeds is larger and is given as SD in supplementary_all_models.csv. Training hospital split 60/10/10/20% (train, isotonic, threshold and conformal calibration, in-site test); the other hospital contributes a 60% test set. AUROC and AUPRC are pooled over hourly rows. Calibration is measured on isotonic-recalibrated probabilities. Utility is normalised (silent model 0, oracle 1) at the threshold that maximises Utility on the training-hospital calibration patients. Alert results use the threshold giving 1 alert-hour per non-septic patient-day at the training hospital; sensitivity is the share of septic patients alerted before onset, lead time the median hours before onset of the first alert among alerted septic patients. Conformal coverage is patient-level class-conditional split conformal at a fixed prediction time of 24 h: one score per patient, the highest model score over hours up to 24, for patients still in the ICU at hour 24 with no sepsis onset by then. The patient-level outcome used to evaluate the sets is septic = sepsis onset within the next 24 h (hours 25 to 48), non-septic = no onset within those 24 h; it is a different quantity from the hourly training label (positive from 6 h before onset) and is used only as the evaluation label. A coverage guarantee applies in-site only (exchangeable eligible patients, that single time and outcome); values at the other hospital are measurements under shift. Interval from resampling test patients given the calibration draw.
Tables 6 and 7 come from the remedy experiments (hour-pooled coverage, five seeds x five random draws of the labeled sample; medians for calibration parameters, means and SD for coverage). They redraw the calibration hour independently of Table 4, so hour-pooled septic coverage can differ between tables by up to about 0.03, within the split-to-split SD. The source-only calibration slope in Table 6 is a median over seeds; Table 1 gives the mean.

## Table 2. AUROC, all models

| Model | A→A (in-site) | A→B (cross-site) | B→B (in-site) | B→A (cross-site) |
| --- | --- | --- | --- | --- |
| GBM, values + mask + temporal | 0.786 (0.777–0.795) | 0.743 (0.736–0.750) | 0.829 (0.817–0.839) | 0.723 (0.717–0.729) |
| GBM, values + mask (primary) | 0.781 (0.772–0.789) | 0.732 (0.725–0.740) | 0.824 (0.813–0.836) | 0.715 (0.709–0.721) |
| GBM, values only | 0.750 (0.740–0.759) | 0.702 (0.694–0.711) | 0.789 (0.775–0.801) | 0.690 (0.684–0.697) |
| GBM, mask and timing only | 0.717 (0.706–0.727) | 0.598 (0.590–0.607) | 0.731 (0.717–0.744) | 0.617 (0.610–0.623) |
| MLP, temporal features | 0.741 (0.732–0.751) | 0.662 (0.656–0.671) | 0.751 (0.739–0.765) | 0.646 (0.640–0.651) |
| GRU-D (exploratory) | 0.730 (0.720–0.740) | 0.701 (0.693–0.709) | 0.787 (0.776–0.799) | 0.673 (0.668–0.679) |
| GRU, no decay (exploratory) | 0.728 (0.718–0.737) | 0.702 (0.695–0.710) | 0.790 (0.779–0.802) | 0.671 (0.665–0.676) |
| SIRS | 0.619 (0.610–0.629) | 0.670 (0.663–0.677) | 0.663 (0.651–0.675) | 0.629 (0.622–0.636) |
| qSOFA (partial) | 0.564 (0.556–0.572) | 0.579 (0.573–0.585) | 0.574 (0.565–0.587) | 0.575 (0.570–0.580) |
| NEWS (partial) | 0.644 (0.634–0.654) | 0.696 (0.688–0.703) | 0.695 (0.683–0.706) | 0.652 (0.646–0.658) |
| SOFA (partial) | 0.560 (0.549–0.572) | 0.629 (0.621–0.638) | 0.630 (0.616–0.645) | 0.558 (0.551–0.564) |

## Table 3. Normalised Utility, all models

| Model | A→A (in-site) | A→B (cross-site) | B→B (in-site) | B→A (cross-site) |
| --- | --- | --- | --- | --- |
| GBM, values + mask + temporal | 0.342 (0.320–0.359) | 0.179 (0.165–0.192) | 0.324 (0.298–0.345) | 0.232 (0.218–0.245) |
| GBM, values + mask (primary) | 0.335 (0.315–0.352) | 0.167 (0.155–0.180) | 0.319 (0.292–0.339) | 0.216 (0.204–0.227) |
| GBM, values only | 0.278 (0.256–0.298) | 0.112 (0.097–0.125) | 0.239 (0.209–0.265) | 0.181 (0.166–0.193) |
| GBM, mask and timing only | 0.218 (0.197–0.237) | 0.034 (0.026–0.044) | 0.179 (0.149–0.205) | 0.077 (0.061–0.088) |
| MLP, temporal features | 0.265 (0.244–0.285) | 0.086 (0.073–0.097) | 0.204 (0.182–0.226) | 0.104 (0.090–0.114) |
| GRU-D (exploratory) | 0.234 (0.213–0.255) | 0.093 (0.078–0.104) | 0.220 (0.195–0.240) | 0.130 (0.117–0.143) |
| GRU, no decay (exploratory) | 0.241 (0.218–0.262) | 0.088 (0.073–0.101) | 0.223 (0.196–0.246) | 0.130 (0.116–0.143) |
| SIRS | 0.078 (0.053–0.097) | 0.019 (-0.002–0.033) | 0.066 (0.047–0.082) | 0.083 (0.074–0.090) |
| qSOFA (partial) | 0.006 (-0.018–0.027) | -0.146 (-0.167–-0.127) | 0.010 (-0.000–0.018) | 0.015 (0.010–0.019) |
| NEWS (partial) | 0.104 (0.083–0.124) | 0.075 (0.057–0.089) | 0.086 (0.066–0.108) | 0.094 (0.086–0.103) |
| SOFA (partial) | 0.028 (0.009–0.047) | 0.001 (-0.018–0.018) | 0.011 (-0.009–0.030) | 0.032 (0.023–0.038) |

## Table 4. Conformal coverage

| Model | Direction and class | Target | Patient-level at 24 h, training hospital | Patient-level at 24 h, other hospital | Hours pooled, training hospital | Hours pooled, other hospital |
| --- | --- | --- | --- | --- | --- | --- |
| GBM, values + mask (primary) | A→B, septic | 0.80 | 0.87 (0.84–0.90) | 0.59 (0.55–0.62) | 0.78 (0.77–0.80) | 0.55 (0.54–0.57) |
| GBM, values + mask (primary) | A→B, non-septic | 0.95 | 0.95 (0.95–0.95) | 0.99 (0.99–0.99) | 0.93 (0.93–0.93) | 0.97 (0.97–0.97) |
| GBM, values + mask (primary) | B→A, septic | 0.80 | 0.87 (0.83–0.92) | 0.91 (0.90–0.93) | 0.79 (0.77–0.81) | 0.78 (0.78–0.79) |
| GBM, values + mask (primary) | B→A, non-septic | 0.95 | 0.95 (0.95–0.96) | 0.87 (0.86–0.87) | 0.93 (0.92–0.93) | 0.84 (0.83–0.84) |
| GRU-D (exploratory) | A→B, septic | 0.80 | 0.83 (0.80–0.87) | 0.59 (0.55–0.63) | 0.78 (0.77–0.80) | 0.63 (0.62–0.65) |
| GRU-D (exploratory) | A→B, non-septic | 0.95 | 0.95 (0.95–0.96) | 0.99 (0.99–0.99) | 0.94 (0.94–0.95) | 0.99 (0.99–0.99) |
| GRU-D (exploratory) | B→A, septic | 0.80 | 0.86 (0.81–0.92) | 0.95 (0.93–0.96) | 0.82 (0.80–0.84) | 0.83 (0.82–0.83) |
| GRU-D (exploratory) | B→A, non-septic | 0.95 | 0.95 (0.95–0.96) | 0.83 (0.82–0.83) | 0.94 (0.94–0.95) | 0.84 (0.84–0.84) |

## Table 5. Guarantee check

| Model | Source hospital | Calibration patients | Septic coverage, mean (SD) over splits | Non-septic coverage, mean (SD) over splits |
| --- | --- | --- | --- | --- |
| GBM, values + mask (primary) | A | 1,528 | 0.811 (0.073) | 0.950 (0.006) |
| GRU-D (exploratory) | A | 1,528 | 0.809 (0.073) | 0.950 (0.006) |
| GBM, values + mask (primary) | B | 1,438 | 0.824 (0.102) | 0.950 (0.007) |
| GRU-D (exploratory) | B | 1,438 | 0.816 (0.104) | 0.950 (0.007) |

## Table A2. Fixed prediction time sensitivity (patient-level coverage, training hospital / other hospital)

| Model | Direction and class | Target | T = 12 h: training / other hospital | T = 24 h: training / other hospital | T = 48 h: training / other hospital |
| --- | --- | --- | --- | --- | --- |
| GBM, values + mask (primary) | A→B, septic | 0.80 | 0.82 / 0.57 | 0.87 / 0.59 | 0.75 / 0.44 |
| GBM, values + mask (primary) | A→B, non-septic | 0.95 | 0.95 / 0.98 | 0.95 / 0.99 | 0.95 / 0.99 |
| GBM, values + mask (primary) | B→A, septic | 0.80 | 0.91 / 0.91 | 0.87 / 0.91 | 0.77 / 0.83 |
| GBM, values + mask (primary) | B→A, non-septic | 0.95 | 0.95 / 0.84 | 0.95 / 0.87 | 0.95 / 0.87 |
| GRU-D (exploratory) | A→B, septic | 0.80 | 0.79 / 0.57 | 0.83 / 0.59 | 0.81 / 0.60 |
| GRU-D (exploratory) | A→B, non-septic | 0.95 | 0.95 / 0.99 | 0.95 / 0.99 | 0.96 / 1.00 |
| GRU-D (exploratory) | B→A, septic | 0.80 | 0.85 / 0.92 | 0.86 / 0.95 | 0.77 / 0.84 |
| GRU-D (exploratory) | B→A, non-septic | 0.95 | 0.96 / 0.83 | 0.95 / 0.83 | 0.95 / 0.86 |

## Table A3. Eligible patients by prediction time (gradient boosting, mean over seeds)

| Direction | Prediction time | Calibration patients (septic / non-septic) | Test patients, training hospital (septic / non-septic) | Test patients, other hospital (septic / non-septic) |
| --- | --- | --- | --- | --- |
| A→B | 12 h | 53 / 1,911 | 97 / 3,830 | 147 / 11,380 |
| A→B | 24 h | 34 / 1,501 | 73 / 2,985 | 119 / 8,506 |
| A→B | 48 h | 26 / 431 | 41 / 813 | 85 / 2,391 |
| B→A | 12 h | 24 / 1,897 | 42 / 3,798 | 289 / 11,493 |
| B→A | 24 h | 20 / 1,433 | 36 / 2,838 | 203 / 8,986 |
| B→A | 48 h | 11 / 408 | 27 / 787 | 131 / 2,496 |

## Table A4. Seed-level intervals for patient-level coverage at 24 h

| Model | Direction and class | Target | Patient-level at 24 h, training hospital: mean (seed SD; 95% interval over seeds) | Patient-level at 24 h, other hospital: mean (seed SD; 95% interval over seeds) |
| --- | --- | --- | --- | --- |
| GBM, values + mask (primary) | A→B, septic | 0.80 | 0.87 (0.09; 0.76–0.98) | 0.59 (0.12; 0.44–0.74) |
| GBM, values + mask (primary) | A→B, non-septic | 0.95 | 0.95 (0.00; 0.95–0.96) | 0.99 (0.01; 0.98–1.00) |
| GBM, values + mask (primary) | B→A, septic | 0.80 | 0.87 (0.07; 0.79–0.96) | 0.91 (0.06; 0.84–0.98) |
| GBM, values + mask (primary) | B→A, non-septic | 0.95 | 0.95 (0.01; 0.94–0.96) | 0.87 (0.04; 0.82–0.91) |
| GRU-D (exploratory) | A→B, septic | 0.80 | 0.83 (0.07; 0.74–0.92) | 0.59 (0.15; 0.41–0.78) |
| GRU-D (exploratory) | A→B, non-septic | 0.95 | 0.95 (0.00; 0.95–0.96) | 0.99 (0.00; 0.99–1.00) |
| GRU-D (exploratory) | B→A, septic | 0.80 | 0.86 (0.09; 0.75–0.97) | 0.95 (0.03; 0.91–0.98) |
| GRU-D (exploratory) | B→A, non-septic | 0.95 | 0.95 (0.01; 0.94–0.97) | 0.83 (0.07; 0.74–0.92) |

## Table A5. Prediction-target sensitivity (horizon of the septic class)

| Model | Direction and class | Target | Septic calibration patients (6 h / 24 h / rest of stay) | Septic = onset within 6 h of T (training-label horizon): training / other hospital | Septic = onset within 24 h of T (main): training / other hospital | Septic = onset any time after T: training / other hospital |
| --- | --- | --- | --- | --- | --- | --- |
| GBM, values + mask (primary) | A→B, septic | 0.80 | 11 / 34 / 107 | 0.83 / 0.62 | 0.87 / 0.59 | 0.79 / 0.51 |
| GBM, values + mask (primary) | A→B, non-septic | 0.95 | 11 / 34 / 107 | 0.95 / 0.99 | 0.95 / 0.99 | 0.95 / 0.99 |
| GBM, values + mask (primary) | B→A, septic | 0.80 | 4 / 20 / 61 | 0.81 / 0.91 | 0.87 / 0.91 | 0.84 / 0.90 |
| GBM, values + mask (primary) | B→A, non-septic | 0.95 | 4 / 20 / 61 | 0.95 / 0.87 | 0.95 / 0.87 | 0.96 / 0.87 |
| GRU-D (exploratory) | A→B, septic | 0.80 | 11 / 34 / 107 | 0.87 / 0.74 | 0.83 / 0.59 | 0.84 / 0.64 |
| GRU-D (exploratory) | A→B, non-septic | 0.95 | 11 / 34 / 107 | 0.95 / 0.99 | 0.95 / 0.99 | 0.95 / 0.99 |
| GRU-D (exploratory) | B→A, septic | 0.80 | 4 / 20 / 61 | 0.84 / 0.97 | 0.86 / 0.95 | 0.85 / 0.91 |
| GRU-D (exploratory) | B→A, non-septic | 0.95 | 4 / 20 / 61 | 0.95 / 0.83 | 0.95 / 0.83 | 0.95 / 0.83 |

## Table A6. Prediction sets at 24 h (gradient boosting)

| Direction | Test hospital | True class | {non-septic} only, % | {septic, non-septic}, % | {septic} only, % |
| --- | --- | --- | --- | --- | --- |
| A→B | A (training hospital) | septic | 13 | 66 | 22 |
| A→B | A (training hospital) | non-septic | 44 | 51 | 5 |
| A→B | B (other hospital) | septic | 41 | 55 | 4 |
| A→B | B (other hospital) | non-septic | 61 | 38 | 1 |
| B→A | B (training hospital) | septic | 13 | 69 | 19 |
| B→A | B (training hospital) | non-septic | 40 | 56 | 5 |
| B→A | A (other hospital) | septic | 9 | 67 | 25 |
| B→A | A (other hospital) | non-septic | 23 | 64 | 13 |

## Table 6. Labeled target patients

| Model | Direction | Labeled target patients | Septic patients in sample | Median calibration slope (IQR) | Septic-hour coverage, mean (SD) | Non-septic-hour coverage, mean (SD) | Sensitivity after threshold reset | Alert-hours per day after reset |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GBM, values + mask (primary) | A→B | 0 (source only) | 0 | 0.63 | 0.54 | 0.97 |  |  |
| GBM, values + mask (primary) | A→B | 100 | 6 | 0.94 (0.73–1.25) | 0.90 (0.10) | 0.93 (0.03) | 0.81 | 13.0 |
| GBM, values + mask (primary) | A→B | 200 | 11 | 0.95 (0.63–1.49) | 0.86 (0.08) | 0.93 (0.02) | 0.80 | 12.1 |
| GBM, values + mask (primary) | A→B | 400 | 23 | 1.07 (0.77–1.29) | 0.84 (0.08) | 0.93 (0.02) | 0.81 | 12.4 |
| GBM, values + mask (primary) | A→B | 800 | 46 | 1.14 (0.90–1.39) | 0.83 (0.07) | 0.93 (0.01) | 0.82 | 12.5 |
| GBM, values + mask (primary) | A→B | 1,600 | 92 | 0.97 (0.92–1.15) | 0.81 (0.05) | 0.93 (0.01) | 0.81 | 12.0 |
| GBM, values + mask (primary) | A→B | 3,200 | 184 | 1.05 (0.93–1.15) | 0.82 (0.04) | 0.93 (0.01) | 0.81 | 12.1 |
| GBM, values + mask (primary) | B→A | 0 (source only) | 0 | 0.61 | 0.81 | 0.84 |  |  |
| GBM, values + mask (primary) | B→A | 100 | 8 | 1.05 (0.82–1.37) | 0.89 (0.09) | 0.94 (0.03) | 0.79 | 11.6 |
| GBM, values + mask (primary) | B→A | 200 | 18 | 1.10 (0.85–1.43) | 0.86 (0.07) | 0.95 (0.02) | 0.80 | 11.8 |
| GBM, values + mask (primary) | B→A | 400 | 35 | 1.22 (0.99–1.41) | 0.86 (0.05) | 0.94 (0.01) | 0.82 | 12.3 |
| GBM, values + mask (primary) | B→A | 800 | 70 | 1.12 (1.02–1.27) | 0.83 (0.04) | 0.94 (0.01) | 0.82 | 12.3 |
| GBM, values + mask (primary) | B→A | 1,600 | 139 | 1.07 (1.01–1.16) | 0.83 (0.03) | 0.93 (0.01) | 0.82 | 12.2 |
| GBM, values + mask (primary) | B→A | 3,200 | 281 | 1.06 (1.01–1.09) | 0.82 (0.02) | 0.94 (0.00) | 0.82 | 12.1 |
| GRU-D (exploratory) | A→B | 0 (source only) | 0 | 0.80 | 0.64 | 0.99 |  |  |
| GRU-D (exploratory) | A→B | 100 | 6 | 0.88 (0.61–1.50) | 0.88 (0.14) | 0.95 (0.02) | 0.77 | 12.4 |
| GRU-D (exploratory) | A→B | 800 | 46 | 1.02 (0.86–1.33) | 0.84 (0.04) | 0.94 (0.01) | 0.82 | 13.1 |
| GRU-D (exploratory) | B→A | 0 (source only) | 0 | 0.50 | 0.83 | 0.84 |  |  |
| GRU-D (exploratory) | B→A | 100 | 8 | 1.06 (0.64–1.56) | 0.87 (0.10) | 0.94 (0.02) | 0.78 | 12.8 |
| GRU-D (exploratory) | B→A | 800 | 70 | 1.14 (1.01–1.25) | 0.82 (0.05) | 0.95 (0.01) | 0.81 | 13.3 |

## Table 7. Label-free weighting

| Model | Direction | Septic-hour coverage, source only | Septic-hour coverage, weighted | Non-septic-hour coverage, source only | Non-septic-hour coverage, weighted | Effective sample size, septic calibration points |
| --- | --- | --- | --- | --- | --- | --- |
| GBM, values + mask (primary) | A→B | 0.54 | 0.74 | 0.97 | 0.92 | 3.4 of 184 |
| GBM, values + mask (primary) | B→A | 0.81 | 0.81 | 0.84 | 0.69 | 6.4 of 109 |
| GRU-D (exploratory) | A→B | 0.64 | 0.78 | 0.99 | 0.91 | 3.4 of 184 |
| GRU-D (exploratory) | B→A | 0.83 | 0.83 | 0.84 | 0.70 | 6.4 of 109 |
