# MIDUS Dataset - Data Dictionary

## Overview
This document describes the variables in the MIDUS (Midlife in the United States) processed dataset, which focuses on stress, inflammation, and cognitive function relationships in aging.

## Dataset Information
- **Source**: MIDUS Biomarker Study
- **Total Variables**: 33
- **Observations**: Variable (see dataset)
- **Study Focus**: Relationships between stress biomarkers, inflammation markers, and cognitive performance over time

---

## Variable Definitions

### Identification Variables

| Variable | Full Name | Description | Type | Typical Values |
|----------|-----------|-------------|------|----------------|
| `M2ID` | MIDUS 2 ID | Unique participant identifier | Integer | 10000-101500 |
| `M2FAMNUM` | MIDUS 2 Family Number | Family unit identifier | Integer | 100000-120000 |
| `SAMPLMAJ` | Sample Major Group | Study sample category | Integer | 1-4 |

**Notes**: 
- SAMPLMAJ: 1=Main sample, 2=Sibling sample, 3=Twin sample, 4=Milwaukee sample

---

### Demographic Variables

| Variable | Full Name | Description | Type | Typical Values | Units |
|----------|-----------|-------------|------|----------------|-------|
| `B1PAGE_M` | Baseline Age | Age at MIDUS 1 baseline assessment | Numeric | 34-80 | years |
| `B1PGENDE` | Baseline Gender | Participant gender | Integer | 1, 2 | 1=Male, 2=Female |

**Scaled versions**: `B1PAGE_M_scaled`, `B1PGENDE_scaled`

---

### Stress & HPA Axis Biomarkers (Log-transformed)

| Variable | Full Name | Description | Type | Typical Values | Original Units |
|----------|-----------|-------------|------|----------------|----------------|
| `M2_LOG_C` | Log Cortisol | Natural log of cortisol levels | Numeric | -2.0 to 3.0 | log(ng/mL) |
| `V7_A` | Log DHEA | Natural log of DHEA levels | Numeric | -2.0 to 4.0 | log(ng/mL) |
| `LOG_M2_N` | Log Norepinephrine | Natural log of norepinephrine | Numeric | 2.0 to 5.0 | log(pg/mL) |

**Clinical Context**:
- **Cortisol**: Primary stress hormone, higher values indicate chronic stress
- **DHEA**: Neurosteroid with neuroprotective properties
- **Norepinephrine**: Sympathetic nervous system marker

**Scaled versions**: `M2_LOG_C_scaled`, `V7_A_scaled`, `LOG_M2_N_scaled`

---

### Inflammation Biomarkers (Original Scale)

| Variable | Full Name | Description | Type | Typical Values | Units |
|----------|-----------|-------------|------|----------------|-------|
| `M2_LOG_I` | Log IL-6 | Natural log of Interleukin-6 | Numeric | -1.5 to 2.5 | log(pg/mL) |
| `M2_DOPAM` | Dopamine | Dopamine levels | Numeric | 40-400 | pg/mL |
| `M2_EPINE` | Epinephrine | Epinephrine (adrenaline) levels | Numeric | 0.4-6.0 | pg/mL |
| `M2_FIBRI` | Fibrinogen | Fibrinogen concentration | Numeric | 130-550 | mg/dL |
| `M2_SICAM` | sICAM-1 | Soluble Intercellular Adhesion Molecule-1 | Numeric | 120-500 | ng/mL |

**Clinical Context**:
- **IL-6**: Pro-inflammatory cytokine, elevated in chronic inflammation
- **Fibrinogen**: Acute phase protein, cardiovascular risk marker
- **sICAM-1**: Endothelial dysfunction marker
- **Dopamine/Epinephrine**: Catecholamines, stress response markers

**Scaled versions**: `M2_DOPAM_scaled`, `M2_EPINE_scaled`, `M2_FIBRI_scaled`, `M2_SICAM_scaled`, `M2_LOG_I_scaled`

---

### Cognitive Function (Standardized Z-scores)

#### MIDUS 2 (Baseline Cognition)

| Variable | Full Name | Description | Type | Typical Values | Interpretation |
|----------|-----------|-------------|------|----------------|----------------|
| `M2_EPISO` | Episodic Memory M2 | Episodic memory z-score at MIDUS 2 | Numeric | -3.0 to 3.0 | Higher = better memory |
| `M2_EXECU` | Executive Function M2 | Executive function z-score at MIDUS 2 | Numeric | -3.0 to 3.0 | Higher = better function |

#### MIDUS 3 (Follow-up Cognition)

| Variable | Full Name | Description | Type | Typical Values | Interpretation |
|----------|-----------|-------------|------|----------------|----------------|
| `M3_EPISO` | Episodic Memory M3 | Episodic memory z-score at MIDUS 3 | Numeric | -3.0 to 3.0 | Higher = better memory |
| `M3_EXECU` | Executive Function M3 | Executive function z-score at MIDUS 3 | Numeric | -3.0 to 3.0 | Higher = better function |

**Clinical Context**:
- **Z-scores**: Standardized scores (mean=0, SD=1)
  - Z > 0.5: Above average performance
  - -0.5 < Z < 0.5: Average performance
  - Z < -0.5: Below average performance
- **Episodic Memory**: Ability to recall specific events and experiences
- **Executive Function**: Planning, cognitive flexibility, working memory

---

### Derived Features - Longitudinal Change

| Variable | Full Name | Description | Type | Typical Values | Interpretation |
|----------|-----------|-------------|------|----------------|----------------|
| `episodic_change` | Episodic Memory Change | Change in episodic memory (M3 - M2) | Numeric | -3.0 to 3.0 | Positive = improvement |
| `executive_change` | Executive Function Change | Change in executive function (M3 - M2) | Numeric | -3.0 to 3.0 | Positive = improvement |

**Clinical Context**:
- **Change > 0.5**: Meaningful cognitive improvement
- **-0.5 to 0.5**: Stable cognitive function
- **Change < -0.5**: Meaningful cognitive decline

---

### Derived Features - Categorical Variables

#### Age Groups

| Variable | Full Name | Description | Type | Categories |
|----------|-----------|-------------|------|------------|
| `age_group` | Age Category | Age group classification | Categorical | Young, Middle, Mature, Older |

**Age Ranges**:
- **Young (0)**: 34-45 years
- **Middle (1)**: 46-55 years
- **Mature (2)**: 56-65 years
- **Older (3)**: 66-80 years

#### Cognitive Profiles

| Variable | Full Name | Description | Type | Categories |
|----------|-----------|-------------|------|------------|
| `cognitive_profile` | Cognitive Profile Type | Cognitive performance pattern | Categorical | Normal, High, Low, Dissociated |

**Profile Definitions**:
- **Normal**: Both memory and executive function z-scores between -0.5 and 0.5
- **High**: Both memory and executive function z-scores > 0.5
- **Low**: Both memory and executive function z-scores < -0.5
- **Dissociated**: One domain high (>0.5) and other low (<-0.5)

---

### Derived Features - Composite Indices

| Variable | Full Name | Description | Type | Typical Values | Components |
|----------|-----------|-------------|------|----------------|------------|
| `inflammation_index` | Systemic Inflammation Index | Composite inflammation marker | Numeric | -2.0 to 2.0 | IL-6, Fibrinogen, sICAM-1 |
| `stress_index` | Stress Biomarker Index | Composite stress marker | Numeric | -2.0 to 2.0 | Cortisol, Norepinephrine, Epinephrine |

**Calculation**: 
- Standardized mean of component variables (z-score transformation)
- Higher values indicate greater inflammation/stress burden

**Clinical Interpretation**:
- **Index > 1.0**: High inflammatory/stress burden
- **-1.0 to 1.0**: Normal range
- **Index < -1.0**: Low inflammatory/stress burden

---

## Data Processing Notes

### Outlier Handling
- **Method**: Values capped at 99th percentile
- **Variables affected**: Original biomarker measurements
- **Rationale**: Preserve extreme but plausible biological values

### Scaling
- **Method**: Robust scaling (median and IQR-based)
- **Applied to**: Demographics, log-transformed biomarkers, original biomarkers
- **Not scaled**: Cognitive z-scores (already standardized)
- **Scaled variables suffix**: `_scaled`

### Missing Data
- Variables may contain missing values (NaN)
- Missing data not imputed in this version
- Analysis should handle missing data appropriately

---

## Usage Recommendations

### For Classification Tasks
**Target variables**:
- `cognitive_profile`: Multi-class classification (4 categories)
- `age_group`: Age stratification

### For Regression Tasks
**Target variables**:
- `episodic_change`: Predict memory trajectory
- `executive_change`: Predict executive function trajectory
- `M3_EPISO`, `M3_EXECU`: Predict future cognitive performance

### Key Predictors
**Biomarkers**:
- `inflammation_index`: Overall inflammatory burden
- `stress_index`: Overall stress burden
- Individual biomarkers for specific hypotheses

**Demographics**:
- `B1PAGE_M`: Age effects
- `B1PGENDE`: Sex differences

---

## Study Timeline

| Assessment | Timepoint | Variables Prefix |
|------------|-----------|------------------|
| MIDUS 1 | Baseline | B1 |
| MIDUS 2 | ~9-10 years later | M2 |
| MIDUS 3 | ~9-10 years after M2 | M3 |

---

## Clinical Significance

### Biomarker Reference Ranges

**Inflammation Markers** (clinical cutoffs):
- IL-6: Normal < 1.8 pg/mL
- Fibrinogen: Normal 200-400 mg/dL
- sICAM-1: Normal 150-300 ng/mL

**Stress Hormones** (typical ranges):
- Cortisol: 5-25 µg/dL (morning)
- Norepinephrine: 80-520 pg/mL
- Epinephrine: 10-200 pg/mL

**Note**: Values in dataset are log-transformed or adjusted; refer to original units for clinical interpretation.

---

## Citation

If using this dataset, please cite:
- MIDUS Biomarker Project
- Original data source: Harvard Dataverse - "Stress, Inflammation, and Cognition"

---

## Contact & Support

For questions about variable definitions or data processing:
- See preprocessing pipeline: [`src/preprocessing.py`](src/preprocessing.py)
- Review main analysis script: [`main.py`](main.py)
- Check visualization outputs: [`results`](results) directory

---

**Last Updated**: October 31, 2025  
**Dataset Version**: Processed v1.0
