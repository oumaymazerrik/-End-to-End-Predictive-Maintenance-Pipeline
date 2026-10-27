# ✈️ End-to-End Predictive Maintenance Pipeline

**NASA C-MAPSS FD001 | Apache Spark | PySpark | Spark MLlib | Structured Streaming**

An end-to-end predictive-maintenance project for estimating the **Remaining Useful Life (RUL)** of aircraft turbofan engines. The pipeline processes historical sensor readings using a **Bronze → Silver → Gold** architecture, trains regression models, and simulates live predictions and maintenance alerts.

**Environment used:** Google Colab (CPU), Apache Spark 3.5.0

## Project objectives

- Ingest and validate raw engine sensor measurements.
- Compute true RUL labels and select informative sensors.
- Engineer time-window features for regression.
- Compare Linear Regression, Random Forest and Gradient-Boosted Trees with an **engine-level** validation split.
- Evaluate the selected model on the last observed cycle of each test engine.
- Simulate Structured Streaming predictions and trigger maintenance alerts.
- Export normalized sensor data and model baseline metrics for a future Deep Learning lab.

## Dataset

The [NASA C-MAPSS dataset](https://data.nasa.gov/dataset/c-mapss-aircraft-engine-simulator-data) simulates turbofan engine degradation. We use **FD001**, which has one operating condition and one fault mode.

| Input file | Purpose | Observations |
|---|---|---:|
| `train_FD001.txt` | Run-to-failure training histories (100 engines) | 20,631 |
| `test_FD001.txt` | Partial histories for 100 unseen test engines | 13,096 |
| `RUL_FD001.txt` | True remaining life at the last test cycle | 100 |

Each sensor record has **26 raw columns**: `unit_id`, `cycle`, three operational settings (`setting_1`–`setting_3`) and 21 sensors (`s1`–`s21`). The RUL file does not contain engine IDs; the line position corresponds to the engine ID starting from 1.

For a training engine:

`rul = max_cycle_of_engine - cycle`

`rul_capped = min(rul, 125)`

For a test engine:

`rul = true_rul_at_last_observation + (last_recorded_cycle - cycle)`

## Pipeline architecture

```text
NASA C-MAPSS .txt
        |
        v
[Bronze] Typed ingestion + metadata -> Parquet
        |
        v
[Silver] Sensor selection + RUL labels -> Parquet
        |
        +----------------------------------> [DL-ready] Min-max sensor data
        |                                              + normalization.json
        v
[Gold] 10-cycle rolling features -> Parquet
        |
        v
[Spark MLlib] Engine-level validation + model selection
        |                         |
        |                         +-> baseline_metrics.json
        v
[Saved PipelineModel]
        |
        v
[Structured Streaming] JSON micro-batches -> RUL predictions + alerts
```

## Technologies

| Component | Technology |
|---|---|
| Notebook / execution | Python, Jupyter, Google Colab (CPU) |
| Distributed data processing | Apache Spark 3.5.0 / PySpark |
| Exploration | Spark SQL, Pandas, Matplotlib |
| Data storage | Apache Parquet, JSON |
| Machine Learning | Spark MLlib |
| Streaming simulation | Spark Structured Streaming (`foreachBatch`) |

## Pipeline implementation

### Part 1 — Bronze: ingestion

1. Read raw text files with `spark.read.text`.
2. Trim and split whitespace-separated values; cast engine IDs and cycles to integers and measurements to doubles.
3. Add `dataset`, `split`, `source_file` and `ingested_at` metadata.
4. Read the RUL truth file through the RDD API and use `zipWithIndex()` to generate engine IDs.
5. Save and reload the three Bronze datasets as Parquet.

**Checks passed:** 26 raw columns, expected data types, 20,631 train records, 13,096 test records, and 100 RUL labels.

### Part 2 — Silver: cleaning and labeling

1. Explore engine lifetimes using Spark SQL.
2. Compute standard deviations for all settings and sensors in one aggregation.
3. Drop nearly constant variables using `STD_THRESHOLD = 0.01`; retain **14 informative sensors** for FD001.
4. Calculate training RUL with `Window.partitionBy("unit_id")`.
5. Join test readings to RUL truth; use a broadcast hint for the small RUL table.
6. Cap the training target at 125 and save the labeled Silver data to Parquet.

**Note:** Feature selection was derived from FD001 training data. It should be recalculated for other operating-condition subsets.

### Part 3 — Gold: feature engineering

For each retained sensor, use the current observation and up to nine preceding cycles from the **same engine** to compute:

- Raw sensor value.
- Rolling mean over 10 cycles.
- Rolling standard deviation over 10 cycles (first-cycle null filled with zero).

Together with `cycle`, this produces **43 predictive features**: `1 + 14 + 14 + 14`.

Created:

- `gold/train` — 20,631 training observations.
- `gold/test` — 13,096 test observations.
- `gold/test_last` — 100 records, one last observed cycle per test engine.

**Spark optimization experiment:**

| Physical plan operator | Without broadcast | With broadcast |
|---|---:|---:|
| Shuffles (`Exchange hashpartitioning`) | 3 | 1 |
| `BroadcastExchange` | 0 | 1 |
| `SortMergeJoin` | 1 | 0 |
| `BroadcastHashJoin` | 0 | 1 |
| `Window` | 1 | 1 |

The original RUL join key had a different integer type from the sensor dataset. Aligning the types reduced the non-broadcast shuffle count **from 3 to 2**.

### Part 4 — Spark MLlib: training and evaluation

The models predict **`rul_capped`** using a pipeline:

`VectorAssembler -> StandardScaler(withMean=True, withStd=True) -> Regressor`

Candidates:

- **Linear Regression**
- **Random Forest:** 100 trees, maximum depth 10
- **Gradient-Boosted Trees (GBT):** 80 iterations, maximum depth 5, step size 0.1

A seeded split holds out **20 complete engines** for validation and uses the remaining **80 engines** for training. Splitting by row would leak nearly identical cycles of the same engine into both sets.

Metrics:

- **RMSE:** root mean squared error (cycles).
- **MAE:** mean absolute error (cycles).
- **NASA Score:** asymmetric penalty, higher for RUL overestimation (late maintenance).

**Best reported validation RMSE: approximately 15.9 cycles** (below the assignment's 25-cycle check).

#### Optional Random Forest tuning

| Max depth | Trees | Validation RMSE | Validation MAE | NASA Score |
|---:|---:|---:|---:|---:|
| 6 | 60 | 16.427 | 11.014 | 37,881.28 |
| 6 | 120 | 16.446 | 11.025 | 37,604.48 |
| 4 | 120 | 17.298 | 12.385 | 42,927.91 |
| 4 | 60 | 17.378 | 12.409 | 43,755.23 |

The best tested tuning configuration had RMSE **16.427**, which did not improve on the original best result (~15.9). The selected model is saved as a reusable Spark `PipelineModel`.

**Final evaluation:** one prediction per unseen test engine at its last recorded cycle (100 engines), against the **uncapped** `rul` label. Exact final test RMSE, MAE and NASA Score are stored in the notebook / `baseline_metrics.json` when generated; they are not reproduced here because they were not supplied.

### Part 5 — Structured Streaming (completed optional extension)

The test fleet is replayed through **13 JSON files / micro-batches**, each file representing up to 25 new cycles for engines with remaining observations.

For each micro-batch, `foreachBatch`:

1. Appends new readings to the history dataset.
2. Recomputes rolling features using available history.
3. Runs the saved ML pipeline on the latest cycle of each engine in the batch.
4. Records `unit_id`, `cycle`, `predicted_rul`, `alert` and `batch_id`.

**Alert rule:** `predicted_rul < 30`.

**Validated outcomes:**

- One micro-batch per input file.
- Every test engine received a final streaming prediction.
- Final streaming predictions matched the batch predictions from Part 4.
- **18 distinct engines** appeared in the final alert-engine list.

For production usage, retain only the recent 10 observations per engine for rolling features and use idempotent/transactional writes keyed by batch or record identifiers. `foreachBatch` may replay a batch after a failure.

### Part 6 — Hand-off to Deep Learning

Compute minimum and maximum sensor values **using only the Silver training set**, then use those same statistics to normalize train and test data. Save:

- `dl_ready/train` and `dl_ready/test` as Parquet.
- `dl_ready/normalization.json` with per-sensor min/max values.
- `baseline_metrics.json` with model comparison and test metrics.

Min-max scaling:

`x_scaled = (x - train_min) / (train_max - train_min)`

This prepares inputs for a **future Lab 6** using sequence models such as **LSTM or 1-D CNN**. **This project does not train an LSTM.**

## Project configuration

| Parameter | Value | Meaning |
|---|---:|---|
| `SUBSET` | `FD001` | NASA dataset subset |
| `RUL_CAP` | `125` | Upper bound for training RUL labels |
| `WINDOW_SIZE` | `10` | Rolling-feature window |
| `STD_THRESHOLD` | `0.01` | Near-constant variable filter |
| `ALERT_THRESHOLD` | `30` | Streaming maintenance alert cutoff |
| `SEED` | `42` | Reproducible engine split |

## Output structure

```text
Assignment1_PredictiveMaintenance.ipynb
README.md
 data/
   CMAPSSData/
     train_FD001.txt
     test_FD001.txt
     RUL_FD001.txt
 lab5_outputs/
   bronze/
     train/
     test/
     rul_truth/
   silver/
     train/
     test/
   gold/
     train/
     test/
     test_last/
   models/
     best_pipeline/
   stream/
     incoming/
     history/
     predictions/
     _checkpoint/
   dl_ready/
     train/
     test/
     normalization.json
   baseline_metrics.json
```

*The diagram shows the expected notebook-generated directories; some appear only after their corresponding cell runs.*

## Run on Google Colab

1. Open [Google Colab](https://colab.research.google.com/) and upload `Assignment1_PredictiveMaintenance.ipynb`.
2. Select a **CPU** runtime.
3. Install dependencies in a first cell:

   ```python
   !pip install -q pyspark==3.5.0 pandas numpy pyarrow matplotlib
   ```

4. Initialize `SparkSession` by running the notebook setup cell.
5. Run the notebook **in order**: Setup → Bronze → Silver → Gold → MLlib → optional Streaming → DL hand-off → Report.
6. Download/export the completed notebook and `lab5_outputs/` before the Colab runtime is reset.

No Docker or local Spark installation is required when using the Colab runtime. Spark UI runs on the remote Colab machine and might require Colab's port proxy to display in a browser.

## Reproducibility and caveats

- Run all cells in order; the pipeline depends on variables created by earlier cells.
- The original NASA dataset is required; the notebook may download it automatically.
- Fit scalers/statistics on training data only, never on test data.
- Avoid row-level training/validation splits to prevent engine-level data leakage.
- A streaming file replay is a simulation, not a production Kafka deployment.
- The Colab filesystem is temporary; back up the notebook, model and exported artifacts.
- No fallback checkpoints were used in the reported run (`USE_FALLBACK = False`).

## Future improvements

- Train an LSTM or 1-D CNN on sensor sequences and compare against the MLlib baseline.
- Add group-aware hyperparameter tuning and error analysis by RUL range.
- Optimize history management and exactly-once-style sinks for streaming reliability.
- Scale to larger fleets using distributed storage and Spark resource tuning.

## Reference

NASA — **C-MAPSS Aircraft Engine Simulator Data** (FD001 subset).

---

*Academic lab project — prepared for reproducible experimentation and future deep-learning extensions.*
