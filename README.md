# 🚗 Used-Car Sale Price Regression — From-Scratch Linear Model

<a id="ai-disclosure"></a>
> ## 🚨 AI DISCLOSURE
> **This project was created solely with artificial-intelligence-generated output.** The model design, Python code, preprocessing strategy, validation analysis, visualizations, report, and documentation were produced with AI assistance. We executed the programs, reviewed the results, tested the outputs, and prepared the presentation materials. No AI-generated result should be treated as automatically correct; the included validation checks document how the output was tested.

This project builds a reproducible linear-regression system that predicts `sale_price_usd` for used-vehicle listings. Predictions are made **after vehicle inspection but before price negotiation and financing**. That timing is central to the project because every predictor must genuinely exist when the prediction is made.

The complete pipeline is implemented in Python without scikit-learn, statsmodels, XGBoost, LightGBM, CatBoost, TensorFlow, PyTorch, ready-made regression models, optimization routines, direct least-squares solvers, matrix inverses, pseudoinverses, or polynomial-fitting shortcuts. Data cleaning, feature construction, categorical encoding, gradient calculation, Adam optimization, L2 regularization, convergence checks, prediction validation, and chart generation are implemented directly with Python and NumPy.

## 📖 Table of Contents

- [AI Disclosure](#ai-disclosure)
- [Project Objective](#project-objective)
- [Runtime Installation and Commands](#runtime-commands)
- [Skills Demonstrated](#skills-demonstrated)
- [Applications and Libraries](#applications-libraries)
- [Dataset](#dataset)
- [Project Process](#project-process)
  - [1. Data Inspection](#data-inspection)
  - [2. Prediction-Time Availability](#prediction-time-review)
  - [3. Missing Values and Outliers](#missing-outliers)
  - [4. Feature Engineering and Encoding](#feature-engineering)
  - [5. Validation Design](#validation-design)
  - [6. Manual Linear Regression](#manual-regression)
  - [7. L2 Regularization](#l2-regularization)
  - [8. Final Prediction Workflow](#prediction-workflow)
  - [9. Visualizations](#visualizations)
- [Model Results](#model-results)
- [Findings](#findings)
- [How to Reproduce](#reproduce)
- [Required Output Format](#output-format)
- [Git Safety and Private Data](#git-safety)
- [Collaboration](#collaboration)
- [File Structure](#file-structure)
- [Limitations](#limitations)

<a id="project-objective"></a>
## 🎯 Project Objective

The target is `sale_price_usd`. The final system accepts a CSV of new vehicle listings for which the target is unknown and writes exactly one finite prediction for every `listing_id`.

Predictions are joined to external labels by `listing_id`; output row order is not used. The program therefore checks that IDs are present and unique and writes exactly these two columns:

```text
listing_id,sale_price_usd_pred
```

The prediction moment is after inspection but before negotiation and financing. This means vehicle specifications, inspection results, history, seller information, and listing geography may be used, while financing outcomes and the final sale price may not.


<a id="runtime-commands"></a>
## ⚙️ Runtime, Installation, and Required Commands

This section contains the minimum setup and execution information required to run the project from a clean checkout.

### Required Python version

Use **Python 3.12**. The completed project was developed and tested with **Python 3.12.14**.

Confirm the active version before installing anything:

```bash
python3 --version
```

The output should begin with `Python 3.12`. If your system keeps several Python versions, use `python3.12` in place of `python3` in every command below.

### Exact dependency-installation command

From the repository directory, install the two permitted runtime dependencies with this exact command:

```bash
python3 -m pip install numpy==2.3.5 matplotlib==3.10.8
```

NumPy is used only for array arithmetic and manually written gradient calculations. Matplotlib is used only to create charts. Neither package supplies the regression model or optimizer.


The project uses only the Python standard library, NumPy, and Matplotlib. The MSE objective, L2 penalty, gradient, Adam updates, convergence rule, preprocessing, and feature encoding are implemented directly in `used_car_price_model.py`.

### Where the train and test paths are configured

Open `used_car_price_model.py`. The three editable paths are grouped near the top of the file under this comment:

```python
# CHANGE ONLY THESE PATHS FOR THE PROJECT DATA FILES.
TRAIN_PATH = Path("car_price_train.csv")
TEST_PATH = Path("car_price_test.csv")
OUTPUT_PATH = Path("predictions.csv")
```

Relative paths are resolved from the directory in which the command is run. Full paths may also be used, for example:

```python
TRAIN_PATH = Path("/full/path/to/car_price_train.csv")
TEST_PATH = Path("/full/path/to/external_test.csv")
OUTPUT_PATH = Path("/full/path/to/predictions.csv")
```

### Command used to train and validate the model

The main program automatically performs its labeled holdout validation, prints accuracy statistics, generates validation charts, and then refits the selected model using all labeled training rows. After confirming the paths above, run:

```bash
python3 used_car_price_model.py
```

For a presentation-only 1,200/600 simulation using the labeled training file, configure `TRAIN_PATH` near the top of `used_car_price_demo.py` and run:

```bash
python3 used_car_price_demo.py
```

### Steps used to generate predictions for a new test CSV

1. Place the new feature-only CSV in the project directory, or note its full path.
2. Confirm that the CSV contains `listing_id` and every required feature column but does not require `sale_price_usd`.
3. Set `TRAIN_PATH` to the labeled training CSV in `used_car_price_model.py`.
4. Set `TEST_PATH` to the new feature-only CSV.
5. Leave `OUTPUT_PATH = Path("predictions.csv")` unless a different output directory is required.
6. From the project directory, run:

   ```bash
   python3 used_car_price_model.py
   ```

7. Confirm that the final console line reports the expected number of written predictions.

The program retrains on all labeled rows before predicting the new test CSV. No private target or external evaluation result is used for tuning.

### Required `predictions.csv` schema and location

By default, the program writes `predictions.csv` into the **current working directory**, which should be the repository root when the command above is used. Its location is controlled by `OUTPUT_PATH` near the top of `used_car_price_model.py`.

The file must contain exactly two columns in this order:

```text
listing_id,sale_price_usd_pred
```

There must be exactly one data row for every test ID. Each `listing_id` must be nonblank and unique, and every `sale_price_usd_pred` must be a finite number. No index column, target column, or extra feature column may appear in the submission file.

<a id="skills-demonstrated"></a>
## 🔧 Skills Demonstrated

- Exploratory data auditing without automated modeling packages
- Prediction-time reasoning and target-leakage prevention
- Missing-value imputation and missingness indicators
- Robust numeric clipping for extreme values
- Fixed nonlinear basis construction for a linear-in-coefficients model
- Manual one-hot encoding with reference categories
- Training-only standardization and preprocessing alignment
- Hand-coded MSE and L2 gradients
- Hand-coded full-batch Adam optimization
- Numerical convergence monitoring
- Holdout and five-fold cross-validation
- Regularization comparison under controlled experimental settings
- RMSE, MAE, R², residual, and coefficient-norm analysis
- Matplotlib model-diagnostic visualizations
- Reproducible CSV prediction output and schema validation

<a id="applications-libraries"></a>
## 💻 Applications and Libraries

- **Python 3.12** — complete data and modeling pipeline
- **NumPy** — array arithmetic only; no model-fitting or least-squares helpers
- **Matplotlib** — presentation and diagnostic charts
- **Python standard library** — CSV input/output, paths, validation, and numeric checks
- **Git and GitHub** — source control and collaboration

Install the permitted runtime dependencies with:

```bash
python3 -m pip install numpy==2.3.5 matplotlib==3.10.8
```

<a id="dataset"></a>
## 📊 Dataset

The training dataset contains **1,800 rows and 19 columns**.

| Role | Column | Project treatment |
|---|---|---|
| Identifier | `listing_id` | Used only to match predictions; never used to estimate price |
| Target | `sale_price_usd` | Value predicted by the model |
| Excluded leakage | `loan_approval_value_usd` | Unavailable before financing and therefore excluded |
| Numeric predictors | 10 columns | Parsed, imputed, clipped, expanded, and standardized |
| Categorical predictors | 6 columns | Normalized and manually one-hot encoded |

### Initial audit summary

| Column | Missing values | Observed range or levels | Treatment |
|---|---:|---|---|
| `model_year` | 0 | 2003–2025 | Numeric predictor |
| `vehicle_age_years` | 0 | 0.1–22.0 | Numeric predictor |
| `odometer_km` | 0 | 300–3,228,000 | Extreme tail clipped |
| `annual_mileage_km` | 0 | 3,500–28,730 | Numeric predictor |
| `engine_liters` | 141 | 0.7–4.141 | Median imputation + missing flag |
| `horsepower` | 0 | 63.91–479.3 | Numeric predictor |
| `condition_score` | 174 | 3.089–10 | Median imputation + missing flag |
| `service_history_score` | 264 | 1.191–10 | Median imputation + missing flag |
| `prior_owners` | 0 | 1–10 | Numeric predictor |
| `accident_count` | 0 | 0–3 | Numeric predictor |
| `brand` | 0 | 9 levels | One-hot encoded |
| `body_style` | 0 | 5 levels | One-hot encoded |
| `fuel_type` | 41 | 4 observed levels | Missing treated as its own category |
| `transmission` | 69 | 3 observed levels | Missing treated as its own category |
| `region` | 0 | 4 levels | One-hot encoded |
| `seller_type` | 0 | 3 levels | One-hot encoded |
| `loan_approval_value_usd` | 216 | $877–$71,410 | Excluded as financing leakage |
| `sale_price_usd` | 0 | $1,200–$75,374.70 | Target |

<a id="project-process"></a>
## 🧩 Process of Project

<a id="data-inspection"></a>
### 1. 🔍 Data Inspection

The CSV is read with Python's `csv` module so the program controls every conversion. Column names are trimmed, listing IDs remain text, and numeric strings are explicitly parsed. Invalid numeric entries and standard missing tokens such as blank strings, `NA`, `N/A`, `null`, and `unknown` become missing numeric values.

The inspection stage checks:

- row and column counts;
- identifier uniqueness;
- data types and parseability;
- missing-value counts;
- numeric minimums and maximums;
- categorical levels;
- duplicate IDs;
- target range and repeated values;
- correlations between numeric predictors and price;
- correlations among predictors.

This stage identified `listing_id` as the identifier and `sale_price_usd` as the target. It also identified a repeated target floor at exactly `$1,200` and an extreme `3,228,000` kilometer odometer observation.

<a id="prediction-time-review"></a>
### 2. ⏱️ Prediction-Time Availability and Leakage Review

Every proposed predictor was evaluated with one question: **Would this value be available after inspection but before negotiation and financing?**

| Predictor group | Available? | Reason |
|---|---|---|
| Vehicle age and identity | Yes | Known from the vehicle record |
| Mileage and mechanical specifications | Yes | Known from the listing or inspection |
| Condition and history | Yes | Established during inspection/history review |
| Brand, body, fuel, and transmission | Yes | Known vehicle attributes |
| Region and seller type | Yes | Known when the listing enters the system |
| `loan_approval_value_usd` | **No** | Created during financing, after prediction time |
| `sale_price_usd` | **No** | Unknown target being predicted |
| `listing_id` | Not a predictor | Identifier used only for matching output |

`loan_approval_value_usd` has a correlation of approximately `+0.935` with sale price, but it remains excluded. A strong statistical relationship does not make an unavailable feature valid. Regularization also cannot repair leakage.

<a id="missing-outliers"></a>
### 3. 🧹 Missing Values and Outlier Handling

All preprocessing values are learned from the fitting portion only. Validation and test rows never influence medians, clipping bounds, category levels, means, or scales.

#### Missing numeric values

For each numeric predictor:

1. Calculate the median using finite fitting-set values.
2. Replace missing entries with that fitting median.
3. Add a separate binary missingness indicator.

The indicator allows the model to learn whether the fact that a value was missing carries information beyond the imputed median.

#### Missing categories

Missing categorical text becomes a literal `<missing>` category. This avoids silently combining missing entries with a real category.

#### Extreme numeric values

Each numeric predictor is clipped to its fitting-set `0.5th` and `99.5th` percentiles. This limits the influence of extreme observations without deleting their rows.

#### Repeated target floor

The training target contains 62 prices at exactly `$1,200`. These observations are treated as floor-censored rather than ordinary continuous measurements. The regression relationship is fitted on prices above the floor, and final predictions are bounded below by `$1,200`.

<a id="feature-engineering"></a>
### 4. 🧱 Feature Engineering, Scaling, and Categorical Encoding

The feature builder creates the same columns in the same order for every dataset.

#### Numeric features

Each numeric field contributes:

- one cleaned numeric value;
- one missingness indicator.

Fixed basis terms add plausible curvature while the fitted model remains linear in its coefficients:

- squared vehicle age;
- `log(1 + odometer)`;
- square-root odometer;
- squared horsepower;
- squared engine size;
- squared condition score;
- squared service-history score.

#### Categorical encoding

Training categories are normalized to lowercase and sorted. One reference level is dropped from every field, and all remaining levels receive a manually constructed zero/one column. A category seen only at validation or test time maps safely to the all-zero reference pattern instead of changing the feature count.

#### Scaling

Constructed numeric features are standardized using fitting-set means and standard deviations:

```text
standardized_value = (value - training_mean) / training_standard_deviation
```

Scaling matters for both optimization and regularization. Without scaling, a coefficient measured per kilometer and a coefficient measured per condition point would have incomparable magnitudes, causing L2 to penalize variables unevenly because of their units.

#### Feature alignment confirmation

`FeatureBuilder.fit()` stores medians, clipping bounds, category levels, feature means, and feature scales. `FeatureBuilder.transform()` reuses those stored values and loops through a fixed predictor sequence. Training, validation, and test matrices therefore contain the same **52 columns including the intercept**, in the same order.

<a id="validation-design"></a>
### 5. 🧪 Validation Design

Three validation approaches were considered:

| Design | Advantages | Disadvantages |
|---|---|---|
| Single holdout | Fast, intuitive, and easy to present | Can be unusually lucky or unlucky |
| Repeated holdouts | Shows sensitivity to several random splits | Some rows may be evaluated repeatedly while others are rarely evaluated |
| Five-fold cross-validation | Every row is validated once and 80% of data trains each fold | Requires five fits and preprocessing must be relearned inside each fold |

**Selected approach:** deterministic five-fold cross-validation for model selection, supported by a fixed 1,200/600 holdout for the presentation and end-to-end output test.

Cross-validation is preferred because the dataset contains only 1,800 labeled rows and a separate external evaluation should not be used for repeated tuning. The fixed holdout remains useful because its 600-row size matches the planned external test-file size.

<a id="manual-regression"></a>
### 6. 📐 Manual Linear Regression and Optimization

For a feature matrix `X`, coefficient vector `w`, intercept `b`, and target `y`, ordinary mean squared error is:

```text
MSE = mean((Xw + b - y)²)
```

The program computes the objective and gradient directly. It then performs deterministic, full-batch Adam updates using NumPy array arithmetic. No regression or optimization package performs the fitting.

| Setting | Value | Purpose |
|---|---:|---|
| Learning rate | `0.02` | Controls update size |
| Maximum iterations | `2,200` | Hard runtime limit |
| Minimum iterations | `200` | Prevents premature stopping |
| Gradient tolerance | `2.5 × 10⁻⁴` | Numerical convergence rule |
| Selected L2 strength | `0.003` | Shrinks unstable slopes |
| Initialization | Zeros | Reproducible starting point |

Training stops after at least 200 iterations when the largest absolute component of the regularized gradient is no greater than `2.5 × 10⁻⁴`, or when 2,200 iterations are reached.

The estimator supports the required interface:

```python
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

<a id="l2-regularization"></a>
### 7. 🛡️ L2 Regularization

The extended objective is:

```text
J(w, b) = mean((Xw + b - y)²) + λ × sum(w²)
```

The intercept `b` is deliberately excluded from the penalty. The coefficient gradient receives the additional term `2λw`, which pulls large slopes toward zero. At `λ = 0`, the penalty and its gradient are exactly zero, reproducing the ordinary MSE objective and gradient. A direct numerical check found objective equality and a maximum gradient difference of `0.0`.

#### Strongly correlated predictor pairs

Pearson correlations were calculated using rows where both numeric values were finite.

| Predictor pair | Pearson correlation | Complete rows | Interpretation |
|---|---:|---:|---|
| `model_year` and `vehicle_age_years` | `-0.997955` | 1,800 | Nearly duplicate age information in opposite directions |
| `engine_liters` and `horsepower` | `+0.822056` | 1,659 | Larger engines generally have more horsepower |
| `vehicle_age_years` and `condition_score` | `-0.775963` | 1,626 | Older vehicles generally have lower condition scores |
| `condition_score` and `service_history_score` | `+0.730071` | 1,387 | Condition and maintenance history move together |

Correlated predictors can divide the same signal among several large and unstable coefficients. L2 reduces that instability by shrinking the coefficient vector.

#### Controlled lambda comparison

The split, features, preprocessing, initialization, learning rate, stopping rule, and price-floor handling were held fixed. Only `lambda` changed.

| Lambda | Validation RMSE | Coefficient L2 norm |
|---:|---:|---:|
| `0` | `$3,673.53` | `12,250.90` |
| `0.0001` | `$3,673.10` | `11,961.02` |
| `0.0003` | `$3,671.98` | `11,748.19` |
| `0.001` | `$3,669.60` | `10,938.80` |
| **`0.003`** | **`$3,667.36`** | **`9,739.70`** |
| `0.01` | `$3,675.96` | `8,266.84` |
| `0.03` | `$3,721.56` | `7,139.76` |
| `0.1` | `$3,861.39` | `6,169.63` |

The selected value `λ = 0.003` produced the best RMSE on the fixed split and reduced the coefficient norm by approximately 20.5% compared with `λ = 0`.

#### Repeatability across five folds

| Lambda | Mean RMSE | RMSE range | Mean coefficient norm |
|---:|---:|---:|---:|
| `0` | `$3,557.55` | `$2,831.83–$3,857.21` | `12,063.30` |
| **`0.003`** | **`$3,557.22`** | **`$2,839.44–$3,869.34`** | **`9,543.56`** |

L2 improved mean RMSE by only `$0.33`, so it is not necessary for competitive predictive accuracy. It was retained because it reduced the mean coefficient norm by approximately **20.9%** while leaving mean RMSE essentially unchanged. This is valuable because the dataset contains extremely correlated predictors.

<a id="prediction-workflow"></a>
### 8. 📤 Final Prediction Workflow

The main program performs two separate jobs:

1. Run a reproducible 600-row validation split and print accuracy statistics.
2. Refit preprocessing and the selected model on all labeled training rows, then predict every row in the feature-only test CSV.

Only three paths near the top of `used_car_price_model.py` normally need to be changed:

```python
TRAIN_PATH = Path("car_price_train.csv")
TEST_PATH = Path("car_price_test.csv")
OUTPUT_PATH = Path("predictions.csv")
```

The program verifies required columns, target validity, unique IDs, prediction count, and finiteness before writing the submission.

<a id="visualizations"></a>
### 9. 📈 Matplotlib Visualizations

Both Python programs create presentation-ready PNG files automatically.

Running the main model creates `model_validation_charts/`. Running the demo creates `presentation_charts/`.

#### Validation dashboard

The dashboard combines the main model evidence into one presentation-ready image.

![Used-car validation dashboard](model_validation_charts/00_validation_dashboard.png)

#### Actual vs. predicted prices

Points near the orange diagonal represent accurate predictions. Most observations follow the diagonal closely, while several higher-priced vehicles show larger underprediction.

![Actual versus predicted sale prices](model_validation_charts/01_actual_vs_predicted.png)

#### Residual analysis

Residuals are defined as predicted price minus actual price. A desirable plot is centered around zero without a strong curve. Most residuals are near zero, but several expensive vehicles produce large negative residuals.

![Residuals versus predicted price](model_validation_charts/02_residuals_vs_predicted.png)

#### Absolute-error distribution

Most errors fall below `$5,000`, while a small number of outliers create a long right tail. The orange line marks the mean absolute error.

![Distribution of absolute prediction errors](model_validation_charts/03_absolute_error_distribution.png)

<a id="model-results"></a>
## 📏 Model Results

### Fixed 600-row presentation holdout

| Metric | Result | Interpretation |
|---|---:|---|
| RMSE | `$3,667.36` | Main error metric; larger misses receive extra weight |
| MAE | `$2,361.95` | Average absolute dollar error |
| Median absolute error | `$1,724.84` | Half of predictions miss by less than this value |
| R² | `0.9068` | Approximately 90.68% of holdout price variation is explained |
| Within `$2,500` | `67.2%` | Roughly two-thirds of predictions |
| Within `$5,000` | `92.7%` | More than nine out of ten predictions |

### Five-fold model-selection result

The selected `λ = 0.003` model produced a mean five-fold RMSE of **`$3,557.22`**, with fold RMSE values ranging from **`$2,839.44` to `$3,869.34`**.

The fixed holdout and five-fold values answer different questions. The holdout provides a simple presentation example that mimics the 600-row final test workflow, while five-fold cross-validation provides stronger model-selection evidence.

<a id="findings"></a>
## 🔍 Findings

**1. Vehicle age is one of the strongest legitimate price signals.**

`vehicle_age_years` has a correlation of approximately `-0.757` with sale price, while `model_year` has a correlation of approximately `+0.755`. Newer vehicles generally sell for more. These two predictors are almost perfectly correlated with one another, which supports coefficient regularization.

**2. Condition and service history carry substantial predictive information.**

`condition_score` correlates approximately `+0.661` with sale price, and `service_history_score` correlates approximately `+0.518`. Better physical condition and stronger maintenance history are associated with higher sale prices.

**3. Mileage and ownership history generally reduce price.**

`odometer_km` correlates approximately `-0.388` with price, and `prior_owners` correlates approximately `-0.477`. Higher vehicle use and more ownership transfers are associated with lower values.

**4. The strongest apparent predictor is invalid because it leaks future information.**

`loan_approval_value_usd` correlates approximately `+0.935` with sale price, but it is produced during financing. Because prediction occurs before financing, including it would create an unrealistic model that could not be used at the stated decision time.

**5. L2 provides stability rather than a meaningful accuracy gain.**

Across five folds, `λ = 0.003` improved mean RMSE by only `$0.33` compared with no penalty. However, it reduced the mean coefficient norm by approximately 20.9%. The selected penalty is justified by coefficient stability in a dataset with highly correlated predictors, not by a claim of a large accuracy improvement.

**6. Most predictions are practically close, but expensive outliers remain difficult.**

On the 600-row holdout, 67.2% of predictions are within `$2,500` and 92.7% are within `$5,000`. The residual and error-distribution plots show a small number of much larger misses, especially among expensive vehicles.

**7. The model explains most, but not all, price variation.**

The holdout R² of `0.9068` indicates that the model explains a large majority of observed price variation. Remaining error may reflect unmeasured equipment, local market effects, nonlinear relationships, negotiation outcomes, and noise in the sale process.

<a id="reproduce"></a>
## 🚀 How to Reproduce

### Presentation demo

1. Clone or download the repository.
2. Place `car_price_train.csv`, `used_car_price_model.py`, and `used_car_price_demo.py` in the same directory.
3. Install dependencies:

   ```bash
   python3 -m pip install numpy==2.3.5 matplotlib==3.10.8
   ```

4. Confirm the training path near the top of `used_car_price_demo.py`:

   ```python
   TRAIN_PATH = Path("car_price_train.csv")
   OUTPUT_PATH = Path("demo_predictions.csv")
   ```

5. Run:

   ```bash
   python3 used_car_price_demo.py
   ```

6. Review the console metrics, `demo_predictions.csv`, and images in `presentation_charts/`.

### Mock external-test workflow

1. Run the mock-data generator:

   ```bash
   python3 create_mock_test_csv.py
   ```

2. Set the model paths:

   ```python
   TRAIN_PATH = Path("car_price_train.csv")
   TEST_PATH = Path("mock_test_features.csv")
   OUTPUT_PATH = Path("predictions.csv")
   ```

3. Run:

   ```bash
   python3 used_car_price_model.py
   ```

4. Confirm that the program reports 600 finite predictions and creates `model_validation_charts/`.

### Final external-test workflow

When a new feature-only CSV is available, change only `TEST_PATH` if the training and output filenames remain unchanged:

```python
TEST_PATH = Path("external_test_file.csv")
```

Then run:

```bash
python3 used_car_price_model.py
```

Do not tune the model after viewing private labels or an external evaluation result.

<a id="output-format"></a>
## ✅ Required Output Format

`predictions.csv` must contain exactly:

```text
listing_id,sale_price_usd_pred
MOCK-CAR-00001,24781.324615
MOCK-CAR-00002,19342.811209
```

The specific values above are illustrative. The program guarantees:

- exactly two output columns;
- one row per test listing;
- no missing or infinite predictions;
- unique, nonblank IDs;
- IDs copied from the test file;
- predictions matched by `listing_id`, independent of row order.

<a id="git-safety"></a>
## 🔒 Git Safety and Private Data

The repository includes a `.gitignore` file. A `.gitignore` is a list of local files and folders that Git should deliberately leave untracked. These files can remain on the computer and still be used by Python, but `git add .` will normally skip them.

This project's `.gitignore` excludes:

- virtual environments such as `.venv/`, `venv/`, and `env/`;
- Python caches such as `__pycache__/` and `*.pyc`;
- test, type-checker, notebook, and coverage caches;
- `.env` files that may contain secrets or machine-specific settings;
- operating-system, editor, log, and temporary files;
- generated prediction outputs that can be recreated;
- folders and filename patterns reserved for private evaluation features and labels.

### Why private evaluation data must not be committed

External feature CSVs and labels may be private evaluation material. Uploading either file to GitHub could expose data that was not licensed or intended for redistribution. The model may read private test features locally, but those bytes should remain only on the authorized local computer.

The safest workflow is to create a local folder named `private_data/` inside the repository and place the external file there:

```text
private_data/external_test.csv
```

The entire `private_data/` folder is ignored. Configure the model locally as:

```python
TEST_PATH = Path("private_data/external_test.csv")
```

Do not add a private label file to the public project. Labels are not needed to generate new predictions.

### Verify that the hidden file is ignored

Before committing, run:

```bash
git check-ignore -v private_data/external_test.csv
```

Git should print the matching `.gitignore` rule. Then inspect the pending commit:

```bash
git status
git diff --cached --name-only
```

Neither command should list the private feature or label file.

### If a hidden file was already staged or tracked

`.gitignore` does not automatically remove a file Git is already tracking. Remove it from Git's index while keeping the local file:

```bash
git rm --cached private_data/external_test.csv
git commit -m "Remove private evaluation data from tracking"
```

Do **not** use plain `git rm` unless the local file should also be deleted. If private data was already pushed to GitHub, deleting it in a later commit does not remove it from earlier history. Stop sharing the repository, review the data owner's requirements, and remove the sensitive history before continuing to push.

### Add the exact private filename when received

The supplied `.gitignore` already blocks common private-test filename patterns and everything inside `private_data/`. When the actual filename is known, add that exact filename to `.gitignore` before placing it in the repository directory.

<a id="collaboration"></a>
## 🤝 Collaboration

| Collaborator | GitHub |
|---|---|---|
| **Mike Green** | `@mikegreen15` |
| **Nate Rivera** | `@the-rivernile` |

Suggested responsibility descriptions include:

- running and verifying the Python pipeline;
- reviewing preprocessing and leakage decisions;
- reproducing validation and regularization experiments;
- checking output schema and test IDs;
- preparing visualizations and presentation explanations;
- maintaining GitHub commits and documentation.

Because the project was built with AI-generated output, this section should describe each collaborator's execution, review, verification, presentation, and repository-management contributions.

<a id="file-structure"></a>
## 📁 File Structure

```text
├── images
│   ├── demo_presentation_charts
│       ├── 00_presentation_dashboard.png
│       ├── 01_actual_vs_predicted.png
│       ├── 02_residuals_vs_predicted.png
│       └── 03_absolute_error_distribution.png
│   ├── model_validation_charts
│       ├── 00_validation_dashboard.png
│       ├── 01_actual_vs_predicted.png
│       ├── 02_residuals_vs_predicted.png
│       └── 03_absolute_error_distribution.png
├── predictions
│   ├── demo_predictions.csv
│   ├── predictions.csv
├── predictions
│   ├── used_car_price_demo.py             # 1,200/600 presentation demonstration
│   ├── used_car_price_model.py            # Final training, validation, chart, and prediction program
├── raw_csv
│   ├── car_price_train.csv                # Labeled training data     
├── sample_test
│   ├── mock_instructor_test.csv
├── .gitignore                             # Blocks caches, environments, outputs, and hidden data
├── used_car_price_project_report.docx     # Detailed audit and presentation report
└── README.md                              # Project documentation
```

<a id="limitations"></a>
## ⚠️ Limitations

- External labels are unavailable, so external-test accuracy cannot be calculated locally.
- The mock test rows are derived from training features and verify workflow—not independent predictive performance.
- Linear basis terms may not capture every nonlinear vehicle-price relationship.
- Rare categories may have limited training evidence.
- Large errors remain for some expensive vehicles and unusual listings.
- The price floor is inferred from repeated `$1,200` targets and may represent censoring or a business rule.
- Correlation describes linear association, not causation.
- Validation results estimate future performance but do not guarantee identical performance on new data.

---

This repository is an educational demonstration of a reproducible, leakage-aware regression pipeline implemented under strict from-scratch modeling constraints.
