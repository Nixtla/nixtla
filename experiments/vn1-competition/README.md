# Score 2nd place in the VN1 Challenge with a few lines of code in under 10 seconds using TimeGPT

We present a fully reproducible experiment demonstrating that Nixtla's **TimeGPT** can achieve the **2nd position** in the [VN1 Forecasting Accuracy Challenge](https://www.datasource.ai/en/home/data-science-competitions-for-startups/phase-2-vn1-forecasting-accuracy-challenge/description) with **zero-shot forecasting**. This result was achieved using the zero-shot capabilities of the foundation model, as most of the code focuses on **data cleaning and preprocessing**, not model training or parameter tuning.

The table below showcases the official competition results, with TimeGPT outperforming the 2nd, 3rd, and other models in the competition.

| **Model**   | **Score**  |
| ----------- | ---------- |
| 1st         | 0.4637     |
| **TimeGPT** | **0.4651** |
| 2nd         | 0.4657     |
| 3rd         | 0.4758     |
| 4th | 0.4774 | 
| 5th | 0.4808 |

---

## **Introduction**

The VN1 Forecasting Accuracy Challenge tasked participants with forecasting future sales using historical sales and pricing data. The goal was to develop robust predictive models capable of anticipating sales trends for various products across different clients and warehouses. Submissions were evaluated based on their accuracy and bias against actual sales figures.

The competition was structured into two phases:

- **Phase 1** (September 12 - October 3, 2024): Participants used the provided Phase 0 sales data to predict sales for Phase 1. This phase lasted three weeks and featured live leaderboard updates to track participant progress.
- **Phase 2** (October 3 - October 17, 2024): Participants utilized both Phase 0 and Phase 1 data to predict sales for Phase 2. Unlike Phase 1, there were no leaderboard updates during this phase until the competition concluded.

One of the competition's key requirements was to use **open-source solutions**. However, as TimeGPT works through an API, we did not upload the forecasts generated during the competition. Instead, we showcase the effectiveness of TimeGPT by presenting the results of our approach.

Our approach leverages the power of **zero-shot forecasting**, where no training, fine-tuning, or manual hyperparameter adjustments are needed. We used only **historical sales data** without any exogenous variables to generate forecasts. With this setting, TimeGPT provides forecasts that achieve an accuracy surpassing nearly all competitors.

Remarkably, the process required only **5 seconds of inference time**, demonstrating the efficiency of TimeGPT.

---

## **Empirical Evaluation**

This study considers time series from multiple datasets provided during the competition. Unlike most competitors, we do not train, fine-tune, or manually adjust TimeGPT. Instead, we rely on **zero-shot learning** to forecast the time series directly.

This study contrasts TimeGPT's zero-shot forecasts against the top 1st, 2nd, and 3rd models submitted to the competition. Our evaluation method follows the official rules and metrics of the VN1 competition.

An R version of this study is also available via `nixtlar`, a CRAN package that provides an interface to Nixtla's TimeGPT. 
---

## **Results**

The table below summarizes the official competition results. Despite using a zero-shot approach, **TimeGPT achieves the 2nd position** with a score of **0.4651**, outperforming the models ranked 2nd and 3rd.

| **Model**   | **Score**  |
| ----------- | ---------- |
| 1st         | 0.4637     |
| **TimeGPT** | **0.4651** |
| 2nd         | 0.4657     |
| 3rd         | 0.4758     |
| 4th | 0.4774 | 
| 5th | 0.4808 |

---

## **Reproducibility**

All necessary code and detailed instructions for reproducing the experiment are available in this repository.

### **Instructions**

1. **Get an API Key** from the [Nixtla Dashboard](https://dashboard.nixtla.io/). Copy it and paste it into the `.env.example` file. Rename the file to `.env`.

2. **Set up [uv](https://github.com/astral-sh/uv):**

```bash
pip install uv
uv venv --python 3.10
source .venv/bin/activate
uv pip sync requirements.txt
```

3. **Download data:**

```bash
make download_data
```

4. **Run the complete pipeline:**

```bash
python -m src.main
```

5. **Tests**

We made sure that the results are comparable by comparing the results against the [official competition results](https://www.datasource.ai/en/home/data-science-competitions-for-startups/phase-2-vn1-forecasting-accuracy-challenge/leaderboard). You can run the tests using:

```bash
pytest
```

6. **R results:**
For the R version of this study using `nixtlar`, run the `main.R` script. Make sure the `functions.R` script is in the same directory.
---

## **References**

- Vandeput, Nicolas. “VN1 Forecasting - Accuracy Challenge.” DataSource.ai, DataSource, 3 Oct. 2024, [https://www.datasource.ai/en/home/data-science-competitions-for-startups/phase-2-vn1-forecasting-accuracy-challenge/description](https://www.datasource.ai/en/home/data-science-competitions-for-startups/phase-2-vn1-forecasting-accuracy-challenge/description)
- [TimeGPT Paper](https://arxiv.org/abs/2310.03589)


