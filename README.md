# 👥 User Behavior Analysis Using API Data 

A data analysis project that fetches user data from the [DummyJSON API](https://dummyjson.com/users), explores and cleans it, then answers key analytical questions through **Seaborn visualizations**.

---
📓 For full analysis, code walkthrough, and visualizations, check the [Jupyter Notebook](analysis.ipynb).

DAY1 & DAY2 are available in another repository.
You can access them here:  🔗[ data-analysis-using-python](https://github.com/mmagdyy1/data-analysis-using-python)

## 📁 Project Structure

```
📦 user-analysis/
├── read.py              # Fetches data from API and saves to CSV
├── main.py              # Full analysis, cleaning, and visualizations
├── analysis.ipynb       # Jupyter Notebook with full analysis & visualizations
├── users.csv            # Generated dataset (100 users)
├── README.md            # Project documentation
└── plots/
    ├── plot1_age_distribution.png
    ├── plot2_avg_age_by_gender.png
    ├── plot3_gender_pie.png
    ├── plot4_top10_cities.png
    ├── plot5_age_vs_height.png
    ├── plot6_age_vs_weight.png
    └── plot7_correlation_heatmap.png
```

---

## ⚙️ Setup & Run

```bash
# 1. Install dependencies
pip install requests pandas seaborn matplotlib

# 2. Fetch data from API → saves users.csv
python read.py

# 3. Run full analysis + generate plots
python main.py
```

---

## 📊 Dataset Overview

- **Source:** [https://dummyjson.com/users?limit=100](https://dummyjson.com/users?limit=100)
- **Records:** 100 users
- **Key columns:** `firstName`, `lastName`, `age`, `gender`, `height`, `weight`, `bloodGroup`, `eyeColor`, `role`, `address`

---

## 🔍 Analysis Questions

| # | Question | Answer |
|---|----------|--------|
| 1 | Average age of users? | ~38 years |
| 2 | Average age by gender? | See Plot 2 |
| 3 | Number of users per gender? | See Plot 3 |
| 4 | Top 10 cities with most users? | See Plot 4 |
| 5 | Average height & weight? | ~170 cm / ~70 kg |
| 6 | Relationship between age & height/weight? | Weak correlation (r ≈ 0) |

---

## 📈 Visualizations

### Plot 1 — Age Distribution
![Age Distribution](plot1_age_distribution.png)

### Plot 2 — Average Age by Gender
![Avg Age by Gender](plot2_avg_age_by_gender.png)

### Plot 3 — Users by Gender
![Gender Pie](plot3_gender_pie.png)

### Plot 4 — Top 10 Cities
![Top 10 Cities](plot4_top10_cities.png)

### Plot 5 — Age vs Height
![Age vs Height](plot5_age_vs_height.png)

### Plot 6 — Age vs Weight
![Age vs Weight](plot6_age_vs_weight.png)

### Plot 7 — Correlation Heatmap
![Correlation Heatmap](plot7_correlation_heatmap.png)

---

## 🛠️ Technologies Used

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-4C72B0)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Plotting-orange)
