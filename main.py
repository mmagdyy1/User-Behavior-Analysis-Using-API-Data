import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import json
import warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

# ═══════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════
print("=" * 60)
print("1. LOAD DATA")
print("=" * 60)

df = pd.read_csv('users.csv')
print(f"✔  Loaded {len(df)} rows, {len(df.columns)} columns\n")


# ═══════════════════════════════════════════════════════════
# 2. BASIC DATA EXPLORATION
# ═══════════════════════════════════════════════════════════
print("=" * 60)
print("2. BASIC DATA EXPLORATION")
print("=" * 60)

# Shape
print(f"\n📐 Shape: {df.shape}")

# Column names
print(f"\n📋 Columns: {list(df.columns)}")

# Data types
print("\n🔠 Data Types:")
print(df.dtypes.to_string())

# Missing values
print("\n❓ Missing Values:")
print(df.isnull().sum().to_string())

# Duplicate rows
print(f"\n♻  Duplicate Rows: {df.duplicated().sum()}")

# Summary statistics
print("\n📊 Summary Statistics:")
print(df.describe().round(2).to_string())

# Value counts for categorical columns
print("\n📂 Value Counts — Categorical Columns:")
for col in ['gender', 'bloodGroup', 'eyeColor', 'role']:
    if col in df.columns:
        print(f"\n  [{col}]")
        print(df[col].value_counts().to_string())


# ═══════════════════════════════════════════════════════════
# 3. DATA CLEANING — Extract city & country from address
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("3. DATA CLEANING")
print("=" * 60)

def parse_address(addr_str, key):
    """Extract a specific key from the address string."""
    try:
        # Handle both single-quote and double-quote JSON-like strings
        addr_dict = json.loads(str(addr_str).replace("'", '"'))
        return addr_dict.get(key, 'Unknown')
    except Exception:
        try:
            # Fallback: eval (safe here since data is from API)
            addr_dict = eval(str(addr_str))
            return addr_dict.get(key, 'Unknown')
        except Exception:
            return 'Unknown'

df['city']    = df['address'].apply(lambda x: parse_address(x, 'city'))
df['country'] = df['address'].apply(lambda x: parse_address(x, 'country'))

print(f"✔  Extracted 'city'    — sample: {df['city'].unique()[:5].tolist()}")
print(f"✔  Extracted 'country' — sample: {df['country'].unique()[:5].tolist()}")

# Fill missing age / height / weight with median
for col in ['age', 'height', 'weight']:
    if col in df.columns:
        missing = df[col].isnull().sum()
        if missing > 0:
            med = df[col].median()
            df[col].fillna(med, inplace=True)
            print(f"✔  Filled {missing} missing '{col}' with median ({med:.1f})")

print()

# Value counts for address.country
print("  [country]")
print(df['country'].value_counts().to_string())


# ═══════════════════════════════════════════════════════════
# 4. ANALYSIS
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("4. ANALYSIS")
print("=" * 60)

# Q1 — Average age
avg_age = df['age'].mean()
print(f"\n① Average age of users: {avg_age:.2f} years")

# Q2 — Average age by gender
avg_age_gender = df.groupby('gender')['age'].mean().round(2)
print("\n② Average age by gender:")
print(avg_age_gender.to_string())

# Q3 — Number of users per gender
users_gender = df['gender'].value_counts()
print("\n③ Number of users per gender:")
print(users_gender.to_string())

# Q4 — Top 10 cities
top_cities = df['city'].value_counts().head(10)
print("\n④ Top 10 cities with most users:")
print(top_cities.to_string())

# Q5 — Average height and weight
avg_h = df['height'].mean()
avg_w = df['weight'].mean()
print(f"\n⑤ Average height: {avg_h:.2f} cm")
print(f"   Average weight: {avg_w:.2f} kg")

# Q6 — Correlation age vs height/weight
corr_h = df[['age', 'height']].corr().loc['age', 'height']
corr_w = df[['age', 'weight']].corr().loc['age', 'weight']
print(f"\n⑥ Pearson r — age ↔ height : {corr_h:.4f}")
print(f"   Pearson r — age ↔ weight : {corr_w:.4f}")
if abs(corr_h) < 0.2 and abs(corr_w) < 0.2:
    print("   → Weak correlation: no strong linear relationship between age and height/weight.")
else:
    print("   → Moderate/strong correlation detected.")


# ═══════════════════════════════════════════════════════════
# 5. VISUALISATIONS
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("5. GENERATING PLOTS …")
print("=" * 60)

# ── Plot 1: Age Distribution ──────────────────────────────
plt.figure(figsize=(8, 5))
sns.histplot(df['age'], bins=20, kde=True, color='steelblue')
plt.axvline(avg_age, color='crimson', linestyle='--', label=f'Mean: {avg_age:.1f}')
plt.title('Distribution of User Ages', fontsize=14, fontweight='bold')
plt.xlabel('Age')
plt.ylabel('Count')
plt.legend()
plt.tight_layout()
plt.savefig('plot1_age_distribution.png')
plt.show()
print("✔  Plot 1: Age Distribution")

# ── Plot 2: Average Age by Gender ────────────────────────
plt.figure(figsize=(6, 5))
ax = avg_age_gender.plot(kind='bar', color=['#4C72B0', '#DD8452'], edgecolor='white')
plt.title('Average Age by Gender', fontsize=14, fontweight='bold')
plt.xlabel('Gender')
plt.ylabel('Average Age')
plt.xticks(rotation=0)
for p in ax.patches:
    ax.annotate(f'{p.get_height():.1f}',
                (p.get_x() + p.get_width() / 2, p.get_height() + 0.2),
                ha='center', fontsize=11)
plt.tight_layout()
plt.savefig('plot2_avg_age_by_gender.png')
plt.show()
print("✔  Plot 2: Avg Age by Gender")

# ── Plot 3: Users per Gender (Pie) ───────────────────────
plt.figure(figsize=(6, 6))
plt.pie(users_gender, labels=users_gender.index, autopct='%1.1f%%',
        colors=['#4C72B0', '#DD8452', '#55A868'],
        startangle=140, wedgeprops=dict(edgecolor='white', linewidth=1.5))
plt.title('User Count by Gender', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plot3_gender_pie.png')
plt.show()
print("✔  Plot 3: Gender Pie")

# ── Plot 4: Top 10 Cities ────────────────────────────────
plt.figure(figsize=(9, 6))
sns.barplot(x=top_cities.values, y=top_cities.index, palette='Blues_r')
plt.title('Top 10 Cities by User Count', fontsize=14, fontweight='bold')
plt.xlabel('Number of Users')
plt.ylabel('City')
plt.tight_layout()
plt.savefig('plot4_top10_cities.png')
plt.show()
print("✔  Plot 4: Top 10 Cities")

# ── Plot 5: Age vs Height (regression) ───────────────────
plt.figure(figsize=(8, 5))
sns.regplot(data=df, x='age', y='height',
            scatter_kws={'alpha': 0.5, 'color': '#4C72B0'},
            line_kws={'color': 'crimson'})
plt.title(f'Age vs Height  (r = {corr_h:.3f})', fontsize=14, fontweight='bold')
plt.xlabel('Age')
plt.ylabel('Height (cm)')
plt.tight_layout()
plt.savefig('plot5_age_vs_height.png')
plt.show()
print("✔  Plot 5: Age vs Height")

# ── Plot 6: Age vs Weight (regression) ───────────────────
plt.figure(figsize=(8, 5))
sns.regplot(data=df, x='age', y='weight',
            scatter_kws={'alpha': 0.5, 'color': '#DD8452'},
            line_kws={'color': 'steelblue'})
plt.title(f'Age vs Weight  (r = {corr_w:.3f})', fontsize=14, fontweight='bold')
plt.xlabel('Age')
plt.ylabel('Weight (kg)')
plt.tight_layout()
plt.savefig('plot6_age_vs_weight.png')
plt.show()
print("✔  Plot 6: Age vs Weight")

# ── Plot 7: Correlation Heatmap ───────────────────────────
plt.figure(figsize=(6, 5))
corr_matrix = df[['age', 'height', 'weight']].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
            linewidths=0.5, square=True, annot_kws={'size': 12})
plt.title('Correlation — Age, Height, Weight', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('plot7_correlation_heatmap.png')
plt.show()
print("✔  Plot 7: Correlation Heatmap")

print("\n" + "=" * 60)
print("✅ DONE! 7 plots saved in your working directory.")
print("=" * 60)