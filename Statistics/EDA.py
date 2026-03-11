import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sqlalchemy import create_engine
import warnings 
warnings.filterwarnings('ignore')

# ============================================================
# CẤU HÌNH KẾT NỐI POSTGRESQL
# ============================================================
DB_CONFIG = {
    "host":     "localhost",
    "port":     5432,
    "database": "postgres",   
    "user":     "postgres",   
    "password": "Tduppostgresql1",   
}

TABLE_NAME = "mentalhealth_cleaned" 

# ============================================================
# CẤU HÌNH THƯ MỤC XUẤT ẢNH
# ============================================================
import os
OUTPUT_DIR = r"D:\FPTlearning\ADY\Visualization"  
os.makedirs(OUTPUT_DIR, exist_ok=True)  

def get_connection():
    url = (
        f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )
    return create_engine(url)

# ============================================================
# 1. ĐỌC DỮ LIỆU
# ============================================================
print("Đang kết nối PostgreSQL và tải dữ liệu...")
engine = get_connection()

query = f"SELECT * FROM {TABLE_NAME};"
df = pd.read_sql(query, engine)

print(f"✅ Tải thành công: {df.shape[0]:,} hàng × {df.shape[1]} cột")
print("\n--- Kiểu dữ liệu ---")
print(df.dtypes)
print("\n--- Thống kê mô tả ---")
print(df.describe())
print("\n--- Missing values ---")
print(df.isnull().sum()[df.isnull().sum() > 0])

# ============================================================
# THIẾT LẬP STYLE CHUNG
# ============================================================
plt.rcParams.update({
    'figure.facecolor': '#0f1117',
    'axes.facecolor':   '#1a1d27',
    'axes.edgecolor':   '#2e3347',
    'axes.labelcolor':  '#c9d1d9',
    'xtick.color':      '#8b949e',
    'ytick.color':      '#8b949e',
    'text.color':       '#c9d1d9',
    'grid.color':       '#21262d',
    'grid.linewidth':   0.6,
    'font.family':      'DejaVu Sans',
    'axes.titlesize':   13,
    'axes.labelsize':   11,
})

ACCENT   = '#58a6ff'
ACCENT2  = '#f78166'
ACCENT3  = '#3fb950'
ACCENT4  = '#d2a8ff'
PALETTE  = [ACCENT, ACCENT2, ACCENT3, ACCENT4, '#ffa657', '#79c0ff']

def save_fig(name):
    path = os.path.join(OUTPUT_DIR, f"{name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor='#0f1117')
    plt.close()
    print(f"  💾 Đã lưu: {path}")


# ============================================================
# PHẦN 1 – UNIVARIATE ANALYSIS
# ============================================================
print("\n" + "="*60)
print("PHẦN 1: PHÂN TÍCH ĐƠN BIẾN")
print("="*60)

# --- 1.1  Phân phối K6_total ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Phân phối K6 Total (Sức khỏe Tâm thần)', fontsize=15,
             color='white', fontweight='bold', y=1.02)

ax = axes[0]
ax.hist(df['k6_total'], bins=20, color=ACCENT, edgecolor='#0f1117',
        alpha=0.85, rwidth=0.9)
ax.axvline(df['k6_total'].mean(), color=ACCENT2, lw=2,
           linestyle='--', label=f"Mean = {df['k6_total'].mean():.1f}")
ax.axvline(13, color=ACCENT3, lw=2, linestyle=':', label='Ngưỡng 13')
ax.set_xlabel('K6 Total Score')
ax.set_ylabel('Số sinh viên')
ax.set_title('Histogram K6 Total')
ax.legend()
ax.grid(axis='y')

ax = axes[1]
# KDE
from scipy.stats import gaussian_kde
x = np.linspace(df['k6_total'].min(), df['k6_total'].max(), 300)
kde = gaussian_kde(df['k6_total'].dropna())
ax.fill_between(x, kde(x), color=ACCENT, alpha=0.3)
ax.plot(x, kde(x), color=ACCENT, lw=2)
ax.axvline(13, color=ACCENT2, lw=2, linestyle='--',
           label='Ngưỡng bất ổn (13)')

normal_pct = (df['k6_total'] < 13).mean() * 100
ax.text(0.05, 0.92, f"Bình thường (<13): {normal_pct:.1f}%",
        transform=ax.transAxes, color=ACCENT3, fontsize=10)
ax.text(0.05, 0.82, f"Có dấu hiệu (≥13): {100-normal_pct:.1f}%",
        transform=ax.transAxes, color=ACCENT2, fontsize=10)
ax.set_title('KDE – Hình dạng phân phối')
ax.legend()
plt.tight_layout()
save_fig("01_k6_distribution")

# --- 1.2  Faculty & Academic Year ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Đặc điểm Học thuật', fontsize=15, color='white',
             fontweight='bold')

ax = axes[0]
faculty_counts = df['faculty'].value_counts()
bars = ax.barh(faculty_counts.index, faculty_counts.values,
               color=PALETTE[:len(faculty_counts)], edgecolor='#0f1117')
for bar, val in zip(bars, faculty_counts.values):
    ax.text(val + 1, bar.get_y() + bar.get_height()/2,
            f'{val:,}', va='center', color='white', fontsize=9)
ax.set_title('Số sinh viên theo Khoa')
ax.set_xlabel('Số lượng')
ax.grid(axis='x')

ax = axes[1]
year_counts = df['academic_year'].value_counts().sort_index()
ax.bar(year_counts.index.astype(str), year_counts.values,
       color=PALETTE, edgecolor='#0f1117', width=0.6)
ax.set_title('Số sinh viên theo Năm học')
ax.set_xlabel('Năm học')
ax.set_ylabel('Số lượng')
ax.grid(axis='y')
plt.tight_layout()
save_fig("02_academic_profile")

# --- 1.3  GPA & Study Hours distribution ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Phân phối GPA và Giờ học', fontsize=15,
             color='white', fontweight='bold')

for ax, col, color, title in zip(
    axes,
    ['gpa', 'study_hours_per_week'],
    [ACCENT3, ACCENT4],
    ['GPA', 'Giờ học / tuần']
):
    skew_val = df[col].skew()
    ax.hist(df[col].dropna(), bins=25, color=color,
            edgecolor='#0f1117', alpha=0.85, rwidth=0.9)
    ax.axvline(df[col].mean(), color=ACCENT2, lw=2,
               linestyle='--', label=f'Mean={df[col].mean():.2f}')
    ax.axvline(df[col].median(), color=ACCENT, lw=2,
               linestyle=':', label=f'Median={df[col].median():.2f}')
    ax.set_title(f'{title}  |  Skew={skew_val:.2f}')
    ax.set_xlabel(col)
    ax.set_ylabel('Số sinh viên')
    ax.legend()
    ax.grid(axis='y')

plt.tight_layout()
save_fig("03_gpa_study_hours")

# --- 1.4  Sleep, Exercise, Caffeine ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Thói quen Sinh hoạt', fontsize=15,
             color='white', fontweight='bold')

# Sleep
ax = axes[0]
ax.hist(df['sleep_hours'].dropna(), bins=20, color=ACCENT,
        edgecolor='#0f1117', alpha=0.85)
ax.axvline(df['sleep_hours'].mean(), color=ACCENT2, lw=2,
           linestyle='--', label=f"Mean={df['sleep_hours'].mean():.1f}h")
ax.set_title('Số giờ ngủ')
ax.set_xlabel('Giờ / ngày')
ax.legend()
ax.grid(axis='y')

# Exercise
ax = axes[1]
ex_counts = df['exercise_frequency'].value_counts().sort_index()
ax.bar(ex_counts.index.astype(str), ex_counts.values,
       color=ACCENT3, edgecolor='#0f1117', width=0.6)
ax.set_title('Tần suất tập thể dục')
ax.set_xlabel('Lần / tuần')
ax.set_ylabel('Số sinh viên')
ax.grid(axis='y')

# Caffeine
ax = axes[2]
ax.hist(df['daily_caffeine_mg'].dropna(), bins=25, color=ACCENT4,
        edgecolor='#0f1117', alpha=0.85)
ax.axvline(df['daily_caffeine_mg'].mean(), color=ACCENT2, lw=2,
           linestyle='--',
           label=f"Mean={df['daily_caffeine_mg'].mean():.0f}mg")
ax.set_title('Caffeine tiêu thụ hàng ngày')
ax.set_xlabel('mg / ngày')
ax.legend()
ax.grid(axis='y')

plt.tight_layout()
save_fig("04_lifestyle_habits")


# ============================================================
# PHẦN 2 – MULTIVARIATE ANALYSIS
# ============================================================
print("\n" + "="*60)
print("PHẦN 2: PHÂN TÍCH ĐA BIẾN")
print("="*60)

# --- 2.1  Correlation Heatmap ---
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(12, 9))
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
            annot=True, fmt='.2f', annot_kws={'size': 8},
            linewidths=0.5, linecolor='#0f1117',
            ax=ax, vmin=-1, vmax=1,
            cbar_kws={'shrink': 0.8})
ax.set_title('Correlation Heatmap – Tất cả biến định lượng',
             fontsize=14, color='white', pad=15)
plt.tight_layout()
save_fig("05_correlation_heatmap")

# --- 2.2  Scatter: Study Hours vs K6 ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Học tập vs Sức khỏe Tâm thần', fontsize=15,
             color='white', fontweight='bold')

ax = axes[0]
sc = ax.scatter(df['study_hours_per_week'], df['k6_total'],
                c=df['k6_total'], cmap='RdYlGn_r', alpha=0.5,
                s=20, edgecolors='none')
# Trend line
z = np.polyfit(df['study_hours_per_week'].dropna(),
               df.loc[df['study_hours_per_week'].notna(), 'k6_total'], 1)
p = np.poly1d(z)
x_line = np.linspace(df['study_hours_per_week'].min(),
                     df['study_hours_per_week'].max(), 200)
ax.plot(x_line, p(x_line), color=ACCENT2, lw=2, linestyle='--',
        label='Xu hướng')
plt.colorbar(sc, ax=ax, label='K6 Score')
ax.set_xlabel('Giờ học / tuần')
ax.set_ylabel('K6 Total Score')
ax.set_title('Giờ học vs Stress')
ax.legend()
ax.grid()

# GPA vs K6 group
ax = axes[1]
df['stress_group'] = np.where(df['k6_total'] >= 13,
                               'Có dấu hiệu (≥13)',
                               'Bình thường (<13)')
for grp, color in zip(['Bình thường (<13)', 'Có dấu hiệu (≥13)'],
                       [ACCENT3, ACCENT2]):
    subset = df[df['stress_group'] == grp]['gpa'].dropna()
    ax.hist(subset, bins=20, alpha=0.6, color=color,
            edgecolor='#0f1117', label=f'{grp} (n={len(subset):,})')
ax.set_xlabel('GPA')
ax.set_ylabel('Số sinh viên')
ax.set_title('GPA theo nhóm K6')
ax.legend()
ax.grid(axis='y')

plt.tight_layout()
save_fig("06_study_vs_mental_health")

# --- 2.3  Boxplots: Housing & Exercise vs K6 ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Lối sống vs Sức khỏe Tâm thần', fontsize=15,
             color='white', fontweight='bold')

ax = axes[0]
housing_order = df['housing_status'].value_counts().index.tolist()
sns.boxplot(data=df, x='housing_status', y='k6_total',
            order=housing_order, palette=PALETTE,
            linewidth=1.2, fliersize=3, ax=ax)
ax.set_title('K6 theo Housing Status')
ax.set_xlabel('Nơi ở')
ax.set_ylabel('K6 Total Score')
ax.axhline(13, color=ACCENT2, lw=1.5, linestyle='--',
           label='Ngưỡng 13')
ax.legend()
ax.grid(axis='y')
plt.setp(ax.get_xticklabels(), rotation=15, ha='right')

ax = axes[1]
exercise_order = sorted(df['exercise_frequency'].dropna().unique())
sns.boxplot(data=df, x='exercise_frequency', y='k6_total',
            order=exercise_order, palette=PALETTE,
            linewidth=1.2, fliersize=3, ax=ax)
ax.set_title('K6 theo Tần suất tập thể dục')
ax.set_xlabel('Lần tập / tuần')
ax.set_ylabel('K6 Total Score')
ax.axhline(13, color=ACCENT2, lw=1.5, linestyle='--',
           label='Ngưỡng 13')
ax.legend()
ax.grid(axis='y')

plt.tight_layout()
save_fig("07_lifestyle_boxplots")

# --- 2.4  Caffeine → Sleep → Stress (triple relationship) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Caffeine → Giờ ngủ → Stress', fontsize=15,
             color='white', fontweight='bold')

ax = axes[0]
sc = ax.scatter(df['daily_caffeine_mg'], df['sleep_hours'],
                c=df['k6_total'], cmap='RdYlGn_r',
                alpha=0.5, s=20, edgecolors='none')
plt.colorbar(sc, ax=ax, label='K6 Score')
ax.set_xlabel('Caffeine (mg/ngày)')
ax.set_ylabel('Giờ ngủ / ngày')
ax.set_title('Caffeine vs Giờ ngủ (màu = Stress)')
ax.grid()

ax = axes[1]
# Phân nhóm caffeine
df['caffeine_group'] = pd.cut(df['daily_caffeine_mg'],
                               bins=[0, 100, 200, 400, 9999],
                               labels=['Thấp (<100)', 'Vừa (100-200)',
                                       'Cao (200-400)', 'Rất cao (>400)'])
caffeine_stress = df.groupby('caffeine_group', observed=True)['k6_total'].mean()
bars = ax.bar(caffeine_stress.index.astype(str), caffeine_stress.values,
              color=PALETTE, edgecolor='#0f1117', width=0.6)
for bar, val in zip(bars, caffeine_stress.values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.1,
            f'{val:.1f}', ha='center', color='white', fontsize=9)
ax.set_title('K6 trung bình theo nhóm Caffeine')
ax.set_ylabel('K6 Mean Score')
ax.axhline(13, color=ACCENT2, lw=1.5, linestyle='--',
           label='Ngưỡng 13')
ax.legend()
ax.grid(axis='y')
plt.setp(ax.get_xticklabels(), rotation=15, ha='right')

plt.tight_layout()
save_fig("08_caffeine_sleep_stress")


# ============================================================
# PHẦN 3 – SEGMENTED ANALYSIS
# ============================================================
print("\n" + "="*60)
print("PHẦN 3: PHÂN TÍCH THEO PHÂN ĐOẠN")
print("="*60)

# --- 3.1  Faculty Comparison ---
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Phân tích theo Khoa (Faculty)', fontsize=15,
             color='white', fontweight='bold')

ax = axes[0]
faculty_order = (df.groupby('faculty')['k6_total']
                 .median().sort_values(ascending=False).index)
sns.boxplot(data=df, x='faculty', y='k6_total',
            order=faculty_order, palette=PALETTE,
            linewidth=1.2, fliersize=3, ax=ax)
ax.axhline(13, color=ACCENT2, lw=1.5, linestyle='--', label='Ngưỡng 13')
ax.set_title('K6 theo Khoa')
ax.set_xlabel('')
ax.set_ylabel('K6 Total Score')
ax.legend()
ax.grid(axis='y')
plt.setp(ax.get_xticklabels(), rotation=20, ha='right')

ax = axes[1]
# Tỷ lệ có dấu hiệu bất ổn (K6 >= 13) theo khoa
stress_rate = (df.groupby('faculty')
               .apply(lambda x: (x['k6_total'] >= 13).mean() * 100)
               .sort_values(ascending=True))
colors_bar = [ACCENT2 if v >= 30 else ACCENT3 for v in stress_rate.values]
bars = ax.barh(stress_rate.index, stress_rate.values,
               color=colors_bar, edgecolor='#0f1117')
for bar, val in zip(bars, stress_rate.values):
    ax.text(val + 0.3, bar.get_y() + bar.get_height()/2,
            f'{val:.1f}%', va='center', color='white', fontsize=9)
ax.set_title('Tỷ lệ sinh viên có K6 ≥ 13 theo Khoa (%)')
ax.set_xlabel('Tỷ lệ (%)')
ax.grid(axis='x')

plt.tight_layout()
save_fig("09_faculty_analysis")

# --- 3.2  Academic Year Trend ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Xu hướng Áp lực theo Năm học', fontsize=15,
             color='white', fontweight='bold')

ax = axes[0]
year_order = sorted(df['academic_year'].dropna().unique())
sns.boxplot(data=df, x='academic_year', y='k6_total',
            order=year_order, palette=PALETTE,
            linewidth=1.2, fliersize=3, ax=ax)
ax.axhline(13, color=ACCENT2, lw=1.5, linestyle='--', label='Ngưỡng 13')
ax.set_title('K6 theo Năm học')
ax.set_xlabel('Năm học')
ax.set_ylabel('K6 Total Score')
ax.legend()
ax.grid(axis='y')

ax = axes[1]
year_stats = df.groupby('academic_year')['k6_total'].agg(['mean', 'std'])
ax.plot(year_stats.index, year_stats['mean'], marker='o',
        color=ACCENT, lw=2, markersize=8, label='Mean K6')
ax.fill_between(year_stats.index,
                year_stats['mean'] - year_stats['std'],
                year_stats['mean'] + year_stats['std'],
                alpha=0.2, color=ACCENT)
ax.axhline(13, color=ACCENT2, lw=1.5, linestyle='--', label='Ngưỡng 13')
ax.set_title('Xu hướng K6 trung bình theo Năm học')
ax.set_xlabel('Năm học')
ax.set_ylabel('K6 Mean Score')
ax.legend()
ax.grid()

plt.tight_layout()
save_fig("10_academic_year_trend")

# --- 3.3  Club Participation vs Mental Health ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Tham gia Câu lạc bộ vs Sức khỏe Tâm thần', fontsize=15,
             color='white', fontweight='bold')

ax = axes[0]
club_data = [
    df[df['club_participation'] == True]['k6_total'].dropna(),
    df[df['club_participation'] == False]['k6_total'].dropna()
]
bp = ax.boxplot(club_data, patch_artist=True,
                labels=['Có tham gia', 'Không tham gia'],
                medianprops={'color': 'white', 'linewidth': 2})
for patch, color in zip(bp['boxes'], [ACCENT3, ACCENT2]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.axhline(13, color=ACCENT4, lw=1.5, linestyle='--', label='Ngưỡng 13')
ax.set_ylabel('K6 Total Score')
ax.set_title('Phân phối K6 theo CLB')
ax.legend()
ax.grid(axis='y')

ax = axes[1]
# Stacked bar theo khoa: tỷ lệ tham gia CLB
club_faculty = (df.groupby('faculty')['club_participation']
                .value_counts(normalize=True)
                .unstack() * 100)
if True in club_faculty.columns and False in club_faculty.columns:
    club_faculty[[True, False]].plot(
        kind='bar', stacked=True, ax=ax,
        color=[ACCENT3, ACCENT2], edgecolor='#0f1117',
        label=['Tham gia', 'Không tham gia'])
ax.set_title('Tỷ lệ tham gia CLB theo Khoa (%)')
ax.set_xlabel('')
ax.set_ylabel('%')
ax.legend(['Tham gia', 'Không tham gia'], loc='upper right')
ax.grid(axis='y')
plt.setp(ax.get_xticklabels(), rotation=20, ha='right')

plt.tight_layout()
save_fig("11_club_participation")

# ============================================================
# BONUS: SUMMARY DASHBOARD (1 trang tổng hợp)
# ============================================================
fig = plt.figure(figsize=(20, 14))
fig.patch.set_facecolor('#0f1117')
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)
fig.suptitle('📊 Student Mental Health — EDA Summary Dashboard',
             fontsize=18, color='white', fontweight='bold', y=1.01)

# K6 histogram
ax1 = fig.add_subplot(gs[0, :2])
ax1.hist(df['k6_total'], bins=20, color=ACCENT, edgecolor='#0f1117',
         alpha=0.85)
ax1.axvline(13, color=ACCENT2, lw=2, linestyle='--', label='Ngưỡng 13')
ax1.set_title('Phân phối K6 Total')
ax1.legend(); ax1.grid(axis='y')

# Heatmap mini (chỉ các cột quan trọng)
ax2 = fig.add_subplot(gs[0, 2:])
key_cols = ['k6_total', 'gpa', 'study_hours_per_week',
            'sleep_hours', 'daily_caffeine_mg', 'exercise_frequency']
key_cols = [c for c in key_cols if c in df.columns]
mini_corr = df[key_cols].corr()
sns.heatmap(mini_corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            linewidths=0.5, linecolor='#0f1117',
            annot_kws={'size': 7}, ax=ax2, cbar=False)
ax2.set_title('Correlation (Biến chính)')

# Boxplot Faculty
ax3 = fig.add_subplot(gs[1, :2])
sns.boxplot(data=df, x='faculty', y='k6_total', palette=PALETTE,
            linewidth=1, fliersize=2, ax=ax3)
ax3.axhline(13, color=ACCENT2, lw=1.2, linestyle='--')
ax3.set_title('K6 theo Khoa')
ax3.set_xlabel('')
ax3.grid(axis='y')
plt.setp(ax3.get_xticklabels(), rotation=20, ha='right', fontsize=8)

# Trend by Year
ax4 = fig.add_subplot(gs[1, 2:])
year_mean = df.groupby('academic_year')['k6_total'].mean()
ax4.plot(year_mean.index, year_mean.values, marker='o',
         color=ACCENT3, lw=2, markersize=7)
ax4.axhline(13, color=ACCENT2, lw=1.2, linestyle='--', label='Ngưỡng 13')
ax4.set_title('Xu hướng K6 theo Năm học')
ax4.legend(); ax4.grid()

# Scatter Study Hours vs K6
ax5 = fig.add_subplot(gs[2, :2])
ax5.scatter(df['study_hours_per_week'], df['k6_total'],
            alpha=0.3, s=15, color=ACCENT4, edgecolors='none')
ax5.set_title('Giờ học vs Stress')
ax5.set_xlabel('Giờ học/tuần')
ax5.set_ylabel('K6 Score')
ax5.grid()

# Club Participation Boxplot
ax6 = fig.add_subplot(gs[2, 2:])
club_map = {True: 'Tham gia CLB', False: 'Không tham gia'}
df['club_label'] = df['club_participation'].map(club_map)
sns.boxplot(data=df, x='club_label', y='k6_total',
            palette=[ACCENT3, ACCENT2], linewidth=1,
            fliersize=2, ax=ax6)
ax6.axhline(13, color=ACCENT2, lw=1.2, linestyle='--')
ax6.set_title('CLB vs K6')
ax6.set_xlabel('')
ax6.grid(axis='y')

save_fig("00_summary_dashboard")

print("\n" + "="*60)
print("✅ HOÀN THÀNH! Các file đã được lưu:")
print("  00_summary_dashboard.png")
print("  01_k6_distribution.png")
print("  02_academic_profile.png")
print("  03_gpa_study_hours.png")
print("  04_lifestyle_habits.png")
print("  05_correlation_heatmap.png")
print("  06_study_vs_mental_health.png")
print("  07_lifestyle_boxplots.png")
print("  08_caffeine_sleep_stress.png")
print("  09_faculty_analysis.png")
print("  10_academic_year_trend.png")
print("  11_club_participation.png")
print("="*60)