import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from scipy import stats
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

# 1. 读取数据
file_path = r"C:\Users\16549\Desktop\deepsider\RQ3结果\编码后的rq3.2.xlsx "  # 请替换为您的文件路径
df = pd.read_excel(file_path)

# 2. 定义自变量列
base_x_cols = [
    'Repository Age (Days)', 'Contributor Count', 'Commit Frequency',
    'Average PR Size', 'Single-author Ratio', 'Cyclomatic Complexity',
    'Duplicate Code Ratio (%)', 'Lines per Function', 'Comment Density (%)'
]

categorical_prefixes = [
    '代码类型_', '项目分组_', '语言_',
    '代码×项目_', '代码×语言_', '项目×语言_', '代码×项目×语言_'
]

X_cols = base_x_cols.copy()
for col in df.columns:
    if any(col.startswith(prefix) for prefix in categorical_prefixes) and col not in X_cols:
        X_cols.append(col)

# 3. 定义因变量列
Y_cols = ['BLOCKER密度', 'CRITICAL密度', 'MAJOR密度', 'MINOR密度', 'INFO密度', '修复时间_分钟']
X = df[X_cols].fillna(0)


# VIP 计算函数
def calculate_vip(model):
    t = model.x_scores_
    w = model.x_weights_
    q = model.y_loadings_
    p, h = w.shape
    s = np.sum(t ** 2, axis=0) * np.sum(q ** 2, axis=0)
    total_s = np.sum(s)
    vip = np.zeros((p,))
    for i in range(p):
        weight = np.array([(w[i, j] / np.linalg.norm(w[:, j])) ** 2 for j in range(h)])
        vip[i] = np.sqrt(p * (s @ weight) / total_s)
    return vip


# 创建一个空列表来存放所有模型的结果 DataFrame
all_results = []

# 4. 循环因变量拟合模型
for y_name in Y_cols:
    if y_name not in df.columns:
        continue

    y = df[y_name].fillna(0)

    # 拟合全局模型，提取 R方 与 VIP
    pls = PLSRegression(n_components=3)
    pls.fit(X, y)
    y_pred = pls.predict(X)
    r2 = r2_score(y, y_pred)
    vip_scores = calculate_vip(pls)

    # 5折交叉验证计算 T检验 p-value
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_coefs = []

    for train_index, test_index in kf.split(X):
        X_train, y_train = X.iloc[train_index], y.iloc[train_index]
        pls_cv = PLSRegression(n_components=3)
        pls_cv.fit(X_train, y_train)
        cv_coefs.append(pls_cv.coef_.ravel())

    cv_coefs = np.array(cv_coefs)
    t_stats, p_values = stats.ttest_1samp(cv_coefs, popmean=0, axis=0)
    p_values = np.nan_to_num(p_values, nan=1.0)
    mean_coefs = np.mean(cv_coefs, axis=0)

    # 将单个因变量的结果整理为 DataFrame
    results_df = pd.DataFrame({
        'Dependent_Variable': y_name,  # 标记属于哪个因变量
        'Model_R_Squared': r2,  # 整个模型的R方
        'Feature': X_cols,
        'Coefficient': mean_coefs,
        'P-Value': p_values,
        'Significant (p<0.05)': p_values < 0.05,
        'VIP': vip_scores
    })

    # 针对当前模型，按VIP降序排序
    results_df = results_df.sort_values(by='VIP', ascending=False)

    # 追加到列表中
    all_results.append(results_df)

# 5. 合并所有 DataFrame 并保存
final_df = pd.concat(all_results, ignore_index=True)
output_file = "combined_pls_results.csv"

# 使用 utf-8-sig 防止中文在 Excel 中乱码
final_df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"成功将所有因变量的分析结果合并并保存至 {output_file}")