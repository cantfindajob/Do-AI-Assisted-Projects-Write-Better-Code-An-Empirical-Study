import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from scipy.stats import t

# 1. 加载数据
df = pd.read_csv(r'C:\Users\16549\Desktop\deepsider\RQ3结果\rq3_normalized.csv')

# 2. 划分自变量
categorical_cols = ['代码类型', '项目分组', '语言', '代码×项目', '代码×语言', '项目×语言', '代码×项目×语言']
numerical_cols = ['Repository Age (Days)', 'Contributor Count', 'Commit Frequency', 'Average PR Size',
                  'Single-author Ratio', 'Cyclomatic Complexity', 'Duplicate Code Ratio (%)',
                  'Lines per Function', 'Comment Density (%)']

# 获取因变量列表 C1 - C16
target_cols = [f'C{i}' for i in range(1, 17)]

# 对类别变量进行独热编码 (One-Hot Encoding)
# drop_first=False 这样你能看到 "AI×java" 具体类别的直接名称和系数
X = pd.get_dummies(df[categorical_cols + numerical_cols], drop_first=False)


# 3. 定义计算 VIP (Variable Importance in Projection) 的函数
def calculate_vip(model):
    t_scores = model.x_scores_
    w_weights = model.x_weights_
    q_loadings = model.y_loadings_
    p, h = w_weights.shape
    vips = np.zeros((p,))

    # 【修改点1】：去掉 .reshape(h, -1)，让 s 保持一维数组 (h,)
    s = np.diag(t_scores.T @ t_scores @ q_loadings.T @ q_loadings)
    total_s = np.sum(s)

    for i in range(p):
        weight = np.array([(w_weights[i, j] / np.linalg.norm(w_weights[:, j])) ** 2 for j in range(h)])
        # 【修改点2】：直接使用 s @ weight，这样出来的就是一个纯标量(Scalar)
        vips[i] = np.sqrt(p * (s @ weight) / total_s)

    return vips


# 4. 定义 Bootstrap 函数：通过重采样评估系数的稳定性、计算 t值和 p值
def pls_bootstrap_pvalues(X, y, n_components, n_boot=500):
    pls = PLSRegression(n_components=n_components)
    pls.fit(X, y)
    coef_orig = pls.coef_.flatten()

    coefs = []
    n_samples = X.shape[0]

    # 交叉验证 / Bootstrap 重采样计算回归系数方差
    for _ in range(n_boot):
        # 有放回抽样
        indices = np.random.choice(n_samples, n_samples, replace=True)
        X_boot, y_boot = X.iloc[indices], y.iloc[indices]
        pls_boot = PLSRegression(n_components=n_components)
        pls_boot.fit(X_boot, y_boot)
        coefs.append(pls_boot.coef_.flatten())

    coefs = np.array(coefs)
    coef_std = np.std(coefs, axis=0)  # 回归系数的标准差

    # 计算 t 统计量 (避免除零错误加了极小值)
    t_stats = coef_orig / (coef_std + 1e-8)

    # 双尾 t 检验计算 p 值
    p_values = 2 * (1 - t.cdf(np.abs(t_stats), df=n_boot - 1))

    return coef_orig, t_stats, p_values


# 5. 主流程：为每个 C1-C16 拟合 PLS 模型
n_components = 3  # 注: 建议先对数据用交叉验证(GridSearchCV)探索最优主成分个数，这里预设为3
results_list = []
model_metrics_list = []  # 【新增】：用来保存每个模型的整体评价指标（如 R方）

for target in target_cols:
    print(f"正在处理因变量: {target} ...")
    y = df[target]

    # 拟合全局 PLS 模型计算 VIP 和 R方
    pls = PLSRegression(n_components=n_components)
    pls.fit(X, y)

    # 【新增】：计算 R方 (R-squared)
    r2_value = pls.score(X, y)
    model_metrics_list.append({
        'Target': target,
        'R_squared': r2_value
    })

    vips = calculate_vip(pls)

    # 执行 Bootstrap (重采样500次)
    coefs_val, t_stats, p_values = pls_bootstrap_pvalues(X, y, n_components=n_components, n_boot=500)

    # 结果封装 (变量级别的结果)
    for i, col_name in enumerate(X.columns):
        results_list.append({
            'Target': target,
            'Feature': col_name,
            'Coefficient': coefs_val[i],
            't_stat': t_stats[i],
            'p_value': p_values[i],
            'VIP': vips[i],
            'Significant (p<0.05)': p_values[i] < 0.05
        })

# 6. 数据导出与结果查看
# (1) 导出变量级别的特征重要性结果
results_df = pd.DataFrame(results_list)
results_df.to_csv('PLS_C1_to_C16_Results.csv', index=False, encoding='utf-8-sig')

# (2) 【新增】：导出每个模型的 R方 结果
metrics_df = pd.DataFrame(model_metrics_list)
metrics_df.to_csv('PLS_Model_R2_Metrics.csv', index=False, encoding='utf-8-sig')

print("\n[成功] 完整分析结果已保存至: PLS_C1_to_C16_Results.csv")
print("[成功] 模型的R方结果已保存至: PLS_Model_R2_Metrics.csv")

# 打印一下 R方的概览
print("\n==== 各个模型(C1-C16)的 R方 概览 ====")
print(metrics_df)