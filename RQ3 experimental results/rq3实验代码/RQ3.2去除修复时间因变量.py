import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from scipy import stats

# 1. 读取数据
df = pd.read_excel(r'C:\Users\16549\Desktop\deepsider\RQ3结果\编码后的rq3.2去除修复时间.xlsx ')

# 2. 定义因变量 (Y)
y_cols = ['BLOCKER密度', 'CRITICAL密度', 'MAJOR密度', 'MINOR密度', 'INFO密度']

# 3. 定义自变量 (X)
# 剔除无关列与因变量列，剩下的作为自变量
exclude_cols = ['项目标题', '项目key', '修复时间_分钟'] + y_cols
x_cols = [c for c in df.columns if c not in exclude_cols]

X = df[x_cols].values

# 4. 修复后的 VIP 计算函数
def calculate_vip(model):
    t = model.x_scores_
    w = model.x_weights_
    q = model.y_loadings_

    p, h = w.shape
    vips = np.zeros((p,))

    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)

    for i in range(p):
        weight = np.array([(w[i, j] / np.linalg.norm(w[:, j])) ** 2 for j in range(h)])
        # 修复点：添加 .item() 将数组显式转换为标量
        vips[i] = np.sqrt(p * (s.T @ weight).item() / total_s)

    return vips

# 5. 模型拟合与10折交叉验证
results = []
n_components = 3  # 您可以根据数据的方差解释率调整组件数量

for y_name in y_cols:
    y = df[y_name].values

    # 拟合完整模型以计算整体的 R2 和 VIP
    pls = PLSRegression(n_components=n_components)
    pls.fit(X, y)
    y_pred = pls.predict(X)
    r2 = r2_score(y, y_pred)
    vip = calculate_vip(pls)

    # 10折交叉验证评估回归系数的稳定性
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    coefs = []

    for train_index, test_index in kf.split(X):
        X_train, y_train = X[train_index], y[train_index]
        fold_pls = PLSRegression(n_components=n_components)
        fold_pls.fit(X_train, y_train)
        coefs.append(fold_pls.coef_.flatten())

    coefs = np.array(coefs)  # 形状: (10, 特征数量)

    # 对10折生成的系数进行单样本 t检验 (检验其均值是否显著不为0)
    t_stats, p_values = stats.ttest_1samp(coefs, 0, axis=0)

    # 计算系数均值
    mean_coefs = np.mean(coefs, axis=0)

    # 将该因变量的结果保存为DataFrame
    res_df = pd.DataFrame({
        'Target': y_name,
        'Feature': x_cols,
        'R2': r2,
        'Mean_Coefficient': mean_coefs,
        'P_Value': p_values,
        'VIP': vip.flatten()
    })

    # =============== 新增逻辑：根据 P值 判断是否显著 ===============
    # 通常显著性水平设为 0.05
    res_df['是否显著影响 (P<0.05)'] = np.where(res_df['P_Value'] < 0.05, '是', '否')
    # ================================================================

    # 根据 VIP 值降序排列，VIP越大的特征排在越前面
    res_df = res_df.sort_values(by='VIP', ascending=False)
    results.append(res_df)

# 6. 合并结果并导出
final_df = pd.concat(results, ignore_index=True)

# 确保在保存CSV时指定编码格式 (utf-8-sig) 避免中文在 Excel 中乱码
final_df.to_csv('去除修复时间因变量的结果.csv', index=False, encoding='utf-8-sig')
print("模型运行完成，结果已保存至 PLS_results.csv")