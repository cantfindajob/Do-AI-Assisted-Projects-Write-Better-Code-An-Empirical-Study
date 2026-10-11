import pandas as pd
import joblib

# ---------------------- 1. 加载并清洗待预测数据 ----------------------
df = pd.read_csv("pdp_no-ai-java_features_106.csv")
repo_names = df["repo_name"]
X_pred = df.drop(columns=["repo_name"])

# 数据清洗
X_pred = X_pred.dropna(how="all")
repo_names = repo_names.loc[X_pred.index]
X_pred = X_pred.fillna(0)
X_pred = X_pred.apply(pd.to_numeric, errors="coerce").fillna(0)

# ---------------------- 2. 加载模型 + 自动特征对齐 ----------------------
clf = joblib.load("pdp_j48_level_readme.pkl")
train_features = clf.feature_names_in_

# 核心修复：自动对齐特征列
for col in train_features:
    if col not in X_pred.columns:
        # 缺失的列补0
        X_pred[col] = 0

# 只保留训练时用到的列，按训练集顺序排列
X_pred = X_pred[train_features]

print(f"有效待预测样本数: {len(X_pred)}")
print(f"特征列已对齐，共 {len(train_features)} 维特征")

# ---------------------- 3. 批量预测 ----------------------
y_pred = clf.predict(X_pred)
y_proba = clf.predict_proba(X_pred)[:, 1]

# ---------------------- 4. 整理并保存结果 ----------------------
result = pd.DataFrame({
    "repo_name": repo_names.values,
    "is_pdp": y_pred.astype(bool),
    "pdp_probability": y_proba.round(4)
})

result = result.sort_values("pdp_probability", ascending=False)
result.to_csv("pdp_predict_result.csv", index=False, encoding="utf-8-sig")

print("\n预测完成！结果已保存为 pdp_predict_result.csv")
print(f"预测为PDP的项目数: {result['is_pdp'].sum()} (占比: {result['is_pdp'].mean():.1%})")