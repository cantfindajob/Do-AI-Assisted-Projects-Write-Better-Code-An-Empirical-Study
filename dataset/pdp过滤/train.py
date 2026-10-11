import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score

# ---------------------- 1. 数据加载与预处理 ----------------------
# 读取Level README特征集数据
df = pd.read_csv(r"C:\Users\16549\Desktop\GitHub_study-master\Experiment2\pdp_level_readme.csv")

# 分离特征与标签：去掉id列和pdp标签列，其余为特征
X = df.drop(columns=["pdp", "id"])
# 标签转换：TRUE -> 1 (PDP), FALSE -> 0 (非PDP)
y = df["pdp"].astype(int)

# 检查缺失值（若无缺失可跳过）
print(f"缺失值总数: {X.isnull().sum().sum()}")
X = X.fillna(0)  # 如有缺失用0填充

# ---------------------- 2. 数据集划分 ----------------------
# 留出法：80%训练，20%测试，固定随机种子保证可复现
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # 分层抽样，保持正负样本比例
)

# ---------------------- 3. 模型训练（对齐J48） ----------------------
#  criterion='entropy' 对应C4.5/J48的信息增益分裂准则
#  可根据需要调整max_depth、min_samples_split等参数控制树规模
clf = DecisionTreeClassifier(
    criterion="entropy",
    random_state=42,
    min_samples_leaf=5,  # 防止过拟合，可微调
    max_depth=None
)

# 训练模型
clf.fit(X_train, y_train)

# ---------------------- 4. 模型评估 ----------------------
print("=" * 50)
print("【留出法测试集评估】")
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=["非PDP", "PDP"]))
print(f"F1值 (PDP类): {f1_score(y_test, y_pred):.3f}")
print(f"精确率 (PDP类): {precision_score(y_test, y_pred):.3f}")
print(f"召回率 (PDP类): {recall_score(y_test, y_pred):.3f}")

print("\n" + "=" * 50)
print("【10折交叉验证评估（与论文一致）】")
# 10折分层交叉验证，评估F1值
cv_f1 = cross_val_score(clf, X, y, cv=10, scoring="f1", n_jobs=-1)
cv_precision = cross_val_score(clf, X, y, cv=10, scoring="precision", n_jobs=-1)
cv_recall = cross_val_score(clf, X, y, cv=10, scoring="recall", n_jobs=-1)

print(f"10折平均F1值: {cv_f1.mean():.3f} (标准差: {cv_f1.std():.3f})")
print(f"10折平均精确率: {cv_precision.mean():.3f}")
print(f"10折平均召回率: {cv_recall.mean():.3f}")

# ---------------------- 5. 特征重要性Top10 ----------------------
print("\n" + "=" * 50)
print("【特征重要性 Top 10】")
feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": clf.feature_importances_
}).sort_values("importance", ascending=False).head(10)
print(feature_importance.to_string(index=False))

# ---------------------- 6. 保存模型 ----------------------
joblib.dump(clf, "pdp_j48_level_readme.pkl")
print("\n模型已保存为 pdp_j48_level_readme.pkl")