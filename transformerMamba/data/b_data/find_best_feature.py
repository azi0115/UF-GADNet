import pandas as pd

analysis = pd.read_csv("traffic_feature_analysis.csv")

# 只看有效特征
effective = analysis[analysis["is_effective"] == True].copy()

# 按单特征 AUC 排序
top_auc = effective.sort_values("single_feature_auc", ascending=False)

print("Top 30 effective features by single_feature_auc:")
print(top_auc[[
    "feature",
    "category",
    "single_feature_auc",
    "cliffs_delta",
    "q_value",
    "phishing_mean",
    "benign_mean"
]].head(30))

# 保存 top 特征
top_auc.head(30).to_csv("top30_effective_traffic_features.csv", index=False)