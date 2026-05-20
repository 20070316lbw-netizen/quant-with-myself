from features import make_features, FEATURE_REGISTRY

# 向后兼容支持 diagnose_collinearity.py 导入
FEATURE_COLS = list(FEATURE_REGISTRY.keys())