import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import joblib
import os

CSV_PATH = "E:\\Test_Lab\\train_data.csv"
MODEL_PATH = "E:\\Test_Lab\\freshness_model.pkl"

def train():
    print("开始训练（包含番茄Lab预测和新鲜度预测）...")
    
    # 1. 准备数据（如果你有真实数据，把这段替换成读CSV；这里用模拟数据保证能跑）
    if not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH) == 0:
        print("未找到真实数据，生成模拟数据...")
        np.random.seed(42)
        n = 200
        df = pd.DataFrame({
            # 输入特征（膜 + 传感器）
            '膜_L': np.random.uniform(40, 60, n),
            '膜_a': np.random.uniform(-20, 20, n),
            '膜_b': np.random.uniform(-20, 20, n),
            'CO2浓度': np.random.uniform(1, 20, n),
            '温度': np.random.uniform(10, 30, n),
            '湿度': np.random.uniform(40, 80, n),
            # 输出目标1：番茄的 Lab（回归）
            '番茄_L': np.random.uniform(35, 50, n),
            '番茄_a': np.random.uniform(30, 50, n),
            '番茄_b': np.random.uniform(20, 40, n),
            # 输出目标2：新鲜度（分类）
            '真实新鲜度标签': np.random.choice(['新鲜', '可接受', '腐败'], n)
        })
    else:
        df = pd.read_csv(CSV_PATH)

    # 2. 定义输入 X
    X = df[['膜_L', '膜_a', '膜_b', 'CO2浓度', '温度', '湿度']]

    # 3. 训练模型A：预测番茄 Lab（多输出回归）
    y_lab = df[['番茄_L', '番茄_a', '番茄_b']]
    model_lab = RandomForestRegressor(n_estimators=100, random_state=42)
    model_lab.fit(X, y_lab)
    print("✅ 番茄 Lab 回归模型训练完成")

    # 4. 训练模型B：预测新鲜度（分类）
    y_class = df['真实新鲜度标签']
    model_class = RandomForestClassifier(n_estimators=100, random_state=42)
    model_class.fit(X, y_class)
    print("✅ 新鲜度分类模型训练完成")

    # 5. 把两个模型打包存成一个文件
    joblib.dump({'lab_model': model_lab, 'class_model': model_class}, MODEL_PATH)
    print(f"✅ 模型已打包保存到：{MODEL_PATH}")

if __name__ == "__main__":
    train()