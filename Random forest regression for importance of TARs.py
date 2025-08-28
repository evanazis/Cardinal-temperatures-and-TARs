import os
import rasterio
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split


def read_tif010(input_tif, flag=False):
    with rasterio.open(input_tif) as src:
        data = src.read(1)
        data[data == src.nodata] = np.nan
        data = np.repeat(np.repeat(data, 2, axis=0), 2, axis=1)
        data = data[1:-1, :]

    if flag:
        data = data - 273.15
    data = data[:3000, :]

    return data

def read_tif005(input_tif):
    with rasterio.open(input_tif) as src:
        data = src.read(1)
        data[data == src.nodata] = np.nan

    data = data[:3000, :]

    return data

# 初始化一个字典来存储所有结果
results = {}

Tmin_tif = r"..\1.1 所有19年全部拟合\v6.6.0 19yr output\Tmin-adjust.tif"
Tmax_tif = r"..\1.1 所有19年全部拟合\v6.6.0 19yr output\Tmax-adjust.tif"
Topt_tif = r"..\1.1 所有19年全部拟合\v6.6.0 19yr output\Topt-fit.tif"
Tmin_array = read_tif005(Tmin_tif)
Topt_array = read_tif005(Topt_tif)
Tmax_array = read_tif005(Tmax_tif)


# -----------------------------------------Tmin---------------------------------------------------------------
input_path = "RF_INPUT"
os.chdir(input_path)
TminAM_array = read_tif005(r"..\..\2.2 v3 三基点时间序列Partial Regression\partial_output\Tmin-MAT_acclimation-magnitude-temp.tif")
TminAM_p_array = read_tif005(r"..\..\2.2 v3 三基点时间序列Partial Regression\partial_output\Tmin-MAT_model-pvalue.tif")
TminAM_array[TminAM_p_array > 0.05] = np.nan


MAT_CV_array = read_tif010("MAT-100CV.tif")
MAP_CV_array = read_tif010("MAP-100CV.tif")
MAT_array = read_tif010("MAT.tif", True)
MAP_array = read_tif010("MAP.tif")
SSR_array = read_tif010("AnnualTotalNSR.tif") / 100
VPD_array = read_tif010("VPD.tif")
SWC_array = read_tif010("SWC2.tif")
SOC_array = read_tif005("SOC.tif")
pH_array = read_tif005("pH.tif")
SN_array = read_tif005("SN.tif")


if all(arr.shape == TminAM_array.shape for arr in
       [Tmin_array, MAT_CV_array, MAT_array, MAP_array,  MAP_CV_array, SSR_array, VPD_array, SWC_array, SOC_array, pH_array, SN_array]):
    print("Tmin所有矩阵形状相同")

env_vars = [Tmin_array, MAT_CV_array, MAT_array, MAP_array, MAP_CV_array,  SSR_array, VPD_array, SWC_array, SOC_array, pH_array, SN_array]
env_names = ["Tmin-adjust", "TEMP-CV", "MAT", "MAP", "MAP-CV","SSR", "VPD", "SWC", "SOC", "pH", "SN"]

valid_mask = ~np.isnan(TminAM_array)
for var in env_vars:
    valid_mask &= ~np.isnan(var)
print("Tmin有效像元个数：", np.count_nonzero(~np.isnan(TminAM_array)))


# 提取有效数据
Y = TminAM_array[valid_mask]
X = np.column_stack([var[valid_mask] for var in env_vars])

# 划分训练集和测试集（例如 80% 训练，20% 测试）
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=66)

# 构建并训练模型
rf = RandomForestRegressor(n_estimators=100, random_state=66, n_jobs=-1, min_samples_leaf=5)
rf.fit(X_train, Y_train)

# 特征重要性
importance = rf.feature_importances_
results['Tmin_importance'] = importance
print("Feature importances:", importance)

# 模型预测
Y_pred = rf.predict(X_test)

# 评估指标
rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
r2 = r2_score(Y_test, Y_pred)
mae = mean_absolute_error(Y_test, Y_pred)

print(f"Test RMSE: {rmse:.4f}")
print(f"Test R²: {r2:.4f}")
print(f"Test MAE: {mae:.4f}")


# -----------------------------------------Topt---------------------------------------------------------------
ToptAM_array = read_tif005(r"..\..\2.2 v3 三基点时间序列Partial Regression\partial_output\Topt-MAT_acclimation-magnitude-temp.tif")
ToptAM_p_array = read_tif005(r"..\..\2.2 v3 三基点时间序列Partial Regression\partial_output\Topt-MAT_model-pvalue.tif")
ToptAM_array[ToptAM_p_array > 0.05] = np.nan

MAT_CV_array = read_tif010("MAT-100CV.tif")
MAT_array = read_tif010("MAT.tif", True)
MAP_CV_array = read_tif010("MAP-100CV.tif")
MAP_array = read_tif010("MAP.tif")
SSR_array = read_tif010("AnnualTotalNSR.tif") / 100
VPD_array = read_tif010("VPD.tif")
SWC_array = read_tif010("SWC2.tif")
SOC_array = read_tif005("SOC.tif")
pH_array = read_tif005("pH.tif")
SN_array = read_tif005("SN.tif")


if all(arr.shape == ToptAM_array.shape for arr in
       [Topt_array, MAT_CV_array, MAT_array, MAP_array, MAP_CV_array,  SSR_array, VPD_array, SWC_array, SOC_array, pH_array, SN_array]):
    print("Topt所有矩阵形状相同")

env_vars = [Topt_array, MAT_CV_array, MAT_array, MAP_array, MAP_CV_array,  SSR_array, VPD_array, SWC_array, SOC_array, pH_array, SN_array]
env_names = ["Topt-fit", "TEMP-CV", "MAT", "MAP", "MAP-CV","SSR", "VPD", "SWC", "SOC", "pH", "SN"]

valid_mask = ~np.isnan(ToptAM_array)
for var in env_vars:
    valid_mask &= ~np.isnan(var)
print("Topt有效像元个数：", np.count_nonzero(~np.isnan(ToptAM_array)))


# 提取有效数据
Y = ToptAM_array[valid_mask]
X = np.column_stack([var[valid_mask] for var in env_vars])

# 划分训练集和测试集（例如 80% 训练，20% 测试）
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=66)

# 构建并训练模型
rf = RandomForestRegressor(n_estimators=100, random_state=66, n_jobs=-1, min_samples_leaf=5)
rf.fit(X_train, Y_train)

# 特征重要性
importance = rf.feature_importances_
results['Topt_importance'] = importance
print("Feature importances:", importance)

# 模型预测
Y_pred = rf.predict(X_test)

# 评估指标
rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
r2 = r2_score(Y_test, Y_pred)
mae = mean_absolute_error(Y_test, Y_pred)

print(f"Test RMSE: {rmse:.4f}")
print(f"Test R²: {r2:.4f}")
print(f"Test MAE: {mae:.4f}")







# -----------------------------------------Tmax---------------------------------------------------------------
TmaxAM_array = read_tif005(r"..\..\2.2 v3 三基点时间序列Partial Regression\partial_output\Tmax-AnnualMaxTmean_acclimation-magnitude-temp.tif")
TmaxAM_p_array = read_tif005(r"..\..\2.2 v3 三基点时间序列Partial Regression\partial_output\Tmax-AnnualMaxTmean_model-pvalue.tif")
TmaxAM_array[TmaxAM_p_array > 0.05] = np.nan

MAT_CV_array = read_tif010("AnnualMaxTmean-100CV.tif")
AnnualMaxTmean_array = read_tif010("AnnualMaxTmean.tif", True)
MAP_array = read_tif010("MAP.tif")
MAP_CV_array = read_tif010("MAP-100CV.tif")
SSR_array = read_tif010("AnnualTotalNSR.tif") / 100
VPD_array = read_tif010("VPD.tif")
SWC_array = read_tif010("SWC2.tif")
SOC_array = read_tif005("SOC.tif")
pH_array = read_tif005("pH.tif")
SN_array = read_tif005("SN.tif")

if all(arr.shape == TmaxAM_array.shape for arr in
       [Tmax_array, MAT_CV_array, AnnualMaxTmean_array, MAP_array, MAP_CV_array, SSR_array, VPD_array, SWC_array, SOC_array, pH_array, SN_array]):
    print("Tmax所有矩阵形状相同")

env_vars = [Tmax_array, MAT_CV_array, AnnualMaxTmean_array, MAP_array,MAP_CV_array, SSR_array, VPD_array, SWC_array, SOC_array, pH_array, SN_array]
env_names = ["Tmax-adjust", "TEMP-CV", "AMaxT", "MAP", "MAP-CV", "SSR", "VPD", "SWC", "SOC", "pH", "SN"]

valid_mask = ~np.isnan(TmaxAM_array)
for var in env_vars:
    valid_mask &= ~np.isnan(var)
print("Tmax有效像元个数：", np.count_nonzero(~np.isnan(TmaxAM_array)))

# 提取有效数据
Y = TmaxAM_array[valid_mask]
X = np.column_stack([var[valid_mask] for var in env_vars])

# 划分训练集和测试集（例如 80% 训练，20% 测试）
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=66)

# 构建并训练模型
rf = RandomForestRegressor(n_estimators=100, random_state=66, n_jobs=-1, min_samples_leaf=5)
rf.fit(X_train, Y_train)

# 特征重要性
importance = rf.feature_importances_
results['Tmax_importance'] = importance
print("Feature importances:", importance)

# 模型预测
Y_pred = rf.predict(X_test)

# 评估指标
rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
r2 = r2_score(Y_test, Y_pred)
mae = mean_absolute_error(Y_test, Y_pred)

print(f"Test RMSE: {rmse:.4f}")
print(f"Test R²: {r2:.4f}")
print(f"Test MAE: {mae:.4f}")


env_names[2] = "Env_Temp"
env_names[0] = "Cardinal_Temp"
results['env_names'] = env_names

# 创建DataFrame并保存为CSV
df = pd.DataFrame({
    'Environmental_Variable': results['env_names'],
    'Tmin_Importance': results['Tmin_importance'],
    'Topt_Importance': results['Topt_importance'],
    'Tmax_Importance': results['Tmax_importance']
})

# 保存到CSV文件
output_filename = "..\\Global_AM_Importance.csv"
df.to_csv(output_filename, index=False)
print(f"结果已保存到 {output_filename}")

# 打印结果
print("\n最终结果：")
print(df)