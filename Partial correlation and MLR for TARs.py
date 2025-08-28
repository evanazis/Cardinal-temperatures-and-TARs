import numpy as np
from tqdm import tqdm
import rasterio
import os
import pingouin as pg
import pandas as pd
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import statsmodels.api as sm

warnings.filterwarnings('ignore', category=RuntimeWarning)


# 读取TIF数据的函数
def read_tif(input_tif, flag=False):
    with rasterio.open(input_tif) as src:
        data = src.read(1)
        data[data == src.nodata] = np.nan
        data = np.repeat(np.repeat(data, 2, axis=0), 2, axis=1)
        data = data[1:-1, :]
        data = np.pad(data, pad_width=((2, 2), (2, 2)), mode='edge')
    if flag:
        data = data - 273.15
    return data


# 定义一个函数来处理每个小块
def process_block(i_start, i_end, j_start, j_end, topt_block, mat_block, map_block, ssr_block, vpd_block):
    local_r_matrix1 = np.full((i_end - i_start, j_end - j_start), np.nan, dtype=np.float32)  # MAT自变量
    local_r_matrix2 = np.full((i_end - i_start, j_end - j_start), np.nan, dtype=np.float32)  # MAP自变量
    local_r_matrix3 = np.full((i_end - i_start, j_end - j_start), np.nan, dtype=np.float32)  # SSR自变量
    local_r_matrix4 = np.full((i_end - i_start, j_end - j_start), np.nan, dtype=np.float32)  # VPD自变量
    local_beta1_matrix = np.full((i_end - i_start, j_end - j_start), np.nan, dtype=np.float32)

    local_r_p_matrix1 = np.full((i_end - i_start, j_end - j_start), np.nan, dtype=np.float32)  # MAT自变量偏相关显著性
    local_temp_p_matrix1 = np.full((i_end - i_start, j_end - j_start), np.nan, dtype=np.float32)  # MAT自变量多元线性回归显著性
    local_model_p_matrix1 = np.full((i_end - i_start, j_end - j_start), np.nan, dtype=np.float32)  # 多元线性回归模型显著性
    local_model_r2 = np.full((i_end - i_start, j_end - j_start), np.nan, dtype=np.float32)  # 多元线性回归模型R2

    for i in range(2, 2 + i_end - i_start):
        for j in range(2, 2 + j_end - j_start):
            # 提取以 (i, j) 为中心的 5 * 5邻域数据
            Topt_neighborhood = topt_block[i - 2:i + 3, j - 2:j + 3, :]  # 5x5x时间序列
            MAT_neighborhood = mat_block[i - 2:i + 3, j - 2:j + 3, :]
            MAP_neighborhood = map_block[i - 2:i + 3, j - 2:j + 3, :]
            SSR_neighborhood = ssr_block[i - 2:i + 3, j - 2:j + 3, :]
            VPD_neighborhood = vpd_block[i - 2:i + 3, j - 2:j + 3, :]
            # print(Topt_neighborhood)

            # 计算每个变量中空值的数量
            nan_count_Topt = np.isnan(Topt_neighborhood).sum()
            nan_count_MAT = np.isnan(MAT_neighborhood).sum()

            if (nan_count_Topt >= 13 or nan_count_MAT >= 13):
                continue  # 检查是否有变量在5x5邻域中空值数量≥13

            # 将 5*5 邻域数据展平成一维数组
            Topt_flat = Topt_neighborhood.ravel()  # 变为 1D 数组
            MAT_flat = MAT_neighborhood.ravel()
            MAP_flat = MAP_neighborhood.ravel()
            SSR_flat = SSR_neighborhood.ravel()
            VPD_flat = VPD_neighborhood.ravel()

            # 对 SSR 数据进行单位转换
            SSR_flat = SSR_flat / (10 ** 6)

            try:
                # 创建数据框用于偏相关分析
                df = pd.DataFrame({
                    'Topt': Topt_flat,
                    'MAT': MAT_flat,
                    'MAP': MAP_flat,
                    'SSR': SSR_flat,
                    'VPD': VPD_flat
                })

                df.dropna(inplace=True)

                if len(df) < 90:
                    continue
                # 合并 MAT 字段相同的记录并取平均值
                # df = df.groupby('MAT', as_index=False).mean()
                # 计算偏相关系数，控制 MAP 和 SSR 的影响
                result1 = pg.partial_corr(data=df, x='MAT', y='Topt', covar=['MAP', 'SSR', "VPD"], method='pearson')
                result2 = pg.partial_corr(data=df, x='MAP', y='Topt', covar=['MAT', 'SSR', "VPD"], method='pearson')
                result3 = pg.partial_corr(data=df, x='SSR', y='Topt', covar=['MAP', 'MAT', "VPD"], method='pearson')
                result4 = pg.partial_corr(data=df, x='VPD', y='Topt', covar=['MAP', 'MAT', "SSR"], method='pearson')

                # Pingouin库计算偏相关系数并存储
                r_fit1 = result1['r'].values[0]
                r_fit2 = result2['r'].values[0]
                r_fit3 = result3['r'].values[0]
                r_fit4 = result4['r'].values[0]

                p_fit1 = result1['p-val'].values[0]

                local_r_matrix1[i - 2, j - 2] = r_fit1
                local_r_matrix2[i - 2, j - 2] = r_fit2
                local_r_matrix3[i - 2, j - 2] = r_fit3
                local_r_matrix4[i - 2, j - 2] = r_fit4

                local_r_p_matrix1[i - 2, j - 2] = p_fit1

                # statsmodel库进行多元线性回归计算偏相关斜率beta1
                X = sm.add_constant(df[['MAT', 'MAP', 'SSR', "VPD"]])  # 显式添加常数项
                y = df['Topt']
                model = sm.OLS(y, X).fit()  # 构建并拟合多元线性回归模型

                local_beta1_matrix[i - 2, j - 2] = model.params['MAT']  # partial_slope
                local_model_p_matrix1[i - 2, j - 2] = model.f_pvalue
                local_temp_p_matrix1[i - 2, j - 2] = model.pvalues['MAT']
                local_model_r2[i - 2, j - 2] = model.rsquared

            except Exception as e:
                print(e)
                continue

    return (i_start, i_end, j_start, j_end, local_r_matrix1, local_r_matrix2, local_r_matrix3, local_r_matrix4,
            local_beta1_matrix, local_r_p_matrix1, local_temp_p_matrix1, local_model_p_matrix1, local_model_r2)


if __name__ == "__main__":
    base_temp_list = ["Tmin-adjust", "Topt-fit", "Tmax-adjust"]
    env_temp_list = ["AnnualMaxTmean", "MATmax", "MAT", "MATmin", "AnnualMinTmean"]

    for base_temp in base_temp_list:
        for env_temp in env_temp_list:
            # if (base_temp == "Tmin-adjust" and env_temp == "AnnualMinTmean") or (
            #         base_temp == "Tmax-adjust" and env_temp == "AnnualMaxTmean") or (
            #         base_temp == "Topt-fit" and env_temp == "MAT"):
            #     None
            # else:
            #     continue

            print(base_temp, " ", env_temp)

            # 保存 r, β1 矩阵为 GeoTIFF
            os.makedirs("partial_output", exist_ok=True)
            output_tif_beta1 = f"partial_output\\{base_temp.split("-")[0]}-{env_temp}_acclimation-magnitude-temp.tif"
            output_tif_r_mat = f"partial_output\\{base_temp.split("-")[0]}-{env_temp}_temp-R.tif"
            output_tif_r_map = f"partial_output\\{base_temp.split("-")[0]}-{env_temp}_prec-R.tif"
            output_tif_r_ssr = f"partial_output\\{base_temp.split("-")[0]}-{env_temp}_nssr-R.tif"
            output_tif_r_vpd = f"partial_output\\{base_temp.split("-")[0]}-{env_temp}_vpd-R.tif"
            output_tif_r_p_mat = f"partial_output\\{base_temp.split("-")[0]}-{env_temp}_r-pvalue.tif"
            output_tif_temp_p_mat = f"partial_output\\{base_temp.split("-")[0]}-{env_temp}_temp-pvalue.tif"
            output_tif_model_p_mat = f"partial_output\\{base_temp.split("-")[0]}-{env_temp}_model-pvalue.tif"
            output_tif_model_r = f"partial_output\\{base_temp.split("-")[0]}-{env_temp}_model-r2.tif"

            if os.path.exists(output_tif_r_mat) and os.path.exists(output_tif_beta1) and os.path.exists(
                    output_tif_r_map) and os.path.exists(output_tif_r_ssr):
                print(f"{base_temp.split("-")[0]}-{env_temp}", "already exists.")
                continue

            # 文件路径设置
            slope_folder = "..\\2.1 三基点时间序列Slope\\5yr tifs"
            rolling_folder = "E:\\Environment Tifs\\Rolling5"

            # 初始化三维数组
            Base_temp_arrays = np.full((3604, 7204, 15), np.nan, dtype=np.float32)
            env_temp_arrays = np.full((3604, 7204, 15), np.nan, dtype=np.float32)
            MAP_arrays = np.full((3604, 7204, 15), np.nan, dtype=np.float32)
            SSR_arrays = np.full((3604, 7204, 15), np.nan, dtype=np.float32)
            VPD_arrays = np.full((3604, 7204, 15), np.nan, dtype=np.float32)

            # 读取TIF数据
            for y in tqdm(range(2001, 2016), desc="Reading TIF Files..."):
                with rasterio.open(os.path.join(slope_folder, f"{base_temp}_{y}.TIF")) as src:
                    data_topt = src.read(1)
                    data_topt[data_topt == src.nodata] = np.nan
                    data_topt = np.pad(data_topt, pad_width=((2, 2), (2, 2)), mode='edge')
                    Base_temp_arrays[:, :, y - 2001] = data_topt

                env_temp_arrays[:, :, y - 2001] = read_tif(os.path.join(rolling_folder, f"Rolling5_{env_temp}_{y}_{y + 4}.tif"), True)
                MAP_arrays[:, :, y - 2001] = read_tif(os.path.join(rolling_folder, f"Rolling5_MAP_{y}_{y + 4}.tif"))
                SSR_arrays[:, :, y - 2001] = read_tif(os.path.join(rolling_folder, f"Rolling5_AnnualTotalNSR_{y}_{y + 4}.tif"))
                VPD_arrays[:, :, y - 2001] = read_tif(os.path.join(rolling_folder, f"Rolling5_VPD_{y}_{y + 4}.tif"))

            # 初始化结果矩阵
            r_matrix_mat = np.full((3600, 7200), np.nan, dtype=np.float32)
            r_matrix_map = np.full((3600, 7200), np.nan, dtype=np.float32)
            r_matrix_ssr = np.full((3600, 7200), np.nan, dtype=np.float32)
            r_matrix_vpd = np.full((3600, 7200), np.nan, dtype=np.float32)
            beta1_matrix = np.full((3600, 7200), np.nan, dtype=np.float32)
            r_p_matrix_mat = np.full((3600, 7200), np.nan, dtype=np.float32)
            temp_p_matrix_mat = np.full((3600, 7200), np.nan, dtype=np.float32)
            model_p_matrix_mat = np.full((3600, 7200), np.nan, dtype=np.float32)
            model_r_matrix_mat = np.full((3600, 7200), np.nan, dtype=np.float32)

            # 划分矩阵为10x10的小块
            block_size_i = 100
            block_size_j = 100
            blocks = []

            # 假设数据维度
            rows, cols, time_steps = Base_temp_arrays.shape

            # 将矩阵分块并准备任务
            for i in range(2, 3602, block_size_i):
                for j in range(2, 7202, block_size_j):
                    i_end = min(i + block_size_i, 3602)
                    j_end = min(j + block_size_j, 7202)
                    # i_end = i + block_size_i
                    # j_end = j + block_size_j
                    topt_block = Base_temp_arrays[i - 2:i_end + 2, j - 2:j_end + 2, :]
                    mat_block = env_temp_arrays[i - 2:i_end + 2, j - 2:j_end + 2, :]
                    map_block = MAP_arrays[i - 2:i_end + 2, j - 2:j_end + 2, :]
                    ssr_block = SSR_arrays[i - 2:i_end + 2, j - 2:j_end + 2, :]
                    vpd_block = VPD_arrays[i - 2:i_end + 2, j - 2:j_end + 2, :]
                    if not np.isnan(topt_block).all():
                        blocks.append((i - 2, i_end - 2, j - 2, j_end - 2, topt_block, mat_block, map_block, ssr_block, vpd_block))

            fit_results = []
            with ProcessPoolExecutor(max_workers=32) as executor:
                futures = [executor.submit(process_block, *block) for block in blocks]
                for future in tqdm(as_completed(futures), total=len(blocks), desc="Processing"):
                    # fit_results.append(future.result())
                    # 获取结果并填充到beta1_matrix中
                    (i_start_output, i_end_output, j_start_output, j_end_output, local_r_matrix_output_mat, local_r_matrix_output_map, local_r_matrix_output_ssr,
                     local_r_matrix_output_vpd, local_beta1_matrix_output, local_r_p_matrix1_output, local_temp_p_matrix1_output, local_model_p_matrix1_output,
                     local_model_r_matrix1_output) = future.result()
                    r_matrix_mat[i_start_output:i_end_output, j_start_output:j_end_output] = local_r_matrix_output_mat
                    r_matrix_map[i_start_output:i_end_output, j_start_output:j_end_output] = local_r_matrix_output_map
                    r_matrix_ssr[i_start_output:i_end_output, j_start_output:j_end_output] = local_r_matrix_output_ssr
                    r_matrix_vpd[i_start_output:i_end_output, j_start_output:j_end_output] = local_r_matrix_output_vpd
                    beta1_matrix[i_start_output:i_end_output, j_start_output:j_end_output] = local_beta1_matrix_output
                    r_p_matrix_mat[i_start_output:i_end_output, j_start_output:j_end_output] = local_r_p_matrix1_output
                    temp_p_matrix_mat[i_start_output:i_end_output, j_start_output:j_end_output] = local_temp_p_matrix1_output
                    model_p_matrix_mat[i_start_output:i_end_output, j_start_output:j_end_output] = local_model_p_matrix1_output
                    model_r_matrix_mat[i_start_output:i_end_output, j_start_output:j_end_output] = local_model_r_matrix1_output

            #  process_block ------> return i_start, i_end, j_start, j_end, local_beta1_matrix

            with rasterio.open(os.path.join(slope_folder, f"Topt-fit_2001.TIF")) as src:
                meta = src.meta
                meta.update(dtype=rasterio.float32, count=1, compress="lzw")
                with rasterio.open(output_tif_beta1, "w", **meta) as dst:
                    dst.write(beta1_matrix, 1)
                with rasterio.open(output_tif_r_mat, "w", **meta) as dst:
                    dst.write(r_matrix_mat, 1)
                with rasterio.open(output_tif_r_map, "w", **meta) as dst:
                    dst.write(r_matrix_map, 1)
                with rasterio.open(output_tif_r_ssr, "w", **meta) as dst:
                    dst.write(r_matrix_ssr, 1)
                with rasterio.open(output_tif_r_vpd, "w", **meta) as dst:
                    dst.write(r_matrix_vpd, 1)
                with rasterio.open(output_tif_r_p_mat, "w", **meta) as dst:
                    dst.write(r_p_matrix_mat, 1)
                with rasterio.open(output_tif_temp_p_mat, "w", **meta) as dst:
                    dst.write(temp_p_matrix_mat, 1)
                with rasterio.open(output_tif_model_p_mat, "w", **meta) as dst:
                    dst.write(model_p_matrix_mat, 1)
                with rasterio.open(output_tif_model_r, "w", **meta) as dst:
                    dst.write(model_r_matrix_mat, 1)


            print(f"Saved {base_temp.split("-")[0]}-{env_temp}")
