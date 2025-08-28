# 空区域（a, b) ：[(0, 0), (3, 9), (4, 0), (4, 1), (6, 1), (6, 4), (6, 7), (7, 1), (7, 4), (7, 5), (8, 0), (8, 1), (8, 5), (8, 6), (8, 7), (8, 8), (9, 0), (9, 1), (9, 2), (9, 3), (9, 4), (9, 5), (9, 6), (9, 7), (9, 8), (9, 9)]
# Contact with evanazis@qq.com
# WE 拟合6.0全新版本，先对19年的GPP序列季节性分解，进行植被的生长期判断。提取出SOS与EOS之后，直接对每个像元进行每五年的WE拟合


import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm
import statistics
import gc
import time
import pandas as pd
import os
from scipy.signal import savgol_filter
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.signal import find_peaks
from scipy.optimize import minimize, root_scalar
from math import log
import warnings
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed


# 计算年积日
def calculate_date_from_doy(year, doy):
    try:
        # 转换输入为整数
        year = int(year)
        doy = int(doy)
        # 创建当年的1月1日日期
        start_of_year = datetime(year, 1, 1)
        target_date = start_of_year + timedelta(days=doy - 1)
        month = f"{target_date.month:02d}"
        day = f"{target_date.day:02d}"

        return month, day
    except ValueError as e:
        raise ValueError(f"输入的年积日或年份无效：{e}")


def verify_peak(prd, dataframe_df):
    date1, date2 = prd

    if date1 > date2:
        TmeanDateRange = dataframe_df[~dataframe_df['month_day'].between(date2, date1)]
        GPPDateRange = dataframe_df[((f"2010-{date2}" >= dataframe_df.index) & (dataframe_df.index >= "2010-01-01")) |
                                    ((f"2010-{date1}" <= dataframe_df.index) & (dataframe_df.index <= "2010-12-31"))]

    else:
        TmeanDateRange = dataframe_df[dataframe_df['month_day'].between(date1, date2)]
        GPPDateRange = dataframe_df[f"2010-{date1}": f"2010-{date2}"]

    peak_GPP_max = np.nanmax(GPPDateRange["GPP_seasonal_normalize"])
    peak_month_day = GPPDateRange.loc[GPPDateRange["GPP_seasonal_normalize"].idxmax(), 'month_day']
    peak_Tmean_avg = np.nanmean(TmeanDateRange["Tmean"])
    peak_Tmean_min = np.nanmin(TmeanDateRange["Tmean"])
    peak_Tmean_max = np.nanmax(TmeanDateRange["Tmean"])

    if (peak_GPP_max > 0.2) and (peak_Tmean_avg > 10):
        return True, peak_Tmean_min, peak_Tmean_max, peak_month_day
    else:
        return False, peak_Tmean_min, peak_Tmean_max, peak_month_day


# 定义 WE 分段函数
def WE_curve(T, Tmin, Topt, Tmax, beta, ki):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    if T <= Tmin or T >= Tmax:
        return 0

    try:
        a = log(2) / log((Tmax - Tmin) / (Topt - Tmin))
        if (not np.isfinite(a)) or a > 300:
            return np.inf
        return ki * ((2 * (T - Tmin) ** a * (Topt - Tmin) ** a - (T - Tmin) ** (2 * a)) / (
                (Topt - Tmin) ** (2 * a))) ** beta
    except ValueError:
        return np.inf


# 让数组能够作为变量进行计算WE曲线上对应的值
def WE_curve_array(T, Tmin, Topt, Tmax, beta, ki):
    return np.array([WE_curve(t, Tmin, Topt, Tmax, beta, ki) for t in T])


# 残差函数，即目标函数，最小化残差以进行拟合
def residuals(params, T_data, GPP_data, ki):
    Tmin, Topt, Tmax, beta = params
    GPP_pred = WE_curve_array(T_data, Tmin, Topt, Tmax, beta, ki)
    resi = np.sum((GPP_pred - GPP_data) ** 2)
    return resi


# 调整最小值和最大值
def adjust_TminTmax(Tmin, Topt, Tmax, beta, ki):
    GPP_lower_bound = 0.05

    def f(T):
        if T == Tmin or T == Tmax:
            f_T = - ki * (GPP_lower_bound + 0.01)  # 加入0.01的penalty迷惑python
        else:
            a = log(2) / log((Tmax - Tmin) / (Topt - Tmin))
            f_T = ki * ((2 * (T - Tmin) ** a * (Topt - Tmin) ** a - (T - Tmin) ** (2 * a)) / (
                    (Topt - Tmin) ** (2 * a))) ** beta - ki * GPP_lower_bound
        return f_T

    Tmin_adj_result = root_scalar(f, bracket=[Tmin, Topt])
    Tmin_adj = Tmin_adj_result.root if Tmin_adj_result.converged else np.nan

    Tmax_adj_result = root_scalar(f, bracket=[Tmax, Topt])
    Tmax_adj = Tmax_adj_result.root if Tmax_adj_result.converged else np.nan

    return Tmin_adj, Tmax_adj


def calculate_topt(Tmean_array, GPP_array, Tmin_actual_input, Tmax_actual_input, f_count):
    x, y = Tmean_array, GPP_array

    # --------------------------------------------数据清洗-------------------------------------------------------------------
    # 取非空值的交集，去除GPP等于0的点
    # valid_indices = ~np.isnan(x) & ~np.isnan(y) & (y > 0)
    valid_indices = ~np.isnan(x) & ~np.isnan(y)
    # valid_indices = (y > 0)
    Tmean = x[valid_indices]
    GPP = y[valid_indices]

    # 取得该站点01-19年间的日最高温和日最低温
    Tmean_Tmax = np.nanmax(Tmean)
    Tmean_Tmin = np.nanmin(Tmean)

    bin_step = (Tmean_Tmax - Tmean_Tmin) / 30

    # 对 Tmean 进行分段，保留每段中 GPP 最大的 10个数
    segment_data = []  # 储存所有筛选出的散点

    mean_gpp_list = []  # 储存GPP筛选出的散点的平均值，温度大于-5
    bin_centers = []  # 储存温度组的平均值，温度大于-5
    mean_gpp_list2 = []  # 储存GPP筛选出的散点的平均值，温度大于0
    bin_centers2 = []  # 储存温度组的平均值，温度大于0

    # 对 Tmean 进行分段，从-5度开始，保留每段中 GPP 最大的20个数
    try:
        for t_bin_max in np.arange(Tmean_Tmax, max(Tmean_Tmin, -5), -bin_step):
            # 对 Tmean 进行分段
            mask_segment = (Tmean >= t_bin_max - bin_step) & (Tmean <= t_bin_max)
            if np.any(mask_segment):
                GPP_segment = GPP[mask_segment]
                Tmean_segment = Tmean[mask_segment]

                # 计算平均值和标准差
                mean_gpp = np.mean(GPP_segment)
                std_gpp = np.std(GPP_segment)

                # 剔除超过平均值 ± 2 倍标准差的离群值
                mask_outliers = (GPP_segment >= mean_gpp - 3 * std_gpp) & (GPP_segment <= mean_gpp + 3 * std_gpp)
                GPP_segment_filtered = GPP_segment[mask_outliers]
                Tmean_segment_filtered = Tmean_segment[mask_outliers]

                # 取出最大的10个元素，若不足10个则全部纳入
                filter_top_count = f_count
                if len(GPP_segment_filtered) > filter_top_count:
                    top_indices = np.argpartition(GPP_segment_filtered, -filter_top_count)[-filter_top_count:]
                    GPP_top_values = GPP_segment_filtered.iloc[top_indices]
                    Tmean_top_values = Tmean_segment_filtered.iloc[top_indices]
                elif len(GPP_segment_filtered) > 5:  # 只有大于5个元素才能计算移动平均
                    GPP_top_values = GPP_segment_filtered
                    Tmean_top_values = Tmean_segment_filtered
                else:
                    continue

                GPP_percentile_threshold = np.mean(GPP_top_values)  # 二次清洗，去除上界防止极高值影响
                valid_indices = np.where(GPP_top_values <= GPP_percentile_threshold)[0]  # 获得所有小于该组GPP平均值的索引

                # 根据索引保留相应的元素
                GPP_top_values = GPP_top_values.iloc[valid_indices]
                Tmean_top_values = Tmean_top_values.iloc[valid_indices]

                segment_data.extend(zip(Tmean_top_values, GPP_top_values))
                mean_gpp_list.append(np.mean(GPP_top_values))
                tmean_bin_center = t_bin_max - 0.5 * bin_step
                bin_centers.append(tmean_bin_center)  # 温度段的中心点

                # 新建俩数组用于分别储存0℃以上的GPP与温度区间
                if tmean_bin_center > 0 and len(GPP_segment_filtered) > 5:
                    bin_centers2.append(tmean_bin_center)
                    mean_gpp_list2.append(np.mean(GPP_top_values))

    except ValueError as e:
        print(e)
        param_fit = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        return param_fit

    # 如果mean_gpp_list2中的有效值太少了，则跳过
    if len(mean_gpp_list2) < 5:
        param_fit = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        return param_fit

    # 转换为数组
    # Tmean_filtered, GPP_filtered = zip(*segment_data)
    # Tmean_filtered = np.array(Tmean_filtered)
    # GPP_filtered = np.array(GPP_filtered)

    # ----------------------------------初始参数估计------------------------------------------------------------
    # 输入参数
    ki_const = max(mean_gpp_list2)  # 最大的 GPP 平均值，作为WE曲线的最大值常数
    max_gpp_index = np.argmax(mean_gpp_list2)  # 最大值的索引
    Topt_input = bin_centers2[max_gpp_index]  # 对应的温度，作为最适温度迭代参数的输入
    Tmax_input = Tmean_Tmax  # 最大温度初始值定义为日最高温
    Tmin_input = Tmean_Tmin  # 最低温度初始值定义为日最低温

    # 参数范围
    Tmin_bound = [max(-20, Tmin_actual_input - 5), min(25, Topt_input)]
    if (Tmin_actual_input - 5) > 25:  # 热带极高温区域特殊处理
        Tmin_bound = [20, 25]
        Tmin_input = 20
    Tmax_bound = [max(5, Topt_input), min(45, Tmax_actual_input + 5)]
    Topt_bound = [max(0, Topt_input - 3, Tmin_actual_input), min(40, Topt_input + 3, Tmax_actual_input)]
    beta_bound = [0.5, 5]

    bounds = [Tmin_bound, Topt_bound, Tmax_bound, beta_bound]

    # 使用 scipy.optimize.minimize 进行拟合，参数顺序依次为Tmin, Topt, Tmax, beta
    initial_guess = [Tmin_input, Topt_input, Tmax_input, 1]  # 初始猜测值

    # print(bounds)
    # print(initial_guess)

    # 定义不等式约束 Tmin + 2 < Topt < Tmax + 2
    constraints = [
        {'type': 'ineq', 'fun': lambda x: x[1] - x[0] - 2},  # Topt > Tmin
        {'type': 'ineq', 'fun': lambda x: x[2] - x[1] - 2},  # Tmax > Topt
    ]

    # ---------------------------使用 minimize 进行拟合，结果单调性判断与输出----------------------------------------------

    try:
        result = minimize(
            residuals,
            initial_guess,
            # args=(Tmean_filtered, GPP_filtered, ki_const),  # 筛选出来的散点进行拟合
            args=(bin_centers, mean_gpp_list, ki_const),  # 分组平均拟合
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 10000}
        )

    except ValueError as e:
        print(e)
        print(Tmin_actual_input)
        print(Tmax_actual_input)
        print(bounds)
        param_fit = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        return param_fit

    # 获取拟合结果
    if result.success:
        Tmin_fit, Topt_fit, Tmax_fit, beta_fit = result.x
        Tmin_adjust, Tmax_adjust = adjust_TminTmax(Tmin_fit, Topt_fit, Tmax_fit, beta_fit, ki_const)
        # 曲线单调性检验
        if Topt_fit >= Tmean_Tmax - bin_step:
            Mono = 1
            Monotonicity = "Increasing"

            # 有低温胁迫，无高温胁迫
            high_T_stress_index = 0
            low_T_stress_index = 1 - WE_curve(Tmean_Tmin, Tmin_adjust, Topt_fit, Tmax_adjust, beta_fit,
                                              ki_const) / ki_const

        elif Topt_fit <= Tmean_Tmin + bin_step:
            Mono = 2
            Monotonicity = "Decreasing"
            # 有高温胁迫，无低温胁迫
            high_T_stress_index = 1 - WE_curve(Tmean_Tmax, Tmin_adjust, Topt_fit, Tmax_adjust, beta_fit,
                                               ki_const) / ki_const
            low_T_stress_index = 0

        else:
            Mono = 3
            Monotonicity = "WE Curve"
            high_T_stress_index = 1 - WE_curve(Tmean_Tmax, Tmin_adjust, Topt_fit, Tmax_adjust, beta_fit,
                                               ki_const) / ki_const
            low_T_stress_index = 1 - WE_curve(Tmean_Tmin, Tmin_adjust, Topt_fit, Tmax_adjust, beta_fit,
                                              ki_const) / ki_const

        param_fit = (
            Topt_input, ki_const, Tmin_fit, Topt_fit,
            Tmax_fit, beta_fit, Tmin_adjust,
            Tmax_adjust, low_T_stress_index, high_T_stress_index, Mono)

    # 拟合失败
    else:
        Mono = 0
        Monotonicity = "Failed"
        param_fit = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, Mono
    return param_fit


# 读取单个 GPP 文件，并返回相应的窗口数据。使用 index 确保返回的数据按文件顺序排列。
def read_gpp_tmean_file(index, gpp_file, a, b, gpp_folder, tmean_folder):
    rows_gpp = (360 * a, 360 * (a + 1))
    cols_gpp = (720 * b, 720 * (b + 1))

    year = gpp_file.split("_")[1][0:4]
    doy = gpp_file.split("_")[1][4:7]
    month_tmean, day_tmean = calculate_date_from_doy(year, doy)
    tmean_file = f"ERA5Land_Tmean_{year}-{month_tmean}-{day_tmean}_{doy}.tif"

    with rasterio.open(os.path.join(gpp_folder, gpp_file)) as gpp_src:
        window_gpp = Window(cols_gpp[0], rows_gpp[0], cols_gpp[1] - cols_gpp[0], rows_gpp[1] - rows_gpp[0])
        data_gpp = gpp_src.read(1, window=window_gpp)

    with rasterio.open(os.path.join(tmean_folder, tmean_file)) as tmean_src:
        data_tmean = tmean_src.read(1, window=window_gpp)

    return index, data_gpp, data_tmean


# 使用并行处理方式读取 GPP 文件，并返回按日期顺序排列的三维数组。
def main_reading(gpp_list, input_gpp_folder, input_tmean_folder, a, b):
    n = len(gpp_list)
    # print(n)
    # 创建空的 GPP Tmean三维数组
    timeseries_gpp_reading = np.zeros((360, 720, n), dtype=np.float32)
    timeseries_tmean_reading = np.zeros((360, 720, n), dtype=np.float32)

    # 使用进程池并行读取 GPP 数据
    with ProcessPoolExecutor(max_workers=32) as executor:
        # 通过 enumerate 给每个文件分配一个索引，并直接调用 read_gpp_file
        results = list(
            executor.map(read_gpp_tmean_file,  # index, gpp_file, a, b, input_gpp_folder, input_tmean_folder
                         range(n), gpp_list, [a] * n, [b] * n,
                         [input_gpp_folder] * n, [input_tmean_folder] * n))

    # 将读取的结果按原始文件顺序填入 timeseries_reading
    for index_i, data_gpp, data_tmean in results:
        timeseries_gpp_reading[:, :, index_i], timeseries_tmean_reading[:, :, index_i] = data_gpp, data_tmean

    return timeseries_gpp_reading, timeseries_tmean_reading


def main_identifying(ax_gpp, ax_tmean, input_array_season1, input_array_season2, day1_array, day2_array):
    # 将数组分块，例如分为36行块和72列块（block代表每块大小为10x10的GPP与Tmean矩阵）
    block_size = 10
    blocks = [
        (i, j, ax_gpp[i:i + block_size, j:j + block_size, :], ax_tmean[i:i + block_size, j:j + block_size, :])
        for i in range(0, 360, block_size)
        for j in range(0, 720, block_size)
    ]

    with ProcessPoolExecutor(max_workers=32) as executor:
        futures = {executor.submit(calculate_block_topt, block): block for block in blocks}

        with tqdm(total=len(blocks), desc="计算进度") as pbar:
            fit_results = []
            for future in as_completed(futures):
                fit_results.append(future.result())  # 获取计算结果
                pbar.update(1)  # 更新进度条

    print("计算完成，正在写入……")
    # 将结果写入 input_arrays 中，根据索引进行逐像元赋值
    # for block_fit in tqdm(as_completed(fit_results), total=len(blocks), desc="Processing"):
    for block_fit in fit_results:
        for i, j, para_result1, para_result2, v_day1, v_day2 in block_fit:
            for year_idx, param_single_year in enumerate(para_result1):
                for variable_idx, variable_value in enumerate(param_single_year):
                    input_array_season1[variable_idx][i, j, year_idx] = variable_value

            for year_idx, param_single_year in enumerate(para_result2):
                for variable_idx, variable_value in enumerate(param_single_year):
                    input_array_season2[variable_idx][i, j, year_idx] = variable_value

            day1_array[i, j] = v_day1
            day2_array[i, j] = v_day2

    return input_array_season1, input_array_season2, day1_array, day2_array


def calculate_block_topt(block_data):
    i_start, j_start, block_gpp, block_tmean = block_data
    results = []
    block_size_x, block_size_y, _ = block_gpp.shape

    for i in range(block_size_x):
        for j in range(block_size_y):
            # 调用逐像元计算逻辑
            pixel_result = identify_phenology(
                (i_start + i, j_start + j, block_gpp[i, j, :], block_tmean[i, j, :]))
            results.append(pixel_result)

    return results


def identify_phenology(pixel_data):
    i_index, j_index, gpp, tmean = pixel_data

    param_nan = [[None] * 11 for _ in range(15)]

    if np.count_nonzero(~np.isnan(gpp)) < 3650 or np.count_nonzero(~np.isnan(tmean)) < 3650:  # GPP/Tmean数据量少于10年则跳过
        return i_index, j_index, param_nan, param_nan, np.nan, np.nan

    # 生成从2001-01-01到2019-12-31的日期范围，每天一条记录，不包含闰年的2月29日
    date_range = pd.date_range(start='2001-01-01', end='2019-12-31', freq='D')
    date_range = date_range[~((date_range.month == 2) & (date_range.day == 29))]

    if len(gpp) != len(date_range):
        print(len(gpp), len(date_range))
        raise ValueError("GPP数据数量与日期索引不匹配")

    df = pd.DataFrame({"Date": date_range, "GPP": gpp, "Tmean": tmean}).set_index("Date")
    # 将生成的日期赋值给DataFrame，作为新的列，将'Date'列设置为DataFrame的索引
    df['Date'] = date_range
    df.set_index('Date', inplace=True)
    GPP = df['GPP']
    df['Tmean'] = df['Tmean'].replace([-9999, '', None], np.nan)

    GPP_fill = GPP.interpolate(method='spline', order=3)
    GPP_fill = GPP_fill.ffill().bfill()
    GPP_fill[GPP_fill < 0] = 0

    # Tmean[Tmean < - 50] = np.nan
    # Tmean_fill = Tmean.interpolate(method='spline', order=3)
    # Tmean_fill = Tmean_fill.ffill().bfill()
    # Tmean_seasonal = seasonal_decompose(Tmean_fill, model='additive', period=365).seasonal
    # Tmean_seasonal_SG = savgol_filter(Tmean_seasonal, window_length=91, polyorder=2)
    # df['Tmean_seasonal_SG'] = Tmean_seasonal_SG + Tmean_fill.mean()

    # SG滤波处理 + 季节性分解的季节（假设年周期为365天）
    GPP_seasonal = seasonal_decompose(GPP_fill, model='additive', period=365).seasonal
    GPP_seasonal_SG = savgol_filter(GPP_seasonal, window_length=91, polyorder=2)
    df['GPP_seasonal_SG'] = GPP_seasonal_SG

    # 取相反数，提取谷值
    GPP_seasonal_SG_negative = - GPP_seasonal_SG
    df['GPP_seasonal_SG_negative'] = GPP_seasonal_SG_negative
    df_09_10 = df["2009-01-01": "2010-12-31"]
    SG_max, SG_min = df_09_10["GPP_seasonal_SG_negative"].max(), df_09_10["GPP_seasonal_SG_negative"].min()
    denominator = SG_max - SG_min

    # 检查分母是否接近于零
    if np.isclose(denominator, 0):
        return i_index, j_index, param_nan, param_nan, np.nan, np.nan

    GPP_seasonal_SG_negative_normalize = (GPP_seasonal_SG_negative - SG_min) / denominator
    df["GPP_seasonal_SG_negative_normalize"] = GPP_seasonal_SG_negative_normalize
    df["GPP_seasonal_normalize"] = 1 - GPP_seasonal_SG_negative_normalize

    # 识别波谷
    peaks_negative, properties_negative = find_peaks(GPP_seasonal_SG_negative_normalize,
                                                     height=0.1,
                                                     distance=90,
                                                     prominence=0.06,
                                                     )

    peak_dates_negative = df.index[peaks_negative]  # 利用peaks索引从df.index取出日期
    # peak_values = df['GPP_seasonal_normalize'].iloc[peaks_negative]  # 从GPP_seasonal_SG中取出相应值
    # filtered_dates = peak_dates_negative[peak_dates_negative >= '2002-01-01']  # 筛选出2002年开始及其之后的所有时间

    valley = []
    for i in range(len(peak_dates_negative[peak_dates_negative.year == 2010])):
        valley.append(peak_dates_negative[peak_dates_negative.year == 2010][i].strftime('%m-%d'))

    period_all = []  # 用于储存不同作物的时间段，根据valley中的日期进行设置
    period = []  # 用于储存不同作物的时间段，period_all筛选后的时间段
    peak_time = []
    period_TminTmax = []

    for v in range(len(valley)):
        period_all.append((valley[v - 1], valley[v]))

    if len(period_all) != 1:
        df['month_day'] = df.index.strftime('%m-%d')  # month_day字段代表了月份-日期，且都是两位字符串

        for i in period_all:
            verified_result = verify_peak(i, df)


            if verified_result[0]:
                period.append(i)
                period_TminTmax.append([verified_result[1], verified_result[2]])
                peak_time.append(verified_result[3])


    else:
        period = period_all

    param_fit_result_season1 = []
    param_fit_result_season2 = []


    def extract_period(time1, time2, dataframe):
        if time1 > time2:
            return ~ dataframe['month_day'].between(time2, time1)
        else:
            return dataframe['month_day'].between(time1, time2)

    is_2 = False
    # ---------------------------------------双峰型拟合------------------------------------------------------------------
    if len(period) == 2:
        is_2 = True
        fit_year = 2001
        while is_2 and fit_year <= 2015:
            df_fit = df[(f"{fit_year}-01-01" <= df.index) & (df.index <= f"{fit_year + 4}-12-31")]

            if peak_time[0] > peak_time[1]:  # period[0]为第二季
                season1_date = extract_period(period[1][0], period[1][1], df_fit)
                season2_date = extract_period(period[0][0], period[0][1], df_fit)
                valley_day1 = (datetime.strptime(period_all[1][0], "%m-%d") - datetime(1900, 1, 1)).days + 1
                valley_day2 = (datetime.strptime(period_all[0][0], "%m-%d") - datetime(1900, 1, 1)).days + 1
                season1_Tmin_actual, season1_Tmax_actual = period_TminTmax[1][0], period_TminTmax[1][1]
                season2_Tmin_actual, season2_Tmax_actual = period_TminTmax[0][0], period_TminTmax[0][1]

            else:  # period[0]为第一季
                season1_date = extract_period(period[0][0], period[0][1], df_fit)
                season2_date = extract_period(period[1][0], period[1][1], df_fit)
                valley_day1 = (datetime.strptime(period_all[0][0], "%m-%d") - datetime(1900, 1, 1)).days + 1
                valley_day2 = (datetime.strptime(period_all[1][0], "%m-%d") - datetime(1900, 1, 1)).days + 1
                season1_Tmin_actual, season1_Tmax_actual = period_TminTmax[0][0], period_TminTmax[0][1]
                season2_Tmin_actual, season2_Tmax_actual = period_TminTmax[1][0], period_TminTmax[1][1]

            # 即，先输出的永远是season1，第一季作物
            GPP_season1 = df_fit["GPP"][season1_date]
            GPP_season2 = df_fit["GPP"][season2_date]
            Tmean_season1 = df_fit["Tmean"][season1_date]
            Tmean_season2 = df_fit["Tmean"][season2_date]

            # if (len(GPP_season2) != len(Tmean_season2)) or (len(GPP_season1) != len(Tmean_season1)):
            #     raise RuntimeError

            # 分别拟合并计算两种作物的温度三基点
            params_season1 = calculate_topt(Tmean_season1, GPP_season1, season1_Tmin_actual, season1_Tmax_actual, 8)
            params_season2 = calculate_topt(Tmean_season2, GPP_season2, season2_Tmin_actual, season2_Tmax_actual, 8)

            if params_season1[3] == np.nan or params_season2[3] == np.nan:  # 存在假峰，按照单峰型处理
                is_2 = False
                break
            else:  # 两个峰均有效
                param_fit_result_season1.append(params_season1)
                param_fit_result_season2.append(params_season2)
            fit_year += 1
        if is_2:
            return i_index, j_index, param_fit_result_season1, param_fit_result_season2, valley_day1, valley_day2
        else:
            param_fit_result_season1 = []
            param_fit_result_season2 = []

    # ---------------------------------------单峰型输出，峰值过多也按照单峰型处理--------------------------------------------
    # df = df.dropna()
    # 单峰拟合，或者三峰的情况也当做单峰进行拟合（因为五年之内，三峰的数据量太少了）len(period) <= 1 or len(period) >= 3
    else:
        for fit_year in range(2001, 2016):
            df_fit = df[(f"{fit_year}-01-01" <= df.index) & (df.index <= f"{fit_year + 4}-12-31")]
            Tmean = df_fit["Tmean"]
            GPP = df_fit["GPP"]

            try:
                param_fit_result = calculate_topt(Tmean, GPP, np.nanmin(tmean), np.nanmax(tmean), 16)
            except Exception as e:
                print(e)
                param_fit_result = [np.nan for _ in range(11)]
            param_fit_result_season1.append(param_fit_result)

        return i_index, j_index, param_fit_result_season1, param_nan, np.nan, np.nan


# (Topt_input, ki_const, Tmin_fit, Topt_fit, Tmax_fit, beta_fit, Tmin_adjust, Tmax_adjust,
#                      low_T_stress_index, high_T_stress_index, Mono)

if __name__ == "__main__":
    # 忽略 RuntimeWarning
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # 存放日尺度的GPP与日均温tif的文件夹路径，过滤出所有 .tif 文件
    tmean_folder = r"E:\ERA5 LAND Tmean"
    gpp_folder = r"F:\FluxSatGPP\FluxSat_GPP_original"

    gpp_list = [file for file in os.listdir(gpp_folder) if file.endswith('.tif')]
    print(f"一共{len(gpp_list)}张GPP文件")
    if len(gpp_list) != 6935:
        raise RuntimeError("GPP文件数量异常")

    # 输出的母文件夹+开始年份
    # output_folder = r"D:\Data\codes\optimum_temperature_output\WE法\WE 6.5.0 每五年拟合一张 双季"
    output_folder = "v6.7.0 5yr output"
    os.makedirs(output_folder, exist_ok=True)
    output_variables_list = ["Topt-initial", "GPPmax", "Tmin-fit", "Topt-fit", "Tmax-fit", "beta-fit", "Tmin-adjust",
                             "Tmax-adjust", "Tmin-stress-index", "Tmax-stress-index", "Monotonicity"]

    # --------------------------Tmean与GPP时间序列读取，每三年读取一次，共分为10*10的100个小块区域-------------------------------------------

    # for a in range(3, 7):
    #     for b in range(10):
    # for b in range(7, 9):
    # for a in range(2, 3):
    #     for b in range(4, 6):

    for a in range(10):
        for b in range(10):
            start_time = time.time()

            # 读取GPP数据的第一张，获取区域坐标系和范围、仿射变换
            rows_gpp = (360 * a, 360 * (a + 1))
            cols_gpp = (720 * b, 720 * (b + 1))
            window_gpp = Window(cols_gpp[0], rows_gpp[0], cols_gpp[1] - cols_gpp[0], rows_gpp[1] - rows_gpp[0])
            with rasterio.open(os.path.join(gpp_folder, gpp_list[0])) as gpp_src:
                transform_gpp = gpp_src.window_transform(window_gpp)  # 仿射变换
                crs_gpp = gpp_src.crs  # 坐标系

            # 生成输出文件并检验11个tif是否已经全部输出
            eleven_tifs_already_exists = True

            for season_number in [1, 2]:
                # 检查15年11个变量
                for yr in range(2001, 2016):
                    for v in output_variables_list:
                        if not os.path.exists(os.path.join(output_folder, str(yr), "season" + str(season_number), v, f"{v}_a={a}_b={b}.tif")):
                            eleven_tifs_already_exists = False
                        os.makedirs(os.path.join(output_folder, str(yr), "season" + str(season_number), v), exist_ok=True)

                # 检查2个物候文件
                if not os.path.exists(os.path.join(output_folder, "SOS" + str(season_number), f"SOS{season_number}_a={a}_b={b}.tif")):
                    eleven_tifs_already_exists = False
                os.makedirs(os.path.join(output_folder, "SOS" + str(season_number)), exist_ok=True)

            if eleven_tifs_already_exists:
                print(f"所有15张 a={a}, b={b} 文件已存在")
                continue

            # ---------------------------如果是空值区域，判断并直接输出全0矩阵------------------------------
            if (a, b) in [(0, 0), (3, 9), (4, 0), (4, 1), (6, 1), (6, 4), (6, 7), (7, 1), (7, 4), (7, 5), (8, 0),
                          (8, 1), (8, 5), (8, 6), (8, 7), (8, 8), (9, 0), (9, 1), (9, 2), (9, 3), (9, 4), (9, 5),
                          (9, 6), (9, 7), (9, 8), (9, 9)]:
                # 初始化存储拟合系数的数组，11个输出变量，默认全为nodata
                array_zero_input = np.full((360, 720), np.nan, dtype=np.float32)
                for season_number in [1, 2]:
                    for yr in range(2001, 2016):
                        for output_variable in output_variables_list:
                            output_subfolder = os.path.join(output_folder, str(yr), "season" + str(season_number), output_variable)
                            output_filename = os.path.join(output_subfolder, f"{output_variable}_a={a}_b={b}.tif")
                            # 保存全0矩阵为 TIF
                            with rasterio.open(
                                    output_filename, 'w',
                                    driver='GTiff', height=array_zero_input.shape[0], width=array_zero_input.shape[1],
                                    count=1, dtype=array_zero_input.dtype, crs=crs_gpp, transform=transform_gpp
                            ) as dst:
                                dst.write(array_zero_input, 1)
                    output_SOS_filename = os.path.join(output_folder, "SOS" + str(season_number), f"SOS{season_number}_a={a}_b={b}.tif")
                    with rasterio.open(
                            output_SOS_filename, 'w',
                            driver='GTiff', height=array_zero_input.shape[0], width=array_zero_input.shape[1],
                            count=1, dtype=array_zero_input.dtype, crs=crs_gpp, transform=transform_gpp
                    ) as dst:
                        dst.write(array_zero_input, 1)

                print(f"所有15张空值 a={a}, b={b}, 输出完毕 nan")
                continue

            # ------------------------------如果是有值区域，提取后计算--------------------------------------------------
            # 读取 GPP 数据并处理
            print(f"\n开始读取a={a}, b={b}……")
            # 读取 GPP 数据
            timeseries_gpp, timeseries_tmean = main_reading(gpp_list, gpp_folder, tmean_folder, a, b)

            end_time1 = time.time()
            print(f"读取用时{end_time1 - start_time:.2f}s")

            # ---------------------------调用函数进行多线程计算，提取GPP物候曲线的谷值----------------------------------------------

            # 初始化存储拟合系数的数组
            array11_zero_season1_input = [np.zeros((360, 720, 15), dtype=np.float32) for _ in range(11)]  # 第二季植被15年11组参数
            array11_zero_season2_input = [np.zeros((360, 720, 15), dtype=np.float32) for _ in range(11)]  # 第二季植被15年11组参数
            day1_array_input, day2_array_input = np.zeros((360, 720), dtype=np.float32), np.zeros((360, 720), dtype=np.float32)

            # 计算时间序列拟合的函数，逐像元多核计算
            array11_season1, array11_season2, day1_array, day2_array = (
                main_identifying(timeseries_gpp, timeseries_tmean,
                                 array11_zero_season1_input, array11_zero_season2_input,
                                 day1_array_input, day2_array_input))
            # ---------------------------------# 两季植被数据输出--------------------------------------------------------------------

            for season_number, array11_zero_output in enumerate([array11_season1, array11_season2]):
                for index, array_output in enumerate(array11_zero_output):
                    output_variable = output_variables_list[index]
                    for y_folder in range(2001, 2016):
                        output_subfolder = os.path.join(output_folder, str(y_folder), "season" + str(season_number + 1), output_variable)
                        output_filename = os.path.join(output_subfolder, f"{output_variable}_a={a}_b={b}.tif")
                        array_year = array_output[:, :, y_folder - 2001]
                        # 保存拟合结果为 TIF
                        with rasterio.open(
                                output_filename,
                                'w',
                                driver='GTiff',
                                height=array_year.shape[0],
                                width=array_year.shape[1],
                                count=1,
                                dtype=array_year.dtype,
                                crs=crs_gpp,
                                transform=transform_gpp
                        ) as dst:
                            dst.write(array_year, 1)

                #    保存两张SOS物候文件
                SOS_output = day1_array if season_number == 0 else day2_array
                output_SOS_filename = os.path.join(output_folder, "SOS" + str(season_number + 1), f"SOS{season_number + 1}_a={a}_b={b}.tif")
                with rasterio.open(
                        output_SOS_filename,
                        'w',
                        driver='GTiff',
                        height=SOS_output.shape[0],
                        width=SOS_output.shape[1],
                        count=1,
                        dtype=SOS_output.dtype,
                        crs=crs_gpp,
                        transform=transform_gpp
                ) as dst:
                    dst.write(SOS_output, 1)

            end_time2 = time.time()
            print(f"任务分配与计算用时{end_time2 - end_time1:.2f}s.")
            print(f"a={a}, b={b}, 输出完毕")

            del timeseries_gpp, timeseries_tmean
            # 释放内存
            gc.collect()
