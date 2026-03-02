import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import re

# ================= 配置区域 =================
MODEL_PATH = r'G:\AI_QW\models\macro_direction_classifier_v2.json'
LOAD_FILE = r'G:\AI_QW\cleaned_power_data_READY.csv'  # 用于提取历史负荷规律
# ===========================================

st.set_page_config(page_title="电力套利 AI 助手", page_icon="⚡", layout="wide")
st.title("⚡ 电力套利 AI 助手 (基于供需比)")
st.markdown("上传日前预测文件，AI 将结合**历史负荷规律**与**当日发电预测**，生成交易指令！")

uploaded_file = st.file_uploader("选择日前预测 Excel 文件", type=["xlsx"])

if uploaded_file is not None:
    st.success(f"已上传：{uploaded_file.name}")

    if st.button("🚀 开始分析"):
        with st.spinner("正在融合数据并生成指令..."):
            try:
                # --- 1. 读取并处理发电数据 (用户输入的 6 月数据) ---
                xl_gen = pd.ExcelFile(uploaded_file)
                target_sheet = None
                for s in xl_gen.sheet_names:
                    if '负荷' in s or '预测' in s:
                        target_sheet = s
                        break
                if not target_sheet:
                    target_sheet = xl_gen.sheet_names[0]

                df_raw = pd.read_excel(uploaded_file, sheet_name=target_sheet, header=None)

                # 智能探测表头
                header_row_idx = -1
                for i in range(min(5, len(df_raw))):
                    row_values = df_raw.iloc[i].astype(str).tolist()
                    if '类型' in row_values and '通道名称' in row_values:
                        header_row_idx = i
                        break

                if header_row_idx == -1:
                    st.error("❌ 无法识别表头")
                    st.stop()

                df_gen = pd.read_excel(uploaded_file, sheet_name=target_sheet, header=header_row_idx)
                df_gen = df_gen.iloc[header_row_idx + 1:].reset_index(drop=True)
                df_gen.columns = [str(c).strip() for c in df_gen.columns]

                if '类型' not in df_gen.columns or '通道名称' not in df_gen.columns:
                    st.error("❌ 找不到必要列")
                    st.stop()

                # 变形 & 聚合发电数据
                id_cols = ['类型', '通道名称']
                time_cols = [c for c in df_gen.columns if c not in id_cols]
                melted_gen = pd.melt(df_gen, id_vars=id_cols, value_vars=time_cols, var_name='预测时刻',
                                     value_name='预测数值')
                melted_gen['预测数值'] = pd.to_numeric(melted_gen['预测数值'], errors='coerce')
                melted_gen = melted_gen.dropna(subset=['预测数值'])

                # 提取日期
                file_name = uploaded_file.name
                date_str = "Unknown"
                match = re.search(r'\((\d{4}-\d{2}-\d{2})\)', file_name)
                if match:
                    date_str = match.group(1)
                melted_gen['交易日期'] = date_str

                # 排除负荷，计算总供给
                gen_rows = melted_gen[~melted_gen['通道名称'].str.contains('负荷', na=False)]
                gen_agg = gen_rows.groupby(['交易日期', '预测时刻'])['预测数值'].sum().reset_index()
                gen_agg.rename(columns={'预测数值': '总预测供给'}, inplace=True)

                # --- 【核心修改】2. 构建“典型负荷曲线” (从 5 月历史数据学习) ---
                if not os.path.exists(LOAD_FILE):
                    st.error(f"❌ 找不到历史负荷文件：{LOAD_FILE}")
                    st.stop()

                df_load_ref = pd.read_csv(LOAD_FILE)
                # 清洗时间
                df_load_ref['预测时刻'] = df_load_ref['Time_Str'].astype(str).apply(
                    lambda x: ':'.join(x.split(':')[:2]))

                # 识别负荷列
                load_cols = [c for c in df_load_ref.columns if
                             'Load' in c and 'Real' not in c and 'RT' not in c and 'Spr' not in c]

                if load_cols:
                    df_load_ref['总预测负荷'] = df_load_ref[load_cols].sum(axis=1)

                    # 计算每个时刻的平均负荷 (典型日负荷曲线)
                    typical_load = df_load_ref.groupby('预测时刻')['总预测负荷'].mean().reset_index()
                    typical_load.rename(columns={'总预测负荷': '典型负荷'}, inplace=True)

                    # 将典型负荷匹配到 6 月数据上
                    gen_agg = pd.merge(gen_agg, typical_load, on='预测时刻', how='left')

                    # 如果匹配失败（比如时刻格式不对）， fallback 到估算
                    if gen_agg['典型负荷'].isna().all():
                        st.warning("⚠️ 时刻匹配失败，使用估算值。")
                        gen_agg['总预测负荷'] = gen_agg['总预测供给'] * 1.2
                    else:
                        gen_agg['总预测负荷'] = gen_agg['典型负荷']
                        st.info(f"✅ 已应用历史典型负荷曲线 (基于 {len(df_load_ref)} 条历史记录)。")
                else:
                    st.error("❌ 未找到负荷列。")
                    st.stop()

                # --- 3. 构造宏观特征 ---
                gen_agg['净负荷'] = gen_agg['总预测负荷'] - gen_agg['总预测供给']
                gen_agg['供需比'] = gen_agg['总预测供给'] / (gen_agg['总预测负荷'] + 1e-6)
                gen_agg['新能源预测'] = 0  # 暂时忽略
                gen_agg['小时'] = gen_agg['预测时刻'].apply(lambda x: int(str(x).split(':')[0]))
                gen_agg['是否周末'] = 0
                gen_agg['Price_DA'] = 350.0  # 假设常数

                # --- 4. 加载模型并预测 ---
                model = xgb.XGBClassifier()
                model.load_model(MODEL_PATH)

                feature_cols = ['净负荷', '供需比', '新能源占比', 'Price_DA', '小时', '是否周末']
                # 确保列存在
                for col in feature_cols:
                    if col not in gen_agg.columns:
                        gen_agg[col] = 0

                X_pred = gen_agg[feature_cols]

                predictions = model.predict(X_pred)
                probabilities = model.predict_proba(X_pred)

                # --- 5. 整理结果 ---
                gen_agg['预测标签'] = predictions
                gen_agg['置信度'] = probabilities.max(axis=1)
                label_map = {0: '买小', 1: '观望', 2: '买大'}
                gen_agg['交易指令'] = gen_agg['预测标签'].map(label_map)

                result_df = gen_agg[
                    ['交易日期', '预测时刻', '总预测供给', '总预测负荷', '净负荷', '交易指令', '置信度']]

                # --- 6. 展示结果 ---
                st.subheader("📊 基于供需比的交易指令")
                st.dataframe(result_df, use_container_width=True)

                # 高亮
                high_conf_buy_big = result_df[(result_df['交易指令'] == '买大') & (result_df['置信度'] > 0.7)]
                high_conf_buy_small = result_df[(result_df['交易指令'] == '买小') & (result_df['置信度'] > 0.7)]

                if not high_conf_buy_big.empty or not high_conf_buy_small.empty:
                    st.subheader("🔥 高置信度机会")
                    col1, col2 = st.columns(2)
                    with col1:
                        if not high_conf_buy_big.empty:
                            st.write("**买大 (预计供不应求)**")
                            for _, row in high_conf_buy_big.iterrows():
                                st.write(
                                    f"- ⏰ **{row['预测时刻']}**: 净负荷 {row['净负荷']:.1f} MW (置信度：{row['置信度']:.2%})")
                    with col2:
                        if not high_conf_buy_small.empty:
                            st.write("**买小 (预计供过于求)**")
                            for _, row in high_conf_buy_small.iterrows():
                                st.write(
                                    f"- ⏰ **{row['预测时刻']}**: 净负荷 {row['净负荷']:.1f} MW (置信度：{row['置信度']:.2%})")

                st.success("✅ 分析完成！模型已利用 5 月规律预测 6 月走势。")

            except Exception as e:
                st.error(f"❌ 处理失败：{e}")
                st.exception(e)