from pathlib import Path
import hashlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from src.data_processing import TARGET, get_feature_info, split_data, build_preprocessor
from src.feature_selection import get_feature_ranking, get_selected_features
from src.modeling import run_experiment

st.set_page_config(page_title='HW2 Multiple Linear Regression', page_icon='📊', layout='wide')
st.title('HW2 - Multiple Linear Regression')
st.caption('CRISP-DM | Step 5 | Two problems + Custom CSV / URL')
BASE = Path(__file__).resolve().parent

from src.data_sources import read_csv_bytes, read_public_https_csv, prepare_dataset

PROBLEMS = {
    'Student Performance Factors': {
        'target': 'Exam_Score', 'filename': 'StudentPerformanceFactors.csv',
        'url': 'https://www.kaggle.com/datasets/lainguyn123/student-performance-factors',
        'purpose': '預測學生考試成績',
    },
    'King County House Sales': {
        'target': 'price', 'filename': 'kc_house_data.csv',
        'url': 'https://www.kaggle.com/datasets/harlfoxem/housesalesprediction',
        'purpose': '預測房屋售價',
    },
}

st.sidebar.header('Problem Selection')
problem = st.sidebar.selectbox('選擇 Problem（不含 50 Startups）', list(PROBLEMS))
config = PROBLEMS[problem]
source = st.sidebar.radio('資料來源', ['內建本地 CSV', '上傳 CSV', '輸入 CSV URL'], index=0)

try:
    if source == '內建本地 CSV':
        path = BASE / 'data' / config['filename']
        if not path.exists():
            st.info(f"{problem} 尚未提供本地資料。請將 {config['filename']} 放到 data/，或改用左側「上傳 CSV」。")
            st.markdown(f"[下載 Kaggle 資料集]({config['url']})")
            st.stop()
        raw = path.read_bytes()
    elif source == '上傳 CSV':
        uploaded = st.sidebar.file_uploader('選擇 .csv 檔案', type=['csv'])
        if uploaded is None:
            st.info('請從左側上傳 CSV。可使用任何合適的回歸資料集。')
            st.stop()
        raw = uploaded.getvalue()
    else:
        url = st.sidebar.text_input('公開 HTTPS CSV 直連網址', placeholder='https://raw.githubusercontent.com/.../data.csv')
        if not url.strip():
            st.info('請輸入可直接下載 CSV 的公開 HTTPS 網址（不是 Kaggle 資料集介紹頁）。')
            st.stop()
        raw = read_public_https_csv(url)

    original_df = read_csv_bytes(raw)
    default_target = config['target'] if config['target'] in original_df.columns else original_df.columns[-1]
    TARGET = st.sidebar.selectbox('預測目標（數值欄位）',
        list(original_df.columns), index=list(original_df.columns).index(default_target))
    excluded_default = [c for c in ['id', 'ID', 'Unnamed: 0'] if c in original_df.columns and c != TARGET]
    excluded_cols = st.sidebar.multiselect('排除非預測欄位（ID / 洩漏欄位等）',
        [c for c in original_df.columns if c != TARGET], default=excluded_default)
    df = prepare_dataset(original_df, problem, TARGET, excluded_cols)
    X, y, numeric_features, categorical_features = get_feature_info(df, TARGET)
    X_train, X_test, y_train, y_test, num_cols, cat_cols = split_data(df, TARGET)

    # All parameters determining a model must be in the cache key.
    digest = hashlib.sha256(raw).hexdigest()
    @st.cache_resource(show_spinner=False, max_entries=4)
    def train_cached(file_hash, prob_name, target_name, exclusions, _df):
        a, b, c, d, n, cat = split_data(_df, target_name)
        return run_experiment(a, b, c, d, n, cat)

    with st.spinner('5-Fold CV + Regression + 95% interval...'):
        result = train_cached(digest, problem, TARGET, tuple(excluded_cols), df)
except Exception as exc:
    st.error(f'資料讀取或模型訓練失敗：{exc}')
    st.stop()

st.sidebar.divider()
st.sidebar.header('Model Summary')
st.sidebar.write('CV: 5-Fold / RMSE')
st.sidebar.metric('Best K', str(result['best_k']))
st.sidebar.metric('95% Interval Radius', f"{result['radius']:.3f}")
st.sidebar.caption('測試集只用於最終評估，不用來選 K 或校準區間。')

st.header('1. Business Understanding')
st.markdown(f"**{problem}**：{config['purpose']}；預測目標 `{TARGET}`。比較全部特徵與交叉驗證選定特徵的線性回歸，並評估 95% 預測區間。")
st.markdown(f"[Kaggle 原始資料來源]({config['url']})")

tabs = st.tabs(['📋 Data Understanding', '📊 EDA', '⚙️ Data Preparation', '🎯 Feature Selection', '📈 Model Evaluation', '📐 95% Prediction Interval'])
with tabs[0]:
    st.header('2. Data Understanding')
    a, b, c, d = st.columns(4)
    a.metric('Samples', len(df)); b.metric('Original Features', X.shape[1])
    c.metric('Numeric', len(numeric_features)); d.metric('Categorical', len(categorical_features))
    if 13 < X.shape[1] < 40:
        st.success('符合老師的 14–39 個原始輸入特徵規定。')
    else:
        st.warning('原始輸入特徵數不在 14–39 個之間。')
    st.subheader('Dataset Preview'); st.dataframe(df.head(20), use_container_width=True)
    st.subheader('Descriptive Statistics'); st.dataframe(df.describe(include='all').T, use_container_width=True)
    colinfo = pd.DataFrame({'Column': df.columns, 'Dtype': df.dtypes.astype(str).values, 'Missing': df.isna().sum().values, 'Unique': df.nunique().values})
    st.subheader('Column Information'); st.dataframe(colinfo, use_container_width=True)
    st.write('Duplicate Rows:', int(df.duplicated().sum()), ' | Total Missing Values:', int(df.isna().sum().sum()))

with tabs[1]:
    st.header('3. Exploratory Data Analysis')
    fig, ax = plt.subplots(figsize=(9, 4)); sns.histplot(y.dropna(), kde=True, bins=25, ax=ax)
    ax.set_title('Distribution of Target'); st.pyplot(fig); plt.close(fig)
    numerics = df.select_dtypes(include='number')
    if numerics.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(11, 7)); sns.heatmap(numerics.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Numeric Correlation Heatmap'); st.pyplot(fig); plt.close(fig)
        if TARGET in numerics.columns:
            st.dataframe(numerics.corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=False).to_frame('Correlation'), use_container_width=True)
    if numeric_features:
        feature = st.selectbox('Feature vs Target', numeric_features)
        fig, ax = plt.subplots(figsize=(9, 4)); sns.regplot(data=df, x=feature, y=TARGET, scatter_kws={'alpha': 0.25}, line_kws={'color': 'red'}, ax=ax)
        st.pyplot(fig); plt.close(fig)
    st.caption('EDA 描述整份資料；正式模型的前處理與選 K 僅使用訓練子集。')

with tabs[2]:
    st.header('4. Data Preparation')
    st.markdown('外層 80/20 Train/Test；訓練集再拆為 75/25 的 **Model Fit / Calibration**。數值補中位數、類別補眾數，並使用 One-Hot Encoding。')
    a, b, c, d = st.columns(4)
    a.metric('Model Fit', result['fit_size']); b.metric('Calibration', result['cal_size'])
    c.metric('Test', result['test_size']); d.metric('Encoded Features', result['n_encoded_features'])
    st.info('前處理、特徵選擇都包在 Pipeline 中。校準資料僅用於計算預測區間寬度。')
    # 僅展示已由訓練子集學習的前處理，不在此重新 fit
    fitted_preprocessor = result['selected'].named_steps['preprocessor']
    preview = pd.DataFrame(fitted_preprocessor.transform(X_train.head(10)), columns=fitted_preprocessor.get_feature_names_out())
    st.subheader('Processed Preview'); st.dataframe(preview, use_container_width=True)

with tabs[3]:
    st.header('5. Feature Selection — Cross Validation')
    st.write('最佳 K：', result['best_k'])
    st.write('選 K 依據：5-Fold Cross-Validation，平均 RMSE 越小越好。')
    cv_df = result['cv_table']
    st.subheader('CV Results'); st.dataframe(cv_df.style.format({'CV RMSE': '{:.4f}', 'CV Std': '{:.4f}', 'Train RMSE': '{:.4f}'}), use_container_width=True, hide_index=True)
    chart_df = cv_df.copy(); chart_df['K Label'] = chart_df['K'].astype(str)
    fig, ax = plt.subplots(figsize=(10, 4)); ax.plot(chart_df['K Label'], chart_df['CV RMSE'], marker='o')
    ax.set_xlabel('K (all = all encoded features)'); ax.set_ylabel('CV RMSE'); ax.set_title('Cross-validation RMSE by K'); ax.tick_params(axis='x', rotation=45)
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    rank = get_feature_ranking(result['selected'])
    st.subheader('Selected Features')
    st.dataframe(rank[rank['Selected']] if result['best_k'] != 'all' else rank, use_container_width=True, hide_index=True)
    st.subheader('Feature Ranking'); st.dataframe(rank, use_container_width=True, hide_index=True)
    st.caption('此 F-score 是單變數線性關聯評分，不代表因果關係。類別欄位會拆成 One-Hot 欄位。')

with tabs[4]:
    st.header('6. Model Evaluation')
    metrics = result['metrics']
    st.subheader('Baseline vs CV-Selected Model')
    st.dataframe(metrics.style.format({'MAE':'{:.4f}', 'MSE':'{:.4f}', 'RMSE':'{:.4f}', 'R2':'{:.4f}'}), hide_index=True, use_container_width=True)
    testmetrics = metrics[(metrics['Model']=='Selected (CV Best K)') & (metrics['Dataset']=='Test')].iloc[0]
    a,b,c,d = st.columns(4)
    a.metric('Test MAE',f"{testmetrics['MAE']:.4f}"); b.metric('Test MSE',f"{testmetrics['MSE']:.4f}")
    c.metric('Test RMSE',f"{testmetrics['RMSE']:.4f}"); d.metric('Test R²',f"{testmetrics['R2']:.4f}")
    detail = result['test_detail']
    fig, ax = plt.subplots(figsize=(7, 5)); ax.scatter(detail['Actual'],detail['Predicted'],alpha=.4)
    mn = min(detail['Actual'].min(),detail['Predicted'].min()); mx = max(detail['Actual'].max(),detail['Predicted'].max())
    ax.plot([mn,mx],[mn,mx],'r--',label='Perfect Prediction'); ax.set_xlabel('Actual'); ax.set_ylabel('Predicted'); ax.legend(); ax.set_title('Actual vs Predicted')
    st.pyplot(fig); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8,4)); ax.scatter(detail['Predicted'], detail['Residual'],alpha=.4); ax.axhline(0,color='red',linestyle='--')
    ax.set_xlabel('Predicted');ax.set_ylabel('Residual (Actual - Predicted)');ax.set_title('Residual Plot'); st.pyplot(fig);plt.close(fig)
    st.subheader('Test Predictions'); st.dataframe(detail.head(30), use_container_width=True)
    st.download_button('Download Test Predictions CSV',detail.to_csv(index=False).encode('utf-8-sig'),'hw2_step4_test_predictions.csv','text/csv')

with tabs[5]:
    st.header('7. 95% Prediction Interval — Split Conformal')
    st.markdown('在獨立 **Calibration** 資料集上計算 `abs(Actual - Predicted)`，以校正後的第 95 百分位數取得誤差半徑，形成 **Predicted ± Radius**。')
    a, b, c = st.columns(3)
    a.metric('Target Coverage','95%');b.metric('Observed Test Coverage',f"{result['interval_stats']['Coverage']:.2%}")
    c.metric('Average Interval Width',f"{result['interval_stats']['Mean Width']:.3f}")
    st.write('Calibration Residual Radius:',round(result['radius'],4))
    st.caption('95% 是在交換性假設下的邊際涵蓋率保證；測試集的實際涵蓋率不一定剛好是 95%。')
    detail = result['test_detail'].sort_values('Predicted').reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11,5))
    xp = range(len(detail))
    ax.fill_between(xp,detail['Lower 95%'],detail['Upper 95%'],alpha=.23,label='95% Prediction Interval')
    ax.plot(xp,detail['Predicted'],linewidth=1.3,label='Prediction')
    ax.scatter(xp,detail['Actual'],s=9,alpha=.55,label='Actual')
    ax.set_xlabel('Test Samples (Sorted by Prediction)');ax.set_ylabel('Target Value');ax.set_title('95% Prediction Intervals — Held-Out Test Set');ax.legend()
    fig.tight_layout();st.pyplot(fig);plt.close(fig)
    st.subheader('Interval Predictions');st.dataframe(detail.head(40),use_container_width=True)
    st.download_button('Download Intervals CSV',detail.to_csv(index=False).encode('utf-8-sig'),'hw2_step4_prediction_intervals.csv','text/csv')
    st.warning('圖中的上下界不刻意裁切至 0–100；這是線性模型搭配共形區間得到的原始範圍。')

st.divider()
st.caption('Step 5：兩個 Problem、CSV 上傳、公開 HTTPS CSV URL。自訂資料集之結果僅代表當次分析。')
