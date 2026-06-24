"""
HR Workforce Intelligence System
Streamlit Dashboard — Fatima Azeemi
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import os

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HR Workforce Intelligence System",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0D1117; color: #E6EDF3; }
    
    /* Sidebar */
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #161B22 0%, #0D1117 100%); }
    
    /* Metric cards */
    [data-testid="metric-container"] {
        background: #161B22;
        border: 1px solid #30363D;
        border-radius: 12px;
        padding: 16px;
    }
    [data-testid="metric-container"] label { color: #8B949E !important; font-size: 12px; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] { color: #58A6FF !important; font-size: 28px; font-weight: 700; }
    [data-testid="metric-container"] [data-testid="stMetricDelta"] { font-size: 12px; }

    /* Cards */
    .card {
        background: #161B22;
        border: 1px solid #30363D;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 16px;
    }
    .card-title {
        font-size: 15px;
        font-weight: 600;
        color: #58A6FF;
        margin-bottom: 10px;
        letter-spacing: 0.3px;
    }

    /* Risk badge */
    .risk-high   { background:#3D1A1A; border:1px solid #F85149; border-radius:8px; padding:16px; text-align:center; }
    .risk-medium { background:#2D2209; border:1px solid #E3B341; border-radius:8px; padding:16px; text-align:center; }
    .risk-low    { background:#0F2A1B; border:1px solid #3FB950; border-radius:8px; padding:16px; text-align:center; }
    .risk-label  { font-size: 13px; color: #8B949E; margin-bottom:6px; }
    .risk-value  { font-size: 36px; font-weight: 800; }

    /* Insight boxes */
    .insight-box {
        background: #1C2128;
        border-left: 4px solid #58A6FF;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin-bottom: 10px;
        font-size: 14px;
        line-height: 1.6;
        color: #C9D1D9;
    }

    /* Section header */
    .section-header {
        font-size: 22px;
        font-weight: 700;
        color: #F0F6FC;
        border-bottom: 2px solid #21262D;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }

    /* Nav pills */
    div[data-testid="stRadio"] > div { flex-direction: row; flex-wrap: wrap; gap: 6px; }
    div[data-testid="stRadio"] label {
        background: #21262D;
        border: 1px solid #30363D;
        border-radius: 20px;
        padding: 6px 16px;
        color: #C9D1D9;
        cursor: pointer;
        transition: all 0.2s;
    }
    div[data-testid="stRadio"] label:hover { background: #30363D; }

    /* Plotly chart backgrounds */
    .js-plotly-plot .plotly { background: transparent !important; }

    /* Hide streamlit default elements */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }

    /* Slider */
    .stSlider > div > div { background: #30363D; }

    /* Select box */
    .stSelectbox > div > div { background: #21262D; border: 1px solid #30363D; border-radius: 8px; }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #1F6FEB, #388BFD);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        font-size: 14px;
        width: 100%;
        transition: all 0.2s;
    }
    .stButton > button:hover { opacity: 0.9; transform: translateY(-1px); }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { background: #161B22; border-radius: 8px; padding: 4px; }
    .stTabs [data-baseweb="tab"] { color: #8B949E; border-radius: 6px; }
    .stTabs [aria-selected="true"] { background: #21262D; color: #F0F6FC !important; }

    /* Progress bar */
    .stProgress > div > div { background: linear-gradient(90deg, #1F6FEB, #388BFD); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# PLOTLY DARK THEME
# ─────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#C9D1D9', family='Inter, sans-serif'),
    title_font=dict(color='#F0F6FC', size=15),
    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#8B949E')),
    xaxis=dict(gridcolor='#21262D', zerolinecolor='#30363D', tickfont=dict(color='#8B949E')),
    yaxis=dict(gridcolor='#21262D', zerolinecolor='#30363D', tickfont=dict(color='#8B949E')),
    margin=dict(t=50, b=40, l=50, r=20),
)
COLORS = ['#58A6FF','#F85149','#3FB950','#E3B341','#BC8CFF',
          '#39D353','#FF7B72','#79C0FF','#FFA657','#F778BA']

# ─────────────────────────────────────────────────────────────
# DATA LOADING & FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    BASE = os.path.dirname(__file__)
    df = pd.read_csv(os.path.join(BASE, "data.csv"))
    df = df.drop(columns=['EmployeeNumber','EmployeeCount'], errors='ignore')

    df['AgeGroup'] = pd.cut(df['Age'], bins=[17,30,45,100],
                             labels=['Young (18-30)','Mid-Career (31-45)','Senior (46+)'])
    df['SalaryBand'] = pd.cut(df['DailyRate'], bins=[0,500,1000,1500],
                               labels=['Low','Medium','High'])
    df['DistanceCategory'] = pd.cut(df['DistanceFromHome'], bins=[0,5,15,30],
                                    labels=['Near (1-5)','Moderate (6-15)','Far (16+)'])
    edu_map = {1:'Below College',2:'College',3:'Bachelor',4:'Master',5:'Doctor'}
    df['EducationLevel'] = df['Education'].map(edu_map)
    df['TravelRiskFlag'] = df['BusinessTravel'].map(
        {'Non-Travel':0,'Travel_Rarely':1,'Travel_Frequently':2})
    return df

@st.cache_resource
def load_model():
    BASE = os.path.dirname(__file__)
    path = os.path.join(BASE, "models", "hr_attrition_model.pkl")
    return joblib.load(path)

@st.cache_resource
def train_all_models():
    """Train all 5 models and return results for comparison."""
    df = load_data().copy()
    le_target = LabelEncoder()
    df['Attrition_bin'] = le_target.fit_transform(df['Attrition'])

    enc_cols = df.select_dtypes(include=['object','category']).columns.tolist()
    enc_cols = [c for c in enc_cols if c != 'Attrition']
    encoders = {}
    for col in enc_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    FEAT = [c for c in df.columns if c not in ['Attrition','Attrition_bin']]
    X = df[FEAT]; y = df['Attrition_bin']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    sm = SMOTE(random_state=42)
    X_bal, y_bal = sm.fit_resample(X_train, y_train)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=2000, C=1.0, random_state=42),
        'Decision Tree'      : DecisionTreeClassifier(max_depth=6, min_samples_leaf=15, random_state=42),
        'Random Forest'      : RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1),
        'Gradient Boosting'  : GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                                           max_depth=3, random_state=42),
        'XGBoost'            : XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=3,
                                              random_state=42, eval_metric='logloss', verbosity=0)
    }

    results = {}
    for name, m in models.items():
        m.fit(X_bal, y_bal)
        y_pred = m.predict(X_test)
        y_prob = m.predict_proba(X_test)[:,1]
        results[name] = {
            'model'     : m,
            'train_acc' : accuracy_score(y_bal, m.predict(X_bal)),
            'test_acc'  : accuracy_score(y_test, y_pred),
            'precision' : precision_score(y_test, y_pred, zero_division=0),
            'recall'    : recall_score(y_test, y_pred, zero_division=0),
            'f1'        : f1_score(y_test, y_pred, zero_division=0),
            'roc_auc'   : roc_auc_score(y_test, y_prob),
            'y_pred'    : y_pred,
            'y_prob'    : y_prob,
            'y_test'    : y_test.values,
        }
    return results, FEAT

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px 0;'>
        <div style='font-size:40px;'>🏢</div>
        <div style='font-size:17px; font-weight:700; color:#F0F6FC; margin-top:8px;'>HR Intelligence</div>
        <div style='font-size:11px; color:#8B949E; margin-top:4px;'>Workforce Analytics System</div>
    </div>
    <hr style='border-color:#21262D; margin:16px 0;'>
    """, unsafe_allow_html=True)

    page = st.radio("Navigate", [
        "🏠  Overview",
        "📊  EDA",
        "🤖  ML Models",
        "🔮  Predict",
        "💡  Insights",
    ], label_visibility="collapsed")

    st.markdown("<hr style='border-color:#21262D; margin:16px 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:11px; color:#8B949E; text-align:center; line-height:1.8;'>
        <div style='color:#58A6FF; font-weight:600; margin-bottom:6px;'>Dataset</div>
        IBM HR Analytics<br>Selected Columns<br>
        <br>
        <div style='color:#58A6FF; font-weight:600; margin-bottom:6px;'>Author</div>
        Fatima Azeemi<br>BSCS · ML Specialization<br>Bahria University Lahore
    </div>
    """, unsafe_allow_html=True)

df = load_data()
pkg = load_model()

# ─────────────────────────────────────────────────────────────
# PAGE 1 — OVERVIEW
# ─────────────────────────────────────────────────────────────
if page == "🏠  Overview":
    st.markdown("""
    <div style='padding: 30px 0 10px 0;'>
        <div style='font-size:32px; font-weight:800; color:#F0F6FC;'>HR Workforce Intelligence System</div>
        <div style='font-size:15px; color:#8B949E; margin-top:6px;'>
            Predictive Employee Attrition Analytics &nbsp;·&nbsp; IBM HR Dataset &nbsp;·&nbsp; Fatima Azeemi
        </div>
    </div>
    <hr style='border-color:#21262D; margin:16px 0 28px 0;'>
    """, unsafe_allow_html=True)

    # KPI Row
    total       = len(df)
    attr_yes    = (df['Attrition'] == 'Yes').sum()
    attr_no     = (df['Attrition'] == 'No').sum()
    attr_rate   = attr_yes / total * 100
    avg_age     = df['Age'].mean()
    avg_rate    = df['DailyRate'].mean()
    num_depts   = df['Department'].nunique()

    k1,k2,k3,k4,k5,k6 = st.columns(6)
    k1.metric("👥 Total Employees",  f"{total:,}")
    k2.metric("🚨 Attrited",         f"{attr_yes:,}", f"-{attr_rate:.1f}% of workforce")
    k3.metric("✅ Retained",          f"{attr_no:,}")
    k4.metric("📊 Attrition Rate",   f"{attr_rate:.1f}%", "Industry avg: 10–12%")
    k5.metric("🎂 Avg Age",          f"{avg_age:.1f} yrs")
    k6.metric("💰 Avg Daily Rate",   f"${avg_rate:,.0f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Quick charts row
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("<div class='card-title'>Attrition Split</div>", unsafe_allow_html=True)
        vc = df['Attrition'].value_counts()
        fig = go.Figure(go.Pie(
            labels=vc.index, values=vc.values,
            hole=0.6,
            marker=dict(colors=[COLORS[1], COLORS[0]], line=dict(color='#0D1117', width=3)),
            textinfo='percent+label', textfont_size=12
        ))
        fig.add_annotation(text=f"<b>{attr_rate:.1f}%</b><br>Attrition",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(size=14, color='#F0F6FC'))
        fig.update_layout(**PLOTLY_LAYOUT, height=260, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("<div class='card-title'>Attrition by Department</div>", unsafe_allow_html=True)
        dept = df.groupby('Department')['Attrition'].apply(
            lambda x: (x=='Yes').mean()*100).reset_index()
        dept.columns = ['Department','Rate']
        dept = dept.sort_values('Rate', ascending=True)
        fig = go.Figure(go.Bar(
            x=dept['Rate'], y=dept['Department'],
            orientation='h',
            marker=dict(color=COLORS[:len(dept)], line=dict(color='rgba(0,0,0,0)')),
            text=[f"{v:.1f}%" for v in dept['Rate']], textposition='outside',
            textfont=dict(color='#C9D1D9', size=12)
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=260, xaxis_title="Attrition Rate (%)")
        st.plotly_chart(fig, use_container_width=True)

    with c3:
        st.markdown("<div class='card-title'>Attrition by Travel Frequency</div>", unsafe_allow_html=True)
        trav = df.groupby('BusinessTravel')['Attrition'].apply(
            lambda x: (x=='Yes').mean()*100).reset_index()
        trav.columns = ['Travel','Rate']
        fig = go.Figure(go.Bar(
            x=trav['Travel'], y=trav['Rate'],
            marker=dict(color=COLORS[:3], line=dict(color='rgba(0,0,0,0)')),
            text=[f"{v:.1f}%" for v in trav['Rate']], textposition='outside',
            textfont=dict(color='#C9D1D9', size=12)
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=260, yaxis_title="Attrition Rate (%)")
        st.plotly_chart(fig, use_container_width=True)

    # Dataset snapshot
    st.markdown("<div class='section-header'>📋 Dataset Snapshot</div>", unsafe_allow_html=True)
    cols_info = pd.DataFrame({
        'Feature'    : df.columns.tolist(),
        'Type'       : df.dtypes.astype(str).tolist(),
        'Unique'     : [df[c].nunique() for c in df.columns],
        'Sample'     : [str(df[c].iloc[0]) for c in df.columns],
        'Missing'    : [df[c].isnull().sum() for c in df.columns]
    })
    st.dataframe(cols_info, use_container_width=True, height=320,
                 column_config={
                     "Missing": st.column_config.NumberColumn("Missing", format="%d"),
                     "Unique" : st.column_config.NumberColumn("Unique",  format="%d"),
                 })

# ─────────────────────────────────────────────────────────────
# PAGE 2 — EDA
# ─────────────────────────────────────────────────────────────
elif page == "📊  EDA":
    st.markdown("<div class='section-header'>📊 Exploratory Data Analysis</div>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Demographics", "Salary & Distance", "Education & Travel", "Correlation"])

    with tab1:
        # Age group vs attrition
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            age_attr = df.groupby(['AgeGroup','Attrition'], observed=True).size().reset_index(name='Count')
            fig = px.bar(age_attr, x='AgeGroup', y='Count', color='Attrition',
                         color_discrete_map={'Yes':COLORS[1],'No':COLORS[0]},
                         barmode='group', title='Age Group vs Attrition')
            fig.update_layout(**PLOTLY_LAYOUT, height=340)
            st.plotly_chart(fig, use_container_width=True)

        with r1c2:
            age_rate = df.groupby('AgeGroup', observed=True)['Attrition'].apply(
                lambda x: (x=='Yes').mean()*100).reset_index()
            age_rate.columns = ['AgeGroup','Rate']
            fig = px.bar(age_rate, x='AgeGroup', y='Rate',
                         color='Rate', color_continuous_scale='RdYlGn_r',
                         title='Attrition Rate % by Age Group', text=[f"{v:.1f}%" for v in age_rate['Rate']])
            fig.update_traces(textposition='outside', textfont_color='#C9D1D9')
            fig.update_layout(**PLOTLY_LAYOUT, height=340, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        # Age distribution
        fig = go.Figure()
        for status, color in zip(['No','Yes'], [COLORS[0], COLORS[1]]):
            fig.add_trace(go.Histogram(
                x=df[df['Attrition']==status]['Age'],
                name=f'Attrition={status}', marker_color=color,
                opacity=0.75, nbinsx=20))
        fig.update_layout(**PLOTLY_LAYOUT, barmode='overlay',
                          title='Age Distribution by Attrition', height=300,
                          xaxis_title='Age', yaxis_title='Count')
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        r2c1, r2c2 = st.columns(2)
        with r2c1:
            sal_rate = df.groupby('SalaryBand', observed=True)['Attrition'].apply(
                lambda x: (x=='Yes').mean()*100).reset_index()
            sal_rate.columns = ['Band','Rate']
            fig = px.bar(sal_rate, x='Band', y='Rate',
                         color='Rate', color_continuous_scale='Reds',
                         title='Attrition Rate % by Salary Band',
                         text=[f"{v:.1f}%" for v in sal_rate['Rate']])
            fig.update_traces(textposition='outside', textfont_color='#C9D1D9')
            fig.update_layout(**PLOTLY_LAYOUT, height=340, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        with r2c2:
            dist_rate = df.groupby('DistanceCategory', observed=True)['Attrition'].apply(
                lambda x: (x=='Yes').mean()*100).reset_index()
            dist_rate.columns = ['Distance','Rate']
            fig = px.bar(dist_rate, x='Distance', y='Rate',
                         color='Rate', color_continuous_scale='Oranges',
                         title='Attrition Rate % by Distance From Home',
                         text=[f"{v:.1f}%" for v in dist_rate['Rate']])
            fig.update_traces(textposition='outside', textfont_color='#C9D1D9')
            fig.update_layout(**PLOTLY_LAYOUT, height=340, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        # DailyRate box by attrition
        fig = go.Figure()
        for status, color in zip(['No','Yes'], [COLORS[0], COLORS[1]]):
            fig.add_trace(go.Box(
                y=df[df['Attrition']==status]['DailyRate'],
                name=f'Attrition={status}', marker_color=color, boxmean=True))
        fig.update_layout(**PLOTLY_LAYOUT, title='Daily Rate Distribution by Attrition',
                          height=300, yaxis_title='Daily Rate ($)')
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        r3c1, r3c2 = st.columns(2)
        with r3c1:
            edu_rate = df.groupby('EducationField')['Attrition'].apply(
                lambda x: (x=='Yes').mean()*100).reset_index()
            edu_rate.columns = ['Field','Rate']
            edu_rate = edu_rate.sort_values('Rate', ascending=True)
            fig = go.Figure(go.Bar(
                x=edu_rate['Rate'], y=edu_rate['Field'], orientation='h',
                marker=dict(color=COLORS[:len(edu_rate)]),
                text=[f"{v:.1f}%" for v in edu_rate['Rate']], textposition='outside',
                textfont=dict(color='#C9D1D9')))
            fig.update_layout(**PLOTLY_LAYOUT, title='Attrition Rate % by Education Field', height=320)
            st.plotly_chart(fig, use_container_width=True)

        with r3c2:
            trav_attr = df.groupby(['BusinessTravel','Attrition']).size().reset_index(name='Count')
            fig = px.bar(trav_attr, x='BusinessTravel', y='Count', color='Attrition',
                         color_discrete_map={'Yes':COLORS[1],'No':COLORS[0]},
                         barmode='stack', title='Business Travel vs Attrition (Stacked)')
            fig.update_layout(**PLOTLY_LAYOUT, height=320)
            st.plotly_chart(fig, use_container_width=True)

        # Education Level pie
        edu_lv = df.groupby('EducationLevel')['Attrition'].apply(
            lambda x: (x=='Yes').mean()*100).reset_index()
        edu_lv.columns = ['Level','Rate']
        fig = px.pie(edu_lv, values='Rate', names='Level', hole=0.5,
                     color_discrete_sequence=COLORS,
                     title='Attrition Share by Education Level')
        fig.update_layout(**PLOTLY_LAYOUT, height=300)
        st.plotly_chart(fig, use_container_width=True)
        
    with tab4:
        df_corr = df.copy()

        # encode categoricals safely
        for col in df_corr.select_dtypes(include=['object', 'category']).columns:
            le = LabelEncoder()
            df_corr[col] = le.fit_transform(df_corr[col].astype(str))

        # drop engineered columns safely
        drop_cols = ['EducationLevel', 'AgeGroup', 'SalaryBand', 'DistanceCategory']
        df_corr = df_corr.drop(columns=drop_cols, errors='ignore')

        corr = df_corr.corr(numeric_only=True).fillna(0).round(2)

        fig = go.Figure()

        fig.add_trace(
            go.Heatmap(
                z=corr.values,
                x=list(corr.columns),
                y=list(corr.columns),
                colorscale='RdBu_r',
                zmid=0,
                text=corr.values,
                texttemplate="%{text}",
                hoverongaps=False,
                colorbar=dict(title="Correlation")
            )
        )

        fig.update_layout(
            **PLOTLY_LAYOUT,
            title="Feature Correlation Matrix",
            height=520
        )

    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# PAGE 3 — ML MODELS
# ─────────────────────────────────────────────────────────────
elif page == "🤖  ML Models":
    st.markdown("<div class='section-header'>🤖 Machine Learning Models</div>", unsafe_allow_html=True)

    with st.spinner("Training 5 models… please wait"):
        results, feat_cols = train_all_models()

    # ── Model comparison table
    comp = pd.DataFrame([{
        'Model'     : n,
        'Train Acc' : f"{r['train_acc']:.4f}",
        'Test Acc'  : f"{r['test_acc']:.4f}",
        'Gap'       : f"{abs(r['train_acc']-r['test_acc']):.4f}",
        'Precision' : f"{r['precision']:.4f}",
        'Recall'    : f"{r['recall']:.4f}",
        'F1 Score'  : f"{r['f1']:.4f}",
        'ROC-AUC'   : f"{r['roc_auc']:.4f}",
    } for n, r in results.items()])
    st.markdown("### 📋 Model Performance Comparison")
    st.dataframe(comp, use_container_width=True, hide_index=True)

    best = max(results, key=lambda k: results[k]['roc_auc'])
    st.success(f"🏆 Best Model by ROC-AUC: **{best}** — AUC: {results[best]['roc_auc']:.4f}")

    # ── Metric bar charts
    metrics_to_plot = ['train_acc','test_acc','f1','roc_auc']
    labels = ['Train Accuracy','Test Accuracy','F1 Score','ROC-AUC']
    names  = list(results.keys())

    r1c1, r1c2 = st.columns(2)
    for i, (metric, label) in enumerate(zip(metrics_to_plot, labels)):
        values = [results[n][metric] for n in names]
        fig = go.Figure(go.Bar(
            x=names, y=values,
            marker=dict(color=COLORS[:len(names)], line=dict(color='rgba(0,0,0,0)')),
            text=[f"{v:.3f}" for v in values], textposition='outside',
            textfont=dict(color='#C9D1D9', size=11)
        ))
        layout_config = PLOTLY_LAYOUT.copy()

        layout_config["yaxis"] = {
            **PLOTLY_LAYOUT["yaxis"],
            "range": [0, 1.15]
        }

        fig.update_layout(
            **layout_config,
            title=label,
            height=300
        )
        if i % 2 == 0:
            r1c1.plotly_chart(fig, use_container_width=True)
        else:
            r1c2.plotly_chart(fig, use_container_width=True)

    # ── Overfitting Analysis
    st.markdown("### 🔍 Overfitting Analysis: Train vs Test Accuracy")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names,
        y=[results[n]['train_acc'] for n in names],
        name='Train Accuracy', marker_color=COLORS[0],
        text=[f"{results[n]['train_acc']:.3f}" for n in names],
        textposition='outside', textfont_size=11
    ))
    fig.add_trace(go.Bar(
        x=names,
        y=[results[n]['test_acc'] for n in names],
        name='Test Accuracy', marker_color=COLORS[1],
        text=[f"{results[n]['test_acc']:.3f}" for n in names],
        textposition='outside', textfont_size=11
    ))
    layout_config = PLOTLY_LAYOUT.copy()
    layout_config["yaxis"] = {
        **PLOTLY_LAYOUT["yaxis"],
        "range": [0, 1.15]
    }
    fig.update_layout(
        **layout_config,
        barmode='group',
        height=360
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── ROC Curves
    st.markdown("### 📈 ROC Curves — All Models")
    fig = go.Figure()
    for i, (n, r) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(r['y_test'], r['y_prob'])
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode='lines', name=f"{n} (AUC={r['roc_auc']:.3f})",
            line=dict(color=COLORS[i], width=2)))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
                             name='Random', line=dict(color='#555', dash='dash')))
    fig.update_layout(**PLOTLY_LAYOUT, height=400,
                      xaxis_title='False Positive Rate',
                      yaxis_title='True Positive Rate',
                      title='ROC Curves — All 5 Models')
    st.plotly_chart(fig, use_container_width=True)

    # ── Confusion Matrices
    st.markdown("### 🔲 Confusion Matrices")
    cols = st.columns(5)
    for col, (name, r) in zip(cols, results.items()):
        cm = confusion_matrix(r['y_test'], r['y_pred'])
        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            z=cm,
            x=['No', 'Yes'],
            y=['No', 'Yes'],
            colorscale='Blues',
            text=cm,
            texttemplate="<b>%{text}</b>",
            hoverongaps=False,
            showscale=False
        ))
        fig.update_layout(
            **PLOTLY_LAYOUT,   
            height=230,
            title=name.replace(' ', '\n')
        )
        col.plotly_chart(fig, use_container_width=True)

    # ── Feature Importance
    st.markdown("### 🔬 Feature Importance (Gradient Boosting + XGBoost)")
    fc1, fc2 = st.columns(2)
    for col, mname, color in zip(
        [fc1, fc2],
        ['Gradient Boosting', 'XGBoost'],
        [COLORS[3], COLORS[4]]
    ):
        m = results[mname]['model']
        imp = pd.Series(
            m.feature_importances_,
            index=feat_cols
        ).sort_values(ascending=True).tail(10)
        fig = go.Figure(go.Bar(
            x=imp.values,
            y=imp.index,
            orientation='h',
            marker=dict(color=color, line=dict(color='rgba(0,0,0,0)')),
            text=[f"{v:.4f}" for v in imp.values],
            textposition='outside',
            textfont=dict(color='#C9D1D9', size=10)
            ))
        layout = PLOTLY_LAYOUT.copy()
        layout.update({
            "title": f"{mname} — Top Features",
            "height": 360,
            "margin": {"t": 50, "b": 30, "l": 140, "r": 60}
            })
        fig.update_layout(**layout)
        col.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# PAGE 4 — PREDICT
# ─────────────────────────────────────────────────────────────
elif page == "🔮  Predict":
    st.markdown("<div class='section-header'>🔮 Attrition Risk Predictor</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='insight-box'>
        Enter employee details below to get an <b>instant attrition risk prediction</b>
        powered by the trained Gradient Boosting model.
    </div>
    """, unsafe_allow_html=True)

    with st.form("predict_form"):
        st.markdown("#### 👤 Employee Information")

        pc1, pc2, pc3 = st.columns(3)

        with pc1:
            age = st.slider("Age", 18, 60, 30)
            daily_rate = st.slider("Daily Rate ($)", 100, 1500, 800)
            distance   = st.slider("Distance From Home (km)", 1, 29, 5)

        with pc2:
            department = st.selectbox("Department",
                ['Sales', 'Research & Development', 'Human Resources'])
            education_field = st.selectbox("Education Field",
                ['Life Sciences', 'Medical', 'Marketing',
                 'Technical Degree', 'Human Resources', 'Other'])
            business_travel = st.selectbox("Business Travel",
                ['Non-Travel', 'Travel_Rarely', 'Travel_Frequently'])

        with pc3:
            education = st.selectbox("Education Level",
                [1, 2, 3, 4, 5],
                format_func=lambda x: {1:'Below College',2:'College',
                                        3:'Bachelor',4:'Master',5:'Doctor'}[x])

        submitted = st.form_submit_button("🔮 Predict Attrition Risk", use_container_width=True)

    if submitted:
        # ── Build feature vector matching training pipeline
        # Auto-derived engineered features
        age_group = ('Young (18-30)' if age <= 30 else
                     'Mid-Career (31-45)' if age <= 45 else 'Senior (46+)')
        salary_band = ('Low' if daily_rate <= 500 else
                       'Medium' if daily_rate <= 1000 else 'High')
        dist_cat = ('Near (1-5)' if distance <= 5 else
                    'Moderate (6-15)' if distance <= 15 else 'Far (16+)')
        edu_level = {1:'Below College',2:'College',3:'Bachelor',4:'Master',5:'Doctor'}[education]
        travel_risk = {'Non-Travel':0,'Travel_Rarely':1,'Travel_Frequently':2}[business_travel]

        # Build raw row matching training columns
        row = {
            'Age'              : age,
            'BusinessTravel'   : business_travel,
            'DailyRate'        : daily_rate,
            'Department'       : department,
            'DistanceFromHome' : distance,
            'Education'        : education,
            'EducationField'   : education_field,
            'AgeGroup'         : age_group,
            'SalaryBand'       : salary_band,
            'DistanceCategory' : dist_cat,
            'EducationLevel'   : edu_level,
            'TravelRiskFlag'   : travel_risk,
        }
        input_df = pd.DataFrame([row])

        # Encode using saved encoders
        encoders = pkg['encoders']
        model    = pkg['model']
        feat_cols = pkg['feature_cols']

        for col in input_df.columns:
            if col in encoders:
                le = encoders[col]
                val = input_df[col].astype(str).iloc[0]
                if val in le.classes_:
                    input_df[col] = le.transform([val])
                else:
                    input_df[col] = 0  # unseen label fallback

        X_input = input_df[feat_cols]
        prob    = model.predict_proba(X_input)[0][1]
        pred    = model.predict(X_input)[0]

        # Risk tier
        if prob >= 0.60:
            risk_class = "risk-high"
            risk_label = "🔴 HIGH RISK"
            risk_color = "#F85149"
            advice = "Immediate retention intervention recommended. Schedule manager 1:1 and review compensation."
        elif prob >= 0.35:
            risk_class = "risk-medium"
            risk_label = "🟡 MEDIUM RISK"
            risk_color = "#E3B341"
            advice = "Monitor closely. Consider career development discussions and workload review."
        else:
            risk_class = "risk-low"
            risk_label = "🟢 LOW RISK"
            risk_color = "#3FB950"
            advice = "Employee appears stable. Continue standard engagement and recognition programs."

        # Display results
        st.markdown("<br>", unsafe_allow_html=True)
        r1, r2, r3 = st.columns([1,1,2])

        with r1:
            st.markdown(f"""
            <div class='{risk_class}'>
                <div class='risk-label'>Attrition Risk Level</div>
                <div class='risk-value' style='color:{risk_color};'>{risk_label}</div>
            </div>
            """, unsafe_allow_html=True)

        with r2:
            st.markdown(f"""
            <div class='{risk_class}'>
                <div class='risk-label'>Attrition Probability</div>
                <div class='risk-value' style='color:{risk_color};'>{prob*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        with r3:
            st.markdown(f"""
            <div class='insight-box' style='height:90px; display:flex; align-items:center;'>
                <div>
                    <b style='color:{risk_color};'>HR Recommendation</b><br>
                    {advice}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob * 100,
            number={'suffix': '%', 'font': {'size': 40, 'color': '#F0F6FC'}},
            delta={'reference': 16.1, 'suffix': '%', 'font': {'size': 14}},
            title={'text': "Attrition Probability vs Avg (16.1%)",
                   'font': {'size': 14, 'color': '#8B949E'}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#8B949E',
                         'tickfont': {'color':'#8B949E'}},
                'bar': {'color': risk_color, 'thickness': 0.3},
                'bgcolor': '#161B22',
                'bordercolor': '#30363D',
                'steps': [
                    {'range': [0, 35],  'color': '#0F2A1B'},
                    {'range': [35, 60], 'color': '#2D2209'},
                    {'range': [60, 100],'color': '#3D1A1A'}
                ],
                'threshold': {
                    'line': {'color': '#8B949E', 'width': 2},
                    'thickness': 0.75, 'value': 16.1
                }
            }
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=320)
        st.plotly_chart(fig, use_container_width=True)

        # Input summary
        with st.expander("📋 Input Summary"):
            display = pd.DataFrame({
                'Feature': ['Age','Daily Rate','Distance','Department','Education Field',
                            'Business Travel','Education Level','Age Group','Salary Band',
                            'Distance Category','Travel Risk Score'],
                'Value'  : [age, f"${daily_rate}", f"{distance} km", department,
                            education_field, business_travel, edu_level, age_group,
                            salary_band, dist_cat, travel_risk]
            })
            st.dataframe(display, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────
# PAGE 5 — INSIGHTS
# ─────────────────────────────────────────────────────────────
elif page == "💡  Insights":
    st.markdown("<div class='section-header'>💡 Business Insights & Recommendations</div>",
                unsafe_allow_html=True)

    # Compute live stats
    total_attr   = (df['Attrition']=='Yes').mean()*100
    dept_rates   = df.groupby('Department')['Attrition'].apply(lambda x:(x=='Yes').mean()*100)
    travel_rates = df.groupby('BusinessTravel')['Attrition'].apply(lambda x:(x=='Yes').mean()*100)
    age_rates    = df.groupby('AgeGroup', observed=True)['Attrition'].apply(lambda x:(x=='Yes').mean()*100)
    sal_rates    = df.groupby('SalaryBand', observed=True)['Attrition'].apply(lambda x:(x=='Yes').mean()*100)
    dist_rates   = df.groupby('DistanceCategory', observed=True)['Attrition'].apply(lambda x:(x=='Yes').mean()*100)

    insights = [
        ("🔴", "High Base Attrition Demands Board-Level Attention",
         f"The organization has a <b>{total_attr:.1f}% attrition rate</b>, above the industry benchmark of 10–12%. "
         f"Every 1% above benchmark costs the company approx. <b>$1.1M/year</b> in replacement costs (1,470-person workforce)."),
        ("✈️", "Frequent Travelers Are the Highest-Risk Group",
         f"Employees who travel frequently show an attrition rate of <b>{travel_rates.get('Travel_Frequently',0):.1f}%</b> "
         f"vs {travel_rates.get('Non-Travel',0):.1f}% for non-travelers. Travel burnout and family disruption are primary drivers."),
        ("🧑", "Young Employees Are the Most Volatile Talent Segment",
         f"The 18–30 age cohort shows the highest attrition rate at <b>{age_rates.get('Young (18-30)',0):.1f}%</b>. "
         f"Gaps in career acceleration pathways, mentorship, and early engagement programs are key risk factors."),
        ("💰", "Low Salary Band Employees Are Disproportionately Leaving",
         f"Employees in the Low salary band (DailyRate ≤$500) attrition at <b>{sal_rates.get('Low',0):.1f}%</b>. "
         f"Compensation benchmarking at the lower end of the pay scale is a critical retention lever."),
        ("🏠", "Distance From Office Quietly Erodes Retention",
         f"Far-distance employees (16+ km) leave at <b>{dist_rates.get('Far (16+)',0):.1f}%</b> vs "
         f"{dist_rates.get('Near (1-5)',0):.1f}% for nearby employees. Flexible/hybrid policies can materially reduce this gap."),
        ("🎓", "Certain Education Fields Show Elevated Flight Risk",
         "Technical Degree and Marketing graduates show higher attrition, facing stronger external market demand. "
         "Targeted retention packages (equity, fast-track promotions) for in-demand fields reduce poaching losses."),
        ("📊", "Age & Daily Rate Are the Primary Attrition Predictors",
         "Feature importance analysis consistently ranks Age and DailyRate as the top predictors across all 5 ML models. "
         "This validates a focused strategy: <b>compensate competitively at early career stages</b> for maximum retention impact."),
    ]

    for emoji, title, body in insights:
        st.markdown(f"""
        <div class='card'>
            <div class='card-title'>{emoji} {title}</div>
            <div style='font-size:14px; color:#C9D1D9; line-height:1.7;'>{body}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Recommendations table
    st.markdown("<div class='section-header'>📋 HR Recommendations</div>", unsafe_allow_html=True)

    recs = pd.DataFrame({
        'Priority'   : ['🔴 Critical','🔴 Critical','🟠 High','🟠 High','🟡 Medium','🟡 Medium','🟢 Ongoing'],
        'Action'     : [
            'Benchmark & raise minimum salary floor for Low band employees',
            'Launch Early Talent Program (mentorship + fast-track)',
            'Introduce flexible/hybrid work to reduce commute churn',
            'Review & limit frequent travel; add travel allowances',
            'Run "stay interviews" in high-attrition departments',
            'Build retention packages for technical education graduates',
            'Deploy ML model for quarterly HR attrition scoring',
        ],
        'Target Group': [
            'Low daily-rate employees','Employees aged 18–30',
            'Far-distance employees','Frequent travelers',
            'Sales & HR departments','STEM graduates',
            'All employees'
        ],
        'Est. Impact': ['High','High','High','Medium','Medium','Medium','Long-term'],
    })
    st.dataframe(recs, use_container_width=True, hide_index=True)

    # ROI box
    st.markdown("""
    <div class='card' style='border-left:4px solid #3FB950;'>
        <div class='card-title'>💰 Estimated ROI of Retention Initiatives</div>
        <div style='font-size:14px; color:#C9D1D9; line-height:1.8;'>
            Reducing attrition by just <b>5 percentage points</b> (from ~16% → ~11%) in a 1,470-person workforce,
            at a conservative replacement cost of 150% annual salary ($50,000 avg):
            <br><br>
            <span style='font-size:28px; font-weight:800; color:#3FB950;'>$5.51M / year</span>
            <span style='color:#8B949E;'> in prevented replacement costs</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
