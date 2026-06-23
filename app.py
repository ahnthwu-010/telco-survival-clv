import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pickle
from pathlib import Path

# ── Page config ──
st.set_page_config(
    page_title="Telco CLV · Survival Analysis",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0f1117;
    color: #e0e0e0;
}
/* Main content area */
[data-testid="stAppViewContainer"] {
    background-color: #0f1117;
}
[data-testid="stMain"] {
    background-color: #0f1117;
}
/* Fix selectbox và slider label màu sáng */
.stSelectbox > label, .stSlider > label, .stCheckbox > label {
    color: #9aa0b4 !important;
    font-size: 0.82rem !important;
}
/* Fix text input */
.stTextInput input {
    background-color: #1a1d2e !important;
    color: #f0f0f0 !important;
    border-color: #2a2d3e !important;
}

/* Dark sidebar */
[data-testid="stSidebar"] {
    background: #0f1117;
    border-right: 1px solid #1e2130;
}
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label { color: #9aa0b4 !important; font-size: 0.78rem !important; }

/* KPI cards */
.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: 20px 0; }
.kpi-card {
    background: #1a1d2e;
    border: 1px solid #2a2d3e;
    border-radius: 10px;
    padding: 20px 24px;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
}
.kpi-card.blue::before   { background: #4f8ef7; }
.kpi-card.amber::before  { background: #f5a623; }
.kpi-card.red::before    { background: #e05c5c; }
.kpi-card.green::before  { background: #4caf7d; }

.kpi-label { font-size: 0.72rem; color: #7a7f94; font-weight: 500; letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 8px; }
.kpi-value { font-size: 1.85rem; font-weight: 700; color: #f0f0f0; font-family: 'JetBrains Mono', monospace; line-height: 1; white-space: nowrap; }
.kpi-sub   { font-size: 0.75rem; color: #5a6070; margin-top: 6px; }

/* Section headers */
.section-header {
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.12em;
    text-transform: uppercase; color: #4f8ef7;
    margin: 32px 0 14px; padding-bottom: 8px;
    border-bottom: 1px solid #1e2130;
}

/* Insight box */
.insight-box {
    background: #0f1a2e; border: 1px solid #1e3a5f;
    border-left: 3px solid #4f8ef7;
    border-radius: 6px; padding: 14px 18px; margin: 10px 0;
    font-size: 0.85rem; color: #c0cce0; line-height: 1.6;
}
.insight-box strong { color: #4f8ef7; }

/* Customer card */
.customer-card {
    background: #1a1d2e; border: 1px solid #2a2d3e;
    border-radius: 10px; padding: 20px 24px; margin: 12px 0;
}
.customer-card .name { font-size: 1.1rem; font-weight: 600; color: #f0f0f0; }
.customer-card .badge {
    display: inline-block; padding: 2px 10px; border-radius: 20px;
    font-size: 0.72rem; font-weight: 600; margin-left: 10px;
}
.badge-mtm  { background: #3d1515; color: #e05c5c; }
.badge-1yr  { background: #3d2e10; color: #f5a623; }
.badge-2yr  { background: #0f2e1a; color: #4caf7d; }

/* Warning banner */
.warn-banner {
    background: #2e1a0f; border: 1px solid #6b3a10;
    border-radius: 6px; padding: 10px 16px;
    font-size: 0.83rem; color: #f5a623; margin: 8px 0;
}

/* Metric row */
.metric-row { display: flex; gap: 16px; flex-wrap: wrap; margin: 12px 0; }
.metric-item { flex: 1; min-width: 120px; background: #1e2235; border: 1px solid #2e3350; border-radius: 8px; padding: 12px 16px; }
.metric-item .m-label { font-size: 0.68rem; color: #9aa0b4; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 500; }
.metric-item .m-value { font-size: 1.1rem; font-weight: 600; color: #ffffff; font-family: 'JetBrains Mono', monospace; margin-top: 4px; white-space: nowrap; }
</style>
""", unsafe_allow_html=True)

# ── Load data & model ──
@st.cache_resource
def load_model():
    with open('app/cph_tv_model.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    results  = pd.read_csv('app/results_clv.csv')
    df_cox   = pd.read_csv('app/df_cox_tv.csv')
    df_orig  = pd.read_csv('app/df_original.csv')
    df_orig['TotalCharges'] = pd.to_numeric(df_orig['TotalCharges'], errors='coerce')
    df_orig.dropna(subset=['TotalCharges'], inplace=True)
    df_orig['Churn_flag'] = (df_orig['Churn'] == 'Yes').astype(int)
    return results, df_cox, df_orig

cph_tv  = load_model()
results_df, df_cox_tv, df_orig = load_data()

HORIZON              = 72
DISCOUNT_RATE        = 0.01
active               = results_df[results_df['Churn_actual'] == 0]
COLOR_MAP = {
    'Month-to-month': '#e05c5c',
    'One year':       '#f5a623',
    'Two year':       '#4caf7d',
}

# ── Sidebar navigation ──
with st.sidebar:
    st.markdown("## 📡 Telco CLV")
    st.markdown("<div style='font-size:0.75rem;color:#5a6070;margin:-10px 0 20px'>Survival-based Customer\nLifetime Value Analysis</div>", unsafe_allow_html=True)
    page = st.radio("", ["Portfolio Overview", "Customer Lookup", "Retention Simulator"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("<div style='font-size:0.72rem;color:#5a6070'>Model: Cox PH · Stratified + Time-Interaction<br>Concordance: <b style='color:#4f8ef7'>0.8840</b><br>Observations: 7,032 · Events: 1,869<br>Discount rate: 1%/month</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════
# PAGE 1 — PORTFOLIO OVERVIEW
# ════════════════════════════════════════════
if page == "Portfolio Overview":
    st.markdown("# Portfolio Overview")
    st.markdown("<div style='color:#7a7f94;margin-top:-10px;margin-bottom:24px'>Toàn bộ 5,163 khách hàng đang active · Giá trị tính tại thời điểm hiện tại</div>", unsafe_allow_html=True)

    # KPI Cards
    total_clv   = active['Remaining_CLV'].sum()
    at_risk     = active[(active['Contract']=='Month-to-month') &
                         (active['Remaining_CLV'] > active['Remaining_CLV'].quantile(0.75))]
    value_saved = at_risk['Remaining_CLV'].sum() * 0.10
    avg_clv     = active['Remaining_CLV'].mean()
    n_active    = len(active)

    st.markdown(f"""
    <div class="kpi-grid">
      <div class="kpi-card blue">
        <div class="kpi-label">Total Portfolio CLV</div>
        <div class="kpi-value">${total_clv/1e6:.2f}M</div>
        <div class="kpi-sub">{n_active:,} active customers</div>
      </div>
      <div class="kpi-card amber">
        <div class="kpi-label">At-Risk Customers</div>
        <div class="kpi-value">{len(at_risk):,}</div>
        <div class="kpi-sub">Month-to-month · Top 25% CLV</div>
      </div>
      <div class="kpi-card red">
        <div class="kpi-label">Value at Risk (10%)</div>
        <div class="kpi-value">${value_saved:,.0f}</div>
        <div class="kpi-sub">Nếu giữ được thêm 10% nhóm at-risk</div>
      </div>
      <div class="kpi-card green">
        <div class="kpi-label">Avg Remaining CLV</div>
        <div class="kpi-value">${avg_clv:,.0f}</div>
        <div class="kpi-sub">Median: ${active['Remaining_CLV'].median():,.0f}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Charts
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor='#0f1117')
    for ax in axes:
        ax.set_facecolor('#1a1d2e')
        ax.tick_params(colors='#7a7f94', labelsize=8)
        for spine in ax.spines.values(): spine.set_color('#2a2d3e')

    contracts = ['Month-to-month', 'One year', 'Two year']
    clr       = ['#e05c5c', '#f5a623', '#4caf7d']

    # Boxplot
    data_by_c = [active[active['Contract']==c]['Remaining_CLV'].values for c in contracts]
    bp = axes[0].boxplot(data_by_c, labels=['M2M','1yr','2yr'], patch_artist=True, notch=True,
                         medianprops=dict(color='white', linewidth=2), flierprops=dict(marker='.', markersize=3))
    for patch, c in zip(bp['boxes'], clr):
        patch.set_facecolor(c); patch.set_alpha(0.75)
    axes[0].set_title('Remaining CLV by Contract', color='#e0e0e0', fontsize=10, pad=10)
    axes[0].set_ylabel('Remaining CLV ($)', color='#7a7f94', fontsize=8)
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:,.0f}'))
    axes[0].grid(True, alpha=0.15, axis='y', color='white')

    # Scatter: CLV vs Tenure
    for c, col in zip(contracts, clr):
        mask = active['Contract'] == c
        axes[1].scatter(active.loc[mask,'tenure_actual'],
                        active.loc[mask,'Remaining_CLV'],
                        c=col, alpha=0.35, s=7, label=c)
    axes[1].set_title('Remaining CLV vs Tenure', color='#e0e0e0', fontsize=10, pad=10)
    axes[1].set_xlabel('Tenure (months)', color='#7a7f94', fontsize=8)
    axes[1].set_ylabel('Remaining CLV ($)', color='#7a7f94', fontsize=8)
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:,.0f}'))
    axes[1].legend(fontsize=7, facecolor='#1a1d2e', labelcolor='white', edgecolor='#2a2d3e')
    axes[1].grid(True, alpha=0.15, color='white')

    # Top 15
    top15 = active.sort_values('Remaining_CLV', ascending=False).head(15).reset_index(drop=True)
    bar_colors = [COLOR_MAP[c] for c in top15['Contract']]
    axes[2].barh(range(15), top15['Remaining_CLV'], color=bar_colors, alpha=0.85, height=0.7)
    for i, (_, r) in enumerate(top15.iterrows()):
        axes[2].text(top15['Remaining_CLV'].iloc[i]+30, i,
                     f"${r['MonthlyCharges']:.0f} · t={r['tenure_actual']}",
                     va='center', fontsize=6.5, color='#9aa0b4')
    axes[2].set_yticks(range(15))
    axes[2].set_yticklabels(top15['customerID'].str[-7:], fontsize=7, color='#9aa0b4')
    axes[2].invert_yaxis()
    axes[2].set_title('Top 15 Retention Priorities', color='#e0e0e0', fontsize=10, pad=10)
    axes[2].set_xlabel('Remaining CLV ($)', color='#7a7f94', fontsize=8)
    axes[2].set_xlim(0, top15['Remaining_CLV'].max()*1.28)
    axes[2].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:,.0f}'))
    axes[2].grid(True, alpha=0.15, axis='x', color='white')

    fig.tight_layout(pad=2)
    st.pyplot(fig)
    plt.close()

    st.markdown('<div class="section-header">Key Insights</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="insight-box">
        <strong>Price sensitivity reverses after month 4.</strong>
        Khách hàng mới (tenure &lt; 4 tháng) rời bỏ khi giá cao — nhưng sau tháng 4, khách trả nhiều tiền hơn
        lại là người trung thành hơn. <em>Đừng giảm giá để giữ khách mới — hãy đầu tư vào onboarding.</em>
    </div>
    <div class="insight-box">
        <strong>TechSupport không có tác dụng tức thì</strong> — nhưng tích lũy giá trị bảo vệ mạnh theo thời gian.
        HR giảm từ ~1.0 về ~0.74 sau 72 tháng. <em>TechSupport là công cụ giữ khách lâu năm, không phải khách mới.</em>
    </div>
    <div class="insight-box">
        <strong>Month-to-month = high risk + high value.</strong>
        Nhóm này có Remaining CLV trung bình $1,522 — cao nhất trong 3 nhóm hợp đồng —
        nhưng cũng là nhóm có đường cong survival dốc nhất. 955 khách trong nhóm này đại diện cho
        <strong>$236,381 giá trị có thể bảo vệ được</strong> chỉ bằng cách giữ thêm 10%.
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════
# PAGE 2 — CUSTOMER LOOKUP
# ════════════════════════════════════════════
elif page == "Customer Lookup":
    st.markdown("# Customer Lookup")
    st.markdown("<div style='color:#7a7f94;margin-top:-10px;margin-bottom:24px'>Tra cứu đường cong survival và Remaining CLV cho từng khách hàng</div>", unsafe_allow_html=True)

    customer_ids = results_df['customerID'].tolist()
    selected_id  = st.selectbox("Nhập hoặc chọn Customer ID", customer_ids,
                                 index=customer_ids.index('9547-ITEFG') if '9547-ITEFG' in customer_ids else 0)

    row_r   = results_df[results_df['customerID'] == selected_id].iloc[0]
    row_cox = df_cox_tv.iloc[results_df[results_df['customerID']==selected_id].index[0]]
    row_o   = df_orig[df_orig['customerID']==selected_id].iloc[0]

    contract = row_r['Contract']
    badge_class = {'Month-to-month':'badge-mtm','One year':'badge-1yr','Two year':'badge-2yr'}[contract]

    # Customer header card
    churn_status = "⚠️ Đã Churn" if row_r['Churn_actual']==1 else "✅ Active"
    st.markdown(f"""
    <div class="customer-card">
        <div class="name">{selected_id}
            <span class="badge {badge_class}">{contract}</span>
        </div>
        <div style="font-size:0.8rem;color:#7a7f94;margin-top:6px">{churn_status}</div>
    </div>
    """, unsafe_allow_html=True)

    # Metrics
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-item"><div class="m-label">Tenure</div><div class="m-value">{int(row_r['tenure_actual'])} tháng</div></div>
        <div class="metric-item"><div class="m-label">Monthly Charges</div><div class="m-value">${row_r['MonthlyCharges']:.2f}</div></div>
        <div class="metric-item"><div class="m-label">RMST (còn lại)</div><div class="m-value">{row_r['RMST_months']:.1f} tháng</div></div>
        <div class="metric-item"><div class="m-label">CLV ban đầu</div><div class="m-value">${row_r['CLV_discounted']:,.0f}</div></div>
        <div class="metric-item"><div class="m-label">Remaining CLV</div><div class="m-value" style="color:#4f8ef7">${row_r['Remaining_CLV']:,.0f}</div></div>
    </div>
    """, unsafe_allow_html=True)

    if row_r['Churn_actual'] == 1:
        st.markdown('<div class="warn-banner">⚠️ Khách hàng này đã churn — Remaining CLV = $0. Đường cong survival hiển thị là dự đoán counterfactual.</div>', unsafe_allow_html=True)

    # Survival curve
    times   = np.arange(1, HORIZON+1)
    pred_df = row_cox.to_frame().T.drop(columns=['tenure','Churn_flag','Contract_OneYear','Contract_TwoYear'])
    surv    = cph_tv.predict_survival_function(pred_df, times=times)
    surv_vals = surv.values.flatten()

    # Conditional survival từ t0
    t0      = int(row_r['tenure_actual'])
    t0_idx  = min(t0-1, len(surv_vals)-1)
    S_t0    = surv_vals[t0_idx] if surv_vals[t0_idx] > 1e-10 else 1e-10
    cond    = np.where(times > t0, surv_vals / S_t0, np.nan)

    fig, ax = plt.subplots(figsize=(10, 4), facecolor='#0f1117')
    ax.set_facecolor('#1a1d2e')
    for spine in ax.spines.values(): spine.set_color('#2a2d3e')
    ax.tick_params(colors='#7a7f94', labelsize=8)

    # Unconditional survival (mờ)
    ax.plot(times, surv_vals, color='#4f8ef7', alpha=0.3, linewidth=1.5, linestyle='--', label='Unconditional S(t)')

    # Conditional survival (nổi bật)
    future_mask = times > t0
    ax.plot(times[future_mask], cond[future_mask],
            color='#4f8ef7', linewidth=2.5, label='Conditional S(t | alive at t₀)')

    # Shade vùng tương lai (CLV area)
    ax.fill_between(times[future_mask], cond[future_mask], 0,
                    color='#4f8ef7', alpha=0.1, label='CLV area (discounted)')

    ax.axvline(x=t0, color='#f5a623', linewidth=1.5, linestyle=':', alpha=0.9)
    ax.text(t0+0.5, 0.95, f't₀ = {t0}', color='#f5a623', fontsize=8, va='top')
    ax.axhline(y=0.5, color='#e05c5c', linewidth=1, linestyle=':', alpha=0.5)
    ax.text(1, 0.52, 'S = 0.5', color='#e05c5c', fontsize=7)

    ax.set_xlim(0, HORIZON)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Tenure (tháng)', color='#7a7f94', fontsize=9)
    ax.set_ylabel('Survival Probability', color='#7a7f94', fontsize=9)
    ax.set_title(f'Survival Curve — {selected_id} · {contract}', color='#e0e0e0', fontsize=11)
    ax.legend(fontsize=8, facecolor='#1a1d2e', labelcolor='white', edgecolor='#2a2d3e')
    ax.grid(True, alpha=0.1, color='white')

    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Percentile ranking
    pct = (active['Remaining_CLV'] < row_r['Remaining_CLV']).mean() * 100
    st.markdown(f"""
    <div class="insight-box">
        Khách hàng này nằm ở <strong>percentile {pct:.0f}</strong> về Remaining CLV trong toàn bộ portfolio active.
        {'<br><strong style="color:#e05c5c">→ Thuộc nhóm ưu tiên retention cao.</strong>' if pct >= 75 else ''}
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════
# PAGE 3 — RETENTION SIMULATOR
# ════════════════════════════════════════════
elif page == "Retention Simulator":
    st.markdown("# Retention Simulator")
    st.markdown("<div style='color:#7a7f94;margin-top:-10px;margin-bottom:24px'>Nhập đặc điểm khách hàng → dự đoán survival curve và CLV tức thì</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown('<div class="section-header">Customer Profile</div>', unsafe_allow_html=True)
        contract       = st.selectbox("Loại hợp đồng", ['Month-to-month','One year','Two year'])
        monthly_charge = st.slider("Monthly Charges ($)", 20.0, 120.0, 65.0, 0.5)
        tenure_current = st.slider("Tenure hiện tại (tháng)", 1, 72, 12)
        senior         = st.checkbox("Senior Citizen")
        fiber          = st.checkbox("Internet: Fiber Optic")
        security       = st.checkbox("Online Security")
        techsupport    = st.checkbox("Tech Support")
        paperless      = st.checkbox("Paperless Billing")

        st.markdown('<div class="section-header">Scenario</div>', unsafe_allow_html=True)
        discount_rate  = st.slider("Discount rate/tháng (%)", 0.5, 3.0, 1.0, 0.1) / 100

    with col2:
        # Build input row cho model
        log_t0 = np.log(max(tenure_current, 1))
        input_row = pd.DataFrame([{
            'tenure'                 : tenure_current,
            'Churn_flag'             : 0,
            'MonthlyCharges'         : monthly_charge,
            'SeniorCitizen'          : int(senior),
            'InternetService_Fiber'  : int(fiber),
            'OnlineSecurity_Yes'     : int(security),
            'TechSupport_Yes'        : int(techsupport),
            'PaperlessBilling_Yes'   : int(paperless),
            'MonthlyCharges_x_logT'  : monthly_charge * log_t0,
            'TechSupport_x_logT'     : int(techsupport) * log_t0,
            'Contract'               : contract,
        }])

        times = np.arange(1, HORIZON+1)
        surv  = cph_tv.predict_survival_function(
            input_row.drop(columns=['tenure','Churn_flag']),
            times=times
        )
        surv_vals = surv.values.flatten()

        # Conditional survival
        t0      = tenure_current
        t0_idx  = min(t0-1, len(surv_vals)-1)
        S_t0    = surv_vals[t0_idx] if surv_vals[t0_idx] > 1e-10 else 1e-10
        future  = times > t0
        cond    = np.where(future, surv_vals / S_t0, np.nan)

        # Remaining CLV
        months_ahead = np.where(future, times - t0, 0)
        disc         = 1 / (1 + discount_rate) ** months_ahead
        remaining_clv = float(np.nansum(np.where(future, cond * disc, 0) * monthly_charge))

        # KPIs
        if np.any(~np.isnan(cond[future])):
            surv_12 = float(cond[future][np.where(future)[0][np.argmin(abs(times[future]-(t0+12)))]]) if t0+12 <= HORIZON else None
            surv_24 = float(cond[future][np.where(future)[0][np.argmin(abs(times[future]-(t0+24)))]]) if t0+24 <= HORIZON else None
        else:
            surv_12 = surv_24 = None

        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-item"><div class="m-label">Remaining CLV</div><div class="m-value" style="color:#4f8ef7">${remaining_clv:,.0f}</div></div>
            <div class="metric-item"><div class="m-label">P(survive +12m)</div><div class="m-value">{"—" if surv_12 is None else f"{surv_12:.1%}"}</div></div>
            <div class="metric-item"><div class="m-label">P(survive +24m)</div><div class="m-value">{"—" if surv_24 is None else f"{surv_24:.1%}"}</div></div>
            <div class="metric-item"><div class="m-label">Monthly</div><div class="m-value">${monthly_charge:.0f}</div></div>
        </div>
        """, unsafe_allow_html=True)

        # Plot
        fig, ax = plt.subplots(figsize=(9, 4.5), facecolor='#0f1117')
        ax.set_facecolor('#1a1d2e')
        for spine in ax.spines.values(): spine.set_color('#2a2d3e')
        ax.tick_params(colors='#7a7f94', labelsize=8)

        c = COLOR_MAP.get(contract, '#4f8ef7')
        ax.plot(times, surv_vals, color=c, alpha=0.25, linewidth=1.5, linestyle='--')
        ax.plot(times[future], cond[future], color=c, linewidth=2.5, label=f'Conditional S(t) · {contract}')
        ax.fill_between(times[future], cond[future], 0, color=c, alpha=0.12, label='Remaining CLV area')
        ax.axvline(x=t0, color='#f5a623', linewidth=1.5, linestyle=':', alpha=0.9)
        ax.text(t0+0.5, 0.97, f't₀={t0}', color='#f5a623', fontsize=8, va='top')

        if surv_12 and t0+12 <= HORIZON:
            ax.scatter([t0+12], [surv_12], color='white', s=40, zorder=5)
            ax.text(t0+12, surv_12+0.04, f'+12m: {surv_12:.0%}', color='white', fontsize=7.5, ha='center')
        if surv_24 and t0+24 <= HORIZON:
            ax.scatter([t0+24], [surv_24], color='white', s=40, zorder=5)
            ax.text(t0+24, surv_24+0.04, f'+24m: {surv_24:.0%}', color='white', fontsize=7.5, ha='center')

        ax.set_xlim(0, HORIZON)
        ax.set_ylim(0, 1.12)
        ax.set_xlabel('Tenure (tháng)', color='#7a7f94', fontsize=9)
        ax.set_ylabel('Survival Probability', color='#7a7f94', fontsize=9)
        ax.set_title('Predicted Survival Curve', color='#e0e0e0', fontsize=11)
        ax.legend(fontsize=8, facecolor='#1a1d2e', labelcolor='white', edgecolor='#2a2d3e')
        ax.grid(True, alpha=0.1, color='white')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

        # What-if scenarios
        st.markdown('<div class="section-header">What-If: Nếu nâng cấp hợp đồng?</div>', unsafe_allow_html=True)

        whatif_data = []
        for c_type in ['Month-to-month','One year','Two year']:
            if c_type == contract:
                continue
            log_t0_ = np.log(max(tenure_current, 1))
            row_ = pd.DataFrame([{
                'tenure'                : tenure_current,
                'Churn_flag'            : 0,
                'MonthlyCharges'        : monthly_charge,
                'SeniorCitizen'         : int(senior),
                'InternetService_Fiber' : int(fiber),
                'OnlineSecurity_Yes'    : int(security),
                'TechSupport_Yes'       : int(techsupport),
                'PaperlessBilling_Yes'  : int(paperless),
                'MonthlyCharges_x_logT' : monthly_charge * log_t0_,
                'TechSupport_x_logT'    : int(techsupport) * log_t0_,
                'Contract'              : c_type,
            }])
            s_   = cph_tv.predict_survival_function(
                row_.drop(columns=['tenure','Churn_flag']), times=times).values.flatten()
            S_t0_ = s_[t0_idx] if s_[t0_idx] > 1e-10 else 1e-10
            cond_ = np.where(future, s_ / S_t0_, 0)
            clv_  = float(np.sum(cond_[future] * disc[future]) * monthly_charge)
            delta = clv_ - remaining_clv
            whatif_data.append({'Contract': c_type, 'Remaining CLV': clv_, 'Δ CLV': delta})

        for w in whatif_data:
            sign  = "+" if w['Δ CLV'] >= 0 else ""
            color = "#4caf7d" if w['Δ CLV'] >= 0 else "#e05c5c"
            st.markdown(f"""
            <div class="insight-box" style="border-left-color:{color}">
                Nếu chuyển sang <strong>{w['Contract']}</strong>:
                Remaining CLV = <strong>${w['Remaining CLV']:,.0f}</strong>
                &nbsp;(<span style="color:{color}">{sign}${w['Δ CLV']:,.0f}</span> so với hiện tại)
            </div>
            """, unsafe_allow_html=True)
