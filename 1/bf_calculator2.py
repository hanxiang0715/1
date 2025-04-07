import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import seaborn as sns

# 全局样式配置
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12
sns.set_style("whitegrid")

COLOR_SCHEME = {
    'baseline': '#2c3e50'
}

# 新增分组颜色方案（每个分组固定颜色）
GROUP_COLORS = ['#d4edda', '#fff3cd', '#F0A73A', '#dc3545']

def styled_metric(label, value):
    """Streamlit 风格化指标展示（未改动）"""
    st.markdown(f"""
    <div style="padding: 1rem;
                background: #ffffff;
                border-radius: 10px;
                margin: 0.8rem 0;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                border-left: 4px solid {COLOR_SCHEME['baseline']};">
        <div style="color: #2c3e50; 
                    font-size: 0.95rem;
                    font-weight: 500;">{label}</div>
        <div style="color: {COLOR_SCHEME['baseline']};
                    font-size: 1.8rem;
                    font-weight: 700;
                    margin-top: 0.5rem;">{value}</div>
    </div>
    """, unsafe_allow_html=True)

def parse_group_label(label):
    """
    解析组标签（未改动）：
      "<20.2" → {'upper':20.2}
      "20.2~<24.0" 或 "20.2-24.0" → {'lower':20.2,'upper':24.0}
      "≥28.6" → {'lower':28.6}
    """
    if '<' in label and '>' not in label:
        return {'upper': float(label.split('<')[1])}
    if '>' in label and '<' not in label:
        return {'lower': float(label.split('>')[1])}
    cleaned = label.replace('~<', '-')
    if '-' in cleaned:
        lo, hi = cleaned.split('-')
        return {'lower': float(lo), 'upper': float(hi)}
    return {}

def custom_forest_plot(risk_data, region, gender):
    """
    修改说明：
    - 拉长图表宽度，确保 "Moderate to high risk" 的文字内容有足够的显示空间
    - 调整文字对齐和间距
    """
    # 1. 取出所有分界点
    boundaries = sorted({v for grp in risk_data for v in parse_group_label(grp['label']).values()})
    # 2. 动态轴范围
    margin = 5
    start = boundaries[0] - margin
    end   = boundaries[-1] + margin
    axis_start = max(0, int(np.floor(start / 5) * 5))
    axis_end   = int(np.ceil(end   / 5) * 5)
    # 3. 分段区间
    segment_bounds = [(axis_start, boundaries[0])]
    for i in range(len(boundaries)-1):
        segment_bounds.append((boundaries[i], boundaries[i+1]))
    segment_bounds.append((boundaries[-1], axis_end))
    # 4. 风险名称映射
    risk_labels = ["Low risk", "Moderate risk", "Moderate to high risk", "High risk"]
    # 5. 计算中点
    x_mids = [(lo+hi)/2 for lo, hi in segment_bounds]
    # 6. 获取每段色块（按分组顺序取颜色）
    block_colors = GROUP_COLORS[:len(risk_data)]

    # 7. 创建画布（拉长宽度）
    fig = plt.figure(figsize=(12, 4))  # 增加宽度
    fig.patch.set_facecolor('#f8f9fa')
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2], hspace=0.08)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

    # —— 顶部彩色块 + 文本
    ax_top.set_xlim(axis_start, axis_end)
    ax_top.set_ylim(0, 1)
    ax_top.axis('off')
    for i, ((lo, hi), grp) in enumerate(zip(segment_bounds, risk_data)):
        ax_top.add_patch(
            Rectangle((lo, 0), hi-lo, 1,
                      transform=ax_top.get_xaxis_transform(),
                      color=block_colors[i], zorder=0)
        )
        # 风险名称
        label = "Reference" if grp['HR']==1.0 else risk_labels[i]
        ax_top.text(x_mids[i], 0.65, label,
                    ha='center', va='center',
                    fontsize=13, fontweight='bold', color='#2c3e50')
        # HR 值
        ax_top.text(x_mids[i], 0.35, f"HR: {grp['HR']:.2f}",
                    ha='center', va='center',
                    fontsize=11, color='#2c3e50')

    # —— 底部分段彩色块
    ax_bot.set_xlim(axis_start, axis_end)
    ax_bot.set_ylim(-0.5, 1)
    ax_bot.set_facecolor('white')
    ax_bot.spines['top'].set_visible(False)
    ax_bot.spines['right'].set_visible(False)
    ax_bot.spines['left'].set_visible(False)
    ax_bot.get_yaxis().set_visible(False)
    for (lo, hi), color in zip(segment_bounds, block_colors):
        ax_bot.add_patch(
            Rectangle((lo, -0.1), hi-lo, 0.2,
                      color=color, zorder=0)
        )

    # 主网格 & 中心线
    ax_bot.grid(axis='x', color='#dddddd', linestyle='--', linewidth=1, zorder=1)
    ax_bot.hlines(0, axis_start, axis_end,
                  color=COLOR_SCHEME['baseline'], linewidth=1.5, zorder=2)

    # 主刻度（每 5 单位）& 次刻度（每 1 单位，短刻度线）
    majors = np.arange(axis_start, axis_end+1, 5)
    minors = np.arange(axis_start, axis_end+1, 1)
    ax_bot.set_xticks(majors, minor=False)
    ax_bot.set_xticks(minors, minor=True)
    ax_bot.tick_params(axis='x', which='major', length=8, labelsize=11, color='#333333')
    ax_bot.tick_params(axis='x', which='minor', length=4, labelsize=0, color='#333333')
    ax_bot.set_xlabel("CUN-BAE", fontsize=13,
                      fontweight='semibold', color='#333333')

    # 箭头 & 阈值文字
    for b in boundaries:
        ax_bot.annotate("",
                        xy=(b, 0), xytext=(b, -0.2),
                        arrowprops=dict(arrowstyle='-|>', lw=1.5, color='#333333'),
                        zorder=3)
        ax_bot.text(b, -0.25, f"{b:.1f}",
                    ha='center', va='top',
                    fontsize=10, color='#333333', zorder=3)

    # 分类标签（可选）
    for grp, (lo, hi), xm in zip(risk_data, segment_bounds, x_mids):
        ax_bot.text(xm, 0.6, grp['label'],
                    ha='center', va='bottom',
                    fontsize=10, color='#2c3e50',
                    bbox=dict(boxstyle='round,pad=0.2',
                              facecolor='white', edgecolor='#cccccc'),
                    zorder=3)

    # 标题
    fig.suptitle(f"{region} - {gender}  Cardiovascular Risk Stratification",
                 fontsize=16, fontweight='bold', color='#2c3e50', y=0.93)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    return fig


def main():
    st.set_page_config(
        page_title="CUN-BAE Risk Analysis System",
        page_icon="❤️",
        layout="wide",  # 使用宽布局
        initial_sidebar_state="expanded"
    )
    st.markdown(f"""
    <style>
        .main {{ background: #f8f9fa; }}
        .stNumberInput > label, .stRadio > label, .stSelectbox > label {{
            font-weight: 500 !important;
            color: #2c3e50 !important;
        }}
        h1 {{
            color: {COLOR_SCHEME['baseline']} !important;
            border-bottom: 3px solid {COLOR_SCHEME['baseline']}50 !important;
            padding-bottom: 0.5rem !important;
        }}
        /* 拉长侧边栏宽度 */
        .css-1y4i2ku {{
            width: 400px !important;  /* 调整侧边栏宽度 */
        }}
        /* 调整主内容区域的边距 */
        .css-1v0mbdj {{
            margin-left: 420px !important;  /* 与侧边栏宽度匹配 */
        }}
    </style>
    """, unsafe_allow_html=True)
    st.title("Cardiovascular Disease Risk Assessment System")

    with st.sidebar:
        st.header("⚙️ Parameter Settings")
        with st.expander("Basic Information", expanded=True):
            height = st.number_input("Height (cm)", 140, 200, 170)
            weight = st.number_input("Weight (kg)", 40, 150, 65)
            age = st.number_input("Age", 50, 100, 50)
            gender = st.radio("Gender", ["Male", "Female"])
            region = st.selectbox("Region", ["China", "Europe"])

    risk_stratification = {
        "China": {
            "Male": [
                {"label": "<20.2",     "HR": 1.0,   "CI": (1.0, 1.0)},
                {"label": "20.2~<24.0", "HR": 1.326, "CI": (1.051, 1.673)},
                {"label": "24.0~<28.6", "HR": 1.747, "CI": (1.376, 2.217)},
                {"label": "≥28.6",     "HR": 2.658, "CI": (1.959, 3.604)}
            ],
            "Female": [
                {"label": "<32.2",     "HR": 1.0,   "CI": (1.0, 1.0)},
                {"label": "32.2~<36.9", "HR": 1.310, "CI": (1.035, 1.658)},
                {"label": "36.9~<41.8", "HR": 1.619, "CI": (1.277, 2.052)},
                {"label": "≥41.8",     "HR": 2.291, "CI": (1.715, 3.060)}
            ]
        },
        "Europe": {
            "Male": [
                {"label": "<25.4",     "HR": 1.0,   "CI": (1.0, 1.0)},
                {"label": "25.4~<30.1", "HR": 1.474, "CI": (1.213, 1.792)},
                {"label": "30.1~<35.3", "HR": 1.604, "CI": (1.303, 1.973)},
                {"label": "≥35.3",     "HR": 1.804, "CI": (1.377, 2.365)}
            ],
            "Female": [
                {"label": "<35.2",     "HR": 1.0,   "CI": (1.0, 1.0)},
                {"label": "35.2~<40.4", "HR": 1.629, "CI": (1.287, 2.061)},
                {"label": "40.4~<46.0", "HR": 2.290, "CI": (1.817, 2.886)},
                {"label": "≥46.0",     "HR": 2.591, "CI": (2.003, 3.352)}
            ]
        }
    }

    if st.sidebar.button("Start Analysis"):
        bmi = weight / ((height/100) ** 2)
        bmi_sq = bmi ** 2
        sex_code = 0 if gender == "Male" else 1
        cun_bae = (
            -44.988 +
            (0.503 * age) +
            (10.689 * sex_code) +
            (3.172 * bmi) -
            (0.026 * bmi_sq) +
            (0.181 * bmi * sex_code) -
            (0.02 * bmi * age) -
            (0.005 * bmi_sq * sex_code) +
            (0.00021 * bmi_sq * age)
        )
        cun_bae = max(0, cun_bae)

        col1, col2 = st.columns(2)
        with col1:
            styled_metric("BMI", f"{bmi:.1f}")
        with col2:
            styled_metric("CUN-BAE", f"{cun_bae:.1f}")

        # 确定当前组（新增索引查找）
        current_group = None
        current_index = 0
        for idx, grp in enumerate(risk_stratification[region][gender]):
            p = parse_group_label(grp["label"])
            if 'upper' in p and cun_bae < p['upper']:
                current_group = grp
                current_index = idx
                break
            if 'lower' in p and 'upper' in p and p['lower'] <= cun_bae < p['upper']:
                current_group = grp
                current_index = idx
                break
            if 'lower' in p and cun_bae >= p['lower']:
                current_group = grp
                current_index = idx
                break
        else:
            current_group = risk_stratification[region][gender][-1]
            current_index = len(risk_stratification[region][gender]) - 1

        # 警告框（颜色改为使用分组索引）
        base_group = risk_stratification[region][gender][0]
        if current_group['HR'] == 1.0:
            alert = [
                "✅ **Optimal Health Status**",
                f"**Current Group:** {current_group['label']}",
                "**Hazard Ratio:** 1.00 (Reference)",
                "**Recommendation:** Maintain healthy lifestyle"
            ]
            show_ci = False
        else:
            alert = [
                "⚠️ **Elevated Risk Warning**",
                f"**Current Group:** {current_group['label']}",
                f"**Hazard Ratio:** {current_group['HR']:.2f} (vs {base_group['label']})",
                "**Recommendation:** Reduce body fat through diet and exercise"
            ]
            show_ci = True

        ci_html = (f'<div style="margin-top:0.8rem;color:#636e72;font-size:0.9rem;">'
                   f'95% CI: ({current_group["CI"][0]:.2f}-{current_group["CI"][1]:.2f})'
                   f'</div>') if show_ci else ''

        st.markdown(
            f'''
            <div style="padding:1.2rem;border-radius:8px;background:{GROUP_COLORS[current_index]};border-left:6px solid {COLOR_SCHEME["baseline"]};margin:1.5rem 0;">
                {'<br>'.join(alert)}
                {ci_html}
            </div>
            ''',
            unsafe_allow_html=True
        )

        # 森林图
        st.subheader("Cardiovascular Risk Stratification Analysis")
        fig = custom_forest_plot(risk_data=risk_stratification[region][gender], region=region, gender=gender)
        st.pyplot(fig)

        # 人群分布图
        plt.figure(figsize=(8, 6))
        sns.kdeplot(x=np.random.normal(bmi, 3, 500),
                   y=np.random.normal(cun_bae, 4, 500),
                   cmap="Blues", fill=True, alpha=0.7)
        plt.scatter(bmi, cun_bae, c='#8b0000',
                    s=120, edgecolor='white', linewidth=1.5,
                    label='Your Position')
        plt.xlabel("BMI")
        plt.ylabel("CUN-BAE")
        plt.title("Population Distribution")
        plt.legend()
        st.sidebar.pyplot(plt.gcf())

if __name__ == "__main__":
    main()