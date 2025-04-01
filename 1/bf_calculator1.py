import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Global style configurations
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12
sns.set_style("whitegrid")

COLOR_SCHEME = {
    'reference': '#95a5a6',
    'low_risk': '#ffa500',
    'medium_risk': '#ff6347',
    'high_risk': '#8b0000',
    'baseline': '#2c3e50'
}

def generate_bmi_cunbae_data():
    """Generate BMI-CUNBAE dataset using standard formulas"""
    height_range = np.arange(140, 201, 1)
    weight_range = np.arange(40, 151, 1)
    
    dataset = []
    for height in height_range:
        for weight in weight_range:
            bmi = weight / ((height/100) ** 2)
            bmi_sq = bmi ** 2
            
            # Male calculation (sex=0, age=30)
            cun_male = (
                -44.988 + (0.503 * 30) + (10.689 * 0) + 
                (3.172 * bmi) - (0.026 * bmi_sq) + 
                (0.181 * bmi * 0) - (0.02 * bmi * 30) - 
                (0.005 * bmi_sq * 0) + (0.00021 * bmi_sq * 30)
            )  # 修复这里缺失的右括号
            
            # Female calculation (sex=1, age=30)
            cun_female = (
                -44.988 + (0.503 * 30) + (10.689 * 1) + 
                (3.172 * bmi) - (0.026 * bmi_sq) + 
                (0.181 * bmi * 1) - (0.02 * bmi * 30) - 
                (0.005 * bmi_sq * 1) + (0.00021 * bmi_sq * 30)
            )
            
            dataset.append([
                height, weight, bmi,
                max(0, cun_male),
                max(0, cun_female)
            ])
    
    return pd.DataFrame(
        dataset,
        columns=['Height', 'Weight', 'BMI', 'CUN-BAE Male', 'CUN-BAE Female']
    )
def styled_metric(label, value):
    """Styled metric component"""
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

def enhanced_forest_plot(risk_data, region, gender):
    """Enhanced forest plot visualization"""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    categories = [g["label"] for g in risk_data]
    hr_values = [g["HR"] for g in risk_data]
    ci_ranges = [g["CI"] for g in risk_data]
    
    colors = []
    for hr in hr_values:
        if hr == 1.0:
            colors.append(COLOR_SCHEME['reference'])
        elif hr < 1.5:
            colors.append(COLOR_SCHEME['low_risk'])
        elif 1.5 <= hr < 2.0:
            colors.append(COLOR_SCHEME['medium_risk'])
        else:
            colors.append(COLOR_SCHEME['high_risk'])
    
    lower_ci = [hr - ci[0] for hr, ci in zip(hr_values, ci_ranges)]
    upper_ci = [ci[1] - hr for hr, ci in zip(hr_values, ci_ranges)]
    y_pos = np.arange(len(categories))
    
    for i, (hr, lci, uci, color) in enumerate(zip(hr_values, lower_ci, upper_ci, colors)):
        ax.errorbar(hr, i,
                    xerr=[[lci], [uci]],
                    fmt='o',
                    color=color,
                    markersize=18,
                    markeredgecolor='white',
                    markeredgewidth=2.5,
                    capsize=14,
                    capthick=3,
                    elinewidth=3,
                    alpha=0.97)
    
    ax.axvline(x=1, color=COLOR_SCHEME['baseline'],
               linestyle=':', linewidth=3, alpha=0.8)
    
    for idx, (hr, ci) in enumerate(zip(hr_values, ci_ranges)):
        if hr == 1.0:
            continue
        
        annotation_text = f"HR: {hr:.2f}\n95% CI: {ci[0]:.2f}-{ci[1]:.2f}"
        x_pos = hr * 1.2 if hr > 1 else hr * 0.8
        ax.text(x_pos, idx, annotation_text,
                va='center', fontsize=12,
                color='#2d3436',
                bbox=dict(facecolor='white', alpha=0.95,
                          edgecolor='#dfe6e9', boxstyle='round,pad=0.5'))
    
    ax.set_ylabel("CUN-BAE Groups", 
                 fontsize=15, 
                 fontweight='semibold',
                 labelpad=15)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories, fontsize=14)
    ax.set_xlabel("Hazard Ratio (HR)", 
                 fontsize=15, 
                 fontweight='semibold',
                 labelpad=15)
    ax.set_title(f"{region} - {gender}  Cardiovascular Risk Stratification\n",
                fontsize=20,
                fontweight='bold',
                pad=25)
    
    ax.xaxis.grid(True, linestyle='--', alpha=0.6)
    for spine in ['right', 'top']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color('#b2bec3')
        ax.spines[spine].set_linewidth=2

    plt.tight_layout()
    return fig

def parse_group_label(label):
    """Parse CUN-BAE group labels"""
    if '<' in label:
        return {'upper': float(label.split('<')[1])}
    elif '>' in label:
        return {'lower': float(label.split('>')[1])}
    elif '-' in label:
        lower, upper = map(float, label.split('-'))
        return {'lower': lower, 'upper': upper}
    return {}

def get_risk_color(hr_value):
    """Dynamic risk color system"""
    if hr_value == 1.0:
        return '#d4edda'  # Light green
    elif 1.0 < hr_value < 1.5:
        return '#fff3cd'  # Light yellow
    elif 1.5 <= hr_value < 2.0:
        return '#f8d7da'  # Light pink
    else:
        return '#dc3545'  # Red

def main():
    st.set_page_config(
        page_title="CUN-BAE Risk Analysis System",
        page_icon="❤️",
        layout="centered",
        initial_sidebar_state="expanded"
    )
    
    st.markdown(f"""
    <style>
        .main {{ background: #f8f9fa; }}
        .stNumberInput > label, .stRadio > label, .stSelectbox > label {{
            font-weight: 500 !important;
            color: #2c3e50 !important;
        }}
        .st-emotion-cache-1y4p8pa {{ 
            background: #ffffff;
            border-right: 2px solid #dfe6e9;
        }}
        h1 {{
            color: {COLOR_SCHEME['baseline']} !important;
            border-bottom: 3px solid {COLOR_SCHEME['baseline']}50 !important;
            padding-bottom: 0.5rem !important;
        }}
    </style>
    """, unsafe_allow_html=True)

    st.title("Cardiovascular Disease Risk Assessment System")
    
    with st.sidebar:
        st.header("⚙️ Parameter Settings")
        with st.expander("Basic Information", expanded=True):
            height = st.number_input("Height (cm)", 140, 200, 170)
            weight = st.number_input("Weight (kg)", 40, 150, 65)
            age = st.number_input("Age", 18, 100, 45)
            gender = st.radio("Gender", ["Male", "Female"])
            region = st.selectbox("Region", ["China", "Europe"])
    
    risk_stratification = {
        "China": {
            "Male": [
                {"label": "<20.2", "HR": 1.0, "CI": (1.0, 1.0)},
                {"label": "20.2-24.0", "HR": 1.326, "CI": (1.051, 1.673)},
                {"label": "24.0-28.6", "HR": 1.747, "CI": (1.376, 2.217)},
                {"label": ">28.6", "HR": 2.658, "CI": (1.959, 3.604)}
            ],
            "Female": [
                {"label": "<32.2", "HR": 1.0, "CI": (1.0, 1.0)},
                {"label": "32.2-36.9", "HR": 1.310, "CI": (1.035, 1.658)},
                {"label": "36.9-41.8", "HR": 1.619, "CI": (1.277, 2.052)},
                {"label": ">41.8", "HR": 2.291, "CI": (1.715, 3.060)}
            ]
        },
        "Europe": {
            "Male": [
                {"label": "<25.4", "HR": 1.0, "CI": (1.0, 1.0)},
                {"label": "25.4-30.1", "HR": 1.474, "CI": (1.213, 1.792)},
                {"label": "30.1-35.3", "HR": 1.604, "CI": (1.303, 1.973)},
                {"label": ">35.3", "HR": 1.804, "CI": (1.377, 2.365)}
            ],
            "Female": [
                {"label": "<35.2", "HR": 1.0, "CI": (1.0, 1.0)},
                {"label": "35.2-40.4", "HR": 1.629, "CI": (1.287, 2.061)},
                {"label": "40.4-46.0", "HR": 2.290, "CI": (1.817, 2.886)},
                {"label": ">46.0", "HR": 2.591, "CI": (2.003, 3.352)}
            ]
        }
    }

    if st.sidebar.button("Start Analysis"):
        bmi = weight / ((height/100) ** 2)
        bmi_sq = bmi ** 2
        sex_code = 0 if gender == "Male" else 1
        
        # Calculate CUN-BAE
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
        cun_bae = max(0, cun_bae)  # Ensure non-negative

        col1, col2 = st.columns(2)
        with col1:
            styled_metric("BMI", f"{bmi:.1f}")
        with col2:
            styled_metric("CUN-BAE", f"{cun_bae:.1f}")

        # Determine risk group
        current_group = None
        for group in risk_stratification[region][gender]:
            parsed = parse_group_label(group["label"])
            if 'upper' in parsed and cun_bae < parsed['upper']:
                current_group = group
                break
            elif 'lower' in parsed and 'upper' in parsed and parsed['lower'] <= cun_bae < parsed['upper']:
                current_group = group
                break
            elif 'lower' in parsed and cun_bae >= parsed['lower']:
                current_group = group
                break
        else:
            current_group = risk_stratification[region][gender][-1]

        # Generate alert content
        base_group = risk_stratification[region][gender][0]
        if current_group['HR'] == 1.0:
            alert_content = [
                "✅ **Optimal Health Status**",
                f"**Current Group:** {current_group['label']}",
                "**Hazard Ratio:** 1.00 (Reference)",
                "**Recommendation:** Maintain healthy lifestyle"
            ]
            show_ci = False
        else:
            alert_content = [
                "⚠️ **Elevated Risk Warning**",
                f"**Current Group:** {current_group['label']}",
                f"**Hazard Ratio:** {current_group['HR']:.2f} (vs {base_group['label']})",
                "**Recommendation:** Reduce body fat through diet and exercise"
            ]
            show_ci = True

        alert_message = '<br>'.join(alert_content)
        bg_color = get_risk_color(current_group['HR'])
        
        # Build HTML content
        ci_html = f'''
        <div style="margin-top: 0.8rem; font-size: 0.9rem; color: #636e72;">
            95% Confidence Interval: ({current_group['CI'][0]:.2f}-{current_group['CI'][1]:.2f})
        </div>
        ''' if show_ci else ''

        st.markdown(
            f'''
            <div style="padding: 1.2rem;
                        border-radius: 8px;
                        background: {bg_color};
                        border-left: 6px solid {COLOR_SCHEME['baseline']};
                        margin: 1.5rem 0;">
                <div style="color: #2c3e50;
                            font-size: 1.05rem;
                            line-height: 1.8;">
                    {alert_message}
                    {ci_html}

            ''',
            unsafe_allow_html=True
        )

        # Visualization
        st.subheader("Cardiovascular Risk Stratification Analysis")
        fig = enhanced_forest_plot(risk_stratification[region][gender], region, gender)
        st.pyplot(fig)

        # Distribution plot
        plt.figure(figsize=(8, 6))
        sns.kdeplot(x=np.random.normal(bmi, 3, 500),
                   y=np.random.normal(cun_bae, 4, 500),
                   cmap="Blues", fill=True, alpha=0.7)
        plt.scatter(bmi, cun_bae, c=COLOR_SCHEME['high_risk'],
                   s=120, edgecolor='white', linewidth=1.5,
                   label='Your Position')
        plt.xlabel("BMI")
        plt.ylabel("CUN-BAE")
        plt.title("Population Distribution")
        plt.legend()
        st.sidebar.pyplot(plt.gcf())

if __name__ == "__main__":
    main()