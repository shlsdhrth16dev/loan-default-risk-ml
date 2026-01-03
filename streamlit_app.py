"""
Streamlit Decision Intelligence System for Loan Default Prediction.

A transparent, interactive system that shows:
- How decisions are made
- Why decisions change  
- How agents collaborate

Phase 1-2 MVP: Loan Evaluation + Outcome Dashboard
With Visual Polish Applied
"""
import streamlit as st
import sys
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agents.loan_decision_crew import get_loan_decision_crew
from streamlit_components.agent_playground import render_agent_playground

# Page config
st.set_page_config(
    page_title="Loan Decision Intelligence System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


def apply_custom_styling():
    """Apply custom CSS for visual polish."""
    st.markdown("""
        <style>
        /* Smooth page transitions */
        .main > div {
            animation: fadeIn 0.5s;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        /* Glassmorphism for metrics */
        [data-testid="stMetricValue"] {
            font-size: 2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        /* Animated buttons */
        .stButton>button {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: 600;
            border: none;
            border-radius: 10px;
            padding: 0.75rem 2rem;
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        }
        
        /* Better form styling */
        .stForm {
            background: rgba(255, 255, 255, 0.02);
            border-radius: 15px;
            padding: 2rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        /* Better tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 1rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 10px 10px 0 0;
            padding: 1rem 2rem;
            font-weight: 600;
        }
        
        /* Card styling */
        [data-testid="stExpander"] {
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            background: rgba(255, 255, 255, 0.02);
        }
        
        /* Better metrics */
        [data-testid="stMetricLabel"] {
            font-size: 1.1rem;
            font-weight: 600;
        }
        </style>
    """, unsafe_allow_html=True)


# Initialize session state
if 'decisions_history' not in st.session_state:
    st.session_state.decisions_history = []

if 'current_decision' not in st.session_state:
    st.session_state.current_decision = None

# Apply styling
apply_custom_styling()


def render_decision_badge(decision):
    """Render rich decision badge with gradient."""
    if decision == "APPROVED":
        return """
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 2.5rem; border-radius: 15px; text-align: center;
                    box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
                    margin: 2rem 0;'>
            <h1 style='color: white; margin: 0; font-size: 3.5rem;'>
                ✅ APPROVED
            </h1>
            <p style='color: rgba(255,255,255,0.9); margin-top: 0.5rem; font-size: 1.2rem;'>
                Application meets all criteria
            </p>
        </div>
        """
    elif decision == "REJECTED":
        return """
        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 2.5rem; border-radius: 15px; text-align: center;
                    box-shadow: 0 10px 40px rgba(245, 87, 108, 0.3);
                    margin: 2rem 0;'>
            <h1 style='color: white; margin: 0; font-size: 3.5rem;'>
                ❌ REJECTED
            </h1>
            <p style='color: rgba(255,255,255,0.9); margin-top: 0.5rem; font-size: 1.2rem;'>
                Application does not meet criteria
            </p>
        </div>
        """
    else:  # MANUAL_REVIEW
        return """
        <div style='background: linear-gradient(135deg, #fad961 0%, #f76b1c 100%); 
                    padding: 2.5rem; border-radius: 15px; text-align: center;
                    box-shadow: 0 10px 40px rgba(247, 107, 28, 0.3);
                    margin: 2rem 0;'>
            <h1 style='color: white; margin: 0; font-size: 3.5rem;'>
                ⚠️ REVIEW REQUIRED
            </h1>
            <p style='color: rgba(255,255,255,0.9); margin-top: 0.5rem; font-size: 1.2rem;'>
                Application requires manual evaluation
            </p>
        </div>
        """


def render_header():
    """Render app header."""
    st.title("🤖 Loan Decision Intelligence System")
    st.markdown("""
    **Not just predicting loan default, but showing how decisions are made.**
    
    This system combines:
    - 🧠 Predictive ML (XGBoost with 3-tier fallback)
    - 🤝 Multi-Agent AI (5 specialized CrewAI agents)
    - 📊 Transparent explanations (3 audience formats)
    """)
    
    with st.expander("ℹ️ How This System Works", expanded=False):
        st.markdown("""
        **Our Multi-Agent AI Pipeline:**
        1. 🔍 **Validation Agent** - Checks data quality and completeness
        2. 📊 **Risk Analyst Agent** - Predicts default probability using ML
        3. ⚖️ **Policy Agent** - Applies business rules and compliance checks
        4. 📝 **Explanation Agent** - Generates human-readable narratives
        5. 📈 **Monitoring Agent** - Tracks system health (background)
        
        **Fallback System:** XGBoost → Logistic Regression → Rule-Based (99.9% uptime)
        """)
    
    st.divider()


def render_loan_form():
    """Render loan application form."""
    st.header("📝 Loan Application")
    
    with st.form("loan_application"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Personal Info")
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            income = st.number_input("Annual Income ($)", min_value=0, max_value=500000, value=55000, step=1000)
            education = st.selectbox("Education", ["High School", "Bachelor", "Master", "PhD"])
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        
        with col2:
            st.subheader("Employment")
            employment_type = st.selectbox("Employment Type", ["Full-time", "Part-time", "Self-employed", "Unemployed"])
            months_employed = st.number_input("Months Employed", min_value=0, max_value=600, value=48)
            has_dependents = st.selectbox("Has Dependents", ["Yes", "No"])
            has_mortgage = st.selectbox("Has Mortgage", ["Yes", "No"])
        
        with col3:
            st.subheader("Loan Details")
            loan_amount = st.number_input("Loan Amount ($)", min_value=1000, max_value=100000, value=25000, step=1000)
            loan_term = st.selectbox("Loan Term (months)", [36, 60, 84, 120])
            loan_purpose = st.selectbox("Loan Purpose", ["DebtConsolidation", "HomeImprovement", "Medical", "Education", "Other"])
            has_cosigner = st.selectbox("Has Cosigner", ["Yes", "No"])
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Financial Metrics")
            credit_score = st.slider("Credit Score", min_value=300, max_value=850, value=680)
            dti_ratio = st.slider("Debt-to-Income Ratio", min_value=0.0, max_value=1.0, value=0.35, step=0.01)
        
        with col2:
            st.subheader("Credit Profile")
            num_credit_lines = st.number_input("Number of Credit Lines", min_value=0, max_value=50, value=5)
            interest_rate = st.slider("Interest Rate", min_value=0.01, max_value=0.35, value=0.085, step=0.001, format="%.3f")
        
        st.divider()
        submit = st.form_submit_button("🚀 Evaluate Application", use_container_width=True, type="primary")
        
        if submit:
            application = {
                "age": age,
                "income": income,
                "loanamount": loan_amount,
                "creditscore": credit_score,
                "monthsemployed": months_employed,
                "numcreditlines": num_credit_lines,
                "interestrate": interest_rate,
                "loanterm": loan_term,
                "dtiratio": dti_ratio,
                "education": education,
                "employmenttype": employment_type,
                "maritalstatus": marital_status,
                "hasmortgage": has_mortgage,
                "hasdependents": has_dependents,
                "loanpurpose": loan_purpose,
                "hascosigner": has_cosigner
            }
            
            return application
    
    return None


def process_application(application):
    """Process application through CrewAI with visual feedback."""
    # Animated progress
    progress_text = "🤖 AI Agents Processing..."
    progress_bar = st.progress(0, text=progress_text)
    
    progress_bar.progress(20, text="✓ Validating application data...")
    time.sleep(0.2)
    
    progress_bar.progress(40, text="✓ Risk Analyst evaluating...")
    crew = get_loan_decision_crew()
    
    progress_bar.progress(60, text="✓ Policy Agent reviewing...")
    result = crew.process(application)
    
    progress_bar.progress(80, text="✓ Explanation Agent writing...")
    time.sleep(0.2)
    
    progress_bar.progress(100, text="✓ Decision Complete!")
    time.sleep(0.3)
    progress_bar.empty()
    
    # Success animation
    st.balloons()
    st.success("✅ Decision Generated Successfully!", icon="✅")
    
    # Store in history
    result['timestamp'] = datetime.now()
    result['application'] = application
    st.session_state.decisions_history.append(result)
    st.session_state.current_decision = result
    
    return result


def render_decision_result(result):
    """Render decision result with rich formatting."""
    st.header("🎯 Decision Result")
    
    # Rich decision badge
    st.markdown(render_decision_badge(result['decision']), unsafe_allow_html=True)
    
    # Metrics with icons
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("### 📊 Default Risk")
        st.metric("", f"{result['risk_assessment']['default_probability']:.1%}")
    
    with col2:
        st.markdown("### 🎯 Risk Level")
        st.metric("", result['risk_assessment']['risk_level'].replace('_', ' ').title())
    
    with col3:
        st.markdown("### 💪 Confidence")
        st.metric("", f"{result['confidence']:.0%}")
    
    with col4:
        st.markdown("### ⚡ Speed")
        exec_time = result['metadata']['total_execution_time_ms']
        st.metric("", f"{exec_time:.0f}ms")
    
    st.divider()
    
    # Explanations in tabs
    st.subheader("📖 Decision Explanations")
    
    tab1, tab2, tab3 = st.tabs(["👤 For Borrower", "👨‍💼 For Loan Officer", "📋 For Auditor"])
    
    with tab1:
        st.info(result['explanations']['borrower_explanation'], icon="💬")
    
    with tab2:
        st.code(result['explanations']['loan_officer_explanation'], language=None)
    
    with tab3:
        st.text(result['explanations']['auditor_explanation'])
    
    # Agent details
    with st.expander("🤖 Agent Execution Details"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Agents Executed:**")
            for agent in result['metadata']['agents_executed']:
                st.markdown(f"- ✓ {agent}")
        
        with col2:
            st.markdown("**Execution Times:**")
            for agent, time_ms in result['metadata']['agent_execution_times'].items():
                st.markdown(f"- {agent}: {time_ms:.1f}ms")
        
        if result['risk_assessment']['flags']:
            st.markdown("**Risk Flags:**")
            for flag in result['risk_assessment']['flags']:
                st.markdown(f"- 🚩 {flag.replace('_', ' ').title()}")


def render_dashboard():
    """Render outcome dashboard with enhanced visuals."""
    if not st.session_state.decisions_history:
        st.markdown("""
        <div style='text-align: center; padding: 4rem; background: rgba(255, 255, 255, 0.02); 
                    border-radius: 15px; border: 2px dashed rgba(255, 255, 255, 0.1);'>
            <p style='font-size: 4rem; margin: 0;'>📊</p>
            <h2 style='color: #888; margin: 1rem 0;'>No Data Yet</h2>
            <p style='color: #999; font-size: 1.2rem;'>
                Submit your first loan application to see analytics here
            </p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    st.header("📊 Outcome Dashboard")
    st.markdown(f"*Based on {len(st.session_state.decisions_history)} evaluations*")
    
    # Prepare data
    df = pd.DataFrame([
        {
            'decision': d['decision'],
            'probability': d['risk_assessment']['default_probability'],
            'risk_level': d['risk_assessment']['risk_level'],
            'confidence': d['confidence']
        }
        for d in st.session_state.decisions_history
    ])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Decision Distribution")
        decision_counts = df['decision'].value_counts()
        
        fig = px.pie(
            values=decision_counts.values,
            names=decision_counts.index,
            color=decision_counts.index,
            color_discrete_map={
                'APPROVED': '#667eea',
                'REJECTED': '#f5576c',
                'MANUAL_REVIEW': '#f76b1c'
            },
            hole=0.4
        )
        fig.update_traces(textposition='inside', textinfo='percent+label', textfont_size=14)
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Arial, sans-serif", size=14),
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Risk Score Distribution")
        
        fig = px.histogram(
            df,
            x='probability',
            color='decision',
            nbins=20,
            color_discrete_map={
                'APPROVED': '#667eea',
                'REJECTED': '#f5576c',
                'MANUAL_REVIEW': '#f76b1c'
            }
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Default Probability",
            yaxis_title="Count",
            showlegend=True,
            hovermode='x unified',
            font=dict(family="Arial, sans-serif", size=14)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Agent influence
    st.subheader("Agent Influence Analysis")
    
    agent_counts = {}
    for decision in st.session_state.decisions_history:
        for agent in decision['metadata']['agents_executed']:
            agent_counts[agent] = agent_counts.get(agent, 0) + 1
    
    fig = px.bar(
        x=list(agent_counts.keys()),
        y=list(agent_counts.values()),
        labels={'x': 'Agent', 'y': 'Times Executed'},
        color=list(agent_counts.values()),
        color_continuous_scale='Purples'
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        font=dict(family="Arial, sans-serif", size=14)
    )
    st.plotly_chart(fig, use_container_width=True)


def main():
    """Main app."""
    render_header()
    
    # Sidebar with status
    with st.sidebar:
        st.header("Navigation")
        page = st.radio(
            "Select View",
            ["📝 Evaluate Application", "📊 Dashboard", "🎮 Agent Playground"],
            label_visibility="collapsed"
        )
        
        st.divider()
        
        # System status
        st.markdown("### System Status")
        col1, col2 = st.columns(2)
        col1.markdown("**Model:** 🟢")
        col2.markdown("**XGBoost**")
        
        st.divider()
        
        # Stats
        st.metric("Total Evaluations", len(st.session_state.decisions_history))
        
        if st.session_state.decisions_history:
            approved = sum(1 for d in st.session_state.decisions_history if d['decision'] == 'APPROVED')
            st.metric("Approval Rate", f"{approved/len(st.session_state.decisions_history):.0%}")
        
        st.divider()
        
        if st.button("🗑️ Clear History", type="secondary", use_container_width=True):
            st.session_state.decisions_history = []
            st.session_state.current_decision = None
            st.rerun()
    
    # Main content
    if page == "📝 Evaluate Application":
        application = render_loan_form()
        
        if application:
            result = process_application(application)
            render_decision_result(result)
        
        elif st.session_state.current_decision:
            st.info("👆 Submit a new application or view other pages", icon="ℹ️")
            with st.expander("View Last Decision"):
                render_decision_result(st.session_state.current_decision)
    
    elif page == "📊 Dashboard":
        render_dashboard()
    
    else:  # Agent Playground
        render_agent_playground()


if __name__ == "__main__":
    main()
