"""
Comparison Lab Module for Streamlit App.

Allows side-by-side comparison of two discrete loan applications.
"""
import streamlit as st
import pandas as pd
from agents.loan_decision_crew import get_loan_decision_crew

def render_comparison_lab():
    """Render the side-by-side comparison lab."""
    st.header("🔬 Decision Comparison Lab")
    st.markdown("""
    Compare two different loan applications side-by-side to understand how the system 
    differentiates between risk profiles.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Application A")
        # Use a simpler form for comparison
        a_age = st.number_input("Age (A)", 18, 100, 35, key="a_age")
        a_income = st.number_input("Income (A)", 0, 500000, 55000, 1000, key="a_income")
        a_credit = st.slider("Credit Score (A)", 300, 850, 680, key="a_credit")
        a_loan = st.number_input("Loan Amount (A)", 1000, 100000, 25000, 1000, key="a_loan")
        a_dti = st.slider("DTI Ratio (A)", 0.0, 1.0, 0.35, 0.01, key="a_dti")
        
    with col2:
        st.subheader("📋 Application B")
        b_age = st.number_input("Age (B)", 18, 100, 35, key="b_age")
        b_income = st.number_input("Income (B)", 0, 500000, 45000, 1000, key="b_income")
        b_credit = st.slider("Credit Score (B)", 300, 850, 620, key="b_credit")
        b_loan = st.number_input("Loan Amount (B)", 1000, 100000, 35000, 1000, key="b_loan")
        b_dti = st.slider("DTI Ratio (B)", 0.0, 1.0, 0.45, 0.01, key="b_dti")
        
    if st.button("🔍 Compare Side-by-Side", type="primary", use_container_width=True):
        # We'll use defaults for other values for simplicity in comparison
        defaults = {
            "education": "Bachelor",
            "employmenttype": "Full-time",
            "maritalstatus": "Married",
            "hasmortgage": "No",
            "hasdependents": "No",
            "loanpurpose": "DebtConsolidation",
            "loanterm": 60,
            "numcreditlines": 5,
            "interestrate": 0.08,
            "hascosigner": "No",
            "monthsemployed": 48
        }
        
        app_a = {**defaults, "age": a_age, "income": a_income, "creditscore": a_credit, "loanamount": a_loan, "dtiratio": a_dti}
        app_b = {**defaults, "age": b_age, "income": b_income, "creditscore": b_credit, "loanamount": b_loan, "dtiratio": b_dti}
        
        with st.spinner("Analyzing both applications..."):
            crew = get_loan_decision_crew()
            res_a = crew.process(app_a)
            res_b = crew.process(app_b)
            
        st.divider()
        
        # Results comparison
        c1, c2, c3 = st.columns([2, 1, 2])
        
        with c1:
            st.markdown(f"<div style='text-align: center'><h3>Application A</h3></div>", unsafe_allow_html=True)
            render_mini_result(res_a)
            
        with c2:
            st.markdown("<div style='text-align: center; padding-top: 100px'><h1>VS</h1></div>", unsafe_allow_html=True)
            
        with c3:
            st.markdown(f"<div style='text-align: center'><h3>Application B</h3></div>", unsafe_allow_html=True)
            render_mini_result(res_b)
            
        st.divider()
        
        # Key differences
        st.subheader("🎯 Why the Difference?")
        diff_prob = res_b['risk_assessment']['default_probability'] - res_a['risk_assessment']['default_probability']
        
        if res_a['decision'] == res_b['decision']:
            st.info(f"Both applications were **{res_a['decision']}**. Application B has a **{abs(diff_prob):.1%} {'higher' if diff_prob > 0 else 'lower'}** predicted risk than Application A.")
        else:
            st.warning(f"Decisions differ! Application A: **{res_a['decision']}** vs Application B: **{res_b['decision']}**. Risk difference: **{diff_prob:+.1%}**.")

def render_mini_result(result):
    """Render a compact version of the result for side-by-side comparison."""
    decision = result['decision']
    prob = result['risk_assessment']['default_probability']
    risk_level = result['risk_assessment']['risk_level']
    
    if decision == "APPROVED": color = "#667eea"
    elif decision == "REJECTED": color = "#f5576c"
    else: color = "#f76b1c"
    
    st.markdown(f"""
    <div style='background: {color}; padding: 1.5rem; border-radius: 10px; text-align: center; color: white; margin-bottom: 1rem;'>
        <h2 style='margin: 0;'>{decision}</h2>
        <p style='margin: 0; opacity: 0.9;'>Default Risk: {prob:.1%}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.metric("Risk Level", risk_level.replace('_', ' ').title())
    st.metric("Confidence", f"{result['confidence']:.0%}")
    
    with st.expander("Show Explanation"):
        st.caption(result['explanations']['borrower_explanation'])
