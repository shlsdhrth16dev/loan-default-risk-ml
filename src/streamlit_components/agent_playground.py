"""
Agent Playground Module for Streamlit App.

Features:
- Agent conversation viewer
- What-if analysis
- Agent battle mode
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from copy import deepcopy


def render_agent_conversation(result):
    """
    Render agent conversation as timeline.
    
    Shows step-by-step agent reasoning.
    """
    st.subheader("🤖 Agent Conversation Timeline")
    
    agents = result['metadata']['agents_executed']
    
    # Timeline visualization
    for i, agent in enumerate(agents):
        with st.container():
            col1, col2 = st.columns([1, 4])
            
            with col1:
                # Agent icon and name
                icons = {
                    'ValidationAgent': '🔍',
                    'RiskAnalystAgent': '📊',
                    'PolicyAgent': '⚖️',
                    'ExplanationAgent': '📝'
                }
                icon = icons.get(agent, '🤖')
                st.markdown(f"### {icon}")
                st.caption(f"**{agent.replace('Agent', '')}**")
            
            with col2:
                # Agent output
                if agent == 'ValidationAgent':
                    st.success("✓ Application validated - all fields present and valid")
                
                elif agent == 'RiskAnalystAgent':
                    prob = result['risk_assessment']['default_probability']
                    risk = result['risk_assessment']['risk_level']
                    st.info(f"""
                    **Analysis Complete**
                    - Default Probability: {prob:.1%}
                    - Risk Classification: {risk.replace('_', ' ').title()}
                    - Category: {result['risk_assessment'].get('category', 'N/A')}
                    """)
                    
                    if result['risk_assessment']['flags']:
                        with st.expander("Risk Flags Identified"):
                            for flag in result['risk_assessment']['flags']:
                                st.markdown(f"- 🚩 {flag.replace('_', ' ').title()}")
                
                elif agent == 'PolicyAgent':
                    if result.get('policy_evaluation'):
                        policy = result['policy_evaluation']
                        
                        if policy.get('override_applied'):
                            st.warning(f"""
                            **Policy Override Applied**
                            - Type: {policy.get('override_type', 'N/A')}
                            - Reason: Business rules triggered
                            """)
                        else:
                            st.info("✓ No policy overrides needed")
                        
                        if policy.get('soft_flags'):
                            with st.expander(f"{len(policy['soft_flags'])} Policy Flags"):
                                for flag in policy['soft_flags']:
                                    st.markdown(f"- ⚠️ {flag.get('description', flag.get('rule', 'Unknown'))}")
                    else:
                        st.info("✓ Risk level clear - policy check skipped")
                
                elif agent == 'ExplanationAgent':
                    st.success("✓ Generated explanations for all audiences")
                    st.caption("Borrower • Loan Officer • Auditor formats created")
            
            if i < len(agents) - 1:
                st.markdown("↓")


def render_what_if_analysis(original_application, original_result):
    """
    Render what-if analysis interface.
    
    Let users tweak parameters and see decision changes.
    """
    st.subheader("🔮 What-If Analysis")
    st.markdown("*Change one parameter and see how the decision changes*")
    
    # Select parameter to change
    parameter = st.selectbox(
        "Select Parameter to Modify",
        [
            "Income",
            "Credit Score",
            "Loan Amount",
            "DTI Ratio",
            "Employment Duration",
            "Has Cosigner"
        ]
    )
    
    # Create modified application
    modified_app = deepcopy(original_application)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Original Value")
        
        if parameter == "Income":
            original_val = original_application['income']
            st.metric("Annual Income", f"${original_val:,}")
            
        elif parameter == "Credit Score":
            original_val = original_application['creditscore']
            st.metric("Credit Score", original_val)
            
        elif parameter == "Loan Amount":
            original_val = original_application['loanamount']
            st.metric("Loan Amount", f"${original_val:,}")
            
        elif parameter == "DTI Ratio":
            original_val = original_application['dtiratio']
            st.metric("DTI Ratio", f"{original_val:.1%}")
            
        elif parameter == "Employment Duration":
            original_val = original_application['monthsemployed']
            st.metric("Months Employed", original_val)
            
        elif parameter == "Has Cosigner":
            original_val = original_application['hascosigner']
            st.metric("Has Cosigner", original_val)
    
    with col2:
        st.markdown("### Modified Value")
        
        if parameter == "Income":
            new_val = st.number_input(
                "New Income",
                min_value=0,
                max_value=500000,
                value=int(original_val * 1.2),
                step=1000
            )
            modified_app['income'] = new_val
            
        elif parameter == "Credit Score":
            new_val = st.slider(
                "New Credit Score",
                min_value=300,
                max_value=850,
                value=min(850, int(original_val + 50))
            )
            modified_app['creditscore'] = new_val
            
        elif parameter == "Loan Amount":
            new_val = st.number_input(
                "New Loan Amount",
                min_value=1000,
                max_value=100000,
                value=int(original_val * 0.8),
                step=1000
            )
            modified_app['loanamount'] = new_val
            
        elif parameter == "DTI Ratio":
            new_val = st.slider(
                "New DTI Ratio",
                min_value=0.0,
                max_value=1.0,
                value=max(0.0, original_val - 0.1),
                step=0.01
            )
            modified_app['dtiratio'] = new_val
            
        elif parameter == "Employment Duration":
            new_val = st.number_input(
                "New Months Employed",
                min_value=0,
                max_value=600,
                value=int(original_val + 24)
            )
            modified_app['monthsemployed'] = new_val
            
        elif parameter == "Has Cosigner":
            new_val = st.selectbox(
                "New Cosigner Status",
                ["Yes", "No"],
                index=0 if original_val == "No" else 1
            )
            modified_app['hascosigner'] = new_val
    
    # Analyze button
    if st.button("🔄 Re-evaluate with New Value", type="primary", use_container_width=True):
        from agents.loan_decision_crew import get_loan_decision_crew
        
        with st.spinner("Re-running AI agents..."):
            crew = get_loan_decision_crew()
            new_result = crew.process(modified_app)
        
        # Comparison
        st.divider()
        st.subheader("📊 Decision Comparison")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### Original")
            st.metric("Decision", original_result['decision'])
            st.metric("Probability", f"{original_result['risk_assessment']['default_probability']:.1%}")
        
        with col2:
            st.markdown("### Modified")
            st.metric("Decision", new_result['decision'])
            st.metric("Probability", f"{new_result['risk_assessment']['default_probability']:.1%}")
        
        with col3:
            st.markdown("### Change")
            
            # Decision change
            decision_changed = original_result['decision'] != new_result['decision']
            if decision_changed:
                st.metric(
                    "Decision",
                    "CHANGED",
                    delta="Different outcome",
                    delta_color="inverse" if new_result['decision'] == 'APPROVED' else "normal"
                )
            else:
                st.metric("Decision", "SAME", delta="No change")
            
            # Probability change
            prob_diff = new_result['risk_assessment']['default_probability'] - original_result['risk_assessment']['default_probability']
            st.metric(
                "Probability",
                f"{prob_diff:+.1%}",
                delta="Risk decreased" if prob_diff < 0 else "Risk increased" if prob_diff > 0 else "No change"
            )
        
        # Explanation
        st.divider()
        st.subheader("💡 Why Did It Change?")
        
        if decision_changed:
            st.success(f"""
            **Decision changed from {original_result['decision']} to {new_result['decision']}**
            
            By changing {parameter}, you affected the risk assessment:
            - Original risk: {original_result['risk_assessment']['default_probability']:.1%}
            - New risk: {new_result['risk_assessment']['default_probability']:.1%}
            - Change: {prob_diff:+.1%}
            
            This crossed the decision threshold, resulting in a different outcome.
            """)
        else:
            if abs(prob_diff) > 0.01:
                st.info(f"""
                **Decision stayed the same ({original_result['decision']})**
                
                While the risk changed by {prob_diff:+.1%}, it wasn't enough to cross the decision threshold.
                The new probability ({new_result['risk_assessment']['default_probability']:.1%}) is still in the same risk category.
                """)
            else:
                st.info(f"""
                **Minimal impact on decision**
                
                Changing {parameter} had very little effect on the risk assessment.
                This suggests other factors are more influential for this application.
                """)


def render_agent_battle_mode(application):
    """
    Compare decisions with/without specific agents.
    
    Shows agent contribution.
    """
    st.subheader("⚔️ Agent Battle Mode")
    st.markdown("*See how decisions change when specific agents are enabled/disabled*")
    
    st.info("""
    **How it works:**
    - **Full System**: All 5 agents collaborate
    - **Model Only**: Just the ML model prediction (no policy/validation)
    - **Comparison**: See the difference agents make
    """)
    
    if st.button("🚀 Run Comparison", type="primary", use_container_width=True):
        from agents.loan_decision_crew import get_loan_decision_crew
        from inference.robust_predict import get_robust_prediction_service
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🤖 Full Agent System")
            with st.spinner("Running full crew..."):
                crew = get_loan_decision_crew()
                full_result = crew.process(application)
            
            st.success(f"**Decision:** {full_result['decision']}")
            st.metric("Default Risk", f"{full_result['risk_assessment']['default_probability']:.1%}")
            st.metric("Agents Used", len(full_result['metadata']['agents_executed']))
            
            if full_result.get('policy_evaluation', {}).get('override_applied'):
                st.warning("✓ Policy override applied")
            
            with st.expander("See Full Reasoning"):
                st.text(full_result['explanations']['loan_officer_explanation'])
        
        with col2:
            st.markdown("### 📊 Model Only")
            with st.spinner("Running model..."):
                pred_service = get_robust_prediction_service()
                model_result = pred_service.predict(application)
            
            decision = "APPROVED" if model_result['prediction'] == 'approved' else "REJECTED"
            st.info(f"**Decision:** {decision}")
            st.metric("Default Risk", f"{model_result['default_probability']:.1%}")
            st.metric("Agents Used", 0)
            st.caption("Direct ML prediction only")
            
            with st.expander("Model Details"):
                st.json(model_result)
        
        # Comparison
        st.divider()
        st.subheader("⚖️ Impact Analysis")
        
        decisions_match = full_result['decision'] == decision
        
        if not decisions_match:
            st.warning(f"""
            **🎯 Agents Changed the Decision!**
            
            - **Model Only:** {decision}
            - **With Agents:** {full_result['decision']}
            
            **Why the difference?**
            
            The agent system didn't just use the model prediction - it also:
            - Validated the application data
            - Applied business policy rules
            - Checked for edge cases
            - Considered compliance requirements
            
            This shows the value of the multi-agent approach versus raw ML predictions!
            """)
        else:
            st.success(f"""
            **✓ Decisions Aligned**
            
            Both approaches reached the same conclusion: **{decision}**
            
            However, the agent system provides:
            - Better explainability
            - Policy compliance checks
            - Audit trail
            - Multi-audience explanations
            """)


def render_agent_playground():
    """Main agent playground interface."""
    st.header("🎮 Agent Playground")
    
    st.markdown("""
    Explore how our multi-agent AI system makes decisions.
    
    **Features:**
    - 🤖 **Conversation Viewer**: See step-by-step agent reasoning
    - 🔮 **What-If Analysis**: Change parameters and see impacts
    - ⚔️ **Battle Mode**: Compare agents vs raw ML model
    """)
    
    # Need a decision to analyze
    if not st.session_state.current_decision:
        st.info("👆 First, submit a loan application in the 'Evaluate Application' page to unlock the playground!")
        return
    
    result = st.session_state.current_decision
    application = result['application']
    
    # Tabs for different features
    tab1, tab2, tab3 = st.tabs(["🤖 Agent Conversation", "🔮 What-If Analysis", "⚔️ Battle Mode"])
    
    with tab1:
        render_agent_conversation(result)
    
    with tab2:
        render_what_if_analysis(application, result)
    
    with tab3:
        render_agent_battle_mode(application)
