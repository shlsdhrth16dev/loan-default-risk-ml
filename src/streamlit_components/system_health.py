"""
System Health Monitoring Component for Streamlit.

Visualizes:
- Circuit breaker states
- Model usage and fallbacks
- Performance metrics
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

def render_system_health(robust_service):
    """Render the system health dashboard."""
    st.header("🛡️ System Health & Reliability")
    
    st.markdown("""
    This dashboard monitors the production-grade reliability features of the system, 
    including the **Circuit Breaker** pattern and **Multi-Tier Fallback** logic.
    """)
    
    metrics = robust_service.get_health_metrics()
    
    # Tier 1: Circuit Breaker Status
    st.subheader("🛠️ Circuit Breakers")
    col1, col2 = st.columns(2)
    
    for i, (name, cb_metrics) in enumerate(metrics['circuit_breakers'].items()):
        with col1 if i == 0 else col2:
            state = cb_metrics['state']
            if state == "CLOSED":
                st.success(f"**{name.upper()}**: {state} (Healthy)")
            elif state == "OPEN":
                st.error(f"**{name.upper()}**: {state} (Failing)")
            else:
                st.warning(f"**{name.upper()}**: {state} (Testing)")
            
            # Detailed CB stats
            inner_col1, inner_col2 = st.columns(2)
            inner_col1.metric("Failures", cb_metrics['failure_count'])
            success_rate = (1 - cb_metrics['failure_rate']) * 100
            inner_col2.metric("Success Rate", f"{success_rate:.1f}%")
            
            st.caption(f"Last state change: {cb_metrics['last_state_change']}")
    
    st.divider()
    
    # Tier 2: Model Usage Distribution
    st.subheader("📊 Model Utilization & Fallbacks")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if metrics['model_usage']:
            usage_df = pd.DataFrame([
                {"Model": m, "Usage": c} for m, c in metrics['model_usage'].items()
            ])
            fig = px.bar(
                usage_df, x="Model", y="Usage", 
                title="Requests per Model",
                color="Model",
                color_discrete_map={
                    'xgboost': '#667eea',
                    'logistic_fallback': '#764ba2',
                    'emergency_rules': '#f5576c'
                }
            )
            fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No model usage recorded yet.")
            
    with col2:
        st.markdown("#### Availability")
        st.markdown(f"**XGBoost:** {'🟢 Available' if metrics['xgboost_available'] else '🔴 Unavailable'}")
        st.markdown(f"**Logistic:** {'🟢 Available' if metrics['logistic_available'] else '🔴 Unavailable'}")
        st.markdown("**Rules:** 🟢 Always Available")
        
        st.markdown("#### Fallback Events")
        if metrics['fallback_usage']:
            for event, count in metrics['fallback_usage'].items():
                st.markdown(f"- {event.replace('_', ' ')}: `{count}`")
        else:
            st.markdown("No fallback events recorded.")

    st.divider()
    
    # Tier 3: Reliability Architecture Visualization
    st.subheader("🏗️ Reliability Architecture")
    
    st.mermaid("""
    graph TD
        A[Incoming Request] --> B{XGBoost Circuit\nClosed?}
        B -- Yes --> C[XGBoost Prediction]
        C -- Success --> D[Return Result]
        C -- Failure --> E[Open Circuit\n& Log Failure]
        B -- No --> F{Logistic Circuit\nClosed?}
        E --> F
        F -- Yes --> G[Logistic Fallback]
        G -- Success --> H[Return Result\n(Fallback Label)]
        G -- Failure --> I[Open Circuit\n& Log Failure]
        F -- No --> J[Rule-Based\nEmergency Decision]
        I --> J
        J --> K[Return Result\n(Emergency Label)]
    """)
