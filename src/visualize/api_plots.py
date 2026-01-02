import matplotlib.pyplot as plt
import io
import base64

def create_risk_chart_buffer(default_prob):
    """
    Generates a risk prediction chart and returns it as a bytes buffer.
    Suitable for serving via FastAPI (e.g., StreamingResponse).
    
    Args:
        default_prob (float): Probability of default.
        
    Returns:
        io.BytesIO: Buffer containing the PNG image.
    """
    labels = ["No Default", "Default"]
    values = [1 - default_prob, default_prob]
    colors = ['green', 'red']

    # Use the Agg backend to prevent showing GUI windows on the server
    current_backend = plt.get_backend()
    plt.switch_backend('Agg')

    try:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(labels, values, color=colors, alpha=0.7)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        ax.set_title("Loan Default Prediction Risk")
        
        for i, v in enumerate(values):
            ax.text(i, v + 0.02, f"{v:.2%}", ha='center', fontweight='bold')
            
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        return buf
    finally:
        # Switch back if needed, though usually Agg is fine for scripts
        plt.switch_backend(current_backend)

def create_risk_chart_base64(default_prob):
    """
    Returns base64 encoded string of the chart.
    Useful for embedding in HTML or JSON.
    """
    buf = create_risk_chart_buffer(default_prob)
    return base64.b64encode(buf.getvalue()).decode('utf-8')
