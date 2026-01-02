import matplotlib.pyplot as plt

def plot_single_prediction(default_prob):
    """
    Plots the probability of default vs no-default for a single prediction.
    
    Args:
        default_prob (float): The probability of default (between 0 and 1).
    """
    labels = ["No Default", "Default"]
    values = [1 - default_prob, default_prob]
    
    # Set colors: Green for No Default, Red for Default
    colors = ['green', 'red']

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values, color=colors, alpha=0.7)
    plt.ylim(0, 1)
    plt.ylabel("Probability")
    plt.title("Loan Default Prediction Risk")
    
    # Add text labels on top of bars
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.2%}", ha='center', fontweight='bold')
        
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.show()

if __name__ == "__main__":
    # Test the function with a dummy value
    plot_single_prediction(0.65)
