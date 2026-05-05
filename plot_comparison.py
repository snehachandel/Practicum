import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC Score']
    lgbm_scores = [0.8800, 0.8387, 0.8102, 0.8217, 0.9333]
    dt_scores = [0.6450, 0.5324, 0.5892, 0.5503, 0.7432]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, lgbm_scores, width, label='LightGBM (Tuned)', color='#1f77b4')
    rects2 = ax.bar(x + width/2, dt_scores, width, label='Decision Tree (Baseline)', color='#ff7f0e')

    ax.set_ylabel('Scores')
    ax.set_title('Performance Comparison: LightGBM vs Decision Tree')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.1)

    # Add text labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_path = os.path.join(script_dir, 'model_comparison_bar_chart.png')
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

if __name__ == '__main__':
    main()
