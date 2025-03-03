import matplotlib.pyplot as plt
import seaborn as sns

class PlotGenerator:
    @staticmethod
    def plot_metric_trend(metrics_history: Dict[str, list], output_path: str):
        plt.figure(figsize=(12, 6))
        for metric, values in metrics_history.items():
            plt.plot(values, label=metric)
        plt.xlabel('Evaluation Iteration')
        plt.ylabel('Metric Value')
        plt.title('Metric Trends Over Time')
        plt.legend()
        plt.savefig(output_path)
        plt.close()
        
    @staticmethod
    def plot_heatmap(similarity_matrix: np.ndarray, item_labels: list, output_path: str):
        plt.figure(figsize=(15, 12))
        sns.heatmap(similarity_matrix, xticklabels=item_labels, yticklabels=item_labels)
        plt.title('Item Similarity Heatmap')
        plt.savefig(output_path)
        plt.close()