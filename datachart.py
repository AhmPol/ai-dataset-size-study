import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import os
import numpy as np # type: ignore

# Load the cleaned CSV
df = pd.read_csv("result.csv")

# Prepare data structure
models = df['Model'].unique()
trials = 8
results = {model: {'accuracy': [], 'avg_time': []} for model in models}

# Loop through each model and trial
for model in models:
    model_df = df[df['Model'] == model]
    for i in range(trials):
        correct_col = f'Correct.{i}' if i > 0 else 'Correct'
        time_col = f'Prediction Time (s).{i}' if i > 0 else 'Prediction Time (s)'

        correct_series = model_df[correct_col].dropna()
        time_series = model_df[time_col].dropna()

        # Calculate accuracy and average prediction time
        total = len(correct_series)
        correct = (correct_series == 'Yes').sum()
        accuracy = (correct / total) * 100 if total > 0 else 0
        avg_time = time_series.mean()

        results[model]['accuracy'].append(accuracy)
        results[model]['avg_time'].append(avg_time)

# Convert to DataFrame for visualization
summary_df = pd.DataFrame({
    'Model': [model for model in models for _ in range(trials)],
    'Trial': [f'Trial {i+1}' for _ in models for i in range(trials)],
    'Accuracy (%)': [acc for model in models for acc in results[model]['accuracy']],
    'Avg Prediction Time (s)': [time for model in models for time in results[model]['avg_time']]
})

# Create charts folder if not exists
os.makedirs("charts", exist_ok=True)

# Plot accuracy bar chart
plt.figure(figsize=(10, 6))
sns.barplot(data=summary_df, x='Trial', y='Accuracy (%)', hue='Model')
plt.title('Model Accuracy per Trial')
plt.ylim(0, 100)
plt.savefig("charts/accuracy_bar_chart.png")
plt.close()

# Plot prediction time bar chart
plt.figure(figsize=(10, 6))
sns.barplot(data=summary_df, x='Trial', y='Avg Prediction Time (s)', hue='Model')
plt.title('Average Prediction Time per Trial')
plt.savefig("charts/prediction_time_bar_chart.png")
plt.close()

# Box Plot for Accuracy across Trials
plt.figure(figsize=(10, 6))
sns.boxplot(data=summary_df, x='Model', y='Accuracy (%)')
plt.title('Accuracy Distribution per Model')
plt.ylim(0, 100)
plt.savefig("charts/accuracy_box_plot.png")
plt.close()

# Box Plot for Prediction Time across Trials
plt.figure(figsize=(10, 6))
sns.boxplot(data=summary_df, x='Model', y='Avg Prediction Time (s)')
plt.title('Prediction Time Distribution per Model')
plt.savefig("charts/prediction_time_box_plot.png")
plt.close()

# Line Plot for Accuracy across Trials
plt.figure(figsize=(10, 6))
for model in models:
    trial_accuracy = results[model]['accuracy']
    plt.plot(range(1, trials + 1), trial_accuracy, label=model, marker='o')
plt.title('Accuracy Trend per Model')
plt.xlabel('Trial Number')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.ylim(0, 100)
plt.savefig("charts/accuracy_line_plot.png")
plt.close()

# Line Plot for Prediction Time across Trials
plt.figure(figsize=(10, 6))
for model in models:
    trial_time = results[model]['avg_time']
    plt.plot(range(1, trials + 1), trial_time, label=model, marker='o')
plt.title('Prediction Time Trend per Model')
plt.xlabel('Trial Number')
plt.ylabel('Average Prediction Time (s)')
plt.legend()
plt.savefig("charts/prediction_time_line_plot.png")
plt.close()

# Scatter Plot for Accuracy vs Prediction Time
plt.figure(figsize=(10, 6))
for model in models:
    accuracy_values = results[model]['accuracy']
    time_values = results[model]['avg_time']
    plt.scatter(accuracy_values, time_values, label=model, marker='x')
plt.title('Accuracy vs Prediction Time')
plt.xlabel('Accuracy (%)')
plt.ylabel('Average Prediction Time (s)')
plt.legend()
plt.savefig("charts/accuracy_vs_prediction_time.png")
plt.close()

# Heatmap of Accuracy across Models and Trials
accuracy_matrix = np.array([results[model]['accuracy'] for model in models])
plt.figure(figsize=(10, 6))
sns.heatmap(accuracy_matrix, annot=True, xticklabels=[f'Trial {i+1}' for i in range(trials)], 
            yticklabels=models, cmap='coolwarm', cbar_kws={'label': 'Accuracy (%)'})
plt.title('Heatmap of Accuracy per Model and Trial')
plt.savefig("charts/accuracy_heatmap.png")
plt.close()

# Heatmap of Prediction Time across Models and Trials
time_matrix = np.array([results[model]['avg_time'] for model in models])
plt.figure(figsize=(10, 6))
sns.heatmap(time_matrix, annot=True, xticklabels=[f'Trial {i+1}' for i in range(trials)], 
            yticklabels=models, cmap='coolwarm', cbar_kws={'label': 'Avg Time (s)'})
plt.title('Heatmap of Prediction Time per Model and Trial')
plt.savefig("charts/prediction_time_heatmap.png")
plt.close()

# Radar Chart for Accuracy per Model
angles = np.linspace(0, 2 * np.pi, trials, endpoint=False).tolist()
angles += angles[:1]
plt.figure(figsize=(10, 6))
for model in models:
    accuracy_values = results[model]['accuracy'] + [results[model]['accuracy'][0]]
    plt.polar(angles, accuracy_values, label=model)
plt.title('Radar Chart of Accuracy per Model')
plt.legend()
plt.savefig("charts/accuracy_radar_chart.png")
plt.close()

# Radar Chart for Prediction Time per Model
plt.figure(figsize=(10, 6))
for model in models:
    time_values = results[model]['avg_time'] + [results[model]['avg_time'][0]]
    plt.polar(angles, time_values, label=model)
plt.title('Radar Chart of Prediction Time per Model')
plt.legend()
plt.savefig("charts/prediction_time_radar_chart.png")
plt.close()

# Violin Plot for Accuracy Distribution
plt.figure(figsize=(10, 6))
sns.violinplot(data=summary_df, x='Model', y='Accuracy (%)')
plt.title('Violin Plot of Accuracy Distribution per Model')
plt.ylim(0, 100)
plt.savefig("charts/accuracy_violin_plot.png")
plt.close()

# Violin Plot for Prediction Time Distribution
plt.figure(figsize=(10, 6))
sns.violinplot(data=summary_df, x='Model', y='Avg Prediction Time (s)')
plt.title('Violin Plot of Prediction Time Distribution per Model')
plt.savefig("charts/prediction_time_violin_plot.png")
plt.close()

# Bar Chart of Average Accuracy per Model
average_accuracy = {model: np.mean(results[model]['accuracy']) for model in models}
plt.figure(figsize=(10, 6))
sns.barplot(x=list(average_accuracy.keys()), y=list(average_accuracy.values()))
plt.title('Average Accuracy per Model')
plt.ylabel('Average Accuracy (%)')
plt.ylim(0, 100)
plt.savefig("charts/average_accuracy_bar_chart.png")
plt.close()

# Bar Chart of Average Prediction Time per Model
average_time = {model: np.mean(results[model]['avg_time']) for model in models}
plt.figure(figsize=(10, 6))
sns.barplot(x=list(average_time.keys()), y=list(average_time.values()))
plt.title('Average Prediction Time per Model')
plt.ylabel('Average Prediction Time (s)')
plt.savefig("charts/average_prediction_time_bar_chart.png")
plt.close()

pivot_acc = summary_df.pivot(index='Trial', columns='Model', values='Accuracy (%)')
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_acc, annot=True, cmap="YlGnBu", fmt=".1f")
plt.title("Heatmap of Accuracy Across Trials")
plt.savefig("charts/accuracy_heatmap.png")
plt.close()

sns.pairplot(summary_df, hue='Model')
plt.suptitle("Pairplot of Model Metrics", y=1.02)
plt.savefig("charts/pairplot_summary.png")
plt.close()

plt.figure(figsize=(10, 6))
for model in models:
    acc_series = pd.Series(results[model]['accuracy']).rolling(window=3, min_periods=1).mean()
    sns.lineplot(x=range(1, trials + 1), y=acc_series, label=model)
plt.title("Rolling Average Accuracy (Window = 3)")
plt.xlabel("Trial")
plt.ylabel("Rolling Accuracy (%)")
plt.savefig(f"charts/rolling_avg_accuracy.png")
plt.close()

acc_extremes = summary_df.groupby('Model')['Accuracy (%)'].agg(['min', 'max']).reset_index()
acc_extremes_melt = acc_extremes.melt(id_vars='Model', value_vars=['min', 'max'], var_name='Type', value_name='Accuracy (%)')
plt.figure(figsize=(10, 6))
sns.barplot(data=acc_extremes_melt, x='Model', y='Accuracy (%)', hue='Type')
plt.title("Max and Min Accuracy per Model")
plt.savefig("charts/max_min_accuracy.png")
plt.close()

plt.figure(figsize=(10, 6))
for model in models:
    sns.histplot(summary_df[summary_df['Model'] == model]['Accuracy (%)'], kde=True, label=model, bins=10)
plt.title('Accuracy Histogram per Model')
plt.xlabel('Accuracy (%)')
plt.ylabel('Frequency')
plt.legend()
plt.savefig("charts/accuracy_histogram.png")
plt.close()

plt.figure(figsize=(10, 6))
for model in models:
    sns.histplot(summary_df[summary_df['Model'] == model]['Avg Prediction Time (s)'], kde=True, label=model, bins=10)
plt.title('Prediction Time Histogram per Model')
plt.xlabel('Time (s)')
plt.ylabel('Frequency')
plt.legend()
plt.savefig("charts/prediction_time_histogram.png")
plt.close()

acc_range = summary_df.groupby('Model')['Accuracy (%)'].agg(lambda x: x.max() - x.min()).reset_index()
acc_range.columns = ['Model', 'Accuracy Range']
plt.figure(figsize=(10, 6))
sns.barplot(data=acc_range, x='Model', y='Accuracy Range')
plt.title('Accuracy Range (Max - Min) per Model')
plt.savefig("charts/accuracy_range_bar_chart.png")
plt.close()

best_trial = summary_df.groupby('Model')['Accuracy (%)'].max().reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(data=best_trial, x='Model', y='Accuracy (%)')
plt.title('Best Trial Accuracy per Model')
plt.savefig("charts/best_trial_accuracy.png")
plt.close()

pivot_norm = summary_df.pivot(index='Trial', columns='Model', values='Accuracy (%)')
pivot_norm = pivot_norm.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_norm, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Normalized Accuracy Heatmap Across Trials")
plt.savefig("charts/normalized_accuracy_heatmap.png")
plt.close()


for model in models:
    subset = summary_df[summary_df['Model'] == model]
    g = sns.jointplot(data=subset, x='Avg Prediction Time (s)', y='Accuracy (%)', kind='reg')
    g.fig.suptitle(f'Accuracy vs Prediction Time – {model}', y=1.02)
    g.savefig("charts/jointplot_accuracy_vs_time_{model}.png")
    plt.close()

# Print results
print("=== Analysis Summary ===")
for model in models:
    print(f"\nModel: {model}")
    for i in range(trials):
        print(f"  Trial {i+1}: Accuracy = {results[model]['accuracy'][i]:.2f}%, "
              f"Avg Time = {results[model]['avg_time'][i]:.4f}s")
print("\nCharts saved in 'charts' folder.")
