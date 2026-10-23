import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import plotly.graph_objects as go  # For better 3D plots

# Use uploaded file
file_path = "C:/Users/SANJAY/Documents/vest train/train/ecommerce_sales_analysis.csv"
df = pd.read_csv(file_path)

# Optional: Preview dataset
print(df.head())

# Group by Month and calculate average rainfall

print(df.columns.tolist())

# Clean them
df.columns = df.columns.str.strip()
line_kwargs = {'color': 'green', 'linestyle': ':'}

# 1. Line Plot
plt.figure()
plt.plot(x, y, label="Avg Rainfall", **line_kwargs)
plt.xlabel("Month Number")
plt.ylabel("Rainfall")
plt.title("Line Plot")
plt.legend(loc="upper left")
plt.show()

# 2. Bar Plot
plt.figure()
plt.bar(x, y, color='green', label="Avg Rainfall")
plt.xlabel("Month Number")
plt.ylabel("Rainfall")
plt.title("Bar Plot")
plt.legend(loc="upper left")
plt.show()

# 3. Histogram
plt.figure()
plt.hist(y, color='green', edgecolor='black', label="Rainfall Histogram")
plt.xlabel("Rainfall")
plt.title("Histogram")
plt.legend(loc="upper left")
plt.show()

# 4. Density Plot
plt.figure()
sns.kdeplot(y, color='green', label="Rainfall Density")
plt.title("Density Plot")
plt.xlabel("Rainfall")
plt.legend(loc="upper left")
plt.show()

# 5. Scatter Plot
plt.figure()
plt.scatter(x, y, color='green', label="Avg Rainfall")
plt.xlabel("Month Number")
plt.ylabel("Rainfall")
plt.title("Scatter Plot")
plt.legend(loc="upper left")
plt.show()

# 6. Area Plot
plt.figure()
plt.fill_between(x, y, color='green', alpha=0.5, label="Avg Rainfall")
plt.plot(x, y, **line_kwargs)
plt.xlabel("Month Number")
plt.ylabel("Rainfall")
plt.title("Area Plot")
plt.legend(loc="upper left")
plt.show()

# 7. Pie Plot
plt.figure()
plt.pie(y, labels=x, autopct='%1.1f%%', colors=['green']*len(x))
plt.title("Pie Chart (Rainfall by Month)")
plt.legend(title="Month", loc="upper left", bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.show()

# 8. 3D Plot using Plotly
fig = go.Figure(data=[go.Scatter3d(
    x=x,
    y=y,
    z=y,
    mode='lines+markers',
    marker=dict(size=5),
    line=dict(color='green', width=4, dash='dot')
)])
fig.update_layout(title='3D Plot',
                  scene=dict(
                      xaxis_title='Month',
                      yaxis_title='Rainfall',
                      zaxis_title='Rainfall (Z)'
                  ))
fig.show()

# 9. Box Plot
plt.figure()
sns.boxplot(y=y, color='green')
plt.title("Box Plot")
plt.ylabel("Rainfall")
plt.show()

# 10. Violin Plot
plt.figure()
sns.violinplot(y=y, color='green')
plt.title("Violin Plot")
plt.ylabel("Rainfall")
plt.show()

# 11. Subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

axs[0, 0].plot(x, y, **line_kwargs)
axs[0, 0].set_title("Line Plot")

axs[0, 1].bar(x, y, color='green')
axs[0, 1].set_title("Bar Plot")

axs[1, 0].scatter(x, y, color='green')
axs[1, 0].set_title("Scatter Plot")

axs[1, 1].fill_between(x, y, color='green', alpha=0.5)
axs[1, 1].plot(x, y, **line_kwargs)
axs[1, 1].set_title("Area Plot")

for ax in axs.flat:
    ax.set(xlabel='Month', ylabel='Rainfall')
    ax.legend(["Rainfall"], loc="upper left")

plt.tight_layout()
plt.show()
