import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load data from CSV file
data = pd.read_csv('flashattention.csv')

# Extract data for plotting
impls = data['impls']
fw = data['fw']
bw = data['bw']
both = data['both']

# Number of groups
num_groups = len(impls)

# Creating a numpy array for the positions of each group
indices = np.arange(num_groups)

# Width of the bars
bar_width = 0.2

# Plotting the bars
plt.bar(indices, fw, bar_width, label='forward')
plt.bar(indices + bar_width, bw, bar_width, label='backward')
plt.bar(indices + 2 * bar_width, both, bar_width, label='both')

# Adding the labels
plt.xlabel('Attention Implementation')
plt.ylabel('Performance (TFLOPS)')
# plt.title('Performance Comparison by Implementation')
plt.xticks(indices + bar_width, impls)

# Adding a legend
plt.legend(loc='upper right')

# Save the plot to a PNG file
plt.savefig('flashattention.png')

# Optionally display the plot
# plt.show()
