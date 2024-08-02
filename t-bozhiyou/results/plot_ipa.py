import matplotlib.pyplot as plt
import pandas as pd

# Load data from CSV file
data = pd.read_csv('ipa.csv')

# Extract data for plotting
batch_size = data['batch_size']
pytorch_fw = data['PyTorch(FW)']
ds_fw = data['DS(FW)']
tk_fw = data['TK(FW)']

print(sum(b/a for a, b, in zip(tk_fw, pytorch_fw))/len(tk_fw))
print(sum(b/a for a, b, in zip(tk_fw, ds_fw))/len(tk_fw))

# Plotting the lines and points
plt.plot(batch_size, pytorch_fw, marker='o', label='PyTorch')
plt.plot(batch_size, ds_fw, marker='o', label='DeepSpeed')
plt.plot(batch_size, tk_fw, marker='o', label='ThunderKittens')

# Adding labels and title
plt.xlabel('Batch Size')
plt.ylabel('Time (s)')
plt.title('RA forward')
plt.grid(True)

# Adding a legend
plt.legend(loc='best')

# Save the plot to a PNG file
plt.savefig('ipa.png')

# Optionally display the plot
# plt.show()
