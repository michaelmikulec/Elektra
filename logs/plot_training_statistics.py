import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('logs/cnn2_training_stats.csv')
df = df.iloc[:50,:]

fig, ax = plt.subplots()
columns = ['training_loss', 'validation_loss', 'validation_accuracy']
for column in columns:
  ax.plot(df.index, df[column], label=column)
ax.legend()
ax.set_title('CNN Model 0.2 Training Statistics')
ax.set_xlabel('Epoch')
ax.set_ylabel('Value')
plt.show()


df = pd.read_csv('logs/t5_training_stats.csv')
df = df.iloc[:10,:]

fig, ax = plt.subplots()
columns = ['training_loss', 'validation_loss', 'validation_accuracy']
for column in columns:
  ax.plot(df.index, df[column], label=column)
ax.legend()
ax.set_title('Transformer Model 0.5 Training Statistics')
ax.set_xlabel('Epoch')
ax.set_ylabel('Value')
plt.show()