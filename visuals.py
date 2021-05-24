# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd 
# sns.set(style="darkgrid")
# df = pd.read_csv('dataR2.csv')

# # Subset the iris dataset by species
# nocancer = df['Classification'] == 1
# cancer = df['Classification'] == 2
 

# f, ax = plt.subplots(figsize=(8, 8))
# ax.set_aspect("equal")

# # Draw the two density plots
# ax = sns.kdeplot(nocancer, shade=True, shade_lowest=False, legend='No Cancer')
# ax = sns.kdeplot(cancer, shade=True,  color='r', legend='Cancer')
# # ax = sns.kdeplot(virginica.sepal_width, virginica.sepal_length,
# #                  cmap="Blues", shade=True, shade_lowest=False)

# # Add labels to the plot
# red = sns.color_palette("Reds")[-2]
# blue = sns.color_palette("Blues")[-2]
# ax.text(2.5, 8.2, "No Cancer", size=16, color=blue)
# ax.text(3.8, 4.5, "Cancer", size=16, color=red)     
# plt.show()

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 
df = pd.read_csv('datasets/breast-cancer-wisconsin.csv')
sns.set(style="dark")
# rs = np.random.RandomState(50)
benign = df[' Class'] == 2
malignant = df[' Class'] == 4
sns.barplot(x=benign, y=df[' Class'], palette="dark")
plt.show()