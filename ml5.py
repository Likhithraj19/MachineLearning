import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('train.csv')
df.head(2)
#plotting pairchart
sns.pairplot(df,hue='Survived')
sns.set_theme(style="darkgrid")
plt.show()
sns.countplot(x='Survived',data=df,hue = 'Sex')
