from wordcloud import WordCloud
import matplotlib.pyplot as plt

cloud = WordCloud(width=800, height=600).generate(" ".join(train_set.astype(str)))
plt.figure(figsize=20,15)
plt.imshow(cloud)
plt.axis('off')
