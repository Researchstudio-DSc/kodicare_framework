
import gensim
import os

model_dir = "models/trec_covid_topic_modelling"

#distance = "kullback_leibler"
distance = "hellinger"
#distance = "jaccard"
#distance = "jensen_shannon"

model_1_path = os.path.join(model_dir, "lda1")
model_2_path = os.path.join(model_dir, "lda2")

model_1 = gensim.models.LdaMulticore.load(model_1_path)
model_2 = gensim.models.LdaMulticore.load(model_2_path)

num_topics = 30


def plot_difference_matplotlib(mdiff, title="", annotation=None):
    """Helper function to plot difference between models.

    Uses matplotlib as the backend."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(18, 14))
    data = ax.imshow(mdiff, cmap='inferno_r', origin='lower')
    plt.title(title)
    plt.colorbar(data)
    plt.show()


mdiff, annotation = model_1.diff(model_2, distance=distance)
plot_difference_matplotlib(mdiff, title=f"Topic difference (one model) [{distance} distance]", annotation=annotation)