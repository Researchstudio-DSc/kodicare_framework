
import gensim
import os

model_dir = "../../models/trec_covid_topic_modelling"

#distance = "kullback_leibler"
#distance = "hellinger"
#distance = "jaccard"
distance = "jensen_shannon"

lda_1_path = os.path.join(model_dir, "lda_1")
lda_2_path = os.path.join(model_dir, "lda_2")

lda_model_tfidf = gensim.models.LdaMulticore.load(lda_1_path)
lda_model_tfidf_2 = gensim.models.LdaMulticore.load(lda_2_path)

num_topics = 30


def plot_difference_matplotlib(mdiff, title="", annotation=None):
    """Helper function to plot difference between models.

    Uses matplotlib as the backend."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(18, 14))
    data = ax.imshow(mdiff, cmap='inferno_r', origin='lower')
    plt.title(title)
    plt.colorbar(data)


mdiff, annotation = lda_model_tfidf.diff(lda_model_tfidf, distance=distance, num_words=100)
plot_difference_matplotlib(mdiff, title=f"Topic difference (one model) [{distance} distance]", annotation=annotation)