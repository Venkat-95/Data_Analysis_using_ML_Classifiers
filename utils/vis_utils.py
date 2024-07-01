import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def visualization_pca(X_test, y_pred):
    y_pred = pd.Series(y_pred, name='label')

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_test)
    pc_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pc_df = pd.concat([pc_df, y_pred], axis=1)

    plt.figure(figsize=(8, 6))
    colors = ['r', 'g']
    labels = [0, 1]
    for label, color in zip(labels, colors):
        plt.scatter(pc_df[pc_df['label'] == label]['PC1'],
                    pc_df[pc_df['label'] == label]['PC2'],
                    c=color, label=f'Label {label}')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.title('PCA of X_test with Binary Labels')
    plt.show()

def visualization_t_SNE(X_test, y_pred):
    y_pred = pd.Series(y_pred, name='label')

    tsne = TSNE(n_components=2, random_state=0)
    tsne_results = tsne.fit_transform(X_test)
    tsne_df = pd.DataFrame(data=tsne_results, columns=['TSNE1', 'TSNE2'])
    tsne_df = pd.concat([tsne_df, y_pred], axis=1)

    plt.figure(figsize=(8, 6))
    colors = ['r', 'g']
    labels = [0, 1]
    for label, color in zip(labels, colors):
        plt.scatter(tsne_df[tsne_df['label'] == label]['TSNE1'],
                    tsne_df[tsne_df['label'] == label]['TSNE2'],
                    c=color, label=f'Label {label}')
    plt.xlabel('TSNE1')
    plt.ylabel('TSNE2')
    plt.legend()
    plt.title('t-SNE of X_test with Binary labels')
    plt.show()
