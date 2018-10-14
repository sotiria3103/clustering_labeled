import csv
import pylab as pl
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from my_functions import cluster2, find_k, cluster_k, pca

with open('semantic_features.csv') as semanticfile:
    print("SEMANTIC FEATURES")

    read_semantic = csv.DictReader(semanticfile)

    labeled_sessions = []
    labeled_robots = []
    humans = robots = 0
    for index, semantic_row in enumerate(read_semantic, start=0):
        if int(semantic_row["Human"]) == 1 or int(semantic_row["Robot"]) == 1:
            if int(semantic_row["Human"]) == 1:
                humans = humans + 1
            if int(semantic_row["Robot"]) == 1:
                robots = robots + 1

            labeled_robots.append(int(semantic_row["Robot"]))

            labeled_sessions.append([int(semantic_row["Total_Topics"]), int(semantic_row["Unique_Topics"]), float(semantic_row["Unique_Percentage"]),
                                     float(semantic_row["Variance"]), float(semantic_row["Variance_Probabilistic"])])

    print("actual number of humans:", humans)
    print("actual number of robots:", robots)

    labels2 = cluster2(labeled_sessions, labeled_robots)

    find_k(labeled_sessions)

    labels6 = cluster_k(labeled_sessions, 6, labeled_robots)  # k = 6

    pca(labeled_sessions, labels2, labels6)

    total_topics = []
    unique_topics = []
    total_topics_all = []
    unique_topics_all = []

    for i in range(0, 6):
        for j, l in enumerate(labeled_sessions, start=0):
            if i == 0:
                total_topics_all.append(l[0])
                unique_topics_all.append(l[1])
            if labels6[j] == i:
                total_topics.append(l[0])
                unique_topics.append(l[1])

        # Draw the plot

        plt.subplot(1, 2, 1)
        plt.title('distribution of total topics for cluster {i} '.format(i=i + 1))
        plt.hist(total_topics_all, color='red', bins=50, range=(0, 2000), label='all', edgecolor='black')
        plt.legend(loc='upper right')
        plt.subplot(1, 2, 2)
        plt.hist(total_topics, color='blue', bins=30, label='cluster', edgecolor='black')
        plt.legend(loc='upper right')
        plt.savefig("total_topics{i}.png".format(i=i + 1), bbox_inches='tight')
        plt.show()

        plt.subplot(1, 2, 1)
        plt.title('distribution of unique topics for cluster {i} '.format(i=i + 1))
        plt.hist(unique_topics_all, color='red', bins=50, range=(0, 1000), label='all', edgecolor='black')
        plt.legend(loc='upper right')
        plt.subplot(1, 2, 2)
        plt.hist(unique_topics, color='blue', bins=30, label='cluster', edgecolor='black')
        plt.legend(loc='upper right')
        plt.savefig("unique_topics{i}.png".format(i=i + 1), bbox_inches='tight')
        plt.show()

        print("min total topics in cluster ", i + 1, ": ", min(total_topics))
        print("max total topics in cluster ", i + 1, ": ", max(total_topics))
        print("range: ", max(total_topics)-min(total_topics))
        print("min unique topics in cluster ", i + 1, ": ", min(unique_topics))
        print("max unique topics in cluster ", i + 1, ": ", max(unique_topics))
        print("range: ", max(unique_topics) - min(unique_topics))

        total_topics[:] = []
        unique_topics[:] = []
