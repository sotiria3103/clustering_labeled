import csv
import pylab as pl
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

with open('semantic_features.csv') as semanticfile:
    print("SEMANTIC FEATURES")

    read_semantic = csv.DictReader(semanticfile)

    labeled = []
    labeled_robots = []
    labeled_index = []
    humans = robots = 0
    for index, semantic_row in enumerate(read_semantic, start=0):
        if int(semantic_row["Human"]) == 1 or int(semantic_row["Robot"]) == 1:
            if int(semantic_row["Human"]) == 1:
                humans = humans + 1
            if int(semantic_row["Robot"]) == 1:
                robots = robots + 1

            labeled_index.append(index)
            labeled_robots.append(int(semantic_row["Robot"]))

            labeled.append([int(semantic_row["Total_Topics"]), int(semantic_row["Unique_Topics"]), float(semantic_row["Unique_Percentage"]),
                            float(semantic_row["Variance"]), float(semantic_row["Variance_Probabilistic"])])

    print("actual number of humans:", humans)
    print("actual number of robots:", robots)

    # Fitting with inputs
    kmeans = KMeans(n_clusters=2).fit(labeled)
    # Predicting the clusters
    labels = kmeans.predict(labeled)
    # Getting the cluster centers
    C = kmeans.cluster_centers_
    print("centers for two clusters (labeled): ", C)
    c0 = c1 = 0
    r0 = r1 = 0
    for i in range(0, len(labeled)):
        if kmeans.labels_[i] == 0:
            c0 += 1  # first cluster counter
            if labeled_robots[i] == 1:
                r0 += 1  # robots in first cluster counter
        if kmeans.labels_[i] == 1:
            c1 += 1  # second cluster counter
            if labeled_robots[i] == 1:
                r1 += 1  # robots in second cluster counter

    print("sessions in first cluster: ", c0)
    print("sessions in second cluster: ", c1)
    print("robots in first cluster: ", r0)
    print("robots in second cluster: ", r1)
    print("percentage of robots in first cluster: ", (r0 / c0) * 100, "%")
    print("percentage of robots in second cluster: ", (r1 / c1) * 100, "%")

    # find "natural" k, where the score for k clusters doesn't have a big difference from the score for k+1 clusters
    N = range(1, 20)
    kmeans_labeled = [KMeans(n_clusters=i) for i in N]
    score = [kmeans_labeled[i].fit(labeled).score(labeled) for i in range(len(kmeans_labeled))]
    pl.plot(N, score)
    pl.xlabel('Number of Clusters')
    pl.ylabel('Score')
    pl.title('Elbow Curve')
    pl.show()


    # Fitting with inputs
    kmeans6 = KMeans(n_clusters=6).fit(labeled) # k=6
    # Predicting the clusters
    labels6 = kmeans6.predict(labeled)
    # Getting the cluster centers
    C = kmeans6.cluster_centers_
    print("centers for six clusters: ", C)

    counter = [0] * 6
    robot_counter = [0] * 6
    for i in range(0, len(labeled)):
        counter[kmeans6.labels_[i]] += 1  # cluster counter
        if labeled_robots[i] == 1:
                robot_counter[kmeans6.labels_[i]] += 1  # counter for robots in cluster

    for i in range(0, 6):
        print("sessions in cluster ", i + 1, " : ", counter[i])
        print("robots in cluster ", i + 1, " : ", robot_counter[i])
        print("percentage of robots in cluster ", i + 1, " : ", (robot_counter[i] / counter[i]) * 100, "%")

    # Create a PCA model.
    pca_2 = PCA(2)
    # Fit the PCA model on the numeric columns from earlier.
    plot_labeled = pca_2.fit_transform(labeled)
    # Make a scatter plot of each labeled session, shaded according to cluster assignment.
    pl.scatter(x=plot_labeled[:, 0], y=plot_labeled[:, 1], c=labels)
    # Show the plot.
    pl.show()

    # Fit the PCA model on the numeric columns from earlier.
    plot_labeled = pca_2.fit_transform(labeled)
    plot_c = pca_2.fit_transform(C)
    # Make a scatter plot of each labeled session, shaded according to cluster assignment.
    pl.scatter(x=plot_labeled[:, 0], y=plot_labeled[:, 1], c=labels6)
    # Show the plot.
    pl.show()

    total_topics = []
    unique_topics = []
    total_topics_all = []
    unique_topics_all = []

    for i in range(0, 6):
        for j, l in enumerate(labeled, start=0):
            if i == 0:
                total_topics_all.append(l[0])
                unique_topics_all.append(l[1])
            if kmeans6.labels_[j] == i:
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
