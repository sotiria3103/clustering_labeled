import pylab as pl
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def cluster2(labeled, labeled_robots):
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
    return labels


def find_k(labeled):
    N = range(1, 20)
    kmeans_labeled = [KMeans(n_clusters=i) for i in N]
    score = [kmeans_labeled[i].fit(labeled).score(labeled) for i in range(len(kmeans_labeled))]
    pl.plot(N, score)
    pl.xlabel('Number of Clusters')
    pl.ylabel('Score')
    pl.title('Elbow Curve')
    pl.show()
    return


def cluster_k(labeled, k, labeled_robots):
    # Fitting with inputs
    kmeans = KMeans(n_clusters=k).fit(labeled)
    # Predicting the clusters
    labels = kmeans.predict(labeled)
    # Getting the cluster centers
    C = kmeans.cluster_centers_
    print("centers for six clusters: ", C)

    counter = [0] * k
    robot_counter = [0] * k
    for i in range(0, len(labeled)):
        counter[kmeans.labels_[i]] += 1  # cluster counter
        if labeled_robots[i] == 1:
            robot_counter[kmeans.labels_[i]] += 1  # counter for robots in cluster

    for i in range(0, k):
        print("sessions in cluster ", i + 1, " : ", counter[i])
        print("robots in cluster ", i + 1, " : ", robot_counter[i])
        print("percentage of robots in cluster ", i + 1, " : ", (robot_counter[i] / counter[i]) * 100, "%")
    return labels


def pca(labeled, labels2, labels6):
    pca_2 = PCA(2)
    # Fit the PCA model on the numeric columns from earlier.
    plot_labeled = pca_2.fit_transform(labeled)
    # Make a scatter plot of each labeled session, shaded according to cluster assignment.
    pl.scatter(x=plot_labeled[:, 0], y=plot_labeled[:, 1], c=labels2)
    # Show the plot.
    pl.show()

    # Fit the PCA model on the numeric columns from earlier.
    plot_labeled = pca_2.fit_transform(labeled)
    # Make a scatter plot of each labeled session, shaded according to cluster assignment.
    pl.scatter(x=plot_labeled[:, 0], y=plot_labeled[:, 1], c=labels6)
    # Show the plot.
    pl.show()
    return
