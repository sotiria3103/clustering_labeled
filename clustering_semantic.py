import csv
import pylab as pl
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
# source: http://www.michaeljgrogan.com/k-means-clustering-python-sklearn/


class Session:
    """class for sessions"""

    def __init__(self, sessionID, totalTopics, uniqueTopics, uniquePercentage, variance, varianceProbabilistic, human, robot):
        # semantic features
        self.sessionID = sessionID
        self.totalTopics = totalTopics
        self.uniqueTopics = uniqueTopics
        self.uniquePercentage = uniquePercentage
        self.variance = variance
        self.varianceProbabilistic = varianceProbabilistic
        self.human = human
        self.robot = robot


with open('semantic_features.csv') as semanticfile:
    print("SEMANTIC FEATURES")
    # in case of a header
    has_header = csv.Sniffer().has_header(semanticfile.readline())
    semanticfile.seek(0)
    readSemantic = csv.reader(semanticfile, delimiter=',')
    if has_header:
        next(readSemantic)

    sessions = []
    for semantic_row in readSemantic:
        session = Session(semantic_row[0], int(semantic_row[1]), int(semantic_row[2]), float(semantic_row[3]),
                          float(semantic_row[4]), float(semantic_row[5]), int(semantic_row[6]), int(semantic_row[7]))
        sessions.append(session)
    unlabeled = []
    labeled = []
    labeled_robots = []
    humans = robots = 0
    for ses in sessions:
        # if ses.human == 0 and ses.robot == 0:
            # 2d matrix where every column is a feature and every line is an unlabeled session
            # unlabeled.append([ses.totalTopics, ses.uniqueTopics, ses.uniquePercentage, ses.variance, ses.varianceProbabilistic])

        if ses.human == 1 or ses.robot == 1:
            if ses.human == 1:
                humans = humans + 1
            if ses.robot == 1:
                robots = robots + 1
                # 2d matrix where every column is a feature and every line is a labeled session
            labeled_robots.append(ses.robot)
            labeled.append([ses.totalTopics, ses.uniqueTopics, ses.uniquePercentage, ses.variance, ses.varianceProbabilistic])

    # Fitting with inputs
    # kmeans = KMeans(n_clusters=2).fit(unlabeled)
    # Predicting the clusters
    # labels = kmeans.predict(unlabeled)
    # Getting the cluster centers
    # C_unlabeled = kmeans.cluster_centers_
    # print("centers for two clusters (unlabeled): ", C_unlabeled)
    # c_u0 = c_u1 = i = 0
    # for u in unlabeled:
    #    if kmeans.labels_[i] == 1:
    #        c_u1 = c_u1 + 1
    #    if kmeans.labels_[i] == 0:
    #        c_u0 = c_u0 + 1
    #    i = i + 1

    # print("sessions in first cluster: ", c_u0)
    # print("sessions in second cluster: ", c_u1)

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
    # N = range(1, 20)
    # kmeans_unlabeled = [KMeans(n_clusters=i) for i in N]
    # score = [kmeans_unlabeled[i].fit(unlabeled).score(unlabeled) for i in range(len(kmeans_unlabeled))]
    # pl.plot(N, score)
    # pl.xlabel('Number of Clusters')
    # pl.ylabel('Score')
    # pl.title('Elbow Curve')
    # pl.show()

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
    kmeans6 = KMeans(n_clusters=6).fit(labeled)
    # Predicting the clusters
    labels6 = kmeans6.predict(labeled)
    # Getting the cluster centers
    C = kmeans6.cluster_centers_
    print("centers for six clusters: ", C)

    counter = [0] * 6
    robot_counter = [0] * 6
    for i in range(0, len(labeled)):
        for j in range(0, 6):
            if kmeans6.labels_[i] == j:
                counter[j] += 1  # j-th cluster counter
                if labeled_robots[i] == 1:
                    robot_counter[j] += 1  # robots in j-th cluster counter

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

    pca_2 = PCA(2)
    # Fit the PCA model on the numeric columns from earlier.
    plot_labeled = pca_2.fit_transform(labeled)
    # Make a scatter plot of each labeled session, shaded according to cluster assignment.
    pl.scatter(x=plot_labeled[:, 0], y=plot_labeled[:, 1], c=labels6)
    # Show the plot.
    pl.show()
