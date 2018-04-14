import csv
import pylab as pl
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas
# https://www.dataquest.io/blog/machine-learning-python/


class Session:
    """class for sessions"""

    def __init__(self, sessionID, numOfRequests, duration, averageTime, standardDeviation, repeated, http0, http2, http3, http4, http5, http7,
                 percentage_pdf, uniqueContent, multiCountries, webService, human, robot):

        # simple features
        self.sessionID = sessionID
        self.numOfRequests = numOfRequests
        self.duration = duration
        self.averageTime = averageTime
        self.standardDeviation = standardDeviation
        self.repeated = repeated
        self.http0 = http0
        self.http2 = http2
        self.http3 = http3
        self.http4 = http4
        self.http5 = http5
        self.http7 = http7
        self.percentage_pdf = percentage_pdf
        self.uniqueContent = uniqueContent
        self.multiCountries = multiCountries
        self.webService = webService
        self.human = human
        self.robot = robot


unlabeled = pandas.read_csv("simple_features.csv", dtype={'SessionID': int, '#Requests': int, 'Duration': int, 'AverageTime': np.float64, 'StandardDeviation': np.float64, 'Repeated': np.float64, 'HTTP0': np.float64, 'HTTP2': np.float64, 'HTTP3': np.float64, 'HTTP4': np.float64, 'HTTP5': np.float64, 'HTTP7': np.float64, '%PDF': np.float64, 'UniqueContent': int, 'MultiCountries': int, 'WebService': int, 'Human': int, 'Robot': int})
unlabeled = unlabeled[(unlabeled["Robot"] == 0) & (unlabeled["Human"] == 0)]
# unlabeled = pandas.Series(pandas.concat(((chunck_df['Robot'] == 0) & (chunck_df['Human'] == 0))
  #                        for chunck_df in pandas.read_csv('simple_features.csv', delimiter=',', chunksize=10000, dtype={'SessionID': int, '#Requests': int, 'Duration': int, 'AverageTime': float, 'StandardDeviation': float, 'Repeated': float, 'HTTP0': float, 'HTTP2': float, 'HTTP3': float, 'HTTP4': float, 'HTTP5': float, 'HTTP7': float, '%PDF': float, 'UniqueContent': int, 'MultiCountries': int, 'WebService': int, 'Human': int, 'Robot': int})))

# unlabeled = unlabeled[unlabeled.columns.difference(['SessionID', 'Human', 'Robot'])]
# unlabeled = pandas.DataFrame(data=unlabeled, columns=['SessionID', '#Requests', 'Duration', 'AverageTime', 'StandardDeviation',
                                                     # 'Repeated', 'HTTP0', 'HTTP2', 'HTTP3', 'HTTP4', 'HTTP5', 'HTTP7', '%PDF', 'UniqueContent', 'MultiCountries', 'WebService', 'Human', 'Robot'])
unlabeled.drop(['SessionID', 'Human', 'Robot'], 1, inplace=True)
# print(unlabeled)
with open('simple_features.csv') as simplefile:
    print("SIMPLE FEATURES")
    # in case of a header
    has_header = csv.Sniffer().has_header(simplefile.readline())
    simplefile.seek(0)
    readSimple = csv.reader(simplefile, delimiter=',')
    if has_header:
        next(readSimple)

    sessions = []
    for simple_row in readSimple:
        session = Session(simple_row[0], int(simple_row[1]), int(simple_row[2]), float(simple_row[3]), float(simple_row[4]),
                          float(simple_row[5]), float(simple_row[6]), float(simple_row[7]), float(simple_row[8]),
                          float(simple_row[9]), float(simple_row[10]), float(simple_row[11]), float(simple_row[12]),
                          int(simple_row[13]), int(simple_row[14]), int(simple_row[15]), int(simple_row[16]), int(simple_row[17]))
        sessions.append(session)
# , dtype={'SessionID': int, '#Requests': int, 'Duration': int, 'AverageTime': float, 'StandardDeviation': float, 'Repeated': float, 'HTTP0': float, 'HTTP2': float, 'HTTP3': float, 'HTTP4': float, 'HTTP5': float, 'HTTP7': float, '%PDF': float, 'UniqueContent': int, 'MultiCountries': int, 'WebService': int, 'Human': int, 'Robot': int}))
    # unlabeled = pandas.concat(((chunck_df['Robot'] == 0) & (chunck_df['Human'] == 0))
                           #   for chunck_df in pandas.read_csv('simple_features.csv', chunksize=10000))
    # unlabeled = unlabeled[unlabeled['Human'] == 0 and unlabeled['Robot'] == 0]
    # unlabeled = unlabeled.drop(['SessionID', 'Human', 'Robot'], 0, inplace=True)
    # unlabeled = unlabeled[unlabeled.columns.difference(['SessionID', 'Human', 'Robot'])]
    # unlabeled = []
    labeled = []
    labeled_robots = []
    humans = robots = 0
    for ses in sessions:
        #if ses.human == 0 and ses.robot == 0:
            # 2d matrix where every column is a feature and every line is an unlabeled session
            #unlabeled.append([ses.numOfRequests, ses.duration, ses.averageTime, ses.standardDeviation, ses.repeated, ses.http0,
                             # ses.http2, ses.http3, ses.http4, ses.http5, ses.http7, ses.percentage_pdf, ses.uniqueContent,
                             # ses.multiCountries, ses.webService])

        if ses.human == 1 or ses.robot == 1:
            if ses.human == 1:
                humans = humans + 1
            if ses.robot == 1:
                robots = robots + 1
                # 2d matrix where every column is a feature and every line is a labeled session
            labeled_robots.append(ses.robot)
            labeled.append([ses.numOfRequests, ses.duration, ses.averageTime, ses.standardDeviation, ses.repeated, ses.http0,
                            ses.http2, ses.http3, ses.http4, ses.http5, ses.http7, ses.percentage_pdf, ses.uniqueContent,
                            ses.multiCountries, ses.webService])

    print("actual number of humans:", humans)
    print("actual number of robots:", robots)
    # Fitting with inputs
    kmeans = KMeans(n_clusters=2).fit(labeled)
    # Predicting the clusters
    labels = kmeans.predict(labeled)
    # Getting the cluster centers
    C = kmeans.cluster_centers_
    print("centers for two clusters (labeled): ", C)
    print("labels: ", labels)
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

    # Fitting with inputs
    # kmeans = KMeans(n_clusters=2).fit(unlabeled)
    # Predicting the clusters
    # labels = kmeans.predict(unlabeled)
    # Getting the cluster centers
    # C_unlabeled = kmeans.cluster_centers_
    # print("centers for two clusters (unlabeled): ", C_unlabeled)
    # c_u0 = c_u1 = i = 0
    # for u in unlabeled:
        # if kmeans.labels_[i] == 1:
            # c_u1 = c_u1 + 1
        # if kmeans.labels_[i] == 0:
            # c_u0 = c_u0 + 1
        # i = i + 1

    # print("sessions in first cluster: ", c_u0)
    # print("sessions in second cluster: ", c_u1)

    # find "natural" k, where the score for k clusters doesn't have a big difference form the score for k+1 clusters
    N = range(1, 20)
    kmeans_labeled = [KMeans(n_clusters=i) for i in N]
    score = [kmeans_labeled[i].fit(labeled).score(labeled) for i in range(len(kmeans_labeled))]
    pl.plot(N, score)
    pl.xlabel('Number of Clusters')
    pl.ylabel('Score')
    pl.title('Elbow Curve')
    pl.show()

    # Fitting with inputs
    kmeans5 = KMeans(n_clusters=5).fit(labeled)
    # Predicting the clusters
    labels5 = kmeans5.predict(labeled)
    # Getting the cluster centers
    C = kmeans5.cluster_centers_
    print("centers for five clusters: ", C)

    counter = [0] * 5
    robot_counter = [0] * 5
    for i in range(0, len(labeled)):
        for j in range(0, 5):
            if kmeans5.labels_[i] == j:
                counter[j] += 1  # j-th cluster counter
                if labeled_robots[i] == 1:
                    robot_counter[j] += 1  # robots in j-th cluster counter

    for i in range(0, 5):
        print("sessions in cluster ", i + 1, " : ", counter[i])
        print("robots in cluster ", i + 1, " : ", robot_counter[i])
        print("percentage of robots in cluster ", i + 1, " : ", (robot_counter[i] / counter[i]) * 100, "%")

    # find "natural" k, where the score for k clusters doesn't have a big difference form the score for k+1 clusters
    # N = range(1, 10)
    # kmeans_unlabeled = [KMeans(n_clusters=i) for i in N]
    # score = [kmeans_unlabeled[i].fit(unlabeled).score(unlabeled) for i in range(len(kmeans_unlabeled))]
    # pl.plot(N, score)
    # pl.xlabel('Number of Clusters')
    # pl.ylabel('Score')
    # pl.title('Elbow Curve')
    # pl.show()

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
    pl.scatter(x=plot_labeled[:, 0], y=plot_labeled[:, 1], c=labels5)
    # Show the plot.
    pl.show()
