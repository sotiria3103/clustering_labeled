import csv
import pylab as pl
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


class Session:
    """class for sessions"""

    def __init__(self, sessionID, totalTopics, uniqueTopics, uniquePercentage, variance, varianceProbabilistic, human, robot,
                 numOfRequests, duration, averageTime, standardDeviation, repeated, http0, http2, http3, http4, http5, http7,
                 percentage_pdf, uniqueContent, multiCountries, webService ):
        # semantic features
        self.sessionID = sessionID
        self.totalTopics = totalTopics
        self.uniqueTopics = uniqueTopics
        self.uniquePercentage = uniquePercentage
        self.variance = variance
        self.varianceProbabilistic = varianceProbabilistic
        # simple features
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
        # common features
        self.human = human
        self.robot = robot


with open('simple_features.csv') as simplefile:
    with open('semantic_features.csv') as semanticfile:
        print("ALL FEATURES")
        # in case of a header
        has_header = csv.Sniffer().has_header(semanticfile.readline())
        semanticfile.seek(0)
        readSemantic = csv.reader(semanticfile, delimiter=',')
        if has_header:
            next(readSemantic)
        has_header = csv.Sniffer().has_header(simplefile.readline())
        simplefile.seek(0)
        readSimple = csv.reader(simplefile, delimiter=',')
        if has_header:
            next(readSimple)

        sessions = []
        for semantic_row, simple_row in zip(readSemantic, readSimple):
            session = Session(semantic_row[0], int(semantic_row[1]), int(semantic_row[2]), float(semantic_row[3]),
                              float(semantic_row[4]), float(semantic_row[5]), int(semantic_row[6]), int(semantic_row[7]),
                              int(simple_row[1]), int(simple_row[2]), float(simple_row[3]), float(simple_row[4]), float(simple_row[5]), float(simple_row[6]),
                              float(simple_row[7]), float(simple_row[8]), float(simple_row[9]), float(simple_row[10]), float(simple_row[11]), float(simple_row[12]),
                              int(simple_row[13]), int(simple_row[14]), int(simple_row[15]))
            sessions.append(session)

        labeled = []
        labeled_robots = []
        humans = robots = 0
        for ses in sessions:
            if ses.human == 1 or ses.robot == 1:
                if ses.human == 1:
                    humans = humans + 1
                if ses.robot == 1:
                    robots = robots + 1
                # list where every column is a feature and every line is a labeled session
                labeled_robots.append(ses.robot)
                labeled.append([ses.totalTopics, ses.uniqueTopics, ses.uniquePercentage, ses.variance, ses.varianceProbabilistic,
                                ses.numOfRequests, ses.duration, ses.averageTime, ses.standardDeviation, ses.repeated, ses.http0,
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
        print("centers for two clusters: ", C)
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

        # (len(labeled_robots))=(len(labeled))=(len(labels))=(len(kmeans.labels_)) labels[j] == kmeans.labels_[j]

        print("sessions in first cluster: ", c0)
        print("sessions in second cluster: ", c1)
        print("robots in first cluster: ", r0)
        print("robots in second cluster: ", r1)
        print("percentage of robots in first cluster: ", (r0 / c0) * 100, "%")
        print("percentage of robots in second cluster: ", (r1 / c1) * 100, "%")
        # find "natural" k, where the score for k clusters doesn't have a big difference form the score for k+1 clusters
        N = range(1, 20)
        kmeans = [KMeans(n_clusters=i) for i in N]
        score = [kmeans[i].fit(labeled).score(labeled) for i in range(len(kmeans))]
        pl.plot(N, score)
        pl.xlabel('Number of Clusters')
        pl.ylabel('Score')
        pl.title('Elbow Curve')
        pl.show()

        # Fitting with inputs
        kmeans10 = KMeans(n_clusters=10).fit(labeled)
        # Predicting the clusters
        labels10 = kmeans10.predict(labeled)
        # Getting the cluster centers
        C = kmeans10.cluster_centers_
        print("centers for ten clusters: ", C)

        counter = [0]*10
        robot_counter = [0]*10
        for i in range(0, len(labeled)):
            for j in range(0, 10):
                if kmeans10.labels_[i] == j:
                    counter[j] += 1  # j-th cluster counter
                    if labeled_robots[i] == 1:
                        robot_counter[j] += 1  # robots in j-th cluster counter

        for i in range(0, 10):
            print("sessions in cluster ", i+1, " : ", counter[i])
            print("robots in cluster ", i+1, " : ", robot_counter[i])
            print("percentage of robots in cluster ", i+1, " : ", (robot_counter[i] / counter[i]) * 100, "%")

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
        pl.scatter(x=plot_labeled[:, 0], y=plot_labeled[:, 1], c=labels10)
        # Show the plot.
        pl.show()
