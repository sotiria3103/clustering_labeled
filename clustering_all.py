import csv
import pylab as pl
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
# source: http://www.michaeljgrogan.com/k-means-clustering-python-sklearn/


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

        unlabeled = []
        labeled = []
        humans = robots = 0
        for ses in sessions:
            if ses.human == 0 and ses.robot == 0:
                unlabeled.append([ses.totalTopics, ses.uniqueTopics, ses.uniquePercentage, ses.variance, ses.varianceProbabilistic,
                                  ses.numOfRequests, ses.duration, ses.averageTime, ses.standardDeviation, ses.repeated, ses.http0,
                                  ses.http2, ses.http3, ses.http4, ses.http5, ses.http7, ses.percentage_pdf, ses.uniqueContent,
                                  ses.multiCountries, ses.webService])

            if ses.human == 1 or ses.robot == 1:
                if ses.human == 1:
                    humans = humans + 1
                if ses.robot == 1:
                    robots = robots + 1
                # 2d matrix where every column is a feature and every line is a labeled session
                labeled.append([ses.totalTopics, ses.uniqueTopics, ses.uniquePercentage, ses.variance, ses.varianceProbabilistic,
                                ses.numOfRequests, ses.duration, ses.averageTime, ses.standardDeviation, ses.repeated, ses.http0,
                                ses.http2, ses.http3, ses.http4, ses.http5, ses.http7, ses.percentage_pdf, ses.uniqueContent,
                                ses.multiCountries, ses.webService])

        # Fitting with inputs
        kmeans = KMeans(n_clusters=2).fit(unlabeled)
        # Predicting the clusters
        labels = kmeans.predict(unlabeled)
        # Getting the cluster centers
        C_unlabeled = kmeans.cluster_centers_
        print("centers for two clusters (unlabeled): ", C_unlabeled)
        c_u0 = c_u1 = i = 0
        for u in unlabeled:
            if kmeans.labels_[i] == 1:
                c_u1 = c_u1 + 1
            if kmeans.labels_[i] == 0:
                c_u0 = c_u0 + 1
            i = i + 1

        print("sessions in first cluster: ", c_u0)
        print("sessions in second cluster: ", c_u1)

        print("actual number of humans:", humans)
        print("actual number of robots:", robots)
        # Fitting with inputs
        kmeans = KMeans(n_clusters=2).fit(labeled)
        # Predicting the clusters
        labels = kmeans.predict(labeled)
        # Getting the cluster centers
        C = kmeans.cluster_centers_
        print("centers for two clusters (labeled): ", C)
        c0 = c1 = j = 0
        for l in labeled:
            if kmeans.labels_[j] == 1:
                c1 = c1+1
            if kmeans.labels_[j] == 0:
                c0 = c0+1
            j = j+1

        print("sessions in first cluster: ", c0)
        print("sessions in second cluster: ", c1)

        # find "natural" k, where the score for k clusters doesn't have a big difference form the score for k+1 clusters
        N = range(1, 20)
        kmeans_labeled = [KMeans(n_clusters=i) for i in N]
        score = [kmeans_labeled[i].fit(labeled).score(labeled) for i in range(len(kmeans_labeled))]
        pl.plot(N, score)
        pl.xlabel('Number of Clusters')
        pl.ylabel('Score')
        pl.title('Elbow Curve')
        pl.show()

        # find "natural" k, where the score for k clusters doesn't have a big difference form the score for k+1 clusters
        N = range(1, 20)
        kmeans_unlabeled = [KMeans(n_clusters=i) for i in N]
        score = [kmeans_unlabeled[i].fit(unlabeled).score(unlabeled) for i in range(len(kmeans_unlabeled))]
        pl.plot(N, score)
        pl.xlabel('Number of Clusters')
        pl.ylabel('Score')
        pl.title('Elbow Curve')
        pl.show()

        pca = PCA(n_components=1).fit(labeled)
        pca_d = pca.transform(labeled)
