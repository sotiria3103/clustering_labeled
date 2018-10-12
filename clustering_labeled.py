import csv
import pylab as pl
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# gets all sessionIDs in semantic_features file and their index to check later if the session exists in simple_features file
with open('semantic_features.csv') as semanticfile:
    semantic_indices = dict((r[0], i) for i, r in enumerate(csv.reader(semanticfile)))

with open('simple_features.csv') as simplefile:
    with open('semantic_features.csv') as semanticfile:
        print("ALL FEATURES")

        read_simple = csv.DictReader(simplefile)
        read_semantic = csv.DictReader(semanticfile)

        labeled = []
        labeled_robots = []
        labeled_index = []
        humans = robots = 0

        for semantic_row, simple_row in zip(read_semantic, read_simple):
            index = semantic_indices.get(simple_row["SessionID"])
            # print(index)
            if index is not None:  # the session with id = simple_row["SessionID"] exists in semantic_features file
                if int(semantic_row["Human"]) == 1 or int(semantic_row["Robot"]) == 1:
                    if int(semantic_row["Human"]) == 1:
                        humans = humans + 1
                    if int(semantic_row["Robot"]) == 1:
                        robots = robots + 1

                    labeled_index.append(index)
                    labeled_robots.append(int(semantic_row["Robot"]))

                    labeled.append([int(semantic_row["Total_Topics"]), int(semantic_row["Unique_Topics"]),
                                    float(semantic_row["Unique_Percentage"]),
                                    float(semantic_row["Variance"]), float(semantic_row["Variance_Probabilistic"]),
                                    int(simple_row["#Requests"]), int(simple_row["Duration"]),
                                    float(simple_row["AverageTime"]), float(simple_row["StandardDeviation"]),
                                    float(simple_row["Repeated"]), float(simple_row["HTTP0"]),
                                    float(simple_row["HTTP2"]), float(simple_row["HTTP3"]),
                                    float(simple_row["HTTP4"]), float(simple_row["HTTP5"]), float(simple_row["HTTP7"]),
                                    float(simple_row["%PDF"]), int(simple_row["UniqueContent"]),
                                    int(simple_row["MultiCountries"]), int(simple_row["WebService"])])

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
            counter[kmeans10.labels_[i]] += 1  # cluster counter
            if labeled_robots[i] == 1:
                robot_counter[kmeans10.labels_[i]] += 1  # counter for robots in cluster

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

        # Fit the PCA model on the numeric columns from earlier.
        plot_labeled = pca_2.fit_transform(labeled)
        # Make a scatter plot of each labeled session, shaded according to cluster assignment.
        pl.scatter(x=plot_labeled[:, 0], y=plot_labeled[:, 1], c=labels10)
        # Show the plot.
        pl.show()

        duration = []
        average_time = []
        standard_deviation = []
        duration_all = []
        average_time_all = []
        standard_deviation_all = []

        for i in range(0, 10):
            for j, l in enumerate(labeled, start=0):
                if i == 0:
                    duration_all.append(l[6])
                    average_time_all.append(l[7])
                    standard_deviation_all.append(l[8])
                if kmeans10.labels_[j] == i:
                    duration.append(l[6])
                    average_time.append(l[7])
                    standard_deviation.append(l[8])

            # Draw the plot
            plt.subplot(1, 2, 1)
            plt.title('distribution of duration for cluster {i} '.format(i=i + 1))
            plt.hist(duration_all, color='red', bins=50, range=(0, 3000), label='all', edgecolor='black')
            plt.legend(loc='upper right')
            plt.subplot(1, 2, 2)
            plt.hist(duration, color='blue', bins=50, label='cluster', edgecolor='black')
            plt.legend(loc='upper right')
            plt.savefig("duration_all_features{i}.png".format(i=i + 1), bbox_inches='tight')
            plt.show()

            plt.subplot(1, 2, 1)
            plt.title('distribution of average time for cluster {i} '.format(i=i + 1))
            plt.hist(average_time_all, color='red', bins=50, range=(0, 1000), label='all', edgecolor='black')
            plt.legend(loc='upper right')
            plt.subplot(1, 2, 2)
            plt.hist(average_time, color='blue', bins=50, label='cluster', edgecolor='black')
            plt.legend(loc='upper right')
            plt.savefig("average_time_all_features{i}.png".format(i=i + 1), bbox_inches='tight')
            plt.show()

            plt.subplot(1, 2, 1)
            plt.title('distribution of standard deviation for cluster {i} '.format(i=i + 1))
            plt.hist(standard_deviation_all, color='red', bins=50, range=(0, 1000), label='all', edgecolor='black')
            plt.legend(loc='upper right')
            plt.subplot(1, 2, 2)
            plt.hist(standard_deviation, color='blue', bins=50, label='cluster', edgecolor='black')
            plt.legend(loc='upper right')
            plt.savefig("standard_deviation_all_features{i}.png".format(i=i + 1), bbox_inches='tight')
            plt.show()

            print("min duration in cluster ", i + 1, ": ", min(duration))
            print("max duration in cluster ", i + 1, ": ", max(duration))
            print("range: ", max(duration) - min(duration))

            print("min average time in cluster ", i + 1, ": ", min(average_time))
            print("max average time in cluster ", i + 1, ": ", max(average_time))
            print("range: ", max(average_time) - min(average_time))

            print("min standard deviation in cluster ", i + 1, ": ", min(standard_deviation))
            print("max standard deviation in cluster ", i + 1, ": ", max(standard_deviation))
            print("range: ", max(standard_deviation) - min(standard_deviation))

            duration[:] = []
            average_time[:] = []
            standard_deviation[:] = []
