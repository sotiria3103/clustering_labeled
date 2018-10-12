import csv
import pylab as pl
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

with open('simple_features.csv') as simplefile:
    print("SIMPLE FEATURES")

    read_simple = csv.DictReader(simplefile)

    labeled = []
    labeled_robots = []
    labeled_index = []
    humans = robots = 0
    for index, simple_row in enumerate(read_simple, start=0):
        if int(simple_row["Human"]) == 1 or int(simple_row["Robot"]) == 1:
            if int(simple_row["Human"]) == 1:
                humans = humans + 1
            if int(simple_row["Robot"]) == 1:
                robots = robots + 1

            labeled_index.append(index)
            labeled_robots.append(int(simple_row["Robot"]))

            labeled.append([int(simple_row["#Requests"]), int(simple_row["Duration"]), float(simple_row["AverageTime"]), float(simple_row["StandardDeviation"]),
                           float(simple_row["Repeated"]), float(simple_row["HTTP0"]), float(simple_row["HTTP2"]), float(simple_row["HTTP3"]),
                           float(simple_row["HTTP4"]), float(simple_row["HTTP5"]), float(simple_row["HTTP7"]), float(simple_row["%PDF"]),
                           int(simple_row["UniqueContent"]), int(simple_row["MultiCountries"]), int(simple_row["WebService"])])

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
        counter[kmeans5.labels_[i]] += 1  # cluster counter
        if labeled_robots[i] == 1:
                robot_counter[kmeans5.labels_[i]] += 1  # counter for robots in cluster

    for i in range(0, 5):
        print("sessions in cluster ", i + 1, " : ", counter[i])
        print("robots in cluster ", i + 1, " : ", robot_counter[i])
        print("percentage of robots in cluster ", i + 1, " : ", (robot_counter[i] / counter[i]) * 100, "%")

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
    pl.scatter(x=plot_labeled[:, 0], y=plot_labeled[:, 1], c=labels5)
    # Show the plot.
    pl.show()

    duration = []
    average_time = []
    standard_deviation = []
    numOfRequests = []
    unique_content = []
    duration_all = []
    average_time_all = []
    standard_deviation_all = []
    numOfRequests_all = []
    unique_content_all = []

    for i in range(0, 5):
        for j, l in enumerate(labeled, start=0):
            if i == 0:
                duration_all.append(l[1])
                average_time_all.append(l[2])
                standard_deviation_all.append(l[3])
                numOfRequests_all.append(l[0])
                unique_content_all.append(l[12])
            if kmeans5.labels_[j] == i:
                duration.append(l[1])
                average_time.append(l[2])
                standard_deviation.append(l[3])
                numOfRequests.append(l[0])
                unique_content.append(l[12])

        # Draw the plot
        plt.subplot(1, 2, 1)
        plt.title('distribution of duration for cluster {i} '.format(i=i+1))
        plt.hist(duration_all, color='red', bins=50, range=(0, 5000), label='all', edgecolor='black')
        plt.legend(loc='upper right')
        plt.subplot(1, 2, 2)
        plt.hist(duration, color='blue', bins=50, label='cluster', edgecolor='black')
        plt.legend(loc='upper right')
        plt.savefig("official_full_range_duration{i}.png".format(i=i+1), bbox_inches='tight')
        plt.show()

        plt.subplot(1, 2, 1)
        plt.title('distribution of average time for cluster {i} '.format(i=i+1))
        plt.hist(average_time_all, color='red', bins=50, range=(0, 2500), label='all', edgecolor='black')
        plt.legend(loc='upper right')
        plt.subplot(1, 2, 2)
        plt.hist(average_time, color='blue', bins=50, label='cluster', edgecolor='black')
        plt.legend(loc='upper right')
        plt.savefig("official_full_range_average_time{i}.png".format(i=i+1), bbox_inches='tight')
        plt.show()

        plt.subplot(1, 2, 1)
        plt.title('distribution of standard deviation for cluster {i} '.format(i=i+1))
        plt.hist(standard_deviation_all, color='red', bins=50, range=(0, 2500), label='all', edgecolor='black')
        plt.legend(loc='upper right')
        plt.subplot(1, 2, 2)
        plt.hist(standard_deviation, color='blue', bins=50, label='cluster', edgecolor='black')
        plt.legend(loc='upper right')
        plt.savefig("official_full_range_standard_deviation{i}.png".format(i=i+1), bbox_inches='tight')
        plt.show()

        plt.subplot(1, 2, 1)
        plt.title('distribution of number of requests for cluster {i} '.format(i=i+1))
        plt.hist(numOfRequests_all, color='red', bins=50, range=(0, 150), label='all', edgecolor='black')
        plt.legend(loc='upper right')
        plt.subplot(1, 2, 2)
        plt.hist(numOfRequests, color='blue', bins=50, label='cluster', edgecolor='black')
        plt.legend(loc='upper right')
        plt.savefig("official_full_range_numOfRequests{i}.png".format(i=i+1), bbox_inches='tight')
        plt.show()

        plt.subplot(1, 2, 1)
        plt.title('distribution of unique content for cluster {i} '.format(i=i+1))
        plt.hist(unique_content_all, color='red', bins=50, range=(0, 100), label='all', edgecolor='black')
        plt.legend(loc='upper right')
        plt.subplot(1, 2, 2)
        plt.hist(unique_content, color='blue', bins=50, label='cluster', edgecolor='black')
        plt.legend(loc='upper right')
        plt.savefig("official_full_range_unique_content{i}.png".format(i=i + 1), bbox_inches='tight')
        plt.show()

        print("min duration in cluster ", i+1, ": ", min(duration))
        print("max duration in cluster ", i+1, ": ", max(duration))
        print("range: ", max(duration) - min(duration))
        print("min average time in cluster ", i + 1, ": ", min(average_time))
        print("max average time in cluster ", i + 1, ": ", max(average_time))
        print("range: ", max(average_time) - min(average_time))
        print("min standard_deviation in cluster ", i + 1, ": ", min(standard_deviation))
        print("max standard_deviation in cluster ", i + 1, ": ", max(standard_deviation))
        print("range: ", max(standard_deviation) - min(standard_deviation))
        print("min numOfRequests in cluster ", i + 1, ": ", min(numOfRequests))
        print("max numOfRequests in cluster ", i + 1, ": ", max(numOfRequests))
        print("range: ", max(numOfRequests) - min(numOfRequests))
        print("min unique content in cluster ", i + 1, ": ", min(unique_content))
        print("max unique content in cluster ", i + 1, ": ", max(unique_content))
        print("range: ", max(unique_content) - min(unique_content))
        duration[:] = []
        average_time[:] = []
        standard_deviation[:] = []
        numOfRequests[:] = []
        unique_content[:] = []
