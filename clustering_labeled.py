import csv
import pylab as pl
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from my_functions import cluster2, find_k, cluster_k, pca

# gets all sessionIDs in semantic_features file and their index to check later if the session exists in simple_features file
with open('semantic_features.csv') as semanticfile:
    semantic_indices = dict((r[0], i) for i, r in enumerate(csv.reader(semanticfile)))

with open('simple_features.csv') as simplefile:
    with open('semantic_features.csv') as semanticfile:
        print("ALL FEATURES")

        read_simple = csv.DictReader(simplefile)
        read_semantic = csv.DictReader(semanticfile)

        labeled_sessions = []
        labeled_robots = []
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

                    labeled_robots.append(int(semantic_row["Robot"]))

                    labeled_sessions.append([int(semantic_row["Total_Topics"]), int(semantic_row["Unique_Topics"]),
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

        labels2 = cluster2(labeled_sessions, labeled_robots)

        find_k(labeled_sessions)

        labels10 = cluster_k(labeled_sessions, 10, labeled_robots)   # k = 10

        pca(labeled_sessions, labels2, labels10)

        duration = []
        average_time = []
        standard_deviation = []
        duration_all = []
        average_time_all = []
        standard_deviation_all = []

        for i in range(0, 10):
            for j, l in enumerate(labeled_sessions, start=0):
                if i == 0:
                    duration_all.append(l[6])
                    average_time_all.append(l[7])
                    standard_deviation_all.append(l[8])
                if labels10[j] == i:
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
