# Implementation of Paper:
#   CurriculumNet: Weakly Supervised Learning from Large-Scale Web Images
#   Github: https://github.com/MalongTech/research-curriculumnet.git

import time
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.utils import check_array, check_consistent_length, gen_batches


def cluster_curriculum_subsets(X, y, n_subsets=3, method='default', density_t=0.6, verbose=False,
                               dim_reduce=256, batch_max=10000, random_state=None, calc_auxiliary=False):

    if not density_t > 0.0:
        raise ValueError("density_thresh must be positive.")
    X = check_array(X, accept_sparse='csr')
    check_consistent_length(X, y)

    unique_categories = set(y)
    t0 = None
    pca = None
    auxiliary_info = []
    if X.shape[1] > dim_reduce:
        pca = PCA(n_components=dim_reduce, copy=False, random_state=random_state)

    # Initialize all labels as negative one which represents un-clustered 'noise'.
    # Post-condition: after clustering, there should be no negatives in the label output.
    all_clustered_labels = np.full(len(y), -1, dtype=np.intp)

    for cluster_idx, current_category in enumerate(unique_categories):
        if verbose:
            t0 = time.time()

        # Collect the "learning material" for this particular category
        dist_list = [i for i, label in enumerate(y) if label == current_category]

        for batch_range in gen_batches(len(dist_list), batch_size=batch_max):
            print("len: {}\tmax: {}".format(len(dist_list), batch_max))
            batch_dist_list = dist_list[batch_range]

            # Load data subset
            subset_vectors = np.zeros((len(batch_dist_list), X.shape[1]), dtype=np.float32)
            for subset_idx, global_idx in enumerate(batch_dist_list):
                subset_vectors[subset_idx, :] = X[global_idx, :]

            # Calc distances
            print("PCA process... ")
            if pca:
                subset_vectors = pca.fit_transform(subset_vectors)
            m = np.dot(subset_vectors, np.transpose(subset_vectors))
            t = np.square(subset_vectors).sum(axis=1)
            distance = np.sqrt(np.abs(-2 * m + t + np.transpose(np.array([t]))))

            # Calc densities
            print("Calc densities")
            if method == 'gaussian':
                densities = np.zeros((len(subset_vectors)), dtype=np.float32)
                distance = distance / np.max(distance)
                for i in range(len(subset_vectors)):
                    densities[i] = np.sum(1 / np.sqrt(2 * np.pi) * np.exp((-1) * np.power(distance[i], 2) / 2.0))
            else:
                densities = np.zeros((len(subset_vectors)), dtype=np.float32)
                flat_distance = distance.reshape(distance.shape[0] * distance.shape[1])
                dist_cutoff = np.sort(flat_distance)[int(distance.shape[0] * distance.shape[1] * density_t)]
                for i in range(len(batch_dist_list)):
                    densities[i] = len(np.where(distance[i] < dist_cutoff)[0]) - 1  # remove itself
            if len(densities) < n_subsets:
                raise ValueError("Cannot cluster into {} subsets due to lack of density diversification,"
                                 " please try a smaller n_subset number.".format(n_subsets))

            # Optionally, calc auxiliary info
            print("calc auxiliary info")
            if calc_auxiliary:
                # Calculate deltas
                deltas = np.zeros((len(subset_vectors)), dtype=np.float32)
                densities_sort_idx = np.argsort(densities)
                for i in range(len(densities_sort_idx) - 1):
                    larger = densities_sort_idx[i + 1:]
                    larger = larger[np.where(larger != densities_sort_idx[i])]  # remove itself
                    deltas[i] = np.min(distance[densities_sort_idx[i], larger])

                # Find the centers and package
                center_id = np.argmax(densities)
                center_delta = np.max(distance[np.argmax(densities)])
                center_density = densities[center_id]
                auxiliary_info.append((center_id, center_delta, center_density))

            print("Start Kmeans clustering....")
            model = KMeans(n_clusters=n_subsets, random_state=random_state)
            model.fit(densities.reshape(len(densities), 1))
            clusters = [densities[np.where(model.labels_ == i)] for i in range(n_subsets)]
            n_clusters_made = len(set([k for j in clusters for k in j]))
            if n_clusters_made < n_subsets:
                raise ValueError("Cannot cluster into {} subsets, please try a smaller n_subset number, such as {}.".
                                 format(n_subsets, n_clusters_made))

            cluster_mins = [np.min(c) for c in clusters]
            bound = np.sort(np.array(cluster_mins))

            # Distribute into curriculum subsets, and package into global adjusted returnable array, optionally aux too
            other_bounds = range(n_subsets - 1)
            for i in range(len(densities)):

                # Check if the most 'clean'
                if densities[i] >= bound[n_subsets - 1]:
                    all_clustered_labels[batch_dist_list[i]] = 0
                # Else, check the others
                else:
                    for j in other_bounds:
                        if bound[j] <= densities[i] < bound[j + 1]:
                            all_clustered_labels[batch_dist_list[i]] = len(bound) - j - 1

        if verbose:
            print("Clustering {} of {} categories into {} curriculum subsets ({:.2f} secs).".format(
                cluster_idx + 1, len(unique_categories), n_subsets, time.time() - t0))

    if (all_clustered_labels > 0).all():
        raise ValueError("A clustering error occurred: incomplete labels detected.")

    return all_clustered_labels, auxiliary_info


class CurriculumClustering(BaseEstimator, ClusterMixin):

    def __init__(self, n_subsets=3, method='default', density_t=0.6, verbose=False,
                 dim_reduce=256, batch_max=30000, random_state=None, calc_auxiliary=False):
        self.n_subsets = n_subsets
        self.method = method
        self.density_t = density_t
        self.verbose = verbose
        self.output_labels = None
        self.random_state = random_state
        self.dim_reduce = dim_reduce
        self.batch_max = batch_max
        self.calc_auxiliary = calc_auxiliary

    def fit(self, X, y):
        X = check_array(X, accept_sparse='csr')
        check_consistent_length(X, y)
        self.output_labels, _ = cluster_curriculum_subsets(X, y, **self.get_params())
        return self

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.output_labels