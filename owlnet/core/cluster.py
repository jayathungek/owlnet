import torch
import torch.nn.functional as F

from owlnet.core.utils import reduce_dimensions


class DBSCANBase:
    def __init__(self, config, eps, minPts, metric="euclidean"):
        self._eps = eps
        self._minPts = minPts
        self._metric = metric
        self.config = config

    def fit(self, X):
        pass

    def predict(self, X):
        pass


class DBSCANTorch(DBSCANBase): # TODO: Why is this slower than numpy?
    def fit(self, X):
        if type(X) is not torch.Tensor:
            self._X = torch.tensor(X).to(self.config['device'])
        else:
            self._X = X.to(self.config['device'])

        dist_matrix = self._get_distance_matrix()
        cluster = 1
        n_objs = self._X.shape[0]
        labels = torch.zeros(n_objs).to(self.config['device'])

        for i in range(n_objs):
            if not labels[i]:
                neighbors = self._get_nearest_neighbours(dist_matrix[i])
                if (
                    len(neighbors) > self._minPts
                ):  # not >= because self is included in neighbours
                    labels[i] = cluster
                    labels = self._expand_cluster(
                        dist_matrix, neighbors, cluster, labels
                    )
                    cluster += 1

        self._centroids = self._get_centroids(labels, cluster - 1)
        self._dist_matrix = dist_matrix
        return labels

    def predict(self, X, as_numpy=True):
        X = X.unsqueeze(0).to(self.config['device'])
        centroids = self._centroids[:, None, ...]
        all_diff = X - centroids
        all_dist = torch.einsum("...k, ...k -> ...", all_diff, all_diff).sqrt()
        preds = torch.argmin(all_dist, dim=0)
        if as_numpy:
            return preds.cpu().numpy()
        else:
            return preds
    
    def _get_distance_matrix(self):
        if self._metric == "euclidean":
            # reimplement cdist
            diff_mat = self._X[:, None] - self._X[None, :]
            diff_mat = torch.einsum("...k, ...k -> ...", diff_mat, diff_mat).sqrt()
            return diff_mat / diff_mat.max()
        elif self._metric == "cosine":
            diff_mat = torch.matmul(self._X, self._X.T)
            diff_mat = (diff_mat + 1.0) / 2
            return diff_mat
        else:
            assert False, f"Metric {self._metric} not implemented for _get_distance_matrix"

    def _get_centroids(self, labels, last_cluster):
        if last_cluster == 0:
            centroids_to_stack = [self._X[0]]
            
        else:
            centroids_to_stack = []
            for i in range(1, last_cluster + 1):
                idx = torch.where(labels == i)[0].cpu()
                mean = self._X[idx].mean(dim=0)
                centroids_to_stack.append(mean)

        return torch.stack(centroids_to_stack)

    def _get_nearest_neighbours(self, x):
        if self._metric == "euclidean":
            return torch.where(x <= self._eps)[0]
        elif self._metric == "cosine":
            return torch.where(x >= self._eps)[0]
        else:
            assert False, f"Metric {self._metric} not implemented for _get_nearest_neighbours"

    def _expand_cluster(self, X, neighbors, cluster, labels):
        for neighbor in neighbors:
            if not labels[neighbor]:  # if point is unassigned
                neighbors_of_neighbor = self._get_nearest_neighbours(X[neighbor])
                if len(neighbors_of_neighbor) >= self._minPts:  # if point is core
                    labels[neighbor] = cluster
                    labels = self._expand_cluster(
                        X, neighbors_of_neighbor, cluster, labels
                    )
        return labels


def get_owlet_clusters(config, embeddings):
    embeddings = F.normalize(embeddings, p=2, dim=1)
    dbscan = DBSCANTorch(config, eps=0.2, minPts=200, metric="euclidean")
    dbscan.fit(embeddings).cpu()
    clusters = torch.tensor(dbscan.predict(embeddings))
    unique_clusters = torch.unique(clusters)
    indices = torch.tensor(list(range(len(embeddings))))
    embeddings = reduce_dimensions(embeddings)


    ret_clusters = []
    ret_indices = []
    for cluster_id in unique_clusters:
        cluster_select = clusters == cluster_id
        cluster_points = embeddings[cluster_select]
        cluster_indices = indices[cluster_select]
        ret_clusters.append(cluster_points)
        ret_indices.append(cluster_indices)

    return ret_clusters, ret_indices



