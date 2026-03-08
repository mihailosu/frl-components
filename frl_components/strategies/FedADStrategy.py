import flwr as fl
import numpy as np
from typing import List, Tuple, Dict, Optional
from flwr.common import FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy
from scipy.spatial.distance import cdist

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

import pandas as pd

from sklearn.metrics import roc_auc_score

from ..models.util import persist_model, persist_memory
from .validation import persist_validation_results, validate_autoencoder

from apricot import FacilityLocationSelection, MaxCoverageSelection, SumRedundancySelection

class FedADStrategy(fl.server.strategy.FedProx):
    '''
    '''


    def __init__(self, data_loader, model_generator_fn, memory_aggregation_method=None, results_path=None, mu=0.01, agg_memory_at_step=None, *args, **kwargs):
        '''
        
        Params:
            memory_aggregation_method -- None, 'random', 'top_n', 'greedy', 'clustered'
        '''
        # super().__init__(*args, **kwargs)
        super().__init__(proximal_mu=mu, *args, **kwargs)
        # 0.1 according to https://arxiv.org/pdf/2408.04442

        self.data_loader = data_loader

        self.model_generator_fn = model_generator_fn

        self.memory_aggregation_method = memory_aggregation_method
        self.validation_results = []

        self.results_path = results_path

        self.model = model_generator_fn(self.data_loader.input_dim)
        self.model(self.data_loader.get_test_data()[0][:1])

        self.best_f1 = 0.0
        self.best_parameters = None

        self.agg_memory_at_step = agg_memory_at_step

        try:
            self.memory_location_index = [w.name for w in self.model.weights].index('memory')
        except:
            self.memory_location_index = None
        


    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ):
        
        if len(results) == 0:
            return None, {}

        results.sort(key=lambda x: int(x[0].cid))

        # Aggregate metrics and params using FedProx        
        params_agr, metrics_agr = super().aggregate_fit(server_round, results, failures)
        # print(fl.common.parameters_to_ndarrays(params_agr))

        # if self.memory_aggregation_method is None or server_round % 5 == 1:

        should_skip_memory_agg = self.agg_memory_at_step is not None and server_round % self.agg_memory_at_step != 0

        if self.memory_aggregation_method is None or should_skip_memory_agg:
            # Regular FedProx result
            return params_agr, metrics_agr

        memory_matrices, middle_index = self._extract_memory_matrices(results)

        all_rows = np.vstack(memory_matrices)
        client_indices = [x // len(memory_matrices) for x in range(len(all_rows))]

        # Aggregate custom matrix
        aggregated_memory_matrix = self.aggregate_memory_matrix(memory_matrices)
        # print("Aggregated memory matrix has NaNs? ", np.isnan(aggregated_memory_matrix).any())

        # chosen_indices = [row in aggregated_memory_matrix for row in all_rows]

        memory_df = pd.DataFrame(all_rows)
        memory_df['client_id'] = client_indices
        # memory_df['chosen'] = chosen_indices
        chosen_df = pd.DataFrame(aggregated_memory_matrix)
        persist_memory(memory_df, chosen_df, f'memory_{server_round}', self.results_path)

        # Replace the memor matrix with the aggregated one
        # 1. Convert params array back to ndarray
        params_ndarr = fl.common.parameters_to_ndarrays(params_agr)

        # print(aggregated_memory_matrix)
        # np.savetxt(f'memory_{server_round}.txt', aggregated_memory_matrix)


        # 2. Insert aggregated memory matrix
        params_ndarr[middle_index] = aggregated_memory_matrix
        # 3. Convert back to param array
        params_agr = fl.common.ndarrays_to_parameters(params_ndarr)

        return params_agr, metrics_agr


    def evaluate(self, server_round: int, parameters: Parameters): 

        if server_round == 0:
            return 0, {}

        print("Server round: ", server_round)
        params_ndarr = fl.common.parameters_to_ndarrays(parameters)
        
        #contains_nan = any(np.isnan(weight).any() for weight in params_ndarr)
        #print("Number of layers in the model: ", len(params_ndarr))
        #print("Do weights contain NaNs? ", contains_nan)

        X, y = self.data_loader.get_test_data()

        # Instantiate model somehow
        # NOTE: Moved model instantiation to init
        # model = self.model_generator_fn(self.data_loader.input_dim)
        # # model.build(input_shape=(None, self.data_loader.input_dim))
        # model(X[:1])
        self.model.set_weights(params_ndarr)

        experiment_name = f'{self.data_loader.experiment_name}-{self.memory_aggregation_method}'

        results_path = self.results_path
        if not results_path:
            results_path = experiment_name

        persist_model(self.model, f'fedad_{server_round}', results_path)

        validation_res = validate_autoencoder(self.model, X, y)

        validation_res['round'] = server_round

        self.validation_results.append(validation_res)
        print("Validation results: ", validation_res, "\n")

        persist_validation_results(self.validation_results, results_path)

        curr_f1 = validation_res['f1']
        if curr_f1 > self.best_f1:
            self.best_parameters = params_ndarr
            self.best_f1 = curr_f1

        return validation_res['f1'], {}


    def _extract_memory_matrices(self, results: List[Tuple[ClientProxy, FitRes]]):
        '''
        The memory matrix is located in the middle of the parameters array.
        '''

        params_list = [
            fl.common.parameters_to_ndarrays(fit_res.parameters)
            for _, fit_res in results
        ]

        # It's not always in the middle...
        middle_index = self.memory_location_index if self.memory_location_index else len(params_list[0]) // 2

        memory_matrices = [
            params[middle_index]
            for params in params_list
        ]

        return memory_matrices, middle_index


    def aggregate_memory_matrix(self, matrices: List[np.ndarray]) -> np.ndarray:

        self._validate_matrix_shapes(matrices)

        N = matrices[0].shape[0]

        # Concatenate all matrices
        all_rows = np.vstack(matrices)

        if self.memory_aggregation_method == 'random':
            return self._random_vector_selection(all_rows, N)
        elif self.memory_aggregation_method == 'top_n':
            return self._top_n_by_distance(all_rows, N)
        elif self.memory_aggregation_method == 'greedy':
            return self._greedy_selection(all_rows, N)
        elif self.memory_aggregation_method == 'facility_location':
            return self._facility_location_selection(all_rows, N)
        elif self.memory_aggregation_method == 'max_coverage':
            return self._max_coverage_selection(all_rows, N)
        elif self.memory_aggregation_method == 'sum_redundancy':
            return self._sum_redundancy_selection(all_rows, N)
        elif self.memory_aggregation_method == 'kmeans':
            return self._kmeans_prototype_selection(all_rows, N)


    def _random_vector_selection(self, all_memory_vectors, N) -> np.ndarray:
        '''
        Random selection of memory vectors.
        '''
        np.random.shuffle(all_memory_vectors)
        return all_memory_vectors[:N]
 

    def _top_n_by_distance(self, all_memory_vectors, N) -> np.ndarray:
        '''
        Calculates the cosine distance between all pairs of vectors. Sorts all vectors
        by their average distance, and selects the top N which are on average the most
        distant from other vectors.
        '''
        # Calculate cosine distances between all rows
        cosine_distances = cdist(all_memory_vectors, all_memory_vectors, metric='cosine')
        
        # Calculate the average distance for each row
        avg_distances = np.mean(cosine_distances, axis=1)

        # Get the indices of the N rows with the highest average distances
        most_different_indices = np.argsort(avg_distances)[-N:]

        print("Selection by distance: ", avg_distances[most_different_indices])
        
        # Return the N most different rows
        return all_memory_vectors[most_different_indices]


    def _greedy_selection(self, all_memory_vectors, N) -> np.ndarray:
        '''
        Calculates the cosine distance between all pairs of vectors. After this,
        an initial vector is chosen. Each subsequent vector is added in such a way
        to maximise the distance between those already selected.
        '''

        cosine_distances = cdist(all_memory_vectors, all_memory_vectors, metric='cosine')

        # Select first vector randomly
        selected_indices = [np.random.randint(len(all_memory_vectors))]
        for _ in range(N - 1):
            min_dist_to_selected = cosine_distances[selected_indices].min(axis=0)

            next_index = np.argmax(min_dist_to_selected)

            selected_indices.append(next_index)

        return all_memory_vectors[selected_indices]


    def _kmeans_prototype_selection(self, all_memory_vectors, N, distance_f='cosine') -> np.ndarray:
        '''
        Select a subset of N most informative vectors (prototypes) using K-Means clustering.
        If distance_f is 'cosine', vectors are normalized to ensure centroids represent 
        directional prototypes.
        '''
        # 1. Handle the Cosine Distance requirement
        # K-Means optimizes Euclidean distance. For Cosine, we project onto the unit sphere.
        if distance_f == 'cosine':
            all_memory_vectors = normalize(all_memory_vectors, axis=1)

        # 2. Fit K-Means
        # n_clusters=N means we want to find N representative prototypes
        kmeans = KMeans(
            n_clusters=N, 
            init='k-means++', 
            n_init=10, 
            random_state=42
        )
        kmeans.fit(all_memory_vectors)

        # 3. Get the Centroids
        # These centroids are the new 'Global Memory Matrix' rows
        global_memory_matrix = kmeans.cluster_centers_

        # 4. Re-normalize for Cosine Similarity (if necessary)
        # Centroids of points on     a sphere are inside the sphere; 
        # we push them back to the surface to maintain cosine similarity properties.
        if distance_f == 'cosine':
            global_memory_matrix = normalize(global_memory_matrix, axis=1)

        return global_memory_matrix.astype(np.float32)  


    def _facility_location_selection(self, all_memory_vectors, N, distance_f='cosine') -> np.ndarray:
        '''
        Select the a subset of most informative vectors based on submodular optimization,
        specifically, the Facility Location Selection method.
        '''
        return FacilityLocationSelection(N, metric=distance_f, random_state=42).fit_transform(all_memory_vectors)


    def _max_coverage_selection(self, all_memory_vectors, N, distance_f='cosine') -> np.ndarray:
        '''
        Select the a subset of most informative vectors based on submodular optimization,
        specifically, the Facility Location Selection method.
        '''
        return MaxCoverageSelection(N, random_state=42).fit_transform(all_memory_vectors)

    def _sum_redundancy_selection(self, all_memory_vectors, N, distance_f='cosine') -> np.ndarray:
        '''
        Select the a subset of most informative vectors based on submodular optimization,
        specifically, the Facility Location Selection method.
        '''
        return SumRedundancySelection(N, metric='cosine', random_state=42).fit_transform(all_memory_vectors)

    def _validate_matrix_shapes(self, matrices: List[np.ndarray]) -> None:
        shapes = [matrix.shape for matrix in matrices]
        if len(set(shapes)) > 1:
            raise ValueError("Matrices have different shapes: {}".format(shapes))
