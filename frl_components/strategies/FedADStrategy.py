import flwr as fl
import numpy as np
from typing import List, Tuple, Dict, Optional
from flwr.common import FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy
from scipy.spatial.distance import cdist

from frl_components.data import Loader
from sklearn.metrics import roc_auc_score

from frl_components.models.util import persist_model
from .validation import persist_validation_results, validate_autoencoder

class FedADStrategy(fl.server.strategy.FedProx):
    '''
    '''


    def __init__(self, data_loader: Loader, model_generator_fn, memory_aggregation_method=None, *args, **kwargs):
        '''
        
        Params:
            memory_aggregation_method -- None, 'random', 'top_n', 'greedy', 'clustered'
        '''
        # super().__init__(*args, **kwargs)
        super().__init__(proximal_mu=0.01, *args, **kwargs)
        # 0.1 according to https://arxiv.org/pdf/2408.04442

        self.data_loader = data_loader

        self.model_generator_fn = model_generator_fn

        self.memory_aggregation_method = memory_aggregation_method
        self.validation_results = []


    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ):
        
        if len(results) == 0:
            return None, {}

        # Aggregate metrics and params using FedProx        
        params_agr, metrics_agr = super().aggregate_fit(server_round, results, failures)
        print(fl.common.parameters_to_ndarrays(params_agr))

        if self.memory_aggregation_method is None:
            # Regular FedProx result
            return params_agr, metrics_agr

        memory_matrices, middle_index = self._extract_memory_matrices(results)

        # Aggregate custom matrix
        aggregated_memory_matrix = self.aggregate_memory_matrix(memory_matrices)
        # print("Aggregated memory matrix has NaNs? ", np.isnan(aggregated_memory_matrix).any())

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
        model = self.model_generator_fn(self.data_loader.input_dim)
        # model.build(input_shape=(None, self.data_loader.input_dim))
        model(X[:1])
        model.set_weights(params_ndarr)

        experiment_name = f'{self.data_loader.experiment_name}-{self.memory_aggregation_method}'

        persist_model(model, f'fedad_{server_round}', experiment_name)

        validation_res = validate_autoencoder(model, X, y)

        validation_res['round'] = server_round

        self.validation_results.append(validation_res)
        print("Validation results: ", validation_res, "\n")


        persist_validation_results(self.validation_results, experiment_name)

        return validation_res['roc_auc'], {}


    def _extract_memory_matrices(self, results: List[Tuple[ClientProxy, FitRes]]):
        '''
        The memory matrix is located in the middle of the parameters array.
        '''

        params_list = [
            fl.common.parameters_to_ndarrays(fit_res.parameters)
            for _, fit_res in results
        ]

        middle_index = len(params_list[0]) // 2

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



    def _validate_matrix_shapes(self, matrices: List[np.ndarray]) -> None:
        shapes = [matrix.shape for matrix in matrices]
        if len(set(shapes)) > 1:
            raise ValueError("Matrices have different shapes: {}".format(shapes))