import numpy as np
import pytest

from frl_components.strategies.FedADStrategy import FedADStrategy

class MockFedADStrategy(FedADStrategy):
    def __init__(self):
        pass # Override the constructor to avoid dependencies


def test_facility_location_selection_shape():
    """
    Tests that _facility_location_selection runs and returns an array
    of the expected shape (N, features).
    """
    NUM_VECTORS = 10
    FEATURE_DIM = 20
    all_memory_vectors = np.random.rand(NUM_VECTORS, FEATURE_DIM)
    
    N_SELECTION = 3
    
    strategy = MockFedADStrategy()
    
    try:
        selected_vectors = strategy._facility_location_selection(
            all_memory_vectors, 
            N=N_SELECTION, 
            distance_f='cosine'
        )
    except Exception as e:
        pytest.fail(f"_facility_location_selection failed to run: {e}")

    assert isinstance(selected_vectors, np.ndarray), "Result is not a numpy array"
    
    expected_shape = (N_SELECTION, FEATURE_DIM)
    breakpoint()
    assert selected_vectors.shape == expected_shape, \
        f"Expected shape {expected_shape}, but got {selected_vectors.shape}"
