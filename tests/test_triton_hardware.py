import torch
import unittest
from timemeshin.layers.triton_lookup import triton_rvq_snapping

class TestTimeMeshinLookup(unittest.TestCase):
    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_elements = 64
        self.embedding_dimension = 32
        self.codebook_slots = 16
        self.test_vectors = torch.randn(self.batch_elements, self.embedding_dimension, device=self.device)
        self.codebook_weights = torch.randn(self.codebook_slots, self.embedding_dimension, device=self.device)

    def test_numerical_precision_parity(self):
        outputs, indices = triton_rvq_snapping(self.test_vectors, self.codebook_weights)
        distances = torch.cdist(self.test_vectors.unsqueeze(1), self.codebook_weights.unsqueeze(0)).squeeze(1)
        expected_indices = torch.argmin(distances, dim=-1)
        expected_outputs = self.codebook_weights[expected_indices.long()]

        index_mismatches = torch.sum(indices.long() - expected_indices).item()
        coordinate_closeness = torch.allclose(outputs, expected_outputs, atol=1e-5)

        self.assertEqual(index_mismatches, 0, "Grid indexing drifted from reference.")
        self.assertTrue(coordinate_closeness, "Output coordinates mismatched reference tensor profiles.")

if __name__ == "__main__":
    unittest.main()
