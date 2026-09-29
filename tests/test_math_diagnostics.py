import torch
import torch.nn as nn
import unittest

class QRCodeLayer(nn.Module):
    def __init__(self, num_codes=16, dim=8):
        super().__init__()
        self.codebook = nn.Parameter(torch.randn(num_codes, dim))

    def forward(self, x, break_gradients=False):
        distances = torch.cdist(x, self.codebook)
        indices = torch.argmin(distances, dim=-1)
        quantized = self.codebook[indices]
        if break_gradients:
            return quantized
        else:
            return x + (quantized - x).detach()

class AbacusLayer(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.bead_weights = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return x + self.bead_weights(x)

class OuijaBoardLayer(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.pointer_force = nn.Linear(dim, dim, bias=False)
        self.decay = 0.7

    def forward(self, x, current_pointer):
        return (self.decay * current_pointer) + ((1.0 - self.decay) * self.pointer_force(x))

class TestMathematicalDiagnostics(unittest.TestCase):
    def test_gradient_flow_in_harmonious_order(self):
        mock_input = torch.randn(1, 8, requires_grad=True)
        mock_target = torch.randn(1, 8)
        initial_pointer = torch.zeros(1, 8)

        qr = QRCodeLayer()
        abacus = AbacusLayer()
        ouija = OuijaBoardLayer()

        out_qr = qr(mock_input, break_gradients=False)
        out_abacus = abacus(out_qr)
        final_pointer = ouija(out_abacus, initial_pointer)

        loss = torch.mean((final_pointer - mock_target) ** 2)
        loss.backward()

        self.assertIsNotNone(mock_input.grad)
        self.assertGreater(torch.norm(mock_input.grad).item(), 0.0)
        self.assertIsNotNone(abacus.bead_weights.weight.grad)
        self.assertGreater(torch.norm(abacus.bead_weights.weight.grad).item(), 0.0)
        self.assertIsNotNone(ouija.pointer_force.weight.grad)
        self.assertGreater(torch.norm(ouija.pointer_force.weight.grad).item(), 0.0)

if __name__ == "__main__":
    unittest.main()
