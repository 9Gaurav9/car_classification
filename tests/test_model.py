import torch
import unittest
from car_classification.model import get_model

class TestModel(unittest.TestCase):
    def test_model_output_shape(self):
        model = get_model(num_classes=196)
        dummy_input = torch.randn(1, 3, 224, 224)
        output = model(dummy_input)
        self.assertEqual(output.shape, (1, 196))

if __name__ == '__main__':
    unittest.main()
