import unittest
import logging
import sys

import torch
from omegaconf import OmegaConf

from qdl.layers.quanvolution import Quanvolution

logger = logging.getLogger(__name__)

class TestQuanvolution(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._batch = torch.randn(5, 1, 28, 28)
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)

    @classmethod
    def tearDownClass(cls):
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.removeHandler(stream_handler)

    def test_forward(self):
        conf = OmegaConf.create({
            "n_blocks": 1,
            "quanvolution_type": "barren",
            "stride": 2,
            "out_channels": 4
        })
        quanv = Quanvolution(conf)
        out = quanv(self._batch)

        # outputs 4 values per channel (one per qubit)
        assert out.shape == (5, 16, 14, 14)
        assert out.requires_grad

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    unittest.main()