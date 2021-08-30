# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import copy
import unittest

import torch

from fcos_core.config import cfg as g_cfg
from fcos_core.modeling import registry
from utils import load_config

# overwrite configs if specified, otherwise default config is used
BACKBONE_CFGS = {
    "R-50-FPN-RETINANET": "/home/citybuster/Projects/FCOS/configs/fcos_nuscenes/fcos_imprv_R_50_FPN_1x_ADD.yaml",
}


class TestBackbones(unittest.TestCase):
    def test_build_backbones(self):
        """Make sure backbones run"""

        # self.assertGreater(len(registry.BACKBONES), 0)

        self.assertGreater(len(registry.BACKBONES), 0)

        for name, backbone_builder in registry.BACKBONES.items():
            print('Testing {}...'.format(name))
            if name in BACKBONE_CFGS:
                cfg = load_config(BACKBONE_CFGS[name])
            else:
                # Use default config if config file is not specified
                cfg = copy.deepcopy(g_cfg)
            backbone = backbone_builder(cfg)

            # make sures the backbone has `out_channels`
            self.assertIsNotNone(
                getattr(backbone, 'out_channels', None),
                'Need to provide out_channels for backbone {}'.format(name)
            )

            N, C_in, H, W = 2, 3, 224, 256
            img_input = torch.rand([N, C_in, H, W], dtype=torch.float32)
            pc_img_input = torch.rand([N, C_in, H, W], dtype=torch.float32)
            out = backbone((img_input, pc_img_input))
            for cur_out in out:
                self.assertEqual(
                    cur_out.shape[:2],
                    torch.Size([N, backbone.out_channels])
                )


if __name__ == "__main__":
    unittest.main()
