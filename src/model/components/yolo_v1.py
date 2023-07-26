import torch
import torch.nn as nn
import hydra

import pyrootutils
from omegaconf import DictConfig, OmegaConf
import hydra

path = pyrootutils.find_root(
    search_from=__file__, indicator=".project-root")
config_path = str(path / "configs" / "model")
output_path = path / "outputs"
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
print("paths", path, config_path, output_path)

architecture_config =[
    (7, 64, 2, 3), # kernel_size, filters, stride, padding
    "M", # MaxPool2d(kernel_size=2, stride=2)
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1,  1),
    (3, 1024, 1, 1),
]
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, bias=False, **kwargs)
        self.batch_norm = nn.BatchNorm2d(num_features=out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)
    def forward(self, x):
        return self.leakyrelu(self.batch_norm(self.conv(x)))
    
class YoloV1(nn.Module):
    def __init__(self, in_channels = 3, architecture = None, **kwargs) -> None:
        super().__init__()
        self.in_channels = in_channels
        # self.out_size = out_size
        self.architecture = architecture
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))
    
    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

            if type(x) == tuple:
                layers += [ConvBlock(in_channels=in_channels, out_channels=x[1], kernel_size=x[0], stride=x[2], padding=x[3])]
                in_channels = x[1]

            if type(x) == list:
                num_repeat = x[len(x)-1]
                for _ in range(num_repeat):
                    for conv in x[:len(x)-1]:
                        layers += [
                            ConvBlock(in_channels=in_channels, out_channels=conv[1], 
                                      kernel_size=conv[0], stride=conv[2], padding=conv[3])
                            ]
                        in_channels = conv[1]
            self.in_channels = in_channels

        return nn.Sequential(*layers)
        
    def _create_fcs(self, grid_cell, num_boxes, num_classes):
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = self.in_channels * grid_cell * grid_cell, out_features=4096),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(in_features=4096, out_features = grid_cell * grid_cell * ((num_boxes * 5) + num_classes))
        )

@hydra.main(version_base="1.3", config_path=config_path, config_name="yolo_v1")
def main(cfg: DictConfig):
    num_classes = 20
    grid_cell = 7
    num_boxes = 2
    image_size = 448
    x = torch.randn((2, 3, image_size, image_size))
    print(OmegaConf.to_yaml(cfg.net))
    # # model = YoloV1(architecture=architecture_config, grid_cell=grid_cell, num_boxes=num_boxes, num_classes=num_classes)
    # model = hydra.utils.instantiate(cfg.get('net'))
    model = YoloV1(architecture=architecture_config, grid_cell = int(cfg.net.grid_cell),
                    num_boxes = int(cfg.net.num_boxes), num_classes = int(cfg.net.num_classes))
    print(model)
    print(model(x).shape)

if __name__ == "__main__":
    main()