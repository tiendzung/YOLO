import torch
from torch.utils.data import DataLoader, Dataset, random_split
from pytorch_lightning import LightningDataModule
from typing import Any, Dict, Optional

import albumentations as A
from albumentations import Compose
from albumentations.pytorch.transforms import ToTensorV2
import torchvision.transforms as transforms
import torchvision

import numpy as np
from PIL import Image, ImageDraw
import pandas as pd
from matplotlib import pyplot as plt

import os
import hydra

class VOCDataset(Dataset):
    def __init__(
        self,
        csv_file = '/Users/tiendzung/Project/YOLO/data/voc_dataset.csv',
        img_dir = '/Users/tiendzung/Project/YOLO/data/images',
        label_dir = '/Users/tiendzung/Project/YOLO/data/labels',
        S: int = 7,
        B: int = 2,
        C: int = 20,
    ):
        super().__init__()
        self.annotations = pd.read_csv(csv_file)
        self.csv_file = csv_file
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.S = S
        self.B = B
        self.C = C
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []
        with open(label_path, 'r') as f:
            for label in f.readlines():
                class_label, x, y, w, h = label.split()
                boxes.append([int(class_label), float(x), float(y), float(w), float(h)])

        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        img = Image.open(img_path) ## No transfromation

        ## Convert to grid cell
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        for box in boxes:
            class_label, x, y, w , h = box

            ## i, j represents the cell row and cell column
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i

            '''
            width_cell = widyh_pixel/(image_width_pixel/S) = width_pixel*S/image_width_pixel = w*s
            '''
            width_cell, height_cell = (w*self.S, h*self.S)
            if label_matrix[i, j, self.C] == 0:
                label_matrix[i, j, self.C] = 1
                label_matrix[i, j, self.C + 1 : self.C + 5] = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                label_matrix[i, j, class_label] = 1
        
        return img, label_matrix

class TransformDataset(Dataset):
    def __init__(
        self, 
        dataset: VOCDataset, 
        transform = None,
    ) -> None:
        super().__init__()
        self.dataset = dataset

        if transform is not None:
            self.transform = transform
        else:
            self.transform = Compose([
                A.Resize(448, 448),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
            # self.transform = transforms.Compose([
            #     transforms.Resize((224, 224)),
            #     transforms.PILToTensor(),
            #     transforms.ConvertImageDtype(torch.float),
            #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # ])

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index) -> Any:
        img, label_matrix = self.dataset[index]
        img = np.array(img)
        transformed = self.transform(image = img)
        return transformed["image"], label_matrix

class VOCDataModule(LightningDataModule):
    def __init__(
        self,
        dataset: VOCDataset = None,
        transform_train = None,
        transform_val = None,
        train_val_test_split = [83, 10, 10],
        batch_size: int = 16,
        num_workers: int = 1,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.dataset = dataset
        self.transform_train = transform_train
        self.transform_val = transform_val
        self.train_val_test_split = train_val_test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.data_train = None
        self.data_val = None
        self.data_test = None

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            # self.data_train = self.data_train
            # self.data_test = self.data_test
            self.data_train, self.data_val, self.data_test = random_split(
                    dataset=self.dataset,
                    lengths=self.train_val_test_split,
                    generator=torch.Generator().manual_seed(42),
                )
            self.data_train = TransformDataset(self.data_train, self.transform_train)
            self.data_val = TransformDataset(self.data_val, self.transform_val)
            self.data_test = TransformDataset(self.data_test,self.transform_val)


    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def predict_dataloader(self):
        return self.test_dataloader()

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass

if __name__ == "__main__":
    import pyrootutils
    from omegaconf import DictConfig
    import hydra
    import numpy as np
    from PIL import Image, ImageDraw
    from tqdm import tqdm

    path = pyrootutils.find_root(
        search_from=__file__, indicator=".project-root")
    config_path = str(path / "configs" / "data")
    output_path = path / "outputs"
    print("root", path, config_path)
    pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

    def test_dataset(cfg: DictConfig):
        dataset: VOCDataset = hydra.utils.instantiate(cfg.dataset)
        # dataset = dataset(data_dir=cfg.data_dir)
        print("dataset", len(dataset))
        image, label = dataset.__getitem__(0)
        print("image", image.size, "label", label.shape)
        # annotated_image = VOCDataset.annotate_image(image, label)
        # annotated_image.save(output_path / "test_dataset_result.png")

    def test_datamodule(cfg: DictConfig):
        datamodule: VOCDataModule = hydra.utils.instantiate(cfg)
        datamodule.prepare_data()
        datamodule.setup()
        loader = datamodule.train_dataloader()
        bx, by = next(iter(loader))
        print("n_batch", len(loader), bx.shape, by.shape, type(by))
        # annotated_batch = TransformDataset.annotate_tensor(bx, by)
        # print("annotated_batch", annotated_batch.shape)
        # torchvision.utils.save_image(annotated_batch, output_path / "test_datamodule_result.png")
        
        for bx, by in tqdm(datamodule.train_dataloader()):
            pass
        print("training data passed")

        for bx, by in tqdm(datamodule.val_dataloader()):
            pass
        print("validation data passed")

        for bx, by in tqdm(datamodule.test_dataloader()):
            pass
        print("test data passed")

    @hydra.main(version_base="1.3", config_path=config_path, config_name="voc")
    def main(cfg: DictConfig):
        # print(cfg)
        test_dataset(cfg)
        test_datamodule(cfg)

    main()
