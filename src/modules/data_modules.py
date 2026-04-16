from typing import List, Dict
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from lightning.pytorch import LightningDataModule
import src.utils.data_utils as data_utils
import src.utils.data_transforms as data_transforms
import torchvision
import pickle
from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate

torchvision.disable_beta_transforms_warning()


class VertexModelDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        split: str = "train",
        quantization_bits: int = 8,
        augmentation: bool = False,
    ) -> None:
        """Initializes Image Dataset

        Args:
            dataset_path: Where model files along with renderings are located
            augmentation: Whether need data augmentation
        """
        self.augmentation = augmentation
        self.quantization_bits = quantization_bits
        self.split = split

        with open(dataset_path, "rb") as pkl_data:
            data = pickle.load(pkl_data)
            self.cached_pts = data[f"pts_{split}"]
            self.cached_vertices = data[f"vertices_{split}"]
            self.cached_faces = data[f"faces_{split}"]
            self.names = data[f"name_{split}"]

        if split == "train" and augmentation:
            self.train_transforms = T.Compose(
                [
                    data_transforms.PointcloudRotate(),
                    data_transforms.PointcloudScale(),
                    data_transforms.PointcloudTranslate(),
                ]
            )
        print(f"VertexModelDataset {split} Number: {len(self.cached_pts)}")

    def __len__(self) -> int:
        return len(self.cached_pts)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """Gets image object along with associated mesh

        Args:
            idx: Index of image to retrieve

        Returns:
            mesh_dict: Dictionary containing vertices, faces of .obj file and image tensor
        """
        pts = self.cached_pts[idx]
        vertices = self.cached_vertices[idx]
        faces = self.cached_faces[idx]
        filename = self.names[idx]

        pts = torch.from_numpy(pts).to(torch.float32)
        vertices = torch.from_numpy(vertices).to(torch.float32)

        # 数据增强
        if self.split == "train" and self.augmentation:
            item_dict = self.train_transforms({"vertices": vertices, "pts": pts})
            vertices, pts = item_dict["vertices"], item_dict["pts"]

        vertices, vertices_scale = data_utils.normalize_vertices_scale(
            vertices, return_scale=True
        )
        pts = (pts / vertices_scale).to(torch.float32)

        pts = torch.clamp(pts, -0.5, 0.5)
        vertices = torch.clamp(vertices, -0.5, 0.5)

        vertices, faces = data_utils.quantize_process_mesh(
            vertices, faces, self.quantization_bits
        )
        pts = data_utils.quantize_verts(pts, self.quantization_bits)

        vertices = vertices.to(torch.int32)
        pts = pts.to(torch.int32)

        data_dict = {
            "vertices": vertices,
            "pts": pts,
            "filename": filename,
        }
        return data_dict


class VertexDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_path: str,
        batch_size: int = 32,
        quantization_bits: int = 8,
        augmentation: bool = True,
    ) -> None:
        """
        Args:
            dataset_path: Root directory for shapenet dataset
            collate_method: Whether to collate vertices or faces
            batch_size: How many 3D objects in one batch
            quantization_bits: How many bits we are using to quantize the vertices
            apply_random_shift: Whether or not we're applying random shift to vertices for face model
            shuffle_vertices: Whether or not we're shuffling the order of vertices during batch generation for face model
        """
        super().__init__()

        self.batch_size = batch_size
        self.quantization_bits = quantization_bits

        self.train_dataset = VertexModelDataset(
            dataset_path=dataset_path,
            split="train",
            quantization_bits=quantization_bits,
            augmentation=augmentation,
        )
        self.val_dataset = VertexModelDataset(
            dataset_path=dataset_path,
            split="val",
            quantization_bits=quantization_bits,
            augmentation=False,
        )

    def collate_vertex_model_batch(
        self, ds: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Applying padding to different length vertex sequences so we can batch them
        Args:
            ds: List of dictionaries where each dictionary has information about a 3D object
        Returns
            vertex_model_batch: A single dictionary which represents the whole batch
        """
        vertex_model_batch = {}
        batch_size = len(ds)

        # vertices
        num_vertices_list = [data_dict["vertices"].shape[0] for data_dict in ds]
        max_vertices = max(num_vertices_list)
        vertices_flat = torch.zeros(
            [batch_size, max_vertices * 3 + 1], dtype=torch.int32
        )
        vertices_flat_mask = torch.zeros_like(vertices_flat, dtype=torch.int32)

        # pointclouds
        pc_sparse_list = []

        # filenames
        filenames_list = []

        for i, element in enumerate(ds):
            # vertices
            vertices = element["vertices"]
            initial_vertex_size = vertices.shape[0]
            vertices_padding_size = max_vertices - initial_vertex_size
            vertices_permuted = torch.stack(
                [vertices[..., 2], vertices[..., 1], vertices[..., 0]], dim=-1
            )
            curr_vertices_flat = vertices_permuted.reshape([-1])
            vertices_flat[i] = F.pad(
                curr_vertices_flat + 1, [0, vertices_padding_size * 3 + 1]
            )[None]
            vertices_flat_mask[i] = torch.zeros_like(
                vertices_flat[i], dtype=torch.float32
            )
            vertices_flat_mask[i, : initial_vertex_size * 3 + 1] = 1

            pts = element["pts"]
            pc_sparse_list.append(
                SparseTensor(
                    coords=pts,
                    feats=torch.cat([pts, torch.ones(pts.shape[0], 1)], dim=1),
                )
            )
            filenames_list.append(element["filename"])

        pc_sparse = sparse_collate(pc_sparse_list)
        
        vertex_model_batch["vertices_flat"] = vertices_flat
        vertex_model_batch["vertices_flat_mask"] = vertices_flat_mask
        vertex_model_batch["pc_sparse"] = pc_sparse
        vertex_model_batch["filename"] = filenames_list
        return vertex_model_batch

    def train_dataloader(self) -> DataLoader:
        """
        Returns:
            train_dataloader: Dataloader used to load training batches
        """
        return DataLoader(
            self.train_dataset,
            self.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=self.collate_vertex_model_batch,
            num_workers=16,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Returns:
            val_dataloader: Dataloader used to load validation batches
        """
        return DataLoader(
            self.val_dataset,
            self.batch_size,
            shuffle=False,
            drop_last=True,
            collate_fn=self.collate_vertex_model_batch,
            num_workers=4,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Returns:
            test_dataloader: Dataloader used to load test batches
        """
        return DataLoader(
            self.val_dataset,
            self.batch_size,
            shuffle=False,
            drop_last=True,
            collate_fn=self.collate_vertex_model_batch,
            num_workers=4,
            persistent_workers=True,
        )


class FaceModelDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        split: str = "train",
        quantization_bits: int = 8,
        augmentation: bool = False,
    ) -> None:
        """Initializes Image Dataset

        Args:
            dataset_path: Where model files along with renderings are located
            augmentation: Whether need data augmentation
        """
        self.augmentation = augmentation
        self.quantization_bits = quantization_bits
        self.split = split

        with open(dataset_path, "rb") as pkl_data:
            data = pickle.load(pkl_data)
            self.cached_pts = data[f"pts_{split}"]
            self.cached_vertices = data[f"vertices_{split}"]
            self.cached_faces = data[f"faces_{split}"]
            self.names = data[f"name_{split}"]

        if split == "train" and augmentation:
            self.train_transforms = T.Compose(
                [
                    data_transforms.PointcloudRotate(),
                    data_transforms.PointcloudScale(),
                    data_transforms.PointcloudTranslate(),
                ]
            )
        print(f"FaceModelDataset {split} Number: {len(self.cached_vertices)}")

    def __len__(self) -> int:
        return len(self.cached_vertices)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """Gets image object along with associated mesh

        Args:
            idx: Index of image to retrieve

        Returns:
            mesh_dict: Dictionary containing vertices, faces of .obj file and image tensor
        """
        pts = self.cached_pts[idx]
        vertices = self.cached_vertices[idx]
        faces = self.cached_faces[idx]
        filename = self.names[idx]

        pts = torch.from_numpy(pts).to(torch.float32)
        vertices = torch.from_numpy(vertices).to(torch.float32)

        # 数据增强
        if self.split == "train" and self.augmentation:
            item_dict = self.train_transforms({"vertices": vertices, "pts": pts})
            vertices, pts = item_dict["vertices"], item_dict["pts"]

        vertices, vertices_scale = data_utils.normalize_vertices_scale(
            vertices, return_scale=True
        )
        pts = (pts / vertices_scale).to(torch.float32)

        pts = torch.clamp(pts, -0.5, 0.5)
        vertices = torch.clamp(vertices, -0.5, 0.5)

        vertices, faces = data_utils.quantize_process_mesh(
            vertices, faces, self.quantization_bits
        )
        pts = data_utils.quantize_verts(pts, self.quantization_bits)

        pts = torch.unique(pts, dim=0)

        vertices = vertices.to(torch.int32)
        pts = pts.to(torch.int32)

        faces = data_utils.flatten_faces(faces)

        vertices = vertices.to(torch.int32)
        pts = pts.to(torch.int32)
        faces = faces.to(torch.int32)

        data_dict = {
            "vertices": vertices,
            "pts": pts,
            "faces": faces,
            "filename": filename,
        }
        return data_dict


class FaceDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_path: str,
        batch_size: int = 32,
        quantization_bits: int = 8,
        augmentation: bool = True,
        shuffle_vertices: bool = True,
    ) -> None:
        """
        Args:
            dataset_path: Root directory for shapenet dataset
            collate_method: Whether to collate vertices or faces
            batch_size: How many 3D objects in one batch
            quantization_bits: How many bits we are using to quantize the vertices
            apply_random_shift: Whether or not we're applying random shift to vertices for face model
            shuffle_vertices: Whether or not we're shuffling the order of vertices during batch generation for face model
        """
        super().__init__()

        self.batch_size = batch_size
        self.shuffle_vertices = shuffle_vertices
        self.quantization_bits = quantization_bits

        self.train_dataset = FaceModelDataset(
            dataset_path=dataset_path,
            split="train",
            quantization_bits=quantization_bits,
            augmentation=augmentation,
        )

        self.val_dataset = FaceModelDataset(
            dataset_path=dataset_path,
            split="val",
            quantization_bits=quantization_bits,
            augmentation=False,
        )

    def collate_face_model_batch(
        self,
        ds: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Applies padding to different length face sequences so we can batch them
        Args:
            ds: List of dictionaries with each dictionary containing info about a specific 3D object

        Returns:
            face_model_batch: A single dictionary which represents the whole face model batch
        """
        face_model_batch = {}
        batch_size = len(ds)

        # faces
        num_faces_list = [data_dict["faces"].shape[0] for data_dict in ds]
        max_faces = max(num_faces_list)

        shuffled_faces = torch.zeros([batch_size, max_faces], dtype=torch.int32)
        faces_mask = torch.zeros_like(shuffled_faces, dtype=torch.int32)

        # vertices
        vs_sparse_list = []

        # pointclouds
        pc_sparse_list = []

        # filenames
        filenames_list = []

        for i, element in enumerate(ds):
            vertices = element["vertices"]
            num_vertices = vertices.shape[0]
            if self.shuffle_vertices:
                permutation = torch.randperm(num_vertices)
                vertices = vertices[permutation]
                face_permutation = torch.cat(
                    [
                        torch.Tensor([0, 1]).to(torch.int32),
                        torch.argsort(permutation).to(torch.int32) + 2,
                    ],
                    dim=0,
                )
                curr_faces = face_permutation[element["faces"].to(torch.int64)][None]
            else:
                curr_faces = element["faces"][None]

            initial_faces_size = curr_faces.shape[1]
            face_padding_size = max_faces - initial_faces_size
            shuffled_faces[i] = F.pad(curr_faces, [0, face_padding_size, 0, 0])
            faces_mask[i] = torch.zeros_like(shuffled_faces[i], dtype=torch.float32)
            faces_mask[i, : initial_faces_size + 1] = 1

            pts = element["pts"]
            pc_sparse_list.append(
                SparseTensor(
                    coords=pts,
                    feats=torch.cat([pts, torch.ones(pts.shape[0], 1)], dim=1),
                )
            )

            vs_sparse_list.append(
                SparseTensor(
                    coords=vertices,
                    feats=torch.cat(
                        [vertices, torch.ones(vertices.shape[0], 1)], dim=1
                    ),
                )
            )

            filenames_list.append(element["filename"])

        pc_sparse = sparse_collate(pc_sparse_list)
        vs_sparse = sparse_collate(vs_sparse_list)

        face_model_batch["faces"] = shuffled_faces
        face_model_batch["faces_mask"] = faces_mask
        face_model_batch["pc_sparse"] = pc_sparse
        face_model_batch["vs_sparse"] = vs_sparse
        face_model_batch["filenames"] = filenames_list
        return face_model_batch

    def train_dataloader(self) -> DataLoader:
        """
        Returns:
            train_dataloader: Dataloader used to load training batches
        """
        return DataLoader(
            self.train_dataset,
            self.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=self.collate_face_model_batch,
            num_workers=16,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Returns:
            val_dataloader: Dataloader used to load validation batches
        """
        return DataLoader(
            self.val_dataset,
            self.batch_size,
            shuffle=False,
            drop_last=True,
            collate_fn=self.collate_face_model_batch,
            num_workers=4,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Returns:
            test_dataloader: Dataloader used to load test batches
        """
        return DataLoader(
            self.val_dataset,
            self.batch_size,
            shuffle=False,
            drop_last=True,
            collate_fn=self.collate_face_model_batch,
            num_workers=4,
            persistent_workers=True,
        )
