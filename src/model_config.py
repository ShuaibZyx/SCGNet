from typing import Any, Dict

import torch
from src.modules.vertex_model import VertexModel
from src.modules.face_model import FaceModel
from src.modules.data_modules import FaceDataModule, VertexDataModule


class VertexModelConfig:
    def __init__(
        self,
        experiment: str,
        dataset_path: str,
        batch_size: int,
        augmentation: bool,
        pooling: bool,
        decoder_config: Dict[str, Any],
        quantization_bits: int,
        max_num_input_verts: int,
        learning_rate: float,
        warmup_steps: int,
        epochs: int,
    ) -> None:
        """Initializes vertex model and vertex data module

        Args:
            dataset_path: Root directory for shapenet dataset
            batch_size: How many 3D objects in one batch
            training_split: What proportion of data to use for training the model
            val_split: What proportion of data to use for validation
            apply_random_shift: Whether or not we're applying random shift to vertices
            decoder_config: Dictionary with TransformerDecoder config. Decoder config has to include num_layers, hidden_size, and fc_size.
            quantization_bits: Number of quantization bits used in mesh preprocessing
            class_conditional: If True, then condition on learned class embeddings
            num_classes: Number of classes to condition on
            max_num_input_verts:  Maximum number of vertices. Used for learned position embeddings.
            use_discrete_embeddings: Discrete embedding layers or linear layers for vertices
            learning_rate: Learning rate for adam optimizer
            step_size: How often to use lr scheduler
            gamma: Decay rate for lr scheduler
            training_steps: How many total steps we want to train for
            image_model: Whether we're training the image model or class-conditioned model
        """

        self.experiment = experiment
        self.epochs = epochs
        self.batch_size = batch_size

        self.vertex_data_module = VertexDataModule(
            dataset_path=dataset_path,
            batch_size=self.batch_size,
            quantization_bits=quantization_bits,
            augmentation=augmentation,
        )

        self.num_gpus = torch.cuda.device_count()
        total_steps = int(
            len(self.vertex_data_module.train_dataloader())
            * self.epochs
            // self.num_gpus
        )

        self.vertex_model = VertexModel(
            batch_size=self.batch_size,
            pooling=pooling,
            decoder_config=decoder_config,
            quantization_bits=quantization_bits,
            max_num_input_verts=max_num_input_verts,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )


# Point Cloud Feature Vertex-wise Aggregation(pcfvwa)
class FaceModelConfig:
    def __init__(
        self,
        experiment: str,
        dataset_path: str,
        batch_size: int,
        shuffle_vertices: bool,
        augmentation: bool,
        encoder_config: Dict[str, Any],
        decoder_config: Dict[str, Any],
        pcfvwa_config: Dict[str, Any],
        quantization_bits: int,
        face_max_count: int,
        face_vertex_max_count: int,
        learning_rate: float,
        warmup_steps: int,
        epochs: int,
    ):
        """Initializes face model and face data module

        Args:
            dataset_path: Root directory for shapenet dataset
            batch_size: How many 3D objects in one batch
            training_split: What proportion of data to use for training the model
            val_split: What proportion of data to use for validation
            apply_random_shift: Whether or not we're applying random shift to vertices
            shuffle_vertices: Whether or not we are randomly shuffling the vertices during batch generation
            encoder_config: Dictionary representing config for PolygenEncoder
            decoder_config: Dictionary representing config for TransformerDecoder
            class_conditional: If we are using global context embeddings based on class labels
            num_classes: How many distinct classes in the dataset
            decoder_cross_attention: If we are using cross attention within the decoder
            use_discrete_vertex_embeddings: Are the inputted vertices quantized
            quantization_bits: How many bits are we using to encode the vertices
            max_seq_length: Max number of face indices we can generate
            learning_rate: Learning rate for adam optimizer
            step_size: How often to use lr scheduler
            gamma: Decay rate for lr scheduler
            training_steps: How many total steps we want to train for
        """

        self.experiment = experiment
        self.epochs = epochs
        self.batch_size = batch_size

        self.face_data_module = FaceDataModule(
            dataset_path=dataset_path,
            batch_size=self.batch_size,
            quantization_bits=quantization_bits,
            augmentation=augmentation,
            shuffle_vertices=shuffle_vertices,
        )

        self.num_gpus = torch.cuda.device_count()
        total_steps = int(
            len(self.face_data_module.train_dataloader()) * self.epochs // self.num_gpus
        )

        radii_quanted = [r * (2**quantization_bits - 1) for r in pcfvwa_config["radii"]]
        pcfvwa_config["radii"] = radii_quanted

        self.face_model = FaceModel(
            batch_size=batch_size,
            encoder_config=encoder_config,
            decoder_config=decoder_config,
            pcfvwa_config=pcfvwa_config,
            quantization_bits=quantization_bits,
            face_max_count=face_max_count,
            face_vertex_max_count=face_vertex_max_count,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )
