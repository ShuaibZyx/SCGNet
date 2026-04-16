from typing import Dict, Optional, Tuple, Any, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from src.utils.init_weights import init_weights_kaiming_uniform
from .polygen_encoder import PolygenEncoder
from .polygen_decoder import TransformerDecoder
from torch.nn.utils.rnn import pad_sequence
from .pcfvwa_module import PCFVWAModule
from src.utils.model_utils import top_k_logits, top_p_logits
from warmup_scheduler import GradualWarmupScheduler
from torchsparse import SparseTensor


class FaceModel(LightningModule):
    def __init__(
        self,
        batch_size: int,
        encoder_config: Dict[str, Any],
        decoder_config: Dict[str, Any],
        pcfvwa_config: Dict[str, Any],
        quantization_bits: int = 8,
        face_max_count: int = 81,
        face_vertex_max_count: int = 59,
        learning_rate: float = 3e-4,
        warmup_steps: int = 10000,
        total_steps: int = 500000,
    ) -> None:
        """Autoregressive generative model of face vertices

        Args:
            batch_size: Batch size
            encoder_config: Dictionary representing config for PolygenEncoder
            decoder_config: Dictionary representing config for TransformerDecoder
            class_conditional: If we are using global context embeddings based on class labels
            num_classes: How many distinct classes in the dataset
            quantization_bits: How many bits are we using to encode the vertices
            max_seq_length: Max number of face indices we can generate
            learning_rate: Learning rate for adamw optimizer
        """
        super(FaceModel, self).__init__()
        self.save_hyperparameters()

        self.batch_size = batch_size
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
        self.pcfvwa_config = pcfvwa_config
        self.quantization_bits = quantization_bits
        self.face_max_count = face_max_count
        self.face_vertex_max_count = face_vertex_max_count
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

        self.embedding_dim = decoder_config["hidden_size"]

        self.encoder = PolygenEncoder(**encoder_config)
        self.decoder = TransformerDecoder(**decoder_config)

        self.pcfvwa_module = PCFVWAModule(pcfvwa_config, self.embedding_dim)

        #  顶点三个坐标xyz嵌入
        self.coord0_embedder = nn.Embedding(
            2**self.quantization_bits, self.embedding_dim, device=self.device
        )
        self.coord1_embedder = nn.Embedding(
            2**self.quantization_bits, self.embedding_dim, device=self.device
        )
        self.coord2_embedder = nn.Embedding(
            2**self.quantization_bits, self.embedding_dim, device=self.device
        )
        # 位置嵌入
        self.pos_index_embedder = nn.Embedding(
            self.face_max_count, self.embedding_dim, device=self.device
        )
        self.pos_arange_embedder = nn.Embedding(
            self.face_vertex_max_count, self.embedding_dim, device=self.device
        )

        self.linear_layer = nn.Linear(
            self.embedding_dim, self.embedding_dim, bias=True, device=self.device
        )
        self.stopping_embeddings = nn.Parameter(
            torch.randn([1, 2, self.embedding_dim], device=self.device)
        )
        self.zero_embed = nn.Parameter(
            torch.randn([1, 1, self.embedding_dim], device=self.device)
        )
        self.apply(lambda m: init_weights_kaiming_uniform(m, nonlinearity="relu"))

    def _prepare_context(
        self, context: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Prepares vertex, global context and sequential context embeddings

        Args:
            context: A dictionary with keys of class_label (if class conditional is true), vertices and vertices_mask

        Returns:
            vertex_embeddings: Value embeddings for vertices of shape [batch_size, num_vertices + 2, embed_size]
            global_context_embedding: Embeddings for class label of shape [batch_size, embed_size]
            sequential_context_embeddings: Result of applying vertex mask to vertex embeddings of shape [batch_size, num_vertices + 2, embed_size]
        """

        vertex_embeddings, vs_feats_batch, vs_batch_mask = self._embed_vertices(
            context["pc_sparse"],
            context["vs_sparse"]
        )

        global_context_embedding = None
        sequential_context_embeddings = vs_feats_batch

        return (
            vertex_embeddings,
            global_context_embedding,
            sequential_context_embeddings,
            vs_batch_mask,
        )

    def _embed_vertices(
        self,
        pc_sparse: List[SparseTensor],
        vs_sparse: List[SparseTensor],
    ) -> torch.Tensor:
        """Provides value embeddings for vertices

        Args:
            vertices: A tensor of shape [batch_size, num_vertices, 3]. Represents vertices in the generated mesh.
            vertices_mask: A tensor of shape [batch_size, num_vertices]. Provides information about which vertices are complete.

        Returns:
            vertex_embeddings: A tensor of shape [batch_size, num_vertices + 2, embed_size]. Represents vertex embeddings with concatenated learnt stopping tokens.
        """
        vs_list, vs_feats_list = self.pcfvwa_module(
            pc_sparse, vs_sparse, self.batch_size
        )

        vs_batch = pad_sequence(vs_list, batch_first=True, padding_value=0).to(
            torch.int32
        )
        vs_batch_mask = vs_batch.sum(dim=-1).ne(0).to(torch.int32)
        vs_feats_batch = pad_sequence(
            vs_feats_list, batch_first=True, padding_value=0.0
        )

        vertex_embeddings = (
            self.coord0_embedder(vs_batch[..., 0])
            + self.coord1_embedder(vs_batch[..., 1])
            + self.coord2_embedder(vs_batch[..., 2])
        )

        vs_feats_batch = vs_feats_batch * vs_batch_mask[..., None]
        vertex_embeddings = vertex_embeddings * vs_batch_mask[..., None]

        stopping_embeddings = torch.repeat_interleave(
            self.stopping_embeddings, vs_batch.shape[0], dim=0
        )
        vertex_embeddings = torch.cat(
            [stopping_embeddings, vertex_embeddings.to(torch.float32)], dim=1
        )

        vertex_embeddings = self.encoder(vertex_embeddings.transpose(0, 1)).transpose(
            0, 1
        )
        # print(f"vertex_embeddings shape: {vertex_embeddings.shape}")
        # print(f"vs_feats_batch shape: {vs_feats_batch.shape}")

        return vertex_embeddings, vs_feats_batch, vs_batch_mask

    def _embed_inputs(
        self,
        faces_long: torch.Tensor,
        vertex_embeddings: torch.Tensor,
        global_context_embedding: Optional[torch.Tensor] = None,
    ):
        """Provides embeddings for sampled faces

        Args:
            faces_long: A tensor of shape [batch_size, sampled_faces]
            vertex_embeddings: A tensor of shape [batch_size, num_vertices + 2, embed_size]
            global_context_embedding: If it exists its a tensor of shape [batch_size, embed_size]

        Returns:
            embeddings: A tensor of shape [num_faces + 1, batch_size, embed_size].
                        The first two dimensions are transposed such that they can be fed directly to the decoder.
        """
        # [batch_size, sampled_faces, embed_size]
        face_embeddings = torch.zeros(
            size=[
                self.batch_size,  # batch_size
                faces_long.shape[1],  # sampled_faces
                self.embedding_dim,  # embed_size
            ],
            device=self.device,
            dtype=torch.float32,
        )
        # pos index
        faces_index = torch.zeros(
            size=[self.batch_size, faces_long.shape[1]],
            device=self.device,
            dtype=torch.int32,
        )
        # pos arrange
        faces_arange = torch.zeros(
            size=[self.batch_size, faces_long.shape[1]],
            device=self.device,
            dtype=torch.int32,
        )

        one_tensor = torch.ones(1, dtype=torch.int32, device=self.device)
        for i in range(self.batch_size):
            face_index = torch.cumsum(faces_long[i] == 1, dim=0, dtype=torch.int32) + 2
            face_index[faces_long[i] == 1] = 1
            face_index[faces_long[i] == 0] = 0
            faces_index[i] = face_index
            face_count = torch.bincount(face_index).to(torch.int32)
            face_arange = []
            for j in range(2, face_count.shape[0]):
                face_arange.extend(
                    torch.arange(
                        start=2,
                        end=face_count[j] + 2,
                        dtype=torch.int32,
                        device=self.device,
                    )
                )
                if len(face_arange) < torch.sum(face_count[1:]):
                    face_arange.extend(one_tensor)
            if face_count.numel() > 0:
                face_arange.extend(
                    torch.zeros(face_count[0], dtype=torch.int32, device=self.device)
                )
            faces_arange[i] = torch.tensor(
                face_arange, dtype=torch.int32, device=self.device
            )
            # 将平面包含的顶点索引的嵌入向量存入face_embeddings
            face_embeddings[i] = vertex_embeddings[i, faces_long[i]]

        faces_index_embeddings = self.pos_index_embedder(faces_index)
        faces_arange_embeddings = self.pos_arange_embedder(faces_arange)

        if global_context_embedding is None:
            zero_embed_tiled = torch.repeat_interleave(
                self.zero_embed, self.batch_size, dim=0
            )
        else:
            zero_embed_tiled = global_context_embedding[:, None]

        embeddings = face_embeddings + (
            faces_index_embeddings + faces_arange_embeddings
        )
        embeddings = (
            torch.cat([zero_embed_tiled, embeddings], dim=1)
            .transpose(0, 1)
            .to(torch.float32)
        )

        return embeddings

    def _project_to_pointers(self, inputs: torch.Tensor) -> torch.Tensor:
        """Passes inputs through a linear layer

        Args:
            inputs: A tensor of shape [...., embed_size]

        Returns:
            linear_outputs: A tensor of shape [..., embed_size]
        """
        return self.linear_layer(inputs)

    def _create_dist(
        self,
        vertex_embeddings: torch.Tensor,
        vertices_mask: torch.Tensor,
        faces_long: torch.Tensor,
        global_context_embedding: Optional[torch.Tensor] = None,
        sequential_context_embeddings: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        cache: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Outputs logits that can be used to create a categorical distribution

        Args:
            vertex_embeddings: A tensor of shape [batch_size, num_vertices + 2, embed_size] representing value embeddings for vertices
            vertices_mask: A tensor of shape [batch_size, num_vertices], representing which vertices are complete
            faces_long: A tensor of shape [batch_size, sampled_faces] representing currently sampled face indices
            global_context_embedding: A tensor of shape [batch_size, embed_size]
            sequential_context_embeddings: A tensor of shape [batch_size, num_vertices + 2, embed_size]
            temperature: A constant to normalize logits
            temperature: Scalar softmax temperature > 0.
            top_k: Number of tokens to keep for top-k sampling.
            top-p: Proportion of probability mass to keep for top-p sampling.
            cache: A list of dictionaries in the following format: {'k': torch.Tensor, 'v': torch.Tensor}. Each dictionary in the list represents the cache at the respective decoder layer.

        Returns:
            logits: Logits of shape [batch_size, sequence_length, num_vertices] that can be used to create a categorical distribution over vertex indices.
        """

        decoder_inputs = self._embed_inputs(
            faces_long.to(torch.int32),
            vertex_embeddings,
            global_context_embedding,
        )

        # check whether we are starting a sequence, or continuing a previous one
        if cache is not None:
            cached_decoder_inputs = decoder_inputs[-1:, :]
        else:
            cached_decoder_inputs = decoder_inputs
        # sequential_context_embeddings即为具有有效值的vertex_embeddings
        decoder_outputs = self.decoder(
            cached_decoder_inputs,
            cache=cache,
            sequential_context_embeddings=sequential_context_embeddings.transpose(0, 1),
        )

        pred_pointers = self._project_to_pointers(decoder_outputs.transpose(0, 1))

        # num_dimensions = 3
        num_dimensions = len(vertex_embeddings.shape)
        # penultimate_dim = 1 last_dim = 2
        penultimate_dim, last_dim = num_dimensions - 2, num_dimensions - 1
        vertex_embeddings_transposed = vertex_embeddings.transpose(
            penultimate_dim, last_dim
        )

        # 矩阵乘法
        logits = torch.matmul(pred_pointers, vertex_embeddings_transposed)
        # 除以self.embedding_dim的平方根进行归一化
        logits = logits / math.sqrt(self.embedding_dim)

        # each example in the batch needs to have max_num_vertices, so that we can create a batch from multiple classes
        f_verts_mask = F.pad(vertices_mask, [2, 0, 0, 0], value=1)[:, None]

        logits = logits * f_verts_mask
        logits = logits - (1.0 - f_verts_mask) * 1e9
        logits = logits / temperature

        logits = top_k_logits(logits, top_k)
        logits = top_p_logits(logits, top_p)

        return logits

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Forward method for Face Model

        Args:
            batch: A dictionary with keys for vertices, vertices_mask and faces

        Returns:
            logits: Logits of shape [batch_size, sequence_length, num_vertices] that can be used to create a categorical distribution over vertex indices.
        """
        vertex_embeddings, global_context, seq_context, vs_batch_mask = (
            self._prepare_context(batch)
        )
        logits = self._create_dist(
            vertex_embeddings,
            vs_batch_mask,
            batch["faces"][:, :-1],
            global_context_embedding=global_context,
            sequential_context_embeddings=seq_context,
        )
        return logits

    def training_step(
        self, face_model_batch: Dict[str, Any], batch_idx: int
    ) -> torch.Tensor:
        """Pytorch Lightning training step method

        Args:
            face_model_batch: A dictionary that contains batch data
            batch_idx: Which batch we are processing

        Returns:
            face_loss: NLL loss of generated categorical distribution
        """
        face_logits = self(face_model_batch)
        face_pred_dist = torch.distributions.categorical.Categorical(logits=face_logits)
        face_loss = -torch.sum(
            face_pred_dist.log_prob(face_model_batch["faces"])
            * face_model_batch["faces_mask"]
        )
        self.log(
            "train_loss",
            face_loss,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        return face_loss

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer and learning rate scheduler

        Returns:
            dict: Dictionary with optimizer and lr scheduler
        """
        face_model_optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate
        )
        face_model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            face_model_optimizer, T_max=self.total_steps, eta_min=0, last_epoch=-1
        )
        face_model_scheduler_warmup = GradualWarmupScheduler(
            face_model_optimizer,
            multiplier=1,
            total_epoch=self.warmup_steps,
            after_scheduler=face_model_scheduler,
        )

        return {
            "optimizer": face_model_optimizer,
            "lr_scheduler": {
                "scheduler": face_model_scheduler_warmup,
                "interval": "step",
            },
        }

    def validation_step(self, val_batch, batch_idx):
        """Pytorch Lightning validation step

        Args:
            val_batch: A dictionary that contains batch data
            batch_idx: Which batch we are processing

        Returns:
            face_loss: NLL loss of generated categorical distribution
        """

        with torch.no_grad():
            face_logits = self(val_batch)
            face_pred_dist = torch.distributions.categorical.Categorical(
                logits=face_logits
            )
            face_loss = -torch.sum(
                face_pred_dist.log_prob(val_batch["faces"]) * val_batch["faces_mask"]
            )
        self.log(
            "val_loss",
            face_loss,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        return face_loss

    def sample(
        self,
        context: Dict[str, Any],
        max_sample_length: int = 1200,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        only_return_complete: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Autoregressive sampling method to generate faces

        Args:
            context: A dictionary with keys for vertices and vertices_mask.
            max_sample_length: Maximum length of sampled faces. Sequences that do not complete are truncated.
            temperature: Scalar softmax temperature > 0.
            top_k: Number of tokens to keep for top-k sampling.
            top-p: Proportion of probability mass to keep for top-p sampling.
            only_return_complete: If True, only return completed samples. Otherwise return all samples along with completed indicator.

        Returns:
            outputs: Output dictionary with fields
                'context': A dictionary that contains modifications made to keys in the original context dictionary
                'completed': Tensor with shape [batch_size,]. Represents which faces have been fully sampled
                'faces': Tensor of shape [batch_size, num_faces]. Represents sampled faces.
                'num_face_indices': A tensor of shape [batch_size,]. Represents ending point of every sampled face.
        """
        vertex_embeddings, global_context, seq_context, vs_batch_mask = (
            self._prepare_context(context)
        )
        num_samples = vertex_embeddings.shape[0]

        def _loop_body(
            i: int, samples: torch.Tensor, cache: Dict
        ) -> Tuple[int, torch.Tensor]:
            """While-loop body for autoregression calculation.

            Args:
                i: Current iteration in the loop
                samples: tensor of shape [batch_size, i].
                cache: A list of dictionaries in the following format: {'k': torch.Tensor, 'v': torch.Tensor}. Each dictionary in the list represents the cache at the respective decoder layer.
            Returns:
                next_iter: i + 1.
                samples: tensor of shape [batch_size, i + 1].
            """

            logits = self._create_dist(
                vertex_embeddings,
                vs_batch_mask,
                samples,
                global_context_embedding=global_context,
                sequential_context_embeddings=seq_context,
                cache=cache,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            pred_dist = torch.distributions.categorical.Categorical(logits=logits)
            next_sample = pred_dist.sample()[:, -1:].to(torch.int32)
            samples = torch.cat([samples, next_sample], dim=1)
            return i + 1, samples

        def _stopping_cond(samples: torch.Tensor) -> bool:
            """Stopping condition for sampling while-loop. Looking for stop token (represented by 0)

            Args:
                samples: tensor of shape of [batch_size, i], where i is the current iteration.
            Returns:
                token_found: Boolean that represents if stop token has been found in every row of samples.
            """
            nonzero_boolean_matrix = torch.ne(samples, 0)
            reduced_matrix = torch.all(
                nonzero_boolean_matrix, dim=-1
            )  # Checks if stopping token is present in every row
            return torch.any(reduced_matrix)

        samples = torch.zeros([num_samples, 0], device=self.device, dtype=torch.int32)
        cache = self.decoder.initialize_cache(num_samples)
        max_sample_length = max_sample_length or self.max_seq_length
        j = 0
        while _stopping_cond(samples) and j < max_sample_length:
            j, samples = _loop_body(j, samples, cache)

        completed_samples_boolean = (
            samples == 0
        )  # Checks for stopping token in every row of sampled faces
        complete_samples = torch.any(
            completed_samples_boolean, dim=-1
        )  # Tells us which samples are complete and which aren't
        sample_length = samples.shape[-1]  # Number of sampled faces
        max_one_ind, _ = torch.max(
            torch.arange(sample_length, device=self.device)[None]
            * (samples == 1).to(torch.int32),
            dim=-1,
        )  # Checking for new face tokens
        max_one_ind = max_one_ind.to(torch.int32)
        zero_inds = (
            torch.argmax((completed_samples_boolean).to(torch.int32), dim=-1)
        ).to(
            torch.int32
        )  # Figuring out where the zeros are in every row
        num_face_indices = (
            torch.where(complete_samples, zero_inds, max_one_ind) + 1
        )  # How many vertices in each face

        faces_mask = (
            torch.arange(sample_length, device=self.device)[None]
            < num_face_indices[:, None] - 1
        ).to(
            torch.int32
        )  # Faces mask turns the last true to false in each row

        samples = samples * faces_mask

        faces_mask = (
            torch.arange(sample_length, device=self.device)[None]
            < num_face_indices[:, None]
        ).to(torch.int32)

        pad_size = max_sample_length - sample_length
        samples = F.pad(samples, [0, pad_size, 0, 0])

        if only_return_complete:
            samples = samples[complete_samples]
            num_face_indices = num_face_indices[complete_samples]
            for key in context:
                context[key] = context[key][complete_samples]
            complete_samples = complete_samples[complete_samples]

        outputs = {
            "context": context,
            "completed": complete_samples,
            "faces": samples,
            "num_face_indices": num_face_indices,
        }

        return outputs

    def sample_mask(
        self,
        context: Dict[str, Any],
        max_sample_length: int = 5000,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        only_return_complete: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Autoregressive sampling method to generate faces

        Args:
            context: A dictionary with keys for vertices and vertices_mask.
            max_sample_length: Maximum length of sampled faces. Sequences that do not complete are truncated.
            temperature: Scalar softmax temperature > 0.
            top_k: Number of tokens to keep for top-k sampling.
            top-p: Proportion of probability mass to keep for top-p sampling.
            only_return_complete: If True, only return completed samples. Otherwise return all samples along with completed indicator.

        Returns:
            outputs: Output dictionary with fields
                'context': A dictionary that contains modifications made to keys in the original context dictionary
                'completed': Tensor with shape [batch_size,]. Represents which faces have been fully sampled
                'faces': Tensor of shape [batch_size, num_faces]. Represents sampled faces.
                'num_face_indices': A tensor of shape [batch_size,]. Represents ending point of every sampled face.
        """
        vertex_embeddings, global_context, seq_context, vs_batch_mask = (
            self._prepare_context(context)
        )
        num_samples = vertex_embeddings.shape[0]

        def _loop_body(
            i: int, samples: torch.Tensor, cache: Dict
        ) -> Tuple[int, torch.Tensor]:
            """While-loop body for autoregression calculation.

            Args:
                i: Current iteration in the loop
                samples: tensor of shape [batch_size, i].
                cache: A list of dictionaries in the following format: {'k': torch.Tensor, 'v': torch.Tensor}. Each dictionary in the list represents the cache at the respective decoder layer.
            Returns:
                next_iter: i + 1.
                samples: tensor of shape [batch_size, i + 1].
            """

            logits = self._create_dist(
                vertex_embeddings,
                vs_batch_mask,
                samples,
                global_context_embedding=global_context,
                sequential_context_embeddings=seq_context,
                cache=cache,
                temperature=temperature,
                top_k=top_k,
                top_p=1.0,
            )

            # 1. Masking the last generated token
            if samples.size(1) > 0:
                last_token = samples[:, -1]
                logits[:, :, last_token.long()] = -1e9

            # 2. Ensuring the first number of a new sub-sequence isn't less than the first number in the previous sub-sequence
            sub_sequence_starts = (samples == 1).nonzero()
            if sub_sequence_starts.size(0) == 1:
                last_start = sub_sequence_starts[-1, 1]
                logits[:, :, 2 : samples[0][0]] = -1e9
            elif sub_sequence_starts.size(0) > 1:
                last_start = sub_sequence_starts[-1, 1]
                second_last_start = sub_sequence_starts[-2, 1]
                min_val = samples[torch.arange(samples.size(0)), second_last_start + 1]
                logits[:, 2:min_val] = -1e9

            # 3. Ensuring numbers within a sub-sequence are greater than the first number in that sub-sequence
            if samples.size(1) > 0:
                if sub_sequence_starts.size(0) == 0:
                    logits[:, :, 2 : samples[0, 0]] = -1e9
                else:
                    if last_start.item() != i - 1:
                        logits[:, :, 2 : samples[0, last_start.item() + 1]] = -1e9

            # 4. Ensuring numbers within a sub-sequence are unique
            if samples.size(1) > 0:
                if sub_sequence_starts.size(0) == 0:
                    for idx in range(samples.size(1)):
                        logits[:, :, samples[0, idx]] = -1e9
                else:
                    if last_start.item() != i - 1:
                        for idx in range(last_start.item() + 1, samples.size(1)):
                            logits[:, :, samples[0, idx]] = -1e9

            pred_dist = torch.distributions.categorical.Categorical(logits=logits)
            next_sample = pred_dist.sample()[:, -1:].to(torch.int32)
            samples = torch.cat([samples, next_sample], dim=1)
            return i + 1, samples

        def _stopping_cond(samples: torch.Tensor) -> bool:
            """Stopping condition for sampling while-loop. Looking for stop token (represented by 0)

            Args:
                samples: tensor of shape of [batch_size, i], where i is the current iteration.
            Returns:
                token_found: Boolean that represents if stop token has been found in every row of samples.
            """
            nonzero_boolean_matrix = torch.ne(samples, 0)
            reduced_matrix = torch.all(
                nonzero_boolean_matrix, dim=-1
            )  # Checks if stopping token is present in every row
            return torch.any(reduced_matrix)

        samples = torch.zeros([num_samples, 0], dtype=torch.int32, device=self.device)
        cache = self.decoder.initialize_cache(num_samples)
        max_sample_length = max_sample_length or self.max_seq_length
        j = 0
        while _stopping_cond(samples) and j < max_sample_length:
            j, samples = _loop_body(j, samples, cache)

        completed_samples_boolean = (
            samples == 0
        )  # Checks for stopping token in every row of sampled faces
        complete_samples = torch.any(
            completed_samples_boolean, dim=-1
        )  # Tells us which samples are complete and which aren't
        sample_length = samples.shape[-1]  # Number of sampled faces
        max_one_ind, _ = torch.max(
            torch.arange(sample_length, device=self.device)[None]
            * (samples == 1).to(torch.int32),
            dim=-1,
        )  # Checking for new face tokens
        max_one_ind = max_one_ind.to(torch.int32)
        zero_inds = (
            torch.argmax((completed_samples_boolean).to(torch.int32), dim=-1)
        ).to(
            torch.int32
        )  # Figuring out where the zeros are in every row
        num_face_indices = (
            torch.where(complete_samples, zero_inds, max_one_ind) + 1
        )  # How many vertices in each face

        faces_mask = (
            torch.arange(sample_length, device=self.device)[None]
            < num_face_indices[:, None] - 1
        ).to(
            torch.int32
        )  # Faces mask turns the last true to false in each row

        samples = samples * faces_mask

        faces_mask = (
            torch.arange(sample_length, device=self.device)[None]
            < num_face_indices[:, None]
        ).to(torch.int32)

        pad_size = max_sample_length - sample_length
        samples = F.pad(samples, [0, pad_size, 0, 0])

        if only_return_complete:
            samples = samples[complete_samples]
            num_face_indices = num_face_indices[complete_samples]
            for key in context:
                if key == "files_list":
                    context[key] = [
                        data
                        for data, flag in zip(context[key], complete_samples)
                        if flag
                    ]
                else:
                    context[key] = context[key][complete_samples]
            complete_samples = complete_samples[complete_samples]

        outputs = {
            "context": context,
            "completed": complete_samples,
            "faces": samples,
            "num_face_indices": num_face_indices,
        }

        return outputs
