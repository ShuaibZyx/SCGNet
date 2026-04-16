from typing import Dict, Optional, Tuple, List, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from lightning.pytorch import LightningModule
from src.utils.data_utils import dequantize_verts
from src.utils.init_weights import init_weights_kaiming_uniform
from .polygen_decoder import TransformerDecoder
from .pointcloud_encoder_vertex import PointCloudEncoderVertex
from src.utils.model_utils import top_k_logits, top_p_logits
from warmup_scheduler import GradualWarmupScheduler


class VertexModel(LightningModule):
    """Autoregressive Generative Model of Quantized Mesh Vertices.
    Operates on flattened vertex sequences with a stopping token:
    [z_0, y_0, x_0, z_1, y_1, x_1, ..., z_n, y_n, x_n, STOP]
    Input Vertex Coordinates are embedded and tagged with learned coordinate and position indicators.
    A transformer decoder outputs logits for a quantized vertex distribution.
    """

    def __init__(
        self,
        batch_size: int,
        pooling: bool,
        decoder_config: Dict[str, Any],
        quantization_bits: int,
        max_num_input_verts: int = 1800,
        learning_rate: float = 3e-4,
        warmup_steps: int = 10000,
        total_steps: int = 500000,
    ) -> None:
        """Initializes VertexModel. The encoder can be a model with a Resnet backbone for image contexts and voxel contexts.
        However for class label context, the encoder is simply the class embedder.

        Args:
            batch_size: Batch size
            decoder_config: Dictionary with TransformerDecoder config. Decoder config has to include num_layers, hidden_size, and fc_size.
            quantization_bits: Number of quantization bits used in mesh preprocessing
            class_conditional: If True, then condition on learned class embeddings
            use_discrete_embeddings: Discrete embedding layers or linear layers for vertices
            num_classes: Number of classes to condition on
            max_num_input_verts:  Maximum number of vertices. Used for learned position embeddings.
            learning_rate: Learning rate for adamw optimizer
        """

        super(VertexModel, self).__init__()
        self.save_hyperparameters()

        self.batch_size = batch_size
        self.pooling = pooling
        self.decoder_config = decoder_config
        self.quantization_bits = quantization_bits
        self.max_num_input_verts = max_num_input_verts
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

        self.embedding_dim = decoder_config["hidden_size"]

        self.pointcloud_encoder = PointCloudEncoderVertex(
            self.embedding_dim, pooling=pooling
        )

        # Transformer解码器
        self.decoder = TransformerDecoder(**decoder_config)

        # 将三个坐标点转换为具有特定维度的嵌入向量,顶点坐标嵌入
        self.coord_embedder = nn.Embedding(
            num_embeddings=3, embedding_dim=self.embedding_dim, device=self.device
        )
        # 顶点坐标位置信息嵌入
        self.pos_embedder = nn.Embedding(
            num_embeddings=self.max_num_input_verts,
            embedding_dim=self.embedding_dim,
            device=self.device,
        )
        # 顶点量化值嵌入
        self.vert_embedder_discrete = nn.Embedding(
            num_embeddings=2**self.quantization_bits + 1,
            embedding_dim=self.embedding_dim,
            device=self.device,
        )
        # 线性层-将self.embedding_dim映射至2 ** self.quantization_bits + 1维
        self.linear_layer = nn.Linear(
            self.embedding_dim, 2**self.quantization_bits + 1, device=self.device
        )
        # 包含随机噪声的嵌入向量张量
        zero_embeddings_tensor = torch.randn(
            [1, 1, self.embedding_dim], device=self.device
        )
        self.zero_embed = nn.Parameter(zero_embeddings_tensor)
        self.apply(lambda m: init_weights_kaiming_uniform(m, nonlinearity="relu"))

    # 准备上下文信息
    def _prepare_context(
        self, context: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Prepares global context embedding

        Args:
            context: A dictionary that contains a key of class_label
            包含键为class_label的字典
        Returns:
            global_context_embeddings: A Tensor of shape [batch_size, embed_size]
            sequential_context_embeddings: None
        """
        pointcloud_features = self.pointcloud_encoder(
            context["pc_sparse"].C, context["pc_sparse"].F, self.batch_size
        )

        sequential_context_embeddings = pad_sequence(
            pointcloud_features, batch_first=True, padding_value=0.0
        )

        global_context_embeddings = None
        return global_context_embeddings, sequential_context_embeddings

    def _embed_inputs(
        self, vertices: torch.Tensor, global_context_embedding: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Embeds flat vertices and adds position and coordinate information.
        嵌入平面顶点并添加位置和坐标信息

        Args:
            vertices: A Tensor of shape [batch_size, sample_length]. Represents current sampled vertices.
            形状为[batch_size, sample_length]的张量。表示当前采样的顶点
            global_context_embedding: A Tensor of shape [batch_size, embed_size]. Represents class label conditioning.
            形状[batch_size, embed_size]的张量。表示类标签条件。
        Returns:
            embeddings: A Tensor of shape [sample_length + 1, batch_size]. Represents combination of embeddings with global context embeddings. The first and second
                        dimensions are transposed for the sake of the decoder.
        """
        # [batch_size, sample_length]
        input_shape = vertices.shape
        batch_size, seq_length = input_shape[0], input_shape[1]
        # 坐标嵌入
        coord_embeddings = self.coord_embedder(
            torch.fmod(torch.arange(seq_length, device=self.device), 3)
        )  # Coord embeddings will be of shape [seq_length, embed_size]
        # 顶点相对位置信息嵌入
        pos_embeddings = self.pos_embedder(
            torch.floor_divide(torch.arange(seq_length, device=self.device), 3)
        )  # Position embeddings will be of shape [seq_length, embed_size]
        # 顶点量化值离散嵌入
        vert_embeddings = self.vert_embedder_discrete(
            vertices
        )  # Vert embeddings will be of shape [batch_size, seq_length, embed_size]

        # 添加上下文嵌入
        if global_context_embedding is None:
            zero_embed_tiled = torch.repeat_interleave(
                self.zero_embed, batch_size, dim=0
            )
        else:
            zero_embed_tiled = global_context_embedding[:, None].to(
                torch.float32
            )  # Zero embed tiled is of shape [batch_size, 1, embed_size]

        embeddings = vert_embeddings + (coord_embeddings + pos_embeddings)[None]

        # Embeddings shape before concatenation is [batch_size, seq_length, embed_size], after concatenation it is [batch_size, seq_length + 1, embed_size]
        embeddings = torch.cat([zero_embed_tiled, embeddings], dim=1)

        # Changing the dimension from [batch_size, seq_length, embed_size] to [seq_length, batch_size, embed_size] for TransformerDecoder
        return embeddings.transpose(0, 1)

    def _project_to_logits(self, inputs: torch.Tensor) -> torch.Tensor:
        """Runs decoder outputs through a linear layer

        Args:
            inputs: Tensor of shape [batch_size, sequence_length, embed_size].
        Returns:
            outputs: Tensor of shape [batch_size, sequence_length, 2 ** self.quantization_bits + 1].
        """
        output = self.linear_layer(inputs)
        return output

    def _create_dist(
        self,
        vertices: torch.Tensor,
        global_context_embedding: Optional[torch.Tensor] = None,
        sequential_context_embedding: Optional[torch.Tensor] = None,
        cache: Optional[List[Dict[str, torch.Tensor]]] = None,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: int = 1,
    ) -> torch.Tensor:
        """Creates a predictive distribution for the next vertex sample
            为下一个顶点样本创建预测分布
        Args:
            vertices: A Tensor of shape [batch_size, sequence_length]. Represents current flattened vertices. Sequence length is at max 3 * the number of vertices.
            形状为[batch_size, sequence_length]的张量。表示当前的平面化顶点。序列长度最大为3 *顶点数。
            global_context_embedding: A Tensor of shape [batch_size, embed_size]. Represents conditioning on class labels.
            形状[batch_size, embed_size]的张量。表示类标签上的条件作用。
            sequential_context_embeddings: A Tensor of shape [batch_size, context_seq_length, context_embed_size]. Represents conditioning on images or voxels.
            形状[batch_size, context_seq_length, context_embed_size]的张量。表示对图像或体素的条件作用。
            cache:  A list of dictionaries in the following format: {'k': torch.Tensor, 'v': torch.Tensor}. Each dictionary in the list represents the cache at the respective decoder layer.
            以下格式的字典列表:{'k': torch。张量,'v': torch.Tensor}。列表中的每个字典表示各自解码器层的缓存。
            temperature: Scalar softmax temperature > 0.
            标量softmax temperature参数> 0。
            top_k: Number of tokens to take out for top-k sampling.
            为top-k抽样取出的令牌数
            top-p: Proportion of probability mass to keep for top-p sampling.
            保留top-p抽样的概率质量比例
        Returns:
            logits: Logits that can be used to create a categorical distribution to sample the next vertex.
            Logits可以用来创建一个分类分布来采样下一个顶点
        """
        decoder_inputs = self._embed_inputs(
            vertices.to(torch.int32), global_context_embedding
        )

        if cache is not None:
            decoder_inputs = decoder_inputs[-1:, :]
        if sequential_context_embedding is not None:
            sequential_context_embedding = sequential_context_embedding.transpose(0, 1)

        outputs = self.decoder(
            inputs=decoder_inputs,
            sequential_context_embeddings=sequential_context_embedding,
            cache=cache,
        ).transpose(
            0, 1
        )  # Transpose to convert from [seq_length, batch_size, embedding_dim] to [batch_size, seq_length, embedding_dim]
        # pass through linear layer
        logits = self._project_to_logits(outputs)
        logits = logits / temperature
        # remove the smaller logits
        logits = top_k_logits(logits, top_k)
        # then choose those that contribute to 90% of mass distribution
        logits = top_p_logits(logits, top_p)
        return logits

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward method for Vertex Model
        Args:
            batch: A dictionary with a key of vertices_flat that represents a flattened input sequence of vertices.
        Returns:
            logits: Logits that can be used to create a categorical distribution to sample the next vertex.
            Logits可以用来创建一个分类分布来采样下一个顶点。
        """
        global_context, seq_context = self._prepare_context(batch)
        vertices = batch["vertices_flat"]
        logits = self._create_dist(
            vertices[:, :-1],
            global_context_embedding=global_context,
            sequential_context_embedding=seq_context,
        )
        return logits

    def training_step(
        self, vertex_model_batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Pytorch Lightning training step method

        Args:
            vertex_model_batch: A dictionary that contains the flat vertices
            包含平面顶点的字典
            batch_idx: Which batch are we processing

        Returns:
            vertex_loss: NLL loss for estimated categorical distribution
        """
        vertex_logits = self(vertex_model_batch)
        vertex_pred_dist = torch.distributions.categorical.Categorical(
            logits=vertex_logits
        )
        vertex_loss = -torch.sum(
            vertex_pred_dist.log_prob(vertex_model_batch["vertices_flat"])
            * vertex_model_batch["vertices_flat_mask"]
        )
        self.log(
            "train_loss",
            vertex_loss,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        return vertex_loss

    def configure_optimizers(self) -> Dict[str, Any]:
        """Method to create optimizer and learning rate scheduler

        Returns:
            dict: A dictionary with optimizer and learning rate scheduler
        """
        vertex_model_optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate
        )

        vertex_model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            vertex_model_optimizer, T_max=self.total_steps, eta_min=0, last_epoch=-1
        )
        vertex_model_scheduler = GradualWarmupScheduler(
            vertex_model_optimizer,
            multiplier=1,
            total_epoch=self.warmup_steps,
            after_scheduler=vertex_model_scheduler,
        )

        return {
            "optimizer": vertex_model_optimizer,
            "lr_scheduler": {
                "scheduler": vertex_model_scheduler,
                "interval": "step",
            },
        }

    def validation_step(
        self, val_batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Validation step for Pytorch Lightning

        Args:
            val_batch: dictionary which contains batch to run validation on
            batch_idx: Which batch we are processing

        Returns:
            vertex_loss: NLL loss for estimated categorical distribution
        """
        with torch.no_grad():
            vertex_logits = self(val_batch)
            vertex_pred_dist = torch.distributions.categorical.Categorical(
                logits=vertex_logits
            )
            vertex_loss = -torch.sum(
                vertex_pred_dist.log_prob(val_batch["vertices_flat"])
                * val_batch["vertices_flat_mask"]
            )
        self.log(
            "val_loss",
            vertex_loss,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        return vertex_loss

    def sample(
        self,
        max_sample_length: int = 50,
        context: Dict[str, torch.Tensor] = None,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        recenter_verts: bool = True,
        only_return_complete: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Autoregressive sampling method to generate vertices
        自回归采样方法生成顶点

        Args:
            num_samples: Number of samples to produce.
            处理样本的数量
            context: A dictionary with the type of context to condition upon. This could be class labels or images or voxels.
            具有上下文类型以作为条件的字典。这可以是类标签、图像或体素
            max_sample_length: Maximum length of sampled vertex samples. Sequences that do not complete are truncated.
            采样顶点样本的最大长度。不完整的序列将被截断
            temperature: Scalar softmax temperature > 0.
            top_k: Number of tokens to keep for top-k sampling.
            top-p: Proportion of probability mass to keep for top-p sampling.
            recenter_verts: If True, center vertex samples around origin. This should be used if model is trained using shift augmentations.
            如果为True,则将顶点采样集中在原点附近。如果模型是使用移位增强来训练的,应该使用这种方法
            only_return_complete: If True, only return completed samples. Otherwise return all samples along with completed indicator.
            如果为Tru,则只返回完整的样品。否则,将所有样品连同完成的指标一并返回

        Returns:
            outputs: Output dictionary with fields
                'completed': Boolean tensor of shape [num_samples]. If True then corresponding sample completed within max_sample_length.
                布尔张量的形状[num_samples]。如果为True,则在max_sample_length范围内完成相应的样本
                'vertices': Tensor of samples with shape [num_samples, num_verts, 3].
                形状为[num_samples, num_vert, 3]的样本张量
                'num_vertices': Tensor indicating number of vertices for each example in padded vertex samples.
                表示填充顶点样本中每个示例的顶点数的张量
                'vertices_mask': Tensor of shape [num_samples, num_verts] that masks corresponding invalid elements in vertices.
                形状为[num_samples, num_vert]的张量，它屏蔽了顶点中相应的无效元素
        """
        global_context, seq_context = self._prepare_context(context)

        num_samples = seq_context.shape[0]

        def _loop_body(
            i: int,
            samples: torch.Tensor,
            cache: Optional[List[Dict[str, torch.Tensor]]] = None,
        ) -> Tuple[int, torch.Tensor]:
            """While-loop body for autoregression calculation.

            Args:
                i: Current iteration in the loop
                samples: tensor of shape [num_samples, i].
                cache: A list of dictionaries in the following format: {'k': torch.Tensor, 'v': torch.Tensor}.
                       Each dictionary in the list represents the cache at the respective decoder layer.
            Returns:
                next_iter: i + 1.
                samples: tensor of shape [num_samples, i + 1] or of shape [num_samples, 2 * i + 1] if cache doesn't exist.
            """
            logits = self._create_dist(
                samples,
                global_context_embedding=global_context,
                sequential_context_embedding=seq_context,
                cache=cache,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            cat_dist = torch.distributions.categorical.Categorical(logits=logits)
            next_sample = cat_dist.sample()
            samples = torch.cat([samples, next_sample.to(torch.int32)], dim=1)
            return i + 1, samples

        def _stopping_cond(samples: torch.Tensor) -> bool:
            """
            Stopping condition for sampling while-loop. Looking for stop token (represented by 0)
            Args:
                samples: tensor of shape [num_samples, i], where i is the current iteration.
            Returns:
                token_found: Boolean that represents if stop token has been found.
            """
            nonzero_boolean_matrix = torch.ne(samples, 0)
            reduced_matrix = torch.all(
                nonzero_boolean_matrix, dim=-1
            )  # Checks if stopping token is present in every row
            return torch.any(reduced_matrix)

        samples = torch.zeros([num_samples, 0], dtype=torch.int32, device=self.device)
        cache = self.decoder.initialize_cache(num_samples)
        max_sample_length = max_sample_length or self.max_num_input_verts
        j = 0
        while _stopping_cond(samples) and j < max_sample_length * 3 + 1:
            j, samples = _loop_body(j, samples, cache)

        completed_samples_boolean = samples == 0  # Checks for stopping token
        completed = torch.any(
            completed_samples_boolean, dim=-1
        )  # Indicates which samples are completed of shape [num_samples,]
        stop_index_completed = torch.argmax(
            completed_samples_boolean.to(torch.int32), dim=-1
        ).to(
            torch.int32
        )  # Indicates where the stopping token occurs in each batch of samples
        stop_index_incomplete = (
            max_sample_length * 3 * torch.ones_like(stop_index_completed)
        )  # Placeholder tensor used to select samples from incomplete samples
        stop_index = torch.where(
            completed, stop_index_completed, stop_index_incomplete
        )  # Stopping Indices of each sample, if completed is true, then stopping index is taken from completed stop index tensor
        num_vertices = torch.floor_divide(stop_index, 3)

        samples = (
            samples[:, : (torch.max(num_vertices) * 3)] - 1
        )  # Selects last possible stopping index
        verts_dequantized = dequantize_verts(samples, self.quantization_bits)
        # Converts vertices to [-1, 1] range
        vertices = torch.reshape(
            verts_dequantized, [num_samples, -1, 3]
        )  # Reshapes into 3D Tensors
        vertices = torch.stack(
            [vertices[..., 2], vertices[..., 1], vertices[..., 0]], dim=-1
        )  # Converts from z-y-x to x-y-z.

        # Pad samples such that samples of different lengths can be concatenated
        pad_size = max_sample_length - vertices.shape[1]
        vertices = F.pad(vertices, [0, 0, 0, pad_size, 0, 0])

        vertices_mask = (
            torch.arange(max_sample_length, device=self.device)[None]
            < num_vertices[:, None]
        ).to(
            torch.float32
        )  # Provides a mask of which vertices to zero out as they were produced after stop token for that batch ended

        if recenter_verts:
            vert_max, _ = torch.max(
                vertices - 1e10 * (1.0 - vertices_mask)[..., None], dim=1, keepdim=True
            )
            vert_min, _ = torch.min(
                vertices + 1e10 * (1.0 - vertices_mask)[..., None], dim=1, keepdim=True
            )
            vert_centers = 0.5 * (vert_max + vert_min)
            vertices = vertices - vert_centers

        vertices = (
            vertices * vertices_mask[..., None]
        )  # Zeros out vertices produced after stop token

        if only_return_complete:
            vertices = vertices[completed]
            num_vertices = num_vertices[completed]
            vertices_mask = vertices_mask[completed]
            completed = completed[completed]

        outputs = {
            "completed": completed,
            "vertices": vertices,
            "num_vertices": num_vertices,
            "vertices_mask": vertices_mask.to(torch.int32),
        }
        return outputs

    def sample_mask(
        self,
        max_sample_length: int = 50,
        context: Dict[str, torch.Tensor] = None,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        recenter_verts: bool = True,
        only_return_complete: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Autoregressive sampling method to generate vertices

        Args:
            num_samples: Number of samples to produce.
            context: A dictionary with the type of context to condition upon. This could be class labels or images or voxels.
            max_sample_length: Maximum length of sampled vertex samples. Sequences that do not complete are truncated.
            temperature: Scalar softmax temperature > 0.
            top_k: Number of tokens to keep for top-k sampling.
            top-p: Proportion of probability mass to keep for top-p sampling.
            recenter_verts: If True, center vertex samples around origin. This should be used if model is trained using shift augmentations.
            only_return_complete: If True, only return completed samples. Otherwise return all samples along with completed indicator.

        Returns:
            outputs: Output dictionary with fields
                'completed': Boolean tensor of shape [num_samples]. If True then corresponding sample completed within max_sample_length.
                'vertices': Tensor of samples with shape [num_samples, num_verts, 3].
                'num_vertices': Tensor indicating number of vertices for each example in padded vertex samples.
                'vertices_mask': Tensor of shape [num_samples, num_verts] that masks corresponding invalid elements in vertices.
        """
        global_context, seq_context = self._prepare_context(context)
        num_samples = seq_context.shape[0]

        def _loop_body(
            i: int,
            samples: torch.Tensor,
            cache: Optional[List[Dict[str, torch.Tensor]]] = None,
        ) -> Tuple[int, torch.Tensor]:
            """While-loop body for autoregression calculation.

            Args:
                i: Current iteration in the loop
                samples: tensor of shape [num_samples, i].
                cache: A list of dictionaries in the following format: {'k': torch.Tensor, 'v': torch.Tensor}.
                    Each dictionary in the list represents the cache at the respective decoder layer.
            Returns:
                next_iter: i + 1.
                samples: tensor of shape [num_samples, i + 1] or of shape [num_samples, 2 * i + 1] if cache doesn't exist.
            """
            logits = self._create_dist(
                samples,
                global_context_embedding=global_context,
                sequential_context_embedding=seq_context,
                cache=cache,
                temperature=temperature,
                top_k=top_k,
                top_p=1.0,
            )

            if i == 0:  # z
                logits[:, :, 0] = -1e9

            elif i == 1:  # y
                logits[:, :, 0] = -1e9

            elif i > 2:
                position = i % 3
                if position == 0:  # z
                    logits[:, :, 1 : samples[:, i - 3]] = -1e9

                elif position == 1:  # y
                    logits[:, :, 0] = -1e9
                    if samples[:, -1] == samples[:, -4]:
                        logits[:, :, 1 : samples[:, i - 3]] = -1e9
                elif position == 2:  # x
                    logits[:, :, 0] = -1e9
                    if (
                        samples[:, -1] == samples[:, -4]
                        and samples[:, -2] == samples[:, -5]
                    ):
                        logits[:, :, 1 : samples[:, i - 3]] = -1e9

            logits = top_p_logits(logits, top_p)
            logits = torch.clip(logits, min=-1e9)
            # probs_tmp = torch.softmax(logits, dim=-1)
            cat_dist = torch.distributions.categorical.Categorical(logits=logits)
            next_sample = cat_dist.sample()
            samples = torch.cat([samples, next_sample.to(torch.int32)], dim=1)
            # next_prob = probs_tmp.squeeze()[next_sample.squeeze()]
            return i + 1, samples  # , next_prob

        def _stopping_cond(samples: torch.Tensor) -> bool:
            """
            Stopping condition for sampling while-loop. Looking for stop token (represented by 0)
            Args:
                samples: tensor of shape [num_samples, i], where i is the current iteration.
            Returns:
                token_found: Boolean that represents if stop token has been found.
            """
            nonzero_boolean_matrix = torch.ne(samples, 0)
            reduced_matrix = torch.all(
                nonzero_boolean_matrix, dim=-1
            )  # Checks if stopping token is present in every row
            return torch.any(reduced_matrix)

        samples = torch.zeros([num_samples, 0], dtype=torch.int32).to(self.device)
        cache = self.decoder.initialize_cache(num_samples)
        max_sample_length = max_sample_length or self.max_num_input_verts
        # samples_probs = []
        j = 0
        while _stopping_cond(samples) and j < max_sample_length * 3 + 1:
            j, samples = _loop_body(j, samples, cache)
            # samples_probs.append(a_prob.item())

        completed_samples_boolean = samples == 0  # Checks for stopping token
        completed = torch.any(
            completed_samples_boolean, dim=-1
        )  # Indicates which samples are completed of shape [num_samples,]
        stop_index_completed = torch.argmax(
            completed_samples_boolean.to(torch.int32), dim=-1
        ).to(
            torch.int32
        )  # Indicates where the stopping token occurs in each batch of samples
        stop_index_incomplete = (
            max_sample_length * 3 * torch.ones_like(stop_index_completed)
        )  # Placeholder tensor used to select samples from incomplete samples
        stop_index = torch.where(
            completed, stop_index_completed, stop_index_incomplete
        )  # Stopping Indices of each sample, if completed is true, then stopping index is taken from completed stop index tensor
        num_vertices = torch.floor_divide(stop_index, 3)

        samples = (
            samples[:, : (torch.max(num_vertices) * 3)] - 1
        )  # Selects last possible stopping index
        # samples_probs = samples_probs[:(torch.max(num_vertices) * 3)]
        verts_dequantized = dequantize_verts(samples, self.quantization_bits)
        # Converts vertices to [-1, 1] range
        vertices = torch.reshape(
            verts_dequantized, [num_samples, -1, 3]
        )  # Reshapes into 3D Tensors
        vertices = torch.stack(
            [vertices[..., 2], vertices[..., 1], vertices[..., 0]], dim=-1
        )  # Converts from z-y-x to x-y-z.

        # # Pad samples such that samples of different lengths can be concatenated
        # pad_size = max_sample_length - vertices.shape[1]
        # vertices = F.pad(vertices, [0, 0, 0, pad_size, 0, 0])

        # vertices_mask = (torch.arange(max_sample_length, device=self.device)[None] < num_vertices[:, None]).to(
        #     torch.float32
        # )  # Provides a mask of which vertices to zero out as they were produced after stop token for that batch ended

        if recenter_verts:
            vert_max, _ = torch.max(
                vertices - 1e10 * (1.0 - vertices_mask)[..., None], dim=1, keepdim=True
            )
            vert_min, _ = torch.min(
                vertices + 1e10 * (1.0 - vertices_mask)[..., None], dim=1, keepdim=True
            )
            vert_centers = 0.5 * (vert_max + vert_min)
            vertices = vertices - vert_centers

        # vertices = vertices * vertices_mask[..., None]  # Zeros out vertices produced after stop token

        if only_return_complete:
            vertices = vertices[completed]
            num_vertices = num_vertices[completed]
            vertices_mask = vertices_mask[completed]
            completed = completed[completed]

        outputs = {
            "completed": completed,
            "vertices": vertices,
            "num_vertices": num_vertices,
            # "probs": samples_probs,
        }
        return outputs
