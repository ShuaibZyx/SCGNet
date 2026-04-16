import sys

sys.path.append("..")
from src.modules.vertex_model import VertexModel
from src.modules.face_model import FaceModel
from src.modules.data_modules import (
    VertexDataModule,
    FaceDataModule,
)
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
import numpy as np
from src.utils import data_utils
import torch
from lightning.pytorch import LightningModule
from typing import Dict
from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate


def batch_to_device(batch, device):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) or isinstance(value, SparseTensor):
            batch[key] = value.to(device)


def load_vertex_model(checkpoint_path, device):
    vertex_model = (
        VertexModel.load_from_checkpoint(checkpoint_path, batch_size=1, pooling=True)
        .eval()
        .to(device)
    )
    return vertex_model


def load_face_model(checkpoint_path, device):

    face_model = (
        FaceModel.load_from_checkpoint(checkpoint_path, batch_size=1).eval().to(device)
    )
    return face_model


def load_GT_dataloader(dataset_path):
    data_module = FaceDataModule(
        dataset_path=dataset_path,
        batch_size=1,
        quantization_bits=8,
        augmentation=False,
        shuffle_vertices=False,
    )
    test_dataloader = data_module.test_dataloader()
    return test_dataloader


def load_test_dataloader(dataset_path):
    data_module = VertexDataModule(
        dataset_path=dataset_path,
        batch_size=1,
        quantization_bits=8,
        augmentation=False,
    )
    test_dataloader = data_module.test_dataloader()
    return test_dataloader


def sample_from_vertex_model(
    vertex_model: LightningModule,
    vertex_batch: Dict[str, torch.Tensor],
    top_p: float,
    sample_mask: bool,
    device,
) -> Dict[str, torch.Tensor]:
    """Runs vertex model sampling procedure

    Args:
        vertex_model: Lightning module with trained weights
        context: Dictionary that contains class labels

    Returns
        samples: Sampled vertices along with masks and other indicator tensors
    """
    with torch.no_grad():
        if sample_mask:
            vertex_samples = vertex_model.sample_mask(
                context=vertex_batch,
                max_sample_length=100,
                top_p=top_p,
                recenter_verts=False,
                only_return_complete=False,
            )
        else:
            vertex_samples = vertex_model.sample(
                context=vertex_batch,
                max_sample_length=100,
                top_p=top_p,
                recenter_verts=False,
                only_return_complete=False,
            )
    curr_vertices = vertex_samples["vertices"][0]
    num_vertices = vertex_samples["num_vertices"][0]
    pred_vertices = curr_vertices[:num_vertices].detach().to(device)
    return pred_vertices


def sample_from_face_model(
    face_model: LightningModule,
    face_batch: Dict[str, torch.Tensor],
    top_p: float,
    sample_mask: bool,
    device,
) -> Dict[str, torch.Tensor]:
    """Runs face model sampling procedure

    Args:
        face_model: Lightning module with trained weights
        context: Dictionary that contains vertices and masks

    Returns:
        samples: Sampled faces along with masks and other indicator tensors
    """
    with torch.no_grad():
        if sample_mask:
            face_samples = face_model.sample_mask(
                context=face_batch,
                max_sample_length=300,
                top_p=top_p,
                only_return_complete=False,
            )
        else:
            face_samples = face_model.sample(
                context=face_batch,
                max_sample_length=100,
                top_p=top_p,
                only_return_complete=False,
            )
    curr_faces = face_samples["faces"][0]
    num_face_indices = face_samples["num_face_indices"][0]
    pred_faces = data_utils.unflatten_faces(
        curr_faces[:num_face_indices].detach().to(device)
    )
    return pred_faces


def preprocess_v(pred_v, pc_sparse, device):
    pred_v = pred_v.to(device)
    pc_sparse = pc_sparse.to(device)

    pred_vertices = data_utils.quantize_verts(pred_v)
    pred_vertices = torch.unique(pred_vertices, dim=0)
    sort_inds = data_utils.torch_lexsort(pred_vertices.T)
    pred_vertices = pred_vertices[sort_inds].to(torch.int32)

    vs_sparse = sparse_collate(
        [
            SparseTensor(
                coords=pred_vertices,
                feats=torch.ones(pred_vertices.shape[0], 1, device=device).to(device),
            )
        ]
    )

    f_batch = {}
    f_batch["pc_sparse"] = pc_sparse
    f_batch["vs_sparse"] = vs_sparse

    return f_batch


def v_have_stop_token(vs):
    return True if len(vs) < 100 else False


def f_have_stop_token(fs):
    return True if len(fs) < 500 else False


def is_floor_covering_pointcloudxy(vs, pts, info, coverage_rate_thres=0.7):
    vs_floor_inds = torch.where(vs[:, -1] < vs[:, -1].min() + 0.5 / info["scale"])[0]
    points1 = vs[vs_floor_inds][:, :2]
    points2 = pts[:, :2]

    if len(points1) < 3:
        return False, None

    hull1 = ConvexHull(points1)
    hull2 = ConvexHull(points2)

    poly1 = Polygon(points1[hull1.vertices])
    poly2 = Polygon(points2[hull2.vertices])

    intersection = poly1.intersection(poly2)
    coverage_rate = intersection.area / poly2.area

    return coverage_rate > coverage_rate_thres, vs_floor_inds


def are_missing_floor_vertices(vs, vs_floor_inds):
    vs_floor = vs[vs_floor_inds][:, :2]
    hull = ConvexHull(vs_floor)
    hull_points = vs_floor[hull.vertices]

    angles = []
    num_points = len(hull_points)
    for i in range(num_points):
        p1 = hull_points[i]
        p2 = hull_points[(i + 1) % num_points]
        p3 = hull_points[(i + 2) % num_points]
        angle = calculate_angle(p1, p2, p3)
        angles.append(angle)

    return any(angle < 60 for angle in angles)


def are_missing_floor_faces(vs, fs, vs_floor_inds):
    floor_f_bool_list = [
        all(element in list(vs_floor_inds) for element in f) for f in fs
    ]

    if np.sum(floor_f_bool_list) == 0:
        return True
    else:
        result_floor_fs = [
            fs[_] for _, mask_value in enumerate(floor_f_bool_list) if mask_value
        ]
        valid_poly_bools = []
        for one_result_floor_fs in result_floor_fs:
            one_result_floor_fs = [t.item() for t in one_result_floor_fs]
            valid_poly_bools.append(Polygon(vs[one_result_floor_fs, :2]).is_valid)
        return not (all(valid_poly_bools))


def calculate_angle(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    angle_rad = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    angle_deg = np.degrees(angle_rad)
    return angle_deg


def extract_edges_from_faces(faces):
    edges = set()

    for face in faces:
        for i in range(len(face)):
            edge = tuple(sorted([face[i], face[(i + 1) % len(face)]]))
            edges.add(edge)

    return list(edges)


def is_almost_vertical(point1, point2, angle_threshold_degrees):
    """
    Check if the line segment defined by two endpoints is almost vertical,
    within a specified angle threshold from the z-axis.

    Args:
    - point1: A tuple (x1, y1, z1) representing the first endpoint of the line segment.
    - point2: A tuple (x2, y2, z2) representing the second endpoint of the line segment.
    - angle_threshold_degrees: The maximum angle (in degrees) allowed from the z-axis.

    Returns:
    - A boolean indicating whether the line segment is almost vertical.
    """

    # Create vectors from points and the z-axis
    line_vector = np.array(point2) - np.array(point1)
    if line_vector[-1] < 0:
        line_vector = np.array(point1) - np.array(point2)
    z_axis_vector = np.array([0, 0, 1])

    # Calculate the angle between the line_vector and the z-axis
    cosine_angle = np.dot(line_vector, z_axis_vector) / (
        np.linalg.norm(line_vector) * np.linalg.norm(z_axis_vector)
    )

    # Ensure the cosine value is within valid range [-1, 1] to avoid NaN errors due to floating-point arithmetic
    cosine_angle = np.clip(cosine_angle, -1, 1)

    angle_radians = np.arccos(cosine_angle)

    # Convert angle to degrees
    angle_degrees = np.degrees(angle_radians)

    # Check if the angle is within the threshold
    return angle_degrees <= angle_threshold_degrees
