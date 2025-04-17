import math

import torch
from icosphere import icosphere
from lib.scene import GaussianModel, Scene
from lib.scene.bounding_box import BoundingBox
from lib.utils.general_utils import (
    build_rotation,
    build_rotation_4d,
    build_scaling_rotation_4d,
    get_expon_lr_func,
    inverse_sigmoid,
)


def buildAABBs(pcd: list[GaussianModel]):
    # TODO: dynamic scene
    pcd = pcd[0]
    vertice = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
        ],
        device="cuda",
    )
    S = pcd.get_scaling * 3
    S_max = torch.max(S, dim=-1).values  # (N)
    vertice = S_max[..., None, None] * vertice + pcd.get_world_xyz().unsqueeze(
        1
    ).repeat(
        1, 8, 1
    )  # (N, 8, 3)
    N = vertice.shape[0]
    base_triangle_indices = torch.tensor(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [1, 5, 6],
            [1, 6, 2],
            [0, 3, 7],
            [0, 7, 4],
            [0, 4, 5],
            [0, 5, 1],
            [3, 2, 6],
            [3, 6, 7],
        ],
        device="cuda",
    )  # 12x3
    offsets = torch.arange(0, N * 8, step=8, device="cuda").view(N, 1, 1)  # (N, 1, 1)
    faces = base_triangle_indices + offsets
    faces = faces.int()
    return vertice.view(-1, 3).contiguous(), faces.view(-1, 3).contiguous()


def buildTighestAABBs(means, scalings, rotations, opacities):
    # TODO: dynamic scene
    vertices = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
        ],
        device="cuda",
    ).float()
    faces = torch.tensor(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [1, 5, 6],
            [1, 6, 2],
            [0, 3, 7],
            [0, 7, 4],
            [0, 4, 5],
            [0, 5, 1],
            [3, 2, 6],
            [3, 6, 7],
        ],
        device="cuda",
    )

    normals = torch.linalg.cross(
        vertices[faces[:, 1]] - vertices[faces[:, 0]],
        vertices[faces[:, 2]] - vertices[faces[:, 0]],
    )
    outer_dir = vertices[faces].mean(dim=1) - 0
    sign_transform = torch.sign(torch.sum(normals * outer_dir, dim=-1))  # (N)

    V = vertices.shape[0]
    N = scalings.shape[0]

    S = torch.zeros((N, 3, 3), dtype=torch.float, device="cuda")
    sx2, sy2, sz2 = scalings[:, 0] ** 2, scalings[:, 1] ** 2, scalings[:, 2] ** 2
    R = build_rotation(rotations)
    S[:, 0, 0] = 3 * torch.sqrt(sx2 * R[:, 0, 0] + sy2 * R[:, 0, 1] + sz2 * R[:, 0, 2])
    S[:, 1, 1] = 3 * torch.sqrt(sx2 * R[:, 1, 0] + sy2 * R[:, 1, 1] + sz2 * R[:, 1, 2])
    S[:, 2, 2] = 3 * torch.sqrt(sx2 * R[:, 2, 0] + sy2 * R[:, 2, 1] + sz2 * R[:, 2, 2])

    vertices = (
        vertices.unsqueeze(0).repeat(N, 1, 1).float()
        @ (R @ S).permute(0, 2, 1)
        * means.unsqueeze(1).repeat(1, V, 1)
    )  # (N, V, 3)

    offsets = torch.arange(0, N * V, step=V, device="cuda").view(N, 1, 1)  # (N, 1, 1)
    faces = faces + offsets  # (N, V, 3)

    vertices, faces = (
        vertices.float().view(-1, 3).contiguous(),
        faces.int().view(-1, 3).contiguous(),
    )

    normals = torch.linalg.cross(
        vertices[faces[:, 1]] - vertices[faces[:, 0]],
        vertices[faces[:, 2]] - vertices[faces[:, 0]],
    )
    normals = normals * sign_transform[None, :, None].repeat(N, 1, 3).view(-1, 3)
    normals = normals.float().contiguous()

    return vertices, faces, normals


def buildCuboid(pcd):
    anchor = torch.tensor([1.0, 1.0, 1.0], device="cuda")
    R_w2l = build_rotation(pcd.get_rotation())
    S = pcd.get_scaling
    S_max = torch.max(S, dim=-1).values


def buildHexahedron(pcd: list[GaussianModel], theta: float = math.pi / 4):
    # TODO: dynamic scene
    pcd = pcd[0]
    sec = 1 / math.sin(theta)
    vertices = torch.tensor(
        [
            [math.sqrt(3) * sec, sec, 0],
            [-math.sqrt(3) * sec, sec, 0],
            [0, -2 * sec, 0],
            [0, 0, math.tan(theta) * sec],
            [0, 0, -math.tan(theta) * sec],
        ],
        device="cuda",
    ).repeat(pcd.get_scaling.shape[0], 1, 1)
    V = vertices.shape[1]  # 12

    S = torch.zeros((pcd.get_scaling.shape[0], 3, 3), dtype=torch.float, device="cuda")
    S[:, 0, 0] = pcd.get_scaling[:, 0] * 3
    S[:, 1, 1] = pcd.get_scaling[:, 1] * 3
    S[:, 2, 2] = pcd.get_scaling[:, 2] * 3
    R = build_rotation(pcd.get_rotation()[1])
    vertices = vertices @ (R @ S).permute(0, 2, 1) + pcd.get_world_xyz().unsqueeze(
        1
    ).repeat(
        1, V, 1
    )  # (N, V, 3)

    N = vertices.shape[0]
    base_triangle_indices = torch.tensor(
        [[0, 1, 3], [1, 2, 3], [0, 2, 3], [0, 1, 4], [1, 2, 4], [0, 2, 4]],
        device="cuda",
    )  # 12x3
    offsets = torch.arange(0, N * V, step=V, device="cuda").view(N, 1, 1)  # (N, 1, 1)
    faces = base_triangle_indices + offsets
    faces = faces.int()
    return vertices.view(-1, 3).contiguous(), faces.view(-1, 3).contiguous()


def build2DRectangle(means, scalings, rotations, opacities):  # for 2dgs
    vertices = (
        torch.tensor(
            [
                [-1, 1, 0],
                [-1, -1, 0],
                [1, 1, 0],
                [1, -1, 0],
            ],
            device="cuda",
        )
        .repeat(scalings.shape[0], 1, 1)
        .float()
    )
    V = vertices.shape[1]  # 4
    N = scalings.shape[0]

    S = torch.zeros((scalings.shape[0], 3, 3), dtype=torch.float, device="cuda")
    alpha_min = 1.0 / 255.0
    factor = torch.sqrt(2 * torch.log(opacities / alpha_min)) + 0.01
    factor = factor.reshape(N)
    S[:, 0, 0] = scalings[:, 0] * factor
    S[:, 1, 1] = scalings[:, 1] * factor
    S[:, 2, 2] = 1
    R = build_rotation(rotations)
    vertices = vertices @ (R @ S).permute(0, 2, 1) + means.unsqueeze(1).repeat(
        1, V, 1
    )  # (N, V, 3)

    N = vertices.shape[0]
    base_triangle_indices = torch.tensor(
        [
            [0, 1, 2],
            [2, 3, 1],
        ],
        device="cuda",
    )  # 2x3
    offsets = torch.arange(0, N * V, step=V, device="cuda").view(N, 1, 1)  # (N, 1, 1)
    faces = base_triangle_indices + offsets
    faces = faces.int()

    normals = torch.Tensor([]).cuda()
    return vertices.view(-1, 3).contiguous(), faces.view(-1, 3).contiguous(), normals


def buildIcosahedron(means, scalings, rotations, opacities):
    vertices, faces = icosphere(1)
    vertices = torch.tensor(vertices, device="cuda")
    faces = torch.tensor(faces, device="cuda")
    normals = torch.linalg.cross(
        vertices[faces[:, 1]] - vertices[faces[:, 0]],
        vertices[faces[:, 2]] - vertices[faces[:, 0]],
    )
    outer_dir = vertices[faces].mean(dim=1) - 0
    sign_transform = torch.sign(torch.sum(normals * outer_dir, dim=-1))  # (N)
    V = vertices.shape[0]  # 12
    N = scalings.shape[0]

    S = torch.zeros((N, 3, 3), dtype=torch.float, device="cuda")

    alpha_min = 1.0 / 255.0
    scale = torch.sqrt(2 * torch.log(opacities / alpha_min)) + 0.01
    scale = scale.reshape(N)
    S[:, 0, 0] = scalings[:, 0] * scale
    S[:, 1, 1] = scalings[:, 1] * scale
    S[:, 2, 2] = scalings[:, 2] * scale

    R = build_rotation(rotations)
    vertices = vertices.unsqueeze(0).repeat(N, 1, 1).float() @ (R @ S).permute(
        0, 2, 1
    ) + means.unsqueeze(1).repeat(
        1, V, 1
    )  # (N, V, 3)

    N = vertices.shape[0]
    offsets = torch.arange(0, N * V, step=V, device="cuda").view(N, 1, 1)  # (N, 1, 1)
    faces = faces + offsets  # (N, 20, 3)
    vertices, faces = (
        vertices.float().view(-1, 3).contiguous(),
        faces.int().view(-1, 3).contiguous(),
    )

    normals = torch.linalg.cross(
        vertices[faces[:, 1]] - vertices[faces[:, 0]],
        vertices[faces[:, 2]] - vertices[faces[:, 0]],
    )
    normals = normals * sign_transform[None, :, None].repeat(N, 1, 3).view(-1, 3)
    normals = normals.float().contiguous()

    return vertices, faces, normals


primitiveTypeCallbacks = {
    "AABBs": buildAABBs,
    "tighestAABBs": buildTighestAABBs,
    "Hexahedron": buildHexahedron,
    "2DRectangle": build2DRectangle,
    "Icosahedron": buildIcosahedron,
}
