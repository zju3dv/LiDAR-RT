import torch
from lib.utils.graphics_utils import image2point, range2point

def depth2normal(depth, frame, sensor):
    points = sensor.range2point(frame, depth).reshape(depth.shape[0], depth.shape[1], 3)
    _, ray_d = sensor.get_range_rays(frame)
    normal = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    normal_sign = torch.sign(-torch.sum(normal_map * ray_d[1:-1, 1:-1], dim=-1, keepdim=True))
    normal_map = normal_map * normal_sign
    normal[1:-1, 1:-1, :] = normal_map
    return normal
