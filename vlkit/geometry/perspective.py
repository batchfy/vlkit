import numpy as np


def random_perspective_matrix(height, width, distortion_scale=0.2):
    half_height, half_width = height // 2, width // 2

    topleft = [
        int(np.random.randint(0, int(distortion_scale * half_width) + 1)),
        int(np.random.randint(0, int(distortion_scale * half_height) + 1)),
    ]
    topright = [
        int(np.random.randint(width - int(distortion_scale * half_width) - 1, width)),
        int(np.random.randint(0, int(distortion_scale * half_height) + 1)),
    ]
    botright = [
        int(np.random.randint(width - int(distortion_scale * half_width) - 1, width)),
        int(np.random.randint(height - int(distortion_scale * half_height) - 1, height)),
    ]
    botleft = [
        int(np.random.randint(0, int(distortion_scale * half_width) + 1)),
        int(np.random.randint(height - int(distortion_scale * half_height) - 1, height)),
    ]
    startpoints = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
    endpoints = [topleft, topright, botright, botleft]

    a_matrix = np.zeros((2 * len(startpoints), 8))
    for i, (p1, p2) in enumerate(zip(endpoints, startpoints)):
        a_matrix[2 * i, :] = np.array([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        a_matrix[2 * i + 1, :] = np.array([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

    b_matrix = np.array(startpoints).reshape(8)
    res = np.linalg.lstsq(a_matrix, b_matrix, rcond=None)[0]

    H = np.ones(9)
    H[:8] = res
    return H.reshape(3, 3), np.array(startpoints), np.array(endpoints)


def warp_points_perspective(points, transform_matrix):
    assert isinstance(points, np.ndarray)
    assert isinstance(transform_matrix, np.ndarray)
    assert points.ndim == 2 and points.shape[1] == 2
    n = points.shape[0]
    augmented_points = np.concatenate((points, np.ones((n, 1))), axis=1).astype(points.dtype)
    points = (transform_matrix @ augmented_points.T).T
    points = points / points[:,-1].reshape(-1, 1)
    return points[:, :2]
