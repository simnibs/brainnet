import torch


def cart_to_sph(points: torch.Tensor) -> torch.Tensor:
    """

    physics/ISO convention

    https://en.wikipedia.org/wiki/Spherical_coordinate_system

    points : x, y, z in columns

    Returns
    -------
    spherical_coordinates: torch.Tensor
        Spherical coordinates of the form (r, theta, phi) where

            theta [  0, pi]   from north to south (lattitude, polar angle)
            phi   [-pi, pi]   around equator (longitude, azimuth)

    """
    r = points.norm(dim=-1)
    # atan2 chooses the correct quadrant
    theta = torch.acos(points[..., 2] / r)
    phi = torch.atan2(points[..., 1], points[..., 0])
    return torch.stack((r, theta, phi), dim=-1)


def sph_to_cart(
    r: float | torch.Tensor, theta: torch.Tensor, phi: torch.Tensor
) -> torch.Tensor:
    """

    Parameters
    ----------

    Returns
    -------
    euclidean_coordinates: torch.Tensor
    """
    theta_sin = theta.sin()
    x = r * theta_sin * phi.cos()
    y = r * theta_sin * phi.sin()
    z = r * theta.cos()
    return torch.stack([x, y, z], dim=-1)


def change_sphere_size(surface, radius) -> None:
    angles = cart_to_sph(surface.vertices)[..., 1:]
    surface.vertices = sph_to_cart(radius, angles[..., 0], angles[..., 1])


def spherical_angle_difference(a: torch.Tensor, b: torch.Tensor):
    """Compute the angle difference between two tensors containing angles. This
    is similar to `b-a` except that it handles the crossings at the poles
    correctly."""
    return torch.angle(torch.exp(1j * a).conj() * torch.exp(1j * b))


def spherical_angle_add(v: torch.Tensor, dv: torch.Tensor):
    """Update the angles in `v` by adding `dv`. `v` and `dv` are tensors such
    that

        v[..., 0] is the angle theta [0, pi]
        v[..., 1] is the angle phi   [-pi, pi]

    as returned by `cart_to_sph` (i.e., physics convention).
    """
    u = torch.angle(torch.exp(1j * v) * torch.exp(1j * dv))

    return u


def compute_central_angle(a, b):
    """Compute the central angle between a and b.

    Parameters
    ----------
    a,b : torch.Tensor
        [..., 0] is the lattitude [0, pi]
        [..., 1] is the longitude [-pi, pi]

    Reference
    ---------
        https://en.wikipedia.org/wiki/Great-circle_distance
    """

    # lattitude: [0,pi]. We call this theta but the reference calls this phi
    # longitude: [-pi,pi]. We call this phi but the reference calls this lambda
    # convert lattitude from [0,pi] to [-pi/2, pi/2]
    half_pi = 0.5 * torch.pi

    lat_a = a[..., 0] - half_pi
    lat_b = b[..., 0] - half_pi
    long_a = a[..., 1]
    long_b = b[..., 1]
    long_delta = spherical_angle_difference(long_a, long_b).abs()

    # Original formula based on spherical law of cosines
    # return torch.acos(phi_a.sin() * phi_b.sin() + phi_a.cos() * phi_b.cos() * theta_delta.cos())

    # Vincenty formula (more numerically stable for small distances)
    cos_lat_a = lat_a.cos()
    cos_lat_b = lat_b.cos()
    sin_lat_a = lat_a.sin()
    sin_lat_b = lat_b.sin()

    cos_long_delta = long_delta.cos()
    sin_long_delta = long_delta.sin()

    y = torch.sqrt(
        (cos_lat_b * sin_long_delta) ** 2
        + (cos_lat_a * sin_lat_b - sin_lat_a * cos_lat_b * cos_long_delta) ** 2
    )
    x = sin_lat_a * sin_lat_b + cos_lat_a * cos_lat_b * cos_long_delta

    return torch.atan2(y, x)


def compute_arc_length(a, b, radius: float | None = None):
    """_summary_

    Parameters
    ----------
    a, b : _type_
        Tensor of shape (..., 2)
    radius : float | None, optional
        _description_, by default None

    Returns
    -------
    _type_
        _description_
    """
    central_angle = compute_central_angle(a, b)
    return central_angle if radius is None else radius * central_angle


def compute_axis_aligned_arc_length(a, b, radius: float | None = None, dim=-1):
    """_summary_

    Parameters
    ----------
    a : _type_
        Tensor containing a single angle
    radius : float | None, optional
        _description_, by default None
    dim : int, optional
        _description_, by default -1

    Returns
    -------
    _type_
        _description_
    """
    angle = spherical_angle_difference(a, b).abs()
    return angle if radius is None else radius * angle


def sph_to_qubit(theta, phi):
    """qubit vector form."""
    sin_theta = theta.sin()
    return torch.stack(
        (sin_theta * phi.cos(), sin_theta * phi.sin(), theta.cos()), dim=-1
    )


def qubit_to_sph(q):
    theta = q[..., -1].acos()
    phi = torch.acos(torch.clamp(q[..., 0] / theta.sin(), -1, 1)) * q[..., 1].sign()
    return theta, phi


# def rotate(v, k, alpha):
#     """Rotate v by alpha (angle) around k (axis).

#     Rodrigues' rotation formula.

#     References
#     ----------
#     https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula

#     """
#     cos_angle = alpha.cos()
#     k_as_v = k.expand_as(v)
#     return (
#         v * cos_angle
#         + torch.cross(v, k_as_v) * alpha.sin()
#         + (v @ k)[..., None] * k_as_v * (1 - cos_angle)
#     )


# def rotate_dim(v, dim, alpha):
#     """Rotate around one of the major axes as specified by `dim`."""
#     cos_angle = alpha.cos()

#     n = v.shape[-1]
#     k = torch.zeros(n, device=v.device)
#     k[dim] = 1.0
#     k_as_v = k.expand_as(v)

#     q = torch.zeros_like(v)
#     q[..., dim] = v[..., dim]

#     return v * cos_angle + torch.cross(v, k_as_v) * alpha.sin() + q * (1 - cos_angle)


# alpha_x = alpha_y = alpha_z = torch.tensor(torch.pi/4.0)

# vv = rotate(vv, alpha_x, torch.tensor([1.0,0.0,0.0]))
# nib.freesurfer.write_geometry("rotx", vv[0].numpy(), f)

# vv = rotate(vv, alpha_y, torch.tensor([0.0,1.0,0.0]))
# nib.freesurfer.write_geometry("rotxy", vv[0].numpy(), f)

# vv = rotate(vv, alpha_z, torch.tensor([0.0,0.0,1.0]))
# nib.freesurfer.write_geometry("rotxyz", vv[0].numpy(), f)


# vv = rotate_major(vv, alpha_x, 0)
# nib.freesurfer.write_geometry("rotx1", vv[0].numpy(), f)

# vv = rotate_major(vv, alpha_y, 1)
# nib.freesurfer.write_geometry("rotxy1", vv[0].numpy(), f)

# vv = rotate_major(vv, alpha_z, 2)
# nib.freesurfer.write_geometry("rotxyz1", vv[0].numpy(), f)

# vv = qubit_rotate(v, alpha/2, k)
# vvv = qubit_rotate(vv, alpha/2, torch.tensor([0.,0.,1.]))
