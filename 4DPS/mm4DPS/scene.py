"""The scene and the person in it.

Two modules merged here by `docs/CONSOLIDATION_AND_OO_PLAN.md` C8, from
`forward/scene.py` and `forward/person.py`. No name collisions.

    scene    range profiles, clutter removal, the trunk range, the diffuse floor
    person   the person point cloud and the transforms applied to it
"""

from __future__ import annotations

import numpy as np
import torch

from dataclasses import dataclass

from mm4DPS.data import range_angle_slow_time_cube, range_bin_spacing_m


# --------------------------------------------------------------------
# scene: range profiles, clutter removal, the diffuse floor
# --------------------------------------------------------------------

def project_onto_steering(
    coefficients: np.ndarray, steering: np.ndarray,
) -> np.ndarray:
    """Contract latent scatterer coefficients with the steering tensor.

    ``coefficients`` is ``(range, scatterer)`` or ``(loop, range, scatterer)``
    and ``steering`` is ``(range, scatterer, TX, RX)``; the result is
    ``(TX, RX, range)`` or ``(loop, TX, RX, range)`` — exactly the einsum
    contractions ``"rs,rstx->txr"`` and ``"lrs,rstx->ltxr"``. ``np.einsum``
    cannot map either index pattern onto a GEMM and falls back to its own
    loop at ~0.7 GFLOP/s; reshaping the same contraction into a matmul
    batched over the range axis reaches BLAS (measured 8.5x on the shipped
    shapes, max_rel 2e-16 against the einsum).
    """
    n_range, n_scatterer, n_tx, n_rx = steering.shape
    matrix = steering.reshape(n_range, n_scatterer, n_tx * n_rx)
    if coefficients.ndim == 2:
        product = np.matmul(coefficients[:, None, :], matrix)
        return product[:, 0, :].T.reshape(n_tx, n_rx, n_range)
    product = np.matmul(coefficients.transpose(1, 0, 2), matrix)
    return product.transpose(1, 2, 0).reshape(-1, n_tx, n_rx, n_range)



def project_onto_steering_torch(coefficients, steering):
    """``project_onto_steering`` for torch tensors, same axes, same order.

    Kept beside the numpy form deliberately: the two are pinned against each
    other in the suite, and the numpy one remains the arbiter.
    """
    n_range, n_scatterer, n_tx, n_rx = steering.shape
    matrix = steering.reshape(n_range, n_scatterer, n_tx * n_rx)
    if coefficients.ndim == 2:
        product = coefficients[:, None, :] @ matrix
        return product[:, 0, :].T.reshape(n_tx, n_rx, n_range)
    product = coefficients.permute(1, 0, 2) @ matrix
    return product.permute(1, 2, 0).reshape(-1, n_tx, n_rx, n_range)



def bistatic_carrier_phase(
    points_xyz: np.ndarray,
    tx_xyz: np.ndarray,
    rx_xyz: np.ndarray,
    wavelength_m: float,
) -> np.ndarray:
    """Geometric carrier phase (point, TX, RX), modulo no particular branch."""
    points = np.asarray(points_xyz, dtype=np.float64)
    tx = np.asarray(tx_xyz, dtype=np.float64)
    rx = np.asarray(rx_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_xyz must have shape (N,3)")
    if tx.ndim != 2 or tx.shape[1] != 3 or rx.ndim != 2 or rx.shape[1] != 3:
        raise ValueError("TX and RX positions must have shape (channels,3)")
    if wavelength_m <= 0:
        raise ValueError("wavelength_m must be positive")
    tx_distance = np.linalg.norm(points[:, None, :] - tx[None, :, :], axis=-1)
    rx_distance = np.linalg.norm(points[:, None, :] - rx[None, :, :], axis=-1)
    path_length = tx_distance[:, :, None] + rx_distance[:, None, :]
    return -2.0 * np.pi * path_length / wavelength_m



@dataclass
class PhaseCoherentStaticJitter:
    """Stationary AR(1) displacement along learned surface normals.

    ``common_variance_fraction`` adds a weak global component shared by all
    points without removing local micro-motion.  It is a fraction of variance,
    not amplitude.  One ``step`` must be called per camera/radar frame and the
    returned positions reused for both radars.
    """

    base_positions: np.ndarray
    normals: np.ndarray
    sigma_m: float = 50e-6
    temporal_correlation: float = 0.6
    common_variance_fraction: float = 0.2
    seed: int = 0

    def __post_init__(self) -> None:
        self.base_positions = np.asarray(self.base_positions, dtype=np.float32).copy()
        normals = np.asarray(self.normals, dtype=np.float32)
        if self.base_positions.ndim != 2 or self.base_positions.shape[1] != 3:
            raise ValueError("base_positions must have shape (N,3)")
        if normals.shape != self.base_positions.shape:
            raise ValueError("normals must match base_positions")
        if self.sigma_m < 0:
            raise ValueError("sigma_m must be nonnegative")
        if not 0.0 <= self.temporal_correlation < 1.0:
            raise ValueError("temporal_correlation must lie in [0,1)")
        if not 0.0 <= self.common_variance_fraction <= 1.0:
            raise ValueError("common_variance_fraction must lie in [0,1]")
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        self.normals = normals / np.maximum(lengths, 1e-12)
        self._rng = np.random.default_rng(self.seed)
        # Draw the initial state from the stationary distribution so frame zero
        # is not artificially quieter than the rest of the sequence.
        self._local_state = self._rng.normal(
            0.0, self.sigma_m, len(self.base_positions),
        ).astype(np.float32)
        self._common_state = float(self._rng.normal(0.0, self.sigma_m))
        self.frame_index = -1

    def step(self, displacement_scale: float = 1.0) -> np.ndarray:
        """Advance the latent state and return one coherent 3-D point set."""
        if not np.isfinite(displacement_scale) or displacement_scale < 0:
            raise ValueError("displacement_scale must be finite and nonnegative")
        rho = self.temporal_correlation
        innovation_scale = np.sqrt(max(1.0 - rho * rho, 0.0)) * self.sigma_m
        self._local_state = (
            rho * self._local_state
            + self._rng.normal(0.0, innovation_scale, len(self._local_state))
        ).astype(np.float32)
        self._common_state = float(
            rho * self._common_state + self._rng.normal(0.0, innovation_scale)
        )
        common_weight = np.sqrt(self.common_variance_fraction)
        local_weight = np.sqrt(1.0 - self.common_variance_fraction)
        displacement = (
            local_weight * self._local_state + common_weight * self._common_state
        )
        self.frame_index += 1
        return (
            self.base_positions
            + float(displacement_scale) * displacement[:, None] * self.normals
        )

    def metadata(self, wavelength_m: float | None = None) -> dict:
        result = {
            "model": "stationary AR(1) displacement along learned surface normals",
            "sigma_m": self.sigma_m,
            "sigma_um": self.sigma_m * 1e6,
            "temporal_correlation": self.temporal_correlation,
            "common_variance_fraction": self.common_variance_fraction,
            "seed": self.seed,
            "phase_coherence_contract": (
                "one perturbed 3-D point set is reused by every TX/RX channel and both "
                "radars; channel phases are recomputed from bistatic path lengths"
            ),
        }
        if wavelength_m is not None:
            result["monostatic_roundtrip_phase_rms_rad_approx"] = (
                4.0 * np.pi * self.sigma_m / wavelength_m
            )
        return result



@dataclass
class PhaseCoherentDiffuseFloor:
    """Radar-specific weak scatterer field for unsupported range bins.

    This is deliberately separate from scene-point jitter.  Every latent
    scatterer has a fixed metric range/direction and one complex AR(1)
    amplitude shared by all TX/RX paths.  Consequently the virtual-array phase
    vector is a valid bistatic steering vector rather than independent channel
    or RA-pixel noise.
    """

    tx_positions: np.ndarray
    rx_positions: np.ndarray
    radar_center: np.ndarray
    array_axis: np.ndarray
    wavelength_m: float
    range_bin_spacing_m: float
    active_range_bins: np.ndarray
    scatterers_per_bin: int = 4
    temporal_correlation: float = 0.3
    seed: int = 0
    spatial_sine_by_range: np.ndarray | None = None
    # Where the spectral cube's inverse FFT and steering projection run.
    # "numpy" is the arbiter and the default. "torch" moves those two stages
    # to `device` in complex128 and leaves **every random draw and the whole
    # AR(1) state on the CPU**, so the nuisance realization is the one the
    # seed has always produced; only the arithmetic that turns it into a
    # profile moves. That is render W1 option 1, and it is why this is an
    # exact change rather than a different sample.
    backend: str = "numpy"
    device: str = "cuda:0"
    # Where the *random draws* and the AR(1) latent state live. "cpu" keeps
    # the seeded numpy stream and is exact against every earlier render.
    # "gpu" draws with torch on the device, which is a **different
    # realization of the same process** -- not a numerical difference, a
    # different sample -- and is render W1(2). It also removes the last
    # large host<->device transfers on the nuisance path: the latent cube is
    # born on the device and the profile never comes back.
    draws: str = "cpu"

    def __post_init__(self) -> None:
        self.tx_positions = np.asarray(self.tx_positions, dtype=np.float64)
        self.rx_positions = np.asarray(self.rx_positions, dtype=np.float64)
        self.radar_center = np.asarray(self.radar_center, dtype=np.float64)
        axis = np.asarray(self.array_axis, dtype=np.float64)
        if self.tx_positions.ndim != 2 or self.tx_positions.shape[1] != 3:
            raise ValueError("tx_positions must have shape (TX,3)")
        if self.rx_positions.ndim != 2 or self.rx_positions.shape[1] != 3:
            raise ValueError("rx_positions must have shape (RX,3)")
        if self.radar_center.shape != (3,) or axis.shape != (3,):
            raise ValueError("radar_center and array_axis must be 3-vectors")
        if self.wavelength_m <= 0 or self.range_bin_spacing_m <= 0:
            raise ValueError("wavelength and range spacing must be positive")
        if self.scatterers_per_bin <= 0:
            raise ValueError("scatterers_per_bin must be positive")
        if not 0.0 <= self.temporal_correlation < 1.0:
            raise ValueError("temporal_correlation must lie in [0,1)")
        axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
        boresight = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        boresight -= axis * float(boresight @ axis)
        boresight /= max(float(np.linalg.norm(boresight)), 1e-12)
        self.array_axis = axis
        self.active_range_bins = np.asarray(self.active_range_bins, dtype=bool)
        if self.active_range_bins.ndim != 1:
            raise ValueError("active_range_bins must be one-dimensional")
        self._rng = np.random.default_rng(self.seed)
        n_range = len(self.active_range_bins)
        if self.spatial_sine_by_range is None:
            spatial_sine = self._rng.uniform(
                -0.9, 0.9, (n_range, self.scatterers_per_bin),
            )
            direction_source = "seeded uniform local-angle hypotheses"
        else:
            spatial_sine = np.asarray(self.spatial_sine_by_range, dtype=np.float64)
            if spatial_sine.shape != (n_range, self.scatterers_per_bin):
                raise ValueError(
                    "spatial_sine_by_range must have shape "
                    "(range_bins, scatterers_per_bin)"
                )
            spatial_sine = np.clip(spatial_sine, -1.0, 1.0)
            direction_source = "externally calibrated local-angle hypotheses"
        self.spatial_sine_by_range = spatial_sine
        self.direction_source = direction_source
        forward = np.sqrt(np.maximum(1.0 - spatial_sine ** 2, 0.0))
        direction = (
            forward[..., None] * boresight[None, None, :]
            + spatial_sine[..., None] * axis[None, None, :]
        )
        ranges = np.arange(n_range, dtype=np.float64) * self.range_bin_spacing_m
        points = (
            self.radar_center[None, None, :]
            + ranges[:, None, None] * direction
        )
        self.latent_points = points
        phase = bistatic_carrier_phase(
            points.reshape(-1, 3), self.tx_positions, self.rx_positions,
            self.wavelength_m,
        )
        self._steering = np.exp(1j * phase).reshape(
            n_range, self.scatterers_per_bin,
            len(self.tx_positions), len(self.rx_positions),
        )
        self._fixed_spectrum = None
        self._spectral_state = None
        self.frame_index = -1
        if self.backend not in ("numpy", "torch"):
            raise ValueError("backend must be 'numpy' or 'torch'")
        if self.draws not in ("cpu", "gpu"):
            raise ValueError("draws must be 'cpu' or 'gpu'")
        if self.draws == "gpu" and self.backend != "torch":
            raise ValueError("draws='gpu' requires backend='torch'")
        self._steering_device = None
        self._torch_rng = None
        if self.draws == "gpu":
            import torch

            self._torch_rng = torch.Generator(device=self.device)
            self._torch_rng.manual_seed(int(self.seed))
        self._state = self._circular_normal(
            (n_range, self.scatterers_per_bin)
        )

    def _device_steering(self):
        """The steering tensor on the device, uploaded once (9.4 MB)."""
        import torch

        if self._steering_device is None:
            self._steering_device = torch.from_numpy(
                np.ascontiguousarray(self._steering)
            ).to(self.device)
        return self._steering_device

    def _circular_normal(self, shape: tuple[int, ...]) -> np.ndarray:
        """One circular-complex Gaussian draw with unit total variance.

        Written as ``(rng.normal(size) + 1j * rng.normal(size)) / sqrt(2)``
        this materializes two full-size complex temporaries — at the shipped
        ``(64, 256, 48)`` latent shape, 12.6 MB each — before the result.
        Drawing both parts in one generator call and assigning them straight
        into the output is the **same values in the same stream order**
        followed by the **same complex division**, and is bitwise identical
        (pinned by the suite, together with the generator's position
        afterwards): 15.6 ms -> 13.2 ms per call.

        That is close to the floor for an exact form. The generator itself is
        10.7 ms of the 13.2, so the draws cannot go much below ~21 ms/frame
        without changing which numbers come out — which is plan W1(2), a
        different realization rather than a faster route to this one.
        """
        count = int(np.prod(shape))
        if self.draws == "gpu":
            import torch

            both = torch.randn(
                2 * count, dtype=torch.float64, device=self.device,
                generator=self._torch_rng,
            )
            return torch.complex(
                both[:count].reshape(shape), both[count:].reshape(shape),
            ) / np.sqrt(2.0)
        both_parts = self._rng.standard_normal(2 * count)
        result = np.empty(shape, dtype=np.complex128)
        result.real = both_parts[:count].reshape(shape)
        result.imag = both_parts[count:].reshape(shape)
        result /= np.sqrt(2.0)
        return result

    def step(self, range_amplitudes: np.ndarray | None = None) -> np.ndarray:
        """Return complex (TX,RX,range) profiles with unit latent RMS."""
        rho = self.temporal_correlation
        self._state = (
            rho * self._state
            + np.sqrt(max(1.0 - rho * rho, 0.0))
            * self._circular_normal(self._state.shape)
        )
        if range_amplitudes is None:
            amplitude = self.active_range_bins.astype(np.float64)
        else:
            amplitude = np.asarray(range_amplitudes, dtype=np.float64)
            if amplitude.shape != self.active_range_bins.shape:
                raise ValueError("range_amplitudes must match active_range_bins")
            amplitude = amplitude * self.active_range_bins
        if self.draws == "gpu":
            import torch

            amplitude = torch.from_numpy(amplitude).to(self.device)
            weighted = (
                self._state * amplitude[:, None]
                / np.sqrt(self.scatterers_per_bin)
            )
            profile = project_onto_steering_torch(
                weighted, self._device_steering(),
            )
            self.frame_index += 1
            return profile.to(torch.complex64)
        weighted = (
            self._state * amplitude[:, None] / np.sqrt(self.scatterers_per_bin)
        )
        profile = project_onto_steering(weighted, self._steering)
        self.frame_index += 1
        return profile.astype(np.complex64)

    def slow_time_cube(
        self, range_amplitudes: np.ndarray | None = None, loops: int = 64,
    ) -> np.ndarray:
        """Return coherent loop-varying profiles with shape (loop,TX,RX,range).

        The latent complex coefficient changes only between slow-time loops.
        Within a loop, every virtual channel still observes the same latent
        scatterer through its exact bistatic steering phase.  This is suitable
        for a weak hardware/diffuse nuisance floor; it is not independent RD or
        virtual-channel noise.
        """
        if loops <= 0:
            raise ValueError("loops must be positive")
        return np.stack([
            self.step(range_amplitudes) for _ in range(loops)
        ]).astype(np.complex64, copy=False)

    def slow_time_cube_from_spectrum(
        self,
        doppler_range_amplitudes: np.ndarray,
        frame_fixed_fraction: float = 0.0,
        temporal_correlation_by_doppler_range: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return a phase-coherent cube with a calibrated Doppler-range spectrum.

        ``doppler_range_amplitudes`` is fftshifted and has shape
        ``(loops, range)``. Each latent angular scatterer receives a
        circular-complex spectral realization with that expected envelope; its
        fluctuating component follows the configured frame-axis AR(1) process.
        The inverse FFT produces slow-time coefficients. A ``sqrt(loops)`` factor
        preserves unit slow-time RMS under NumPy's normalized inverse FFT, so
        the amplitude scale is compatible with :meth:`slow_time_cube`. The same
        coefficient and exact bistatic steering vector are still shared across
        every TX/RX channel. This matches background RD statistics without
        copying measured noise samples or injecting independent RD pixels.
        """
        if not 0.0 <= frame_fixed_fraction <= 1.0:
            raise ValueError("frame_fixed_fraction must lie in [0,1]")
        amplitude = np.asarray(doppler_range_amplitudes, dtype=np.float64)
        if amplitude.ndim != 2 or amplitude.shape[1] != len(self.active_range_bins):
            raise ValueError(
                "doppler_range_amplitudes must have shape (loops, range_bins)"
            )
        if not np.all(np.isfinite(amplitude)) or np.any(amplitude < 0):
            raise ValueError("doppler_range_amplitudes must be finite and nonnegative")
        amplitude = amplitude * self.active_range_bins[None, :]
        latent_shape = (
            amplitude.shape[0],
            amplitude.shape[1],
            self.scatterers_per_bin,
        )
        innovation = self._circular_normal(latent_shape)
        on_device = self.draws == "gpu"
        if on_device:
            import torch

            # Only the small per-frame maps cross: (loops, range) float64 is
            # 131 kB each against the 6.3 MB latent cube, which is born on
            # the device and never leaves it.
            amplitude = torch.from_numpy(amplitude).to(self.device)
        if temporal_correlation_by_doppler_range is None:
            rho: float | np.ndarray = self.temporal_correlation
        else:
            correlation = np.asarray(
                temporal_correlation_by_doppler_range, dtype=np.float64
            )
            if correlation.shape != amplitude.shape:
                raise ValueError(
                    "temporal correlation map must match Doppler×range amplitudes"
                )
            if (
                not np.all(np.isfinite(correlation))
                or np.any(correlation < 0)
                or np.any(correlation >= 1)
            ):
                raise ValueError(
                    "temporal correlation map values must lie in [0,1)"
                )
            rho = correlation[:, :, None]
            if on_device:
                rho = torch.from_numpy(correlation).to(self.device)[:, :, None]
        if (
            self._spectral_state is None
            or self._spectral_state.shape != innovation.shape
        ):
            self._spectral_state = innovation
        elif on_device:
            self._spectral_state = (
                rho * self._spectral_state
                + torch.clamp(1.0 - rho * rho, min=0.0).sqrt() * innovation
            )
        else:
            self._spectral_state = (
                rho * self._spectral_state
                + np.sqrt(np.maximum(1.0 - rho * rho, 0.0)) * innovation
            )
        if frame_fixed_fraction > 0:
            if (
                self._fixed_spectrum is None
                or self._fixed_spectrum.shape != latent_shape
            ):
                self._fixed_spectrum = self._circular_normal(latent_shape)
            latent = (
                np.sqrt(frame_fixed_fraction) * self._fixed_spectrum
                + np.sqrt(1.0 - frame_fixed_fraction) * self._spectral_state
            )
        else:
            latent = self._spectral_state
        spectrum = (
            latent
            * amplitude[:, :, None]
            * np.sqrt(amplitude.shape[0])
            / np.sqrt(self.scatterers_per_bin)
        )
        self.frame_index += 1
        if self.backend == "torch":
            import torch

            # With draws="cpu" the realization was produced on the host and
            # only the inverse FFT and the projection move, so the sample is
            # unchanged. With draws="gpu" the whole chain is already on the
            # device and nothing crosses back.
            spectrum_device = (
                spectrum if on_device
                else torch.from_numpy(spectrum).to(self.device)
            )
            slow_time = torch.fft.ifft(
                torch.fft.ifftshift(spectrum_device, dim=0), dim=0,
            )
            profile = project_onto_steering_torch(
                slow_time, self._device_steering(),
            )
            profile = profile.to(torch.complex64)
            return profile if on_device else profile.cpu().numpy()
        slow_time = np.fft.ifft(
            np.fft.ifftshift(spectrum, axes=0), axis=0,
        )
        profile = project_onto_steering(slow_time, self._steering)
        return profile.astype(np.complex64)

    def metadata(self) -> dict:
        return {
            "model": "radar-specific metric latent scatterers with complex AR(1) amplitude",
            "active_range_bin_count": int(self.active_range_bins.sum()),
            "scatterers_per_bin": self.scatterers_per_bin,
            "temporal_correlation": self.temporal_correlation,
            "seed": self.seed,
            "direction_source": self.direction_source,
            "spectral_cube_backend": self.backend,
            "random_draws": self.draws,
            "random_stream_device": (
                "seeded torch Philox on the device; a DIFFERENT realization "
                "of the same process from the numpy stream, not a rounding "
                "difference" if self.draws == "gpu" else
                "seeded numpy PCG64 on the host; the realization does not "
                "depend on where the inverse FFT and projection run"
            ),
            "phase_coherence_contract": (
                "each latent scatterer's complex amplitude is shared across every TX/RX; "
                "relative phases are exact bistatic steering phases"
            ),
            "scope_warning": (
                "nuisance-floor model for unsupported range bins, not claimed scene geometry"
            ),
        }

# --------------------------------------------------------------------
# person: the point cloud and its transforms
# --------------------------------------------------------------------

def transform_person_pcl(
    pcl: np.ndarray,
    depth_scale: float = 1.0,
    depth_offset_m: float = 0.0,
) -> np.ndarray:
    """Apply ``y'=scale*y+offset`` and inverse-transpose-transform normals."""
    values = np.asarray(pcl, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError("person pcl must have shape (N,6)")
    if not np.isfinite(depth_scale) or depth_scale <= 0:
        raise ValueError("depth_scale must be positive and finite")
    if not np.isfinite(depth_offset_m):
        raise ValueError("depth_offset_m must be finite")
    result = values.copy()
    result[:, 1] = float(depth_scale) * result[:, 1] + float(depth_offset_m)
    # For A=diag(1,s,1), normals transform by A^{-T}.
    result[:, 4] /= float(depth_scale)
    result[:, 3:6] /= np.maximum(
        np.linalg.norm(result[:, 3:6], axis=1, keepdims=True), 1e-12,
    )
    return result



def transform_person_joints(
    joints_xyz: np.ndarray,
    depth_scale: float = 1.0,
    depth_offset_m: float = 0.0,
) -> np.ndarray:
    values = np.asarray(joints_xyz, dtype=np.float32)
    if values.shape[-1] != 3:
        raise ValueError("joints must end in XYZ coordinates")
    if not np.isfinite(depth_scale) or depth_scale <= 0:
        raise ValueError("depth_scale must be positive and finite")
    result = values.copy()
    result[..., 1] = float(depth_scale) * result[..., 1] + float(depth_offset_m)
    return result



def surface_solid_angle_weights(
    points_xyz: np.ndarray,
    normals_xyz: np.ndarray,
    radar_centre_xyz: np.ndarray,
    normalize: bool = True,
    minimum_cosine: float = 1e-3,
) -> np.ndarray:
    """The Monte-Carlo weight `areas` is supposed to carry, per point.

    `mm3DPS/rasterizer_factorized.py` documents its `areas` slot as

        Hemisphere solid angle: density-ratio MC weight (p/q x opacity)
        or surface integral weight (A x cos_o / d_rx^2)

    and forms the receive amplitude as ``sqrt(G_rx * dOmega)`` with **no
    explicit 1/d_rx** -- the receive-side range falloff and the obliquity live
    inside that slot. 3DPS can pass 1.0 because it cosine-hemisphere
    importance-resamples its point set (``train_gaussian.py``, "Cosine-
    hemisphere importance resample", prob proportional to cos_bore), which
    makes p/q = 1 by construction.

    This project's person cloud is a metric-depth unprojection intersected
    with a skeleton proxy and is **not** resampled, so p/q != 1 and a constant
    is not a valid weight. Measured on a sequence-102 frame, the correct
    weight varies by **6,600x** across the cloud, almost all of it obliquity:
    grazing points, which should contribute almost nothing, are currently
    weighted the same as face-on ones.

    Returns ``cos_o / d^2`` per point, with ``A`` treated as constant across
    the cloud (a uniform-density surface sampling assumption). With
    ``normalize`` the result has unit mean, so this installs the *relative*
    per-point structure and leaves the global level exactly where it was --
    the level is a separate question, and mixing the two would make this
    change untestable.

    Note what this does **not** fix. The two HuPR radars are 3 cm apart, so
    ``cos_o / d^2`` differs between them by a factor of 0.9984: a geometric
    weight cannot explain any per-radar amplitude asymmetry, and the
    ``PERSON_VERTICAL_MULTIPLIER`` constant is not a missing Jacobian.
    """
    points = np.asarray(points_xyz, dtype=np.float64)
    normals = np.asarray(normals_xyz, dtype=np.float64)
    centre = np.asarray(radar_centre_xyz, dtype=np.float64).reshape(3)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N,3)")
    if normals.shape != points.shape:
        raise ValueError("normals must match points")
    delta = points - centre[None, :]
    distance = np.linalg.norm(delta, axis=1)
    if not np.all(np.isfinite(distance)) or np.any(distance <= 0):
        raise ValueError("points must be at a positive finite range")
    unit = normals / np.maximum(
        np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    towards_radar = -delta / distance[:, None]
    # |cos| rather than cos: the cloud's normals are not consistently oriented
    # and a sign flip is an orientation artefact, not a back-facing surface.
    cosine = np.abs((unit * towards_radar).sum(axis=1))
    weight = np.maximum(cosine, minimum_cosine) / distance ** 2
    if normalize:
        weight = weight / max(float(weight.mean()), 1e-30)
    return weight.astype(np.float32)


def surface_solid_angle_weights_torch(points_xyz, normals_xyz, radar_centre_xyz,
                                      normalize: bool = True, minimum_cosine: float = 1e-3):
    """:func:`surface_solid_angle_weights` on the device, ``(N,)`` in the points' dtype (float64 inside)."""
    import torch
    points = points_xyz.double()
    normals = normals_xyz.double()
    centre = torch.as_tensor(radar_centre_xyz, dtype=torch.float64, device=points.device).reshape(3)
    delta = points - centre[None, :]
    distance = delta.norm(dim=1)
    unit = normals / normals.norm(dim=1, keepdim=True).clamp_min(1e-12)
    towards_radar = -delta / distance[:, None]
    cosine = (unit * towards_radar).sum(dim=1).abs()
    weight = torch.clamp(cosine, min=minimum_cosine) / distance ** 2
    if normalize:
        weight = weight / weight.mean()
    return weight.to(points_xyz.dtype)



# ---------------------------------------------------------------------------
# The physical solid-angle weight (docs/STATIC_SEQS_PLAN.md, Option B, 2026-09-19).
#
# The renderer's `areas` slot is the receiver-centred solid angle a point represents, dOmega_i = A_i |cos theta_i| /
# d_i^2 (3DPS section 3: the surface-to-hemisphere Jacobian; the receive amplitude is sqrt(G_rx dOmega) with no
# other 1/d_rx). Neither point set of this project is solid-angle uniform (the scene is voxel + farthest-point
# sampled, the person is SMPL's vertices), so the slot is computed from the geometry: A_i is the barycentric vertex
# area of the posed SMPL mesh for the person and the local area per point of the cloud for the scene. Both sets are
# expressed in one unit, the scene's median dOmega, so the absolute receiver gain keeps its scale; nothing in the
# slot is learned.
# ---------------------------------------------------------------------------
def smpl_faces() -> np.ndarray:
    from mm4DPS.data import DEFAULT_MODEL
    return np.asarray(np.load(DEFAULT_MODEL)["faces"], dtype=np.int64)


def mesh_vertex_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """(V,) barycentric vertex areas in m^2: a third of every adjacent triangle's area."""
    v = np.asarray(vertices, dtype=np.float64)
    a, b, c = v[faces[:, 0]], v[faces[:, 1]], v[faces[:, 2]]
    face_area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    out = np.zeros(len(v), dtype=np.float64)
    for k in range(3):
        np.add.at(out, faces[:, k], face_area / 3.0)
    return out


def cloud_point_areas_torch(points: "torch.Tensor", neighbours: int = 6) -> "torch.Tensor":
    """(N,) local area per point of a surface cloud, m^2: (sqrt(3)/2) s^2 with s the mean distance to the nearest
    `neighbours` points (a hexagonal packing of spacing s covers that area per point). float64 inside."""
    import torch
    p = points.double()
    areas = torch.empty(len(p), dtype=torch.float64, device=p.device)
    for start in range(0, len(p), 4096):
        d = torch.cdist(p[start:start + 4096], p)                                   # (chunk, N)
        spacing = d.topk(neighbours + 1, dim=1, largest=False).values[:, 1:].mean(dim=1)
        areas[start:start + 4096] = (3.0 ** 0.5 / 2.0) * spacing.square()
    return areas.to(points.dtype)


def physical_solid_angle_torch(points_xyz, normals_xyz, radar_centre_xyz, areas_m2, minimum_cosine: float = 1e-3):
    """dOmega_i = A_i |cos theta_i| / d_i^2 per point, steradians, in the points' dtype (float64 inside). No normalisation."""
    import torch
    points = points_xyz.double(); normals = normals_xyz.double()
    centre = torch.as_tensor(radar_centre_xyz, dtype=torch.float64, device=points.device).reshape(3)
    delta = points - centre[None, :]
    distance = delta.norm(dim=1).clamp_min(1e-6)
    unit = normals / normals.norm(dim=1, keepdim=True).clamp_min(1e-12)
    cosine = (unit * (-delta / distance[:, None])).sum(dim=1).abs().clamp_min(minimum_cosine)
    return (torch.as_tensor(areas_m2, dtype=torch.float64, device=points.device) * cosine / distance.square()).to(points_xyz.dtype)


def scene_solid_angle_weights(positions, normals, radar_centre_xyz) -> tuple:
    """The scene's slot values and the unit: (weights (N,) with median 1, unit in sr)."""
    raw = physical_solid_angle_torch(positions, normals, radar_centre_xyz, cloud_point_areas_torch(positions))
    unit = float(raw.double().median())
    return (raw / unit).to(positions.dtype), unit


def person_solid_angle_weights(full_pcl: np.ndarray, faces: np.ndarray, index, radar_centre_xyz, unit: float, device):
    """The person's slot values for one frame: the posed FULL mesh's vertex areas (m^2), taken at `index` (the
    rendered vertices), times |cos theta| / d^2 from the mesh normals, divided by the scene's unit."""
    import torch
    areas = mesh_vertex_areas(full_pcl[:, :3], faces)[np.asarray(index)]
    pcl = torch.as_tensor(full_pcl[np.asarray(index)], dtype=torch.float32, device=device)
    return physical_solid_angle_torch(pcl[:, :3], pcl[:, 3:6], radar_centre_xyz, areas) / unit


def assert_persistent_person_geometry(sequence_dir, probe_frames: int = 60) -> int:
    """Refuse a person point set whose points do not persist across frames.

    The person's material is per-point and its normals are trained per point,
    so both are meaningless on a cloud that is rebuilt every frame with a
    different length: a per-point parameter would be seen by exactly one frame
    and could not be learned. The monocular depth cloud is exactly that -- 437
    distinct point counts over 600 frames on sequence 102. The GVHMR SMPL mesh
    is 6890 vertices in every frame, with a per-frame visibility mask, so
    point identity survives.

    This is a hard check rather than a flag because the failure is silent:
    training on a non-persistent cloud does not crash, it just quietly
    resolves the material to a single shared vector and fits normals that no
    later frame reuses. Two sequences were trained that way for a whole cycle
    before anyone noticed, because the only symptom was slightly worse
    numbers.

    The depth model is still what reconstructs the *scene*, and the GVHMR
    conversion is built on a depth-derived reference sequence -- it inherits
    the masks, intrinsics, scene points and radar calibration and replaces
    only the person. So this rejects the depth cloud as a *person geometry*,
    which is a narrower thing than rejecting the depth pipeline.

    Returns the point count.
    """
    from pathlib import Path

    import numpy as np

    frames = sorted((Path(sequence_dir) / "person_frames").glob("frame_*.npz"))
    if not frames:
        raise FileNotFoundError(
            f"no person frames under {sequence_dir}/person_frames")
    counts = {
        int(np.load(frame)["pcl"].shape[0])
        for frame in frames[:max(1, probe_frames)]
    }
    if len(counts) != 1:
        raise ValueError(
            f"{sequence_dir} has a person cloud that changes length frame to "
            f"frame ({len(counts)} distinct counts in the first "
            f"{min(len(frames), probe_frames)} frames: "
            f"{sorted(counts)[:4]}...). That is the monocular depth cloud, "
            f"which carries no point identity and cannot hold a per-point "
            f"material or trained per-point normals. Use the GVHMR mesh from "
            f"seeds/gvhmr_person_geometry/."
        )
    return counts.pop()



def vertex_normals(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Area-weighted vertex normals for a batch of meshes, (batch,verts,3)."""
    a, b, c = (vertices[:, faces[:, i]] for i in range(3))
    face_normal = torch.cross(b - a, c - a, dim=-1)
    normals = torch.zeros_like(vertices)
    for corner in range(3):
        normals.index_add_(1, faces[:, corner], face_normal)
    return torch.nn.functional.normalize(normals, dim=-1)

# ----------------------------------------------------------------------
# person_forward_model: The person's `areas` weight -- one definition, used by training and render.
# ----------------------------------------------------------------------


def person_areas(area_scale, point_gains, solid_angle, radar_gain):
    """The ``areas`` argument for a person render. See the module docstring.

    Every argument is a torch tensor or a scalar broadcastable against
    ``point_gains``; this function is deliberately backend-agnostic so the
    trainer (autograd, per-point parameters) and the renderer (frozen
    checkpoint values) run the identical expression.
    """
    return area_scale * point_gains * solid_angle * radar_gain
