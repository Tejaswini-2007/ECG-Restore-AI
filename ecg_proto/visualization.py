from typing import Optional

import numpy as np
import plotly.graph_objects as go


def create_time_series_figure(
    time: np.ndarray,
    amplitude: np.ndarray,
    peaks: Optional[np.ndarray] = None,
) -> go.Figure:
    """
    Create a 2D time-series ECG plot with optional R-peak markers.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=time,
            y=amplitude,
            mode="lines",
            name="ECG Signal",
            line=dict(color="#1f77b4", width=2),
        )
    )

    if peaks is not None and len(peaks) > 0:
        fig.add_trace(
            go.Scatter(
                x=time[peaks],
                y=amplitude[peaks],
                mode="markers",
                name="Detected Beats",
                marker=dict(color="#d62728", size=8, symbol="circle-open"),
            )
        )

    fig.update_layout(
        title="Reconstructed ECG Signal",
        xaxis_title="Time (s)",
        yaxis_title="Normalized Amplitude",
        template="plotly_white",
        margin=dict(l=60, r=20, t=60, b=40),
    )

    return fig


def create_3d_figure(
    time: np.ndarray,
    amplitude: np.ndarray,
) -> go.Figure:
    """
    Create a 3D ECG visualization.

    x = time, y = amplitude, z = intensity/depth effect.
    The z dimension is derived from a smoothed version of the amplitude
    to create a visually pleasing depth effect.
    """
    # Simple depth effect: smoothed amplitude + gradual drift
    if len(amplitude) > 5:
        window = max(len(amplitude) // 200, 3)
        if window % 2 == 0:
            window += 1
        kernel = np.ones(window) / window
        z_base = np.convolve(amplitude, kernel, mode="same")
    else:
        z_base = amplitude.copy()

    depth_ramp = np.linspace(-0.3, 0.3, len(time))
    z = z_base + depth_ramp

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=time,
                y=amplitude,
                z=z,
                mode="lines",
                line=dict(
                    color=amplitude,
                    colorscale="Viridis",
                    width=4,
                ),
            )
        ]
    )

    fig.update_layout(
        title="3D ECG Visualization",
        scene=dict(
            xaxis_title="Time (s)",
            yaxis_title="Normalized Amplitude",
            zaxis_title="Depth / Intensity",
        ),
        template="plotly_white",
        margin=dict(l=0, r=0, t=50, b=0),
    )

    return fig

