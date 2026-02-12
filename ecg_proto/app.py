import io

import numpy as np
import pandas as pd
import streamlit as st

from ecg_processing import extract_waveform, load_image_from_bytes
from signal_analysis import detect_heartbeats
from visualization import create_3d_figure, create_time_series_figure


st.set_page_config(
    page_title="ECG Image → Digital Signal Lab",
    layout="wide",
)


def main() -> None:
    st.title("ECG Image → Digital Signal Lab")
    st.caption(
        "Convert ECG images into approximate digital signals with automatic heartbeat "
        "detection and immersive 3D visualization. Prototype – not for clinical use."
    )

    with st.sidebar:
        st.header("1. Input & Assumptions")
        uploaded = st.file_uploader(
            "Upload ECG image",
            type=["png", "jpg", "jpeg"],
            help="Use a single-lead ECG with a light background for best results.",
        )

        duration_seconds = st.slider(
            "Duration represented by image (seconds)",
            min_value=5.0,
            max_value=20.0,
            value=10.0,
            step=0.5,
            help="Maps image width to this total time. Typical full-width ECG strips are ~10 seconds.",
        )

        st.markdown("---")
        st.header("2. Beat Detection")
        min_peak_height = st.slider(
            "Minimum relative peak height",
            min_value=0.05,
            max_value=0.8,
            value=0.25,
            step=0.05,
            help="Higher values reduce false positives; lower values capture smaller beats.",
        )
        min_bpm = st.slider(
            "Min plausible heart rate (BPM)",
            min_value=20,
            max_value=80,
            value=30,
            step=5,
        )
        max_bpm = st.slider(
            "Max plausible heart rate (BPM)",
            min_value=100,
            max_value=220,
            value=180,
            step=10,
        )

        st.markdown("---")
        st.info(
            "This prototype estimates signal and heart rate from image geometry only. "
            "Results are approximate and for exploration/demo purposes."
        )

    if not uploaded:
        st.subheader("Awaiting ECG image...")
        st.write(
            "Upload a clear ECG strip image in the sidebar to see the reconstructed waveform, "
            "automatic heartbeat detection, and 3D visualization."
        )
        return

    # Layout: top row (images), middle (2D/3D), bottom (data & exports)
    image_bytes = uploaded.read()

    try:
        image_bgr = load_image_from_bytes(image_bytes)
    except Exception as exc:  # pragma: no cover - defensive
        st.error(f"Could not read image: {exc}")
        return

    # Run extraction
    try:
        time, amplitude, debug = extract_waveform(
            image_bgr=image_bgr,
            duration_seconds=duration_seconds,
        )
    except Exception as exc:  # pragma: no cover - defensive
        st.error(str(exc))
        return

    # Heartbeat detection
    peaks, hr_bpm, hb_debug = detect_heartbeats(
        time=time,
        amplitude=amplitude,
        min_bpm=float(min_bpm),
        max_bpm=float(max_bpm),
        min_peak_height=float(min_peak_height),
    )

    # === TOP: Image views ===
    st.subheader("Image Views")
    col_orig, col_binary, col_cleaned = st.columns(3)

    with col_orig:
        st.markdown("**Original ECG Image**")
        st.image(
            io.BytesIO(image_bytes),
            use_column_width=True,
        )

    import cv2  # local import to avoid unused warning when app is imported

    gray = debug["gray"]
    binary = debug["binary"]
    cleaned = debug["cleaned"]

    with col_binary:
        st.markdown("**Thresholded / Binary View**")
        st.image(binary, clamp=True, use_column_width=True)

    with col_cleaned:
        st.markdown("**Cleaned Trace Mask**")
        st.image(cleaned, clamp=True, use_column_width=True)

    # === MIDDLE: Plots & metrics ===
    st.subheader("Reconstructed Signal & Insights")
    col_plot, col_metrics = st.columns([2.3, 1.2])

    with col_plot:
        tabs = st.tabs(["2D Signal", "3D ECG"])

        with tabs[0]:
            fig_2d = create_time_series_figure(time, amplitude, peaks)
            st.plotly_chart(fig_2d, use_container_width=True)

        with tabs[1]:
            fig_3d = create_3d_figure(time, amplitude)
            st.plotly_chart(fig_3d, use_container_width=True)

    with col_metrics:
        st.markdown("**Automatic Heartbeat Detection**")
        if hr_bpm is not None:
            st.metric(
                label="Estimated Heart Rate",
                value=f"{hr_bpm:0.1f} BPM",
            )
        else:
            st.warning(
                "Not enough reliable peaks detected to estimate heart rate. "
                "Try adjusting the peak height or uploading a clearer strip."
            )

        st.markdown("---")
        st.markdown("**Signal Snapshot**")
        st.write(f"Detected beats: **{len(peaks)}**")
        st.write(f"Signal duration: **{duration_seconds:0.2f} s**")
        st.write(f"Samples: **{len(time)}**")

        if hb_debug.get("rr_valid") is not None and hb_debug["rr_valid"].size > 0:
            rr_ms = hb_debug["rr_valid"] * 1000.0
            st.write(f"Median RR: **{np.median(rr_ms):0.0f} ms**")
            st.write(f"RR range: **{np.min(rr_ms):0.0f} – {np.max(rr_ms):0.0f} ms**")

    # === BOTTOM: Data & export ===
    st.subheader("Data & Export")
    df = pd.DataFrame(
        {
            "time_s": time,
            "amplitude_norm": amplitude,
        }
    )

    st.dataframe(df.head(500), use_container_width=True, height=220)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Reconstructed Signal as CSV",
        data=csv_bytes,
        file_name="ecg_reconstructed_signal.csv",
        mime="text/csv",
    )

    st.caption(
        "Prototype dashboard: visualizations and metrics are approximate and should "
        "not be used for diagnosis."
    )


if __name__ == "__main__":
    main()

