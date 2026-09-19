import numpy as np
import pandas as pd


def extract_features_from_file(filepath):

    # Read CSV
    # header=None is kept so this also works if the CSV structure changes.
    # Any text/header row will be converted to NaN and removed below.
    df = pd.read_csv(
        filepath,
        header=None,
        low_memory=False
    )

    # Convert everything to numeric
    # Text such as column names becomes NaN
    df = df.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Remove rows that contain no numeric data
    # This removes the CSV header row if it was read as data
    df = df.dropna(how="all")

    # Check that there are no unexpected non-numeric/missing values left
    if df.isna().any().any():
        raise ValueError(
            "CSV contains missing or non-numeric sensor values "
            "after removing the header."
        )

    data = df.to_numpy(dtype=float)

    # First column = speed sensor
    speed_col = data[:, 0]

    # Remaining 128 columns = vibration + shock sensor data
    sensor_data = data[:, 1:]

    # Basic structure check
    if sensor_data.shape[1] != 128:
        raise ValueError(
            f"Expected 128 sensor columns, "
            f"but found {sensor_data.shape[1]}."
        )

    # Estimate speed activity from pulse transitions
    speed_transitions = np.sum(
        np.diff(speed_col) != 0
    )

    # Split alternating vibration / shock channels
    vib_data = sensor_data[:, 0::2]
    shock_data = sensor_data[:, 1::2]

    # Side I and Side II axle indices
    side1_indices = []
    side2_indices = []

    for car in range(8):

        for pos in range(1, 9):

            idx = car * 8 + (pos - 1)

            if pos in [1, 3, 5, 7]:
                side1_indices.append(idx)

            else:
                side2_indices.append(idx)

    # Separate signals by side
    s1_vib = vib_data[:, side1_indices]
    s2_vib = vib_data[:, side2_indices]

    s1_shk = shock_data[:, side1_indices]
    s2_shk = shock_data[:, side2_indices]

    # Calculate summary statistics
    def calc_stats(arr):

        rms = np.sqrt(
            np.mean(arr ** 2, axis=0)
        )

        peak = np.max(
            np.abs(arr),
            axis=0
        )

        std = np.std(
            arr,
            axis=0
        )

        return [
            np.mean(rms),
            np.max(rms),
            np.mean(peak),
            np.max(peak),
            np.mean(std),
            np.max(std)
        ]

    # Statistics for each sensor group
    s1_v_stats = calc_stats(s1_vib)
    s2_v_stats = calc_stats(s2_vib)

    s1_s_stats = calc_stats(s1_shk)
    s2_s_stats = calc_stats(s2_shk)

    # Side I vs Side II asymmetry
    vib_asymmetry = (
        s1_v_stats[0]
        - s2_v_stats[0]
    )

    shk_asymmetry = (
        s1_s_stats[0]
        - s2_s_stats[0]
    )

    # Final 27-feature vector
    features = (
        [
            speed_transitions,
            vib_asymmetry,
            shk_asymmetry
        ]
        + s1_v_stats
        + s2_v_stats
        + s1_s_stats
        + s2_s_stats
    )

    features = np.array(
        features,
        dtype=float
    )

    # Final safety check
    if len(features) != 27:
        raise ValueError(
            f"Expected 27 features, "
            f"but extracted {len(features)}."
        )

    return features