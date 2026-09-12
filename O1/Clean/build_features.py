"""Model-agnostic feature construction for the merged, cleaned frame.

Adds the six cyclical time encodings (hour, day of week, month as sin/cos pairs)
by delegating to add_temporal_features. Wind u/v are already built upstream by
transform_openmeteo, so this stage exists mainly to make feature construction an
explicit, extensible step in the pipeline.
"""

from common_preprocess import add_temporal_features


def build_features(df):
    """Add cyclical time features to a frame, returning it unchanged if empty."""
    return df if df.empty else add_temporal_features(df)