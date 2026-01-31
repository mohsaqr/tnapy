"""Data preparation functions for TNA package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class TNAData:
    """Container for prepared sequence data.

    Attributes
    ----------
    long_data : pd.DataFrame
        Processed data in long format with session information
    sequence_data : pd.DataFrame
        Actions in wide format (one row per sequence/session)
    meta_data : pd.DataFrame
        Other variables in wide format
    time_data : pd.DataFrame or None
        Time values in wide format if time column was provided
    statistics : dict
        Summary metrics about the data
    """

    long_data: pd.DataFrame
    sequence_data: pd.DataFrame
    meta_data: pd.DataFrame
    time_data: pd.DataFrame | None = None
    statistics: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        n_sessions = len(self.sequence_data)
        n_actions = self.statistics.get('n_unique_actions', '?')
        n_actors = self.statistics.get('n_actors', '?')
        return f"TNAData(sessions={n_sessions}, actions={n_actions}, actors={n_actors})"


def _parse_timestamp(
    series: pd.Series,
    custom_format: str | None = None,
    is_unix_time: bool = False,
    unix_time_unit: str = "seconds"
) -> pd.Series:
    """Parse timestamps with multiple format support.

    Parameters
    ----------
    series : pd.Series
        Column containing timestamp values
    custom_format : str, optional
        strptime format string for custom parsing
    is_unix_time : bool
        If True, treat values as Unix timestamps
    unix_time_unit : str
        Unit for Unix timestamps: 'seconds', 'milliseconds', 'microseconds'

    Returns
    -------
    pd.Series
        Datetime series
    """
    if is_unix_time:
        unit_map = {
            'seconds': 's',
            'milliseconds': 'ms',
            'microseconds': 'us'
        }
        unit = unit_map.get(unix_time_unit, 's')
        return pd.to_datetime(series, unit=unit)

    if custom_format is not None:
        return pd.to_datetime(series, format=custom_format)

    # Try automatic parsing (handles ISO8601 and many other formats)
    return pd.to_datetime(series)


def _create_sessions(
    df: pd.DataFrame,
    actor_col: str,
    time_col: str | None,
    time_threshold: int
) -> pd.Series:
    """Create session IDs based on time gaps.

    Parameters
    ----------
    df : pd.DataFrame
        Input data (must be sorted by actor and time)
    actor_col : str
        Name of the actor column
    time_col : str or None
        Name of the time column
    time_threshold : int
        Gap in seconds that starts a new session

    Returns
    -------
    pd.Series
        Session IDs
    """
    if time_col is None:
        # Without time, each actor gets one session
        return df.groupby(actor_col).ngroup()

    # Calculate time differences within each actor
    time_diff = df.groupby(actor_col)[time_col].diff()

    # Convert to seconds if timedelta
    if pd.api.types.is_timedelta64_dtype(time_diff):
        time_diff = time_diff.dt.total_seconds()

    # Mark session boundaries where gap exceeds threshold or first row
    new_session = (time_diff > time_threshold) | time_diff.isna()

    # Cumulative sum gives session number within actor
    session_within_actor = new_session.groupby(df[actor_col]).cumsum()

    # Create unique session ID by combining actor and session number
    session_id = df[actor_col].astype(str) + '_' + session_within_actor.astype(int).astype(str)

    return session_id


def prepare_data(
    data: pd.DataFrame,
    actor: str | list[str] | None = None,
    time: str | None = None,
    action: str = "action",
    order: str | None = None,
    time_threshold: int = 900,
    custom_format: str | None = None,
    is_unix_time: bool = False,
    unix_time_unit: str = "seconds"
) -> TNAData:
    """Prepare raw event/log data into sequence format for TNA analysis.

    Parameters
    ----------
    data : pd.DataFrame
        Input data in long format (one row per event)
    actor : str or list of str, optional
        Column name(s) identifying actors/users. If multiple columns,
        they are concatenated to form a unique actor ID.
    time : str, optional
        Column name containing timestamps
    action : str
        Column name containing the action/event type
    order : str, optional
        Column name for explicit ordering (used instead of time for sorting)
    time_threshold : int
        Gap in seconds that starts a new session (default: 900 = 15 minutes)
    custom_format : str, optional
        strptime format string for timestamp parsing
    is_unix_time : bool
        If True, treat time column as Unix timestamps
    unix_time_unit : str
        Unit for Unix timestamps: 'seconds', 'milliseconds', 'microseconds'

    Returns
    -------
    TNAData
        Container with processed data in various formats
    """
    df = data.copy()

    # Handle actor column(s)
    if actor is None:
        # Create a single actor for all data
        df['.actor'] = 'all'
        actor_col = '.actor'
    elif isinstance(actor, list):
        # Concatenate multiple actor columns
        df['.actor'] = df[actor].astype(str).agg('_'.join, axis=1)
        actor_col = '.actor'
    else:
        actor_col = actor

    # Parse timestamps if provided
    if time is not None:
        df['.standardized_time'] = _parse_timestamp(
            df[time],
            custom_format=custom_format,
            is_unix_time=is_unix_time,
            unix_time_unit=unix_time_unit
        )
        time_col = '.standardized_time'
    else:
        time_col = None

    # Sort data
    if order is not None:
        sort_cols = [actor_col, order]
    elif time_col is not None:
        sort_cols = [actor_col, time_col]
    else:
        sort_cols = [actor_col]

    df = df.sort_values(sort_cols).reset_index(drop=True)

    # Create session IDs
    df['.session_id'] = _create_sessions(df, actor_col, time_col, time_threshold)

    # Create sequence number within session
    df['.sequence'] = df.groupby('.session_id').cumcount() + 1

    # Get unique sessions and their max sequence length
    sessions = df.groupby('.session_id').agg({
        action: list,
        '.sequence': 'max'
    }).reset_index()

    max_len = sessions['.sequence'].max()

    # Create wide format for sequences
    sequence_rows = []
    for _, row in sessions.iterrows():
        actions = row[action]
        # Pad with None to max length
        padded = actions + [None] * (max_len - len(actions))
        sequence_rows.append(padded)

    sequence_data = pd.DataFrame(
        sequence_rows,
        columns=[f"action_{i+1}" for i in range(max_len)],
        index=sessions['.session_id']
    )

    # Create time data in wide format if available
    time_data = None
    if time_col is not None:
        time_sessions = df.groupby('.session_id')[time_col].apply(list).reset_index()
        time_rows = []
        for _, row in time_sessions.iterrows():
            times = row[time_col]
            padded = times + [None] * (max_len - len(times))
            time_rows.append(padded)
        time_data = pd.DataFrame(
            time_rows,
            columns=[f"time_{i+1}" for i in range(max_len)],
            index=sessions['.session_id']
        )

    # Create metadata (actor info per session)
    # Get first row of each session for metadata
    meta_cols = [col for col in df.columns
                 if col not in [action, time_col, '.sequence', '.session_id', '.standardized_time']
                 and not col.startswith('.')]
    if actor_col not in meta_cols and actor_col in df.columns:
        meta_cols = [actor_col] + meta_cols

    meta_data = df.groupby('.session_id').first()[meta_cols].reset_index()
    meta_data = meta_data.set_index('.session_id')

    # Compute statistics
    unique_actions = df[action].dropna().unique()
    statistics = {
        'n_sessions': len(sessions),
        'n_actors': df[actor_col].nunique(),
        'n_events': len(df),
        'n_unique_actions': len(unique_actions),
        'unique_actions': list(unique_actions),
        'max_sequence_length': max_len,
        'mean_sequence_length': sessions['.sequence'].mean(),
        'action_frequencies': df[action].value_counts().to_dict()
    }

    return TNAData(
        long_data=df,
        sequence_data=sequence_data,
        meta_data=meta_data,
        time_data=time_data,
        statistics=statistics
    )


def create_seqdata(
    x: pd.DataFrame | np.ndarray,
    cols: list[str] | None = None,
    concat: int = 1,
    begin_state: str | None = None,
    end_state: str | None = None
) -> tuple[np.ndarray, list[str], list[str]]:
    """Create sequence data from wide-format DataFrame.

    Parameters
    ----------
    x : pd.DataFrame or np.ndarray
        Wide-format data where rows are sequences and columns are time steps
    cols : list of str, optional
        Column names to use (if DataFrame)
    concat : int
        Number of consecutive sequences to concatenate (for n-gram)
    begin_state : str, optional
        Add this state at the beginning of each sequence
    end_state : str, optional
        Add this state at the end of each sequence

    Returns
    -------
    tuple
        (sequence_array, state_labels, column_names)
    """
    if isinstance(x, pd.DataFrame):
        if cols is not None:
            data = x[cols].values
            col_names = cols
        else:
            data = x.values
            col_names = list(x.columns)
    else:
        data = x
        col_names = [f"V{i+1}" for i in range(x.shape[1])]

    # Get unique states
    flat = data.flatten()
    unique_states = pd.unique(flat[~pd.isna(flat)])
    state_labels = sorted([str(s) for s in unique_states])

    # Add begin/end states if specified
    if begin_state is not None and begin_state not in state_labels:
        state_labels = [begin_state] + state_labels
    if end_state is not None and end_state not in state_labels:
        state_labels = state_labels + [end_state]

    # Concatenate sequences if requested
    if concat > 1:
        new_rows = []
        for i in range(0, len(data), concat):
            rows_to_concat = data[i:min(i + concat, len(data))]
            concatenated = np.concatenate([r for r in rows_to_concat])
            new_rows.append(concatenated)
        # Pad to same length
        max_len = max(len(r) for r in new_rows)
        data = np.array([
            np.concatenate([r, [None] * (max_len - len(r))])
            for r in new_rows
        ])
        col_names = [f"V{i+1}" for i in range(data.shape[1])]

    # Add begin/end states
    if begin_state is not None:
        begin_col = np.full((data.shape[0], 1), begin_state)
        data = np.hstack([begin_col, data])
        col_names = ['begin'] + col_names

    if end_state is not None:
        end_col = np.full((data.shape[0], 1), end_state)
        data = np.hstack([data, end_col])
        col_names = col_names + ['end']

    return data, state_labels, col_names
