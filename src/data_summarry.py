from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Markdown, display
from src.formatting import display_dictionary, series_to_frame, hide_index
from src.data_types import infer_data_type

def data_summary(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    additional_rows: Optional[Dict[str, Any]] = None,
) -> None:
    """Renders a quick info table about df in the calling notebook.

    Parameters
    ----------
    df : pd.DataFrame
        Data which to analyze.
    columns : List[str], optional
        List of columns of df to analyze. All columns of df are used by default.
    additional_rows : Dict[str, Any], optional
        Additional custom rows to add to the table.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a Pandas DataFrame")

    if columns is not None:
        df = df[columns]

    missing_cells = df.isna().sum().sum()
    missing_cells_percent = 100 * missing_cells / (df.shape[0] * df.shape[1])

    zeros = (df == 0).sum().sum()
    zeros_percent = 100 * zeros / (df.shape[0] * df.shape[1])

    duplicate_rows = df.duplicated().sum()
    duplicate_rows_percent = 100 * duplicate_rows / len(df)

    df_info_rows = {
        "Rows": df.shape[0],
        "Columns": df.shape[1],
        "Missing cells": f"{missing_cells} ({missing_cells_percent:,.02f} %)",
        "Zeros": f"{zeros} ({zeros_percent:.02f} %)",
        "Duplicate rows": f"{duplicate_rows} ({duplicate_rows_percent:,.02f} %)",
    }

    if additional_rows is not None:
        df_info_rows = {**df_info_rows, **additional_rows}
    
    print("Snapshot:")
    display_dictionary(df_info_rows)

def data_types(df: pd.DataFrame, columns: Optional[List[str]] = None) -> None:
    """Renders a table with inferred data types in the calling notebook.

    Parameters
    ----------
    df : pd.DataFrame:
        Data for which to infer data types
    columns : List[str], optional
        List of columns for which to infer data type. All columns of df are used by default.
    """
    if columns is not None:
        df = df[columns]

    # Type ignored because the apply is not properly typed: the type hints for
    # the parameter `func` do not cover the complete set of possible inputs.
    dtypes: pd.Series[str] = df.apply(
        func=lambda x_: str(infer_data_type(x_)),  # type: ignore
        axis=0,
        result_type="expand",
    )

    # Convert result to frame for viewing
    dtypes_frame = series_to_frame(
        series=dtypes, index_name="Column Name", column_name="Inferred Data Type"
    )

    display(hide_index(dtypes_frame))


def missing_values(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    bar_plot: bool = True,
    bar_plot_figsize: Tuple[int, int] = (15, 6),
    bar_plot_title: str = "Missing Values Percentage of Each Column",
    bar_plot_ylim: float = 0,
    bar_plot_color: str = "#FFA07A",
    **bar_plot_args: Any,
) -> None:
    """Displays a table of missing values percentages for each column of df and a bar plot
    of the percentages.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe for which to calculate missing values.
    columns : Optional[List[str]], optional
        Subset of columns for which to calculate missing values percentage.
        If None, all columns of df are used.
    bar_plot : bool (default = False)
        Whether to also display a bar plot visualizing missing values percentages for each
        column.
    bar_plot_figsize : Tuple[int, int]
        Width and height of the bar plot.
    bar_plot_title : str
        Title of the bar plot.
    bar_plot_ylim : float
        Bar plot y axis bottom limit.
    bar_plot_color : str
        Color of bars in the bar plot in hex format.
    bar_plot_args : Any
        Additional kwargs passed to pandas.Series.bar.
    """
    if columns is not None:
        df = df[columns]

    # Count null values
    null_count = df.isna().sum()
    null_percentage = 100 * null_count / len(df)

    if null_count.sum() == 0:
        print("There are no missing values")
        return

    # Convert series to frames
    null_count_frame = series_to_frame(
        series=null_count, index_name="Column Name", column_name="Null Count"
    )
    null_percentage_frame = series_to_frame(
        series=null_percentage, index_name="Column Name", column_name="Null %"
    )
    # Merge null count and percentage into one frame
    null_stats_frame = null_count_frame.merge(null_percentage_frame, on="Column Name").sort_values(
        "Null Count", ascending=False
    )

    display(
        hide_index(null_stats_frame)
        .bar(color="#FFA07A", subset=["Null %"], vmax=100)
        .format({"Null %": "{:.03f}"})
    )

    # Display bar plot of missing value percentages
    if bar_plot:
        (
            null_percentage_frame.sort_values("Null %", ascending=False)
            .plot.bar(
                x="Column Name",
                figsize=bar_plot_figsize,
                title=bar_plot_title,
                ylim=bar_plot_ylim,
                color=bar_plot_color,
                **bar_plot_args,
            )
            .set_ylabel("Missing Values [%]")
        )
        plt.show()


def constant_occurrence(
    df: pd.DataFrame, columns: Optional[List[str]] = None, constant: Any = 0
) -> None:
    """Displays a table with occurrence of a constant in each column.

    By default, check for 0 occurrence.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe for which to calculate constant values occurrence.
    columns : Optional[List[str]], optional
        Subset of columns for which to calculate constant values occurrence.
        If None, all columns of df are used.
    constant : Any
        Constant for which to check occurrence in df, by default 0.
    """
    if columns is not None:
        df = df[columns]

    # Count constant counts
    constant_count = (df == constant).sum()
    constant_percentage = 100 * constant_count / len(df)

    constant_formatted = f"<i>{constant!r}</i>"
    constant_count_name = f"{constant_formatted} Count"
    constant_perc_name = f"{constant_formatted} %"

    # Convert series to frames
    constant_count_frame = series_to_frame(
        series=constant_count,
        index_name="Column Name",
        column_name=constant_count_name,
    )

    constant_percentage_frame = series_to_frame(
        series=constant_percentage,
        index_name="Column Name",
        column_name=constant_perc_name,
    )

    # Merge absolute and relative counts
    constant_stats_frame = constant_count_frame.merge(
        constant_percentage_frame, on="Column Name"
    ).sort_values(constant_perc_name, ascending=False)

    # Display table
    display(
        hide_index(constant_stats_frame)
        .bar(color="#FFA07A", subset=[constant_perc_name], vmax=100)
        .format({constant_perc_name: "{:.03f}"})
    )



def duplicate_row_count(df: pd.DataFrame, columns: Optional[List[str]] = None) -> None:
    """Displays a table with duplicated row count and percentage.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe for which to count missing value rows.
    columns : Optional[List[str]], optional
        List of columns to consider when counting. If None, all columns are used.
    """
    if columns is not None:
        df = df[columns]

    # Count duplicated rows
    num_duplicated_rows = df.duplicated().sum()

    if num_duplicated_rows == 0:
        print("There are no duplicated rows")
        return

    # Relative count
    percentage_duplicated_rows = 100 * num_duplicated_rows / len(df)

    duplicate_rows_info = {
        "Duplicate rows column subset": "all columns" if columns is None else columns,
        "Duplicate row count": num_duplicated_rows,
        "Duplicate row percentage": f"{percentage_duplicated_rows:.02f} %",
    }
    
    print("Duplicate Row")
    # Display table
    display_dictionary(duplicate_rows_info)