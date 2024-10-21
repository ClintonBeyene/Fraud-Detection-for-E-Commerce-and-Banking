from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Markdown, display
from typing import Any, Dict, List, Union
import pandas as pd
from IPython.display import display
from pandas.io.formats.style import Styler
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple



def display_dictionary(dictionary: Dict[str, Any]) -> None:
    """
    Converts a dictionary to a dataframe and renders it in the report notebook.

    Parameters
    ----------
    dictionary : Dict[str, Any]
        Dictionary to be rendered
    """
    dictionary = {key: str(value) for key, value in dictionary.items()}

    display(pd.DataFrame.from_dict(dictionary, orient="index", columns=[""]))


def series_to_frame(series: pd.Series, index_name: str, column_name: str) -> pd.DataFrame:
    """Converts a pandas.Series to a pandas.DataFrame by putting the series index into a separate
    column.

    Parameters
    ---
    series : pd.Series
        Input series
    index_name : str
        Name of the new column into which the series index will be put
    column_name : str
        Name of the series values column

    Returns
    ---
    pd.DataFrame
        Dataframe with two columns index_name and column_name with values of series.index and
        series.values respectively
    """
    return series.rename_axis(index=index_name).to_frame(name=column_name).reset_index()


def hide_index(df: pd.DataFrame) -> Styler:
    """
    Hides the index of a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame where the index should be hidden.

    Returns
    -------
    Styler
        Styler object with the index hidden.
    """
    return df.style.hide(axis="index")