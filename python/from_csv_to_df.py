import pandas as pd


def get_select_data(filepath: str, all_cols: list, select_cols: list) -> object:
    """Get Select Columns of Data from GDELT Latest Update CSV
    :param filepath: full filename and path to CSV file to be imported
    :param all_cols: column names of the CSV imported
    :param select_cols: derivative list of columns from all_cols
    :rtype: dataframe
    :return: dataframe
    """

    # Import entire CSV
    latest_update_df = pd.DataFrame(pd.read_csv(filepath,
                                                names=all_cols, delimiter="\t"))

    # Select specific columns
    return latest_update_df[select_cols]
