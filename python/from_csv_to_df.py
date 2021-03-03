import matplotlib.pyplot as plt
import pandas as pd
from pylab import rcParams
from random import randint


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


def plot_piechart(df: pd, target_cols: str, pie_title: str, save_png: str) -> pd:
    """ Create colorful pie chart showing percent of events by variable label
    :param df: Dataframe created via the get_var_percentages() function
    :param target_cols: Target column in the get_var_percentages() function
    :param pie_title: Target column string for pie chart title
    :param save_png: String based on the target column for storing png
    """
    # Create color list for pie chart visualization
    labels = df[target_cols]
    colors = []
    for i in range(len(labels)):
        colors.append('#%06X' % randint(0, 0xFFFFFF))

    if len(np.unique(labels)) > 10:
        rcParams['figure.figsize'] = 30, 30
    else:
        rcParams['figure.figsize'] = 16, 16

    # Plot
    plt.pie(df['%'], colors=colors, labels=labels, autopct='%1.1f%%')
    plt.title('Percentage of Events by ' + pie_title)
    plt.savefig('../' + save_png + '.png', dpi=300)
    plt.show()
