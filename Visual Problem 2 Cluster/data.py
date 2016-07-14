import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import datetime
import PIL.Image


dirname = os.path.dirname(__file__)
data_in_path = os.path.join(dirname, 'datain')
data_out_path = os.path.join(dirname, 'dataout')
days = ['Fri', 'Sat', 'Sun']

tableau10 = ['#1F77B4', '#FF7F0E',
             '#2CA02C', '#D62728',
             '#9467BD', '#8C564B',
             '#E377C2', '#7F7F7F',
             '#BCBD22', '#17BECF']
tableau20 = ['#1F77B4', '#AEC7E8', '#FF7F0E', '#FFBB78',
             '#2CA02C', '#98DF8A', '#D62728', '#FF9896',
             '#9467BD', '#C5B0D5', '#8C564B', '#C49C94',
             '#E377C2', '#F7B6D2', '#7F7F7F', '#C7C7C7',
             '#BCBD22', '#DBDB8D', '#17BECF', '#9EDAE5']

palette10 = tableau10
palette20 = tableau20


def read_data(day='Fri', type='2', grouped=False) -> pd.DataFrame:
    """
    Reads main trajectory data

    Essentially this is just a wrapper for pandas' read_csv method,
    this handles the file paths nicely.

    :param day: The day to read from, either 'Fri', 'Sat' or 'Sun'
    :param type: What type of data to read. This is the part at the end of the file name, e.g. 'subset-300'
    :param grouped: Whether to read group data or individual data
    :return: DataFrame with the read data.
    """
    if type is None:
        type = 2

    if str(type) == '2' or 'subset' in type:
        dtypes = {'id': 'int64',
                  'X': 'int64',
                  'Y': 'int64'}
    elif type == 'key-points':
        dtypes = {'id': 'int64',
                  'place_id': 'int64',
                  'timespan': 'int64'}
    else:
        dtypes = None
    if grouped:
        group_str = 'groups-'
    else:
        group_str = ''
    data_path = 'park-movement-{}{}-{}.csv'.format(group_str, day, type)
    full_path = os.path.join(data_in_path, data_path)
    return pd.read_csv(full_path, parse_dates=['Timestamp'], dtype=dtypes)


def read_data_with_timespans(day, type='2') -> pd.DataFrame:
    """
    Reads data and calculates the spent in each place

    :param day: The day to read from, either 'Fri', 'Sat' or 'Sun'
    :param type: What type of data to read. This is the part at the end of the file name, e.g. 'subset-300'
    :return: DataFrame with the read data and timespans
    """
    if type is None:
        type = 2
    df = read_data(day, type).sort_values(by=['id', 'Timestamp'])
    df = df[(df.X.diff() != 0) | (df.Y.diff() != 0)]  # drop consecutive duplicates
    tdiff = df.Timestamp.diff().shift(-1)
    tdiff[df.id != df.id.shift(-1)] = pd.NaT
    tdiff /= np.timedelta64(1, 's')  # convert to seconds so it works nicer
    df['timespan'] = tdiff
    return df


def read_key_points() -> pd.DataFrame:
    """
    Reads in information of the key points

    :return: DataFrame with place_ids, categories and x & y coordinates
    """
    path = os.path.join(data_in_path, 'key points.csv')
    return pd.read_csv(path)


def read_visited_key_points(day, extra=None, grouped=False) -> pd.DataFrame:
    """
    Reads what key points a person visited

    :param day: The day to read from, either 'Fri', 'Sat' or 'Sun'
    :param extra: Extra data to about the key_points to include, e.g. names, categories, etc
    :param grouped: Whether to read group data or individual data
    """
    df = read_data(day, 'key-points', grouped)
    # no extra points -> no need to read key points or do merge
    if extra is None or len(extra) == 0:
        return df
    kp = read_key_points()
    columns = ['place_id']
    columns.extend(extra)
    # this changes the sort order in a weird way. Up the to caller to resort if necessary
    return pd.merge(df, kp.loc[:, columns], on='place_id', sort=False)


def read_groups(day) -> pd.DataFrame:
    """
    Gets the list of each id and the group they are in.

    :param day: The day to read from, either 'Fri', 'Sat' or 'Sun'
    :return: Pandas Dataframe with the columns 'id' and 'group_id' for each person.
    """
    path = os.path.join(data_out_path, 'groups/groups-{}.csv'.format(day))
    return pd.read_csv(path)


def read_group_typical_ids(day) -> pd.DataFrame:
    """
    Gets the each group, and the id of the most typical, or median person.

    :param day: The day to read from, either 'Fri', 'Sat' or 'Sun'
    :return: Pandas Dataframe with the columns 'group_id' and 'id' for each group.
    """
    path = os.path.join(data_out_path, 'groups/group-typical-id-{}.csv'.format(day))
    return pd.read_csv(path)


def read_group_info(day) -> pd.DataFrame:
    """
    Gets info for each groups, including the typical id.

    :param day: The day to read from, either 'Fri', 'Sat' or 'Sun'
    :return: Pandas Dataframe with the columns 'group_id' and 'id' for each group.
    """
    path = os.path.join(data_out_path, 'groups/group-info-{}.csv'.format(day))
    return pd.read_csv(path)


def read_group_distances(day) -> np.array:
    """
    Gets the distance matrix for the distance between each group

    :param day: The day to read from, either 'Fri', 'Sat' or 'Sun'
    :return: Distance matrix as a numpy array
    """
    path = os.path.join(data_out_path, 'group_dist/{}.npy'.format(day))
    matrix = np.load(path)
    return matrix


def get_image_path(type='Light') -> str:
    """
    Gets the file path of the map image

    :param type: The type of image. Have a look at the files in datain/image for the options
    :return: The file path of the map image
    """
    return os.path.join(data_in_path, 'image/Park Map - {}.jpg'.format(type))


def read_image(type='Light', size=None) -> np.ndarray:
    """
    Reads in a map image

    :param type: The type of image. Have a look at the files in datain/image for the options
    :param size: The size in pixels to rescale the image to. If None, no rescaling occurs.
    :return: An image that can be displayed with plt.imshow()
    """
    path = get_image_path(type)
    if size is None:
        return plt.imread(path)
    else:
        im = PIL.Image.open(path)
        im = im.resize((size, size))
        return np.array(im)


def read_dtw(type):
    """
    Reads in dtw data for two ids

    :param type: Either 'distance' or 'speed'
    :return: Pandas DataFrame
    """
    path = os.path.join(data_in_path, 'dtw/output{} 2672 4343.txt'.format(type))
    return pd.read_csv(path)


def read_dist_speed(id):
    """
    Reads the distance and speed of a person

    :param id: The id of the person to read
    :return: Pandas DataFrame
    """
    path = os.path.join(data_out_path, 'distspeed/dist_speed_{}.csv'.format(id))
    return pd.read_csv(path, parse_dates=[0])


def read_matrix(day, type='freqs', grouped=False):
    """
    Reads in a file from dataout/matrices
    """
    group_str = ''
    if grouped:
        group_str = '_grouped'
    path = os.path.join(data_out_path, 'matrices/{}_{}{}.csv'.format(type, day, group_str))
    return pd.read_csv(path, index_col=0)


# def read_manifold(type='freqs'):
#     path = os.path.join(data_out_path, 'manifold_{}.npz'.format(type))
#     return np.load(path)


def read_manifold(day, type='freqs'):
    """
    Reads in a manifold from dataout/manifold
    """
    path = os.path.join(data_out_path, 'manifold/{}_{}.npz'.format(type, day))
    return np.load(path)


def read_position_totals():
    """
    Reads in the position totals data.

    The data has how many people are at each place each minute. If a place is empty, it isn't stored
    """
    path = os.path.join(data_out_path, 'position_totals.csv')
    return pd.read_csv(path, parse_dates=[0])


def get_attraction_totals():
    """
    Gets a dataframe of how many people are at each attraction each minute.

    If a place is empty, there will be no entry for that minute
    """
    kp = read_key_points()
    pos = read_position_totals()
    totals = (
        pd.merge(pos, kp.loc[:, ['X', 'Y', 'place_id']], on=['X', 'Y'], sort=False)
        .drop(['X', 'Y'], 1))
    return totals


def _time_ticks(x, pos):
    """
    Turns a time in seconds into a human-readable form, for Matplotlib plots.
    """
    d = datetime.timedelta(seconds=x)
    return str(d)

timedelta_formatter = matplotlib.ticker.FuncFormatter(_time_ticks)

