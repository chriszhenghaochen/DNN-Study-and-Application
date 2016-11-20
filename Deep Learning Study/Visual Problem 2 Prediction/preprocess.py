import data
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd


kp = data.read_key_points()
common_categories = kp['category']
common_categories = common_categories[~common_categories.isin(['Restrooms', 'Entry/Exit'])].unique()


def get_sequence_data(days=None, cutoff_time=12, categories=None, return_prev=False, return_ids=False):
    xs = []
    ys = []
    prevs = []
    ids = []

    print('Getting data')
    for day in days:
        print(day)
        df = data.read_visited_key_points(day, extra=['category'], grouped=True)
        if categories is not None:
            df = df[df['category'].isin(categories)]

        first_time = df['Timestamp'].min()
        if isinstance(cutoff_time, timedelta):
            cutoff = datetime(first_time.year, first_time.month, first_time.day) + cutoff_time
        else:
            cutoff = datetime(first_time.year, first_time.month, first_time.day, cutoff_time)
        df_pre = df[df['Timestamp'] <= cutoff].sort_values('Timestamp').copy()
        df_pre['reverse_order_visited'] = df_pre.groupby('group_id').cumcount(ascending=False)
        df2 = (
            # df_pre[df_pre['reverse_order_visited'] < 20]
            df_pre
            .pivot(index='group_id', columns='reverse_order_visited', values='place_id')
            .fillna(0)
        )
        xs.append(df2.values)
        # The next place visited
        next_places = get_next_place(df, cutoff).reindex(df2.index, fill_value=0)
        ys.append(next_places.values)
        if return_prev:
            # The previous place visited
            prev_places = get_prev_place(df, cutoff).reindex(df2.index, fill_value=0)
            prevs.append(prev_places.values)
        if return_ids:
            id_df = pd.DataFrame(data=df2.index)
            id_df.columns = ['group_id']
            id_df['Day'] = day
            ids.append(id_df)

    # add the results to one big array
    rows = sum(x.shape[0] for x in xs)
    cols = max(x.shape[1] for x in xs)
    x = np.zeros((rows, cols))
    insert_index = 0
    for x_, y_ in zip(xs, ys):
        x[insert_index:insert_index + x_.shape[0], :x_.shape[1]] = x_
        insert_index += x_.shape[0]

    out = x, np.concatenate(ys)
    if return_prev:
        out += np.concatenate(prevs),
    if return_ids:
        out += pd.concat(ids),
    return out


def get_bag_data(days=None, cutoff_time=12, categories=None, return_prev=False, return_ids=False):
    xs = []
    ys = []
    prevs = []
    ids = []

    print('Getting data')
    for day in days:
        print('  {}'.format(day))
        df = data.read_visited_key_points(day, extra=['category'], grouped=True)
        if categories is not None:
            df = df[df['category'].isin(categories)]
            place_ids = kp.loc[kp['category'].isin(categories), 'place_id']
        else:
            place_ids = kp['place_id']

        first_time = df['Timestamp'].min()
        if isinstance(cutoff_time, timedelta):
            cutoff = datetime(first_time.year, first_time.month, first_time.day) + cutoff_time
        else:
            cutoff = datetime(first_time.year, first_time.month, first_time.day, cutoff_time)
        df_pre = df[df['Timestamp'] <= cutoff]
        x = (
            # group by the group and place id, and then count the totals
            df_pre.groupby(['group_id', 'place_id'])
            .size()
            # rearrange into matrix format, with each place_id as the columns (and so group_id as the rows)
            .unstack('place_id')
            # makes sure there are columns for each place id (without this, there won't be any zero columns)
            .reindex(columns=place_ids)
            .fillna(0)
            .astype('int64')
        )
        xs.append(x.values)
        # The next place visited
        next_places = get_next_place(df, cutoff).reindex(x.index, fill_value=0)
        ys.append(next_places.values)
        if return_prev:
            # The previous place visited
            prev_places = get_prev_place(df, cutoff).reindex(x.index, fill_value=0)
            prevs.append(prev_places.values)
        if return_ids:
            id_df = pd.DataFrame(data=x.index)
            id_df.columns = ['group_id']
            id_df['Day'] = day
            ids.append(id_df)

    # x = (x > 0).astype('int64')
    out = np.concatenate(xs), np.concatenate(ys)
    if return_prev:
        out += np.concatenate(prevs),
    if return_ids:
        out += pd.concat(ids),
    return out


def get_next_place(df, cutoff):
    return (
        df[df['Timestamp'] > cutoff]
        .sort_values('Timestamp')  # oops... without this it doesn't work at all
        .groupby('group_id')
        .first()
        ['place_id']
    )


def get_prev_place(df, cutoff):
    return (
        df[df['Timestamp'] <= cutoff]
        .sort_values('Timestamp')  # oops... without this it doesn't work at all
        .groupby('group_id')
        .last()
        ['place_id']
    )


def main():
    """
    Sanity check for if the 'y' (i.e. the next attraction someone went to) is right

    (I'm pretty sure it's wrong at the moment
    """
    day = 'Fri'
    df = data.read_visited_key_points(day, extra=['category'], grouped=True)
    df = df[df['category'].isin(common_categories)]
    first_time = df['Timestamp'].min()
    first_day = datetime(first_time.year, first_time.month, first_time.day)
    df['Timestamp Seconds'] = (df['Timestamp'] - first_day) / np.timedelta64(1, 's')

    cutoff = datetime(first_time.year, first_time.month, first_time.day, 12)
    cutoff_seconds = np.timedelta64(cutoff - first_day, 's') / np.timedelta64(1, 's')

    # next place, without sorting
    fig, ax1 = plt.subplots()
    fig, ax2 = plt.subplots()
    fig, ax3 = plt.subplots()

    for ax in (ax1, ax2, ax3):
        ax.scatter(x=df['Timestamp Seconds'], y=df['group_id'], lw=0, color='blue', s=30)
        ax.plot([cutoff_seconds, cutoff_seconds], [df['group_id'].min(), df['group_id'].max()], color='red', lw=2)
        ax.xaxis.set_major_formatter(data.timedelta_formatter)
        ax.set_ylabel('id')
        ax.set_xlabel('time')

    np1 = df[df['Timestamp'] > cutoff].groupby('group_id').first().reset_index()
    ax2.scatter(x=np1['Timestamp Seconds'], y=np1['group_id'], lw=0, color='red', s=30)
    ax2.set_title('Without sorting')

    # next place, with sorting
    np2 = df[df['Timestamp'] > cutoff].sort_values('Timestamp').groupby('group_id').first().reset_index()
    ax3.scatter(x=np2['Timestamp Seconds'], y=np2['group_id'], lw=0, color='red')
    ax3.set_title('With sorting')

    plt.show()


if __name__ == '__main__':
    main()