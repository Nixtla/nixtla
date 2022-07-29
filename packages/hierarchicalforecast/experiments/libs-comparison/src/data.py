import fire
from datasetsforecast.hierarchical import HierarchicalData


def hierarchical_cols(group: str):
    if group == 'Wiki2':
        return ['Country', 'Access', 'Agent', 'Topic'], ['Country', 'Access', 'Agent', 'Topic'], '_'
    elif group == 'Labour':
        return ['Employment', 'Gender', 'Region'], ['Region', 'Employment', 'Gender'], ','
    elif group == 'TourismSmall':
        return ['State', 'Purpose', 'CityNonCity'], ['Purpose', 'State', 'CityNonCity'], '-'
    raise Exception(f'Unknown group {group}')

def parse_data(group: str):
    #Get bottom time series to use in R
    init_cols, hier_cols, sep = hierarchical_cols(group)
    Y_df, S, tags = HierarchicalData.load('data', group)
    Y_df = Y_df.query('unique_id in @S.columns')
    Y_df[init_cols] = Y_df['unique_id'].str.split(sep, expand=True)
    Y_df = Y_df[init_cols + ['ds', 'y']]
    Y_df = Y_df.groupby(init_cols + ['ds']).sum().reset_index()
    Y_df.to_csv(f'data/{group}.csv', index=False)


if __name__=="__main__":
    fire.Fire(parse_data)

