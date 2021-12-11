import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('data/Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv')
    renamer = {'RegionID': 'unique_id'}
    df.rename(columns=renamer, inplace=True)
    df.drop(columns=['SizeRank', 'RegionName', 'RegionType', 'StateName'], inplace=True)
    df = df.set_index('unique_id').stack(dropna=False).rename_axis(['unique_id', 'ds']).rename('y').reset_index()
    df['y'] = df.groupby('unique_id')['y'].ffill()
    df.dropna(inplace=True)

    test = df.groupby('unique_id').tail(4)
    train = df.drop(test.index)
    print(train.groupby('unique_id').size())
    print(test.groupby('unique_id').size())


    train.to_csv('data/prepared-data-train.csv', index=False)
    test.to_csv('data/prepared-data-test.csv', index=False)
