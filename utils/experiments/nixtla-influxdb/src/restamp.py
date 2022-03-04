import pandas as pd


if __name__=="__main__":
    temps = pd.read_csv('dataset/temps.txt', 
                        sep=' ',
                        names=['unique_id', 'y', 'ds'],
                        header=None)
    temps['ds'] = pd.to_datetime(temps['ds'])
    temps['ds'] -= temps['ds'].min() 
    temps['ds'] = pd.to_timedelta(temps['ds']) + pd.Timestamp.now()

    # reconvert as int
    temps['ds'] = temps['ds'].view(int)
    temps.to_csv('dataset/temps-stamped.txt', 
                 header=None, 
                 index=None, 
                 sep=' ')

    
