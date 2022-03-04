from time import sleep

import pandas as pd


if __name__=="__main__":
    temps = pd.read_csv('dataset/temps-stamped.txt',
                        sep=' ',            
                        names=['unique_id', 'y', 'ds'],
                        header=None)

    for idrow, (uid, y, ds) in temps.iterrows():
        wait_segs = ds / 1e9 - pd.Timestamp.now().timestamp()
        if wait_segs > 0:
            sleep(wait_segs)
        print(f'{uid} {y} {ds}', flush=True)
                        
