import argparse

import pandas as pd


if __name__ == '__main__':
   parser = argparse.ArgumentParser()

   parser.add_argument('--filename1', type=str, required=True)
   parser.add_argument('--filename2', type=str, required=True)

   args = parser.parse_args()

   df1 = pd.read_csv(args.filename1, names=['unique_id', 'ds', 'y'], skiprows=1)
   df2 = pd.read_csv(args.filename2, names=['unique_id', 'ds', 'y'], skiprows=1)

   for df in [df1, df2]:
       df.sort_values(['unique_id', 'ds'], inplace=True)
       df.reset_index(inplace=True, drop=True)
   print(df1.head(), df2.head(), sep='\n')
   
   if not df1.equals(df2):
       raise Exception('Data frames are not equal')
