"""Example: Using the balanced pool for diverse time series generation.

balanced_pool() returns 42 pre-configured generators covering 15 distinct
behavioral niches, avoiding implicit bias toward any single domain.
"""

from synforecast import SynSet, balanced_pool

generators = balanced_pool(
    min_length=200, max_length=200, freq="D", engine="polars", seed=42
)
dataset = SynSet(generators)

# 1 series per generator = 42 diverse time series
df = dataset.generate(n_series_per_generator=1)

print(f"Generated {df['unique_id'].n_unique()} series")
print(f"Total observations: {len(df)}")
print(f"Columns: {df.columns}")
print()

print("Contributing generators:")
for gen in generators:
    print(f"  {gen.alias}")
