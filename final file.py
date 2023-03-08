import pandas as pd

# Read in the two CSV files
df1 = pd.read_csv("submission_oily_ML_DTS.csv")
df2 = pd.read_csv("submission_oily_ML_DTC.csv")

# Merge the columns of the two dataframes
merged_df = pd.concat([df2, df1], axis=1)

# Save the merged dataframe to a new CSV file
merged_df.to_csv("final_file.csv", index=False)
