# datacleaning.py
import pandas as pd

class BreastCancerData:
    def __init__(self, filepath):
        self.filepath = filepath
        self.column_names = [
            "Sample code number",
            "Clump Thickness",
            "Uniformity of Cell Size",
            "Uniformity of Cell Shape",
            "Marginal Adhension",
            "Single Epithelial Cell Size",
            "Bare Nuclei",
            "Bland Chromatin",
            "Normal Nucleoli",
            "Mitoses",
            "Class"
        ]
        self.df = None

    def load_data(self):
        self.df = pd.read_csv(self.filepath, header=None, names=self.column_names, na_values=["?"])
        print("[INFO] Data Loaded successfully")

    def get_info(self):
        if self.df is not None:
            return self.df.info()
        else:
            raise ValueError("Data not Loaded")

    def show_head(self, n=5):
        if self.df is not None:
            return self.df.head(n)
        else:
            raise ValueError("Data not Loaded")

    def save_csv(self, output_path="new_breast_cancer.csv"):
        if self.df is not None:
            self.df.to_csv(output_path, index=False)
            print(f"[INFO] Data Saved to {output_path}")
        else:
            raise ValueError("No Data to save, Make sure to load first")

class DataCleaner:
    def __init__(self, df):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Expected a pandas DataFrame")
        self.df = df

    def handling_missing(self, strategy="drop"):
        if strategy == "drop":
            initial_shape = self.df.shape
            self.df.dropna(inplace=True)
            print(f"[INFO] Dropped {initial_shape[0] - self.df.shape[0]} rows with missing values")
        elif strategy == "mean":
            self.df.fillna(self.df.mean(numeric_only=True), inplace=True)
            print("[INFO] Filled missing values with column means")
        elif strategy == "median":
            self.df.fillna(self.df.median(numeric_only=True), inplace=True)
            print("[INFO] Filled missing values with column medians")
        else:
            raise ValueError("Invalid Strategy. Use 'drop', 'mean' or 'median'")

    def get_clean_data(self):
        return self.df
