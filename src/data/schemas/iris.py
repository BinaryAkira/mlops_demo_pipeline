import pandera.pandas as pa   # new recommended import

schema = pa.DataFrameSchema(
    {
        "sepal length (cm)": pa.Column(float, pa.Check.gt(0)),
        "sepal width (cm)": pa.Column(float, pa.Check.gt(0)),
        "petal length (cm)": pa.Column(float, pa.Check.gt(0)),
        "petal width (cm)": pa.Column(float, pa.Check.gt(0)),
    }
)
