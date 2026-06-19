from src.data.schemas import iris

SCHEMAS = {
    "iris": iris.schema,
}


def validate_data(df, dataset_name: str):
    schema = SCHEMAS[dataset_name]
    return schema.validate(df)
