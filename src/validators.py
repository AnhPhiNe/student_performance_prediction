# src/validators.py

import pandas as pd

from src.config import CATEGORICAL_OPTIONS, NUMERIC_RANGES


def validate_required_columns(df: pd.DataFrame, required_columns: list[str]) -> dict:
    """
    Check required and unexpected columns.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    extra_columns = [col for col in df.columns if col not in required_columns]

    return {
        "missing_columns": missing_columns,
        "extra_columns": extra_columns,
    }


def validate_numeric_ranges(df: pd.DataFrame) -> list[str]:
    """
    Check that numeric columns contain valid numbers inside configured ranges.
    """
    errors = []

    for col, (min_val, max_val) in NUMERIC_RANGES.items():
        if col in df.columns:
            numeric_series = pd.to_numeric(df[col], errors="coerce")

            if numeric_series.isna().any():
                errors.append(f"Column '{col}' contains invalid numeric values.")
                continue

            invalid_mask = (numeric_series < min_val) | (numeric_series > max_val)
            if invalid_mask.any():
                errors.append(f"Column '{col}' must be between {min_val} and {max_val}.")

    return errors


def validate_categorical_values(df: pd.DataFrame) -> list[str]:
    """
    Check that categorical columns contain only configured allowed values.
    """
    errors = []

    for col, valid_options in CATEGORICAL_OPTIONS.items():
        if col in df.columns:
            actual_values = set(df[col].dropna().astype(str).unique())
            allowed_values = set(valid_options)

            invalid_values = sorted(actual_values - allowed_values)

            if invalid_values:
                errors.append(
                    f"Column '{col}' has invalid values: {invalid_values}. Allowed values: {valid_options}"
                )

    return errors


def validate_no_missing_values(df: pd.DataFrame, required_columns: list[str]) -> list[str]:
    """
    Check that required columns do not contain missing values.
    """
    errors = []

    for col in required_columns:
        if col in df.columns and df[col].isna().any():
            errors.append(f"Column '{col}' contains missing values.")

    return errors


def validate_input_dataframe(df: pd.DataFrame, required_columns: list[str]) -> dict:
    """
    Validate the full input DataFrame contract.
    """
    errors = []
    warnings = []

    col_check = validate_required_columns(df, required_columns)

    if col_check["missing_columns"]:
        errors.append(f"Missing required columns: {col_check['missing_columns']}")

    if col_check["extra_columns"]:
        warnings.append(f"Unexpected extra columns will be ignored: {col_check['extra_columns']}")

    if not errors:
        errors.extend(validate_no_missing_values(df, required_columns))
        errors.extend(validate_numeric_ranges(df))
        errors.extend(validate_categorical_values(df))

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }


def validate_single_input_dict(input_dict: dict, required_columns: list[str]) -> dict:
    """
    Validate a single prediction payload.
    """
    df = pd.DataFrame([input_dict], columns=required_columns)
    return validate_input_dataframe(df, required_columns)
