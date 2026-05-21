from typing import get_args

import joblib

from api.schemas import StudentProfileRequest
from src.config import (
    CATEGORICAL_DEFAULTS,
    CATEGORICAL_OPTIONS,
    DEFAULT_VALUES,
    NUMERIC_RANGES,
    RAW_FEATURES_PATH,
    SAMPLE_PROFILES,
)


def _field_bound(field, bound_name: str):
    for item in field.metadata:
        value = getattr(item, bound_name, None)
        if value is not None:
            return value
    return None


def test_api_schema_fields_match_configured_feature_contract():
    raw_feature_names = list(joblib.load(RAW_FEATURES_PATH))
    schema_fields = list(StudentProfileRequest.model_fields.keys())

    assert set(schema_fields) == set(raw_feature_names)
    assert set(NUMERIC_RANGES) | set(CATEGORICAL_OPTIONS) == set(raw_feature_names)


def test_api_numeric_bounds_match_config():
    for field_name, (min_value, max_value) in NUMERIC_RANGES.items():
        field = StudentProfileRequest.model_fields[field_name]

        assert _field_bound(field, "ge") == min_value
        assert _field_bound(field, "le") == max_value


def test_api_categorical_literals_match_config():
    for field_name, allowed_values in CATEGORICAL_OPTIONS.items():
        annotation = StudentProfileRequest.model_fields[field_name].annotation

        assert list(get_args(annotation)) == allowed_values


def test_defaults_are_valid_against_feature_config():
    for field_name, default_value in DEFAULT_VALUES.items():
        if field_name in NUMERIC_RANGES:
            min_value, max_value = NUMERIC_RANGES[field_name]
            assert min_value <= default_value <= max_value
        elif field_name in CATEGORICAL_OPTIONS:
            assert default_value in CATEGORICAL_OPTIONS[field_name]


def test_categorical_missing_defaults_are_valid_and_explicit():
    assert set(CATEGORICAL_DEFAULTS).issubset(CATEGORICAL_OPTIONS)

    for field_name, default_value in CATEGORICAL_DEFAULTS.items():
        assert default_value in CATEGORICAL_OPTIONS[field_name]


def test_sample_profiles_follow_schema_contract():
    expected_fields = set(StudentProfileRequest.model_fields)

    for profile in SAMPLE_PROFILES.values():
        assert set(profile) == expected_fields
        StudentProfileRequest(**profile)
