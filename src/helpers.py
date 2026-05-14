# src/helpers.py

import pandas as pd
from src.config import DEFAULT_VALUES, FRIENDLY_LABELS


def get_friendly_label(column_name: str) -> str:
    """
    Đổi tên cột kỹ thuật sang tên dễ đọc hơn cho giao diện.
    Nếu không có trong FRIENDLY_LABELS thì thay dấu _ bằng khoảng trắng.
    """
    return FRIENDLY_LABELS.get(column_name, column_name.replace("_", " "))


def build_default_input(raw_feature_names: list[str]) -> dict:
    """
    Tạo một dictionary mặc định cho tất cả các input field.
    Dùng khi khởi tạo form hoặc tạo template CSV.
    """
    data = {}

    for feature in raw_feature_names:
        data[feature] = DEFAULT_VALUES.get(feature, None)

    return data


def dict_to_single_row_df(data: dict, raw_feature_names: list[str]) -> pd.DataFrame:
    """
    Chuyển dictionary input thành DataFrame 1 dòng
    và đảm bảo thứ tự cột đúng như model mong muốn.
    """
    row = {
        feature: data.get(feature, DEFAULT_VALUES.get(feature, None))
        for feature in raw_feature_names
    }
    return pd.DataFrame([row])


def format_score(score: float) -> str:
    """
    Format điểm cho đẹp khi hiển thị ra UI.
    """
    try:
        return f"{float(score):.1f}"
    except Exception:
        return "N/A"


def safe_get_first_value(df: pd.DataFrame, column_name: str, default=None):
    """
    Lấy giá trị đầu tiên của một cột trong DataFrame cho an toàn.
    Nếu cột không tồn tại hoặc lỗi thì trả về default.
    """
    try:
        return df[column_name].iloc[0]
    except Exception:
        return default