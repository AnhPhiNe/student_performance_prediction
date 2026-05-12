# Trong file src/utils.py
import pandas as pd

def add_new_features(df):
    X = df.copy()
    # Tính toán thẳng, không dùng .dtype hay vòng lặp phức tạp
    X['Study_Efficiency'] = X['Previous_Scores'] / (X['Hours_Studied'] + 1)
    X['Total_Study_Time'] = X['Hours_Studied'] + (X['Tutoring_Sessions'] * 1.5)
    
    # Ép kiểu cho chắc ăn
    is_extra = (X['Extracurricular_Activities'] == 'Yes').astype(int)
    X['Engagement_Index'] = (X['Attendance'] + (is_extra * 10)) / 100
    
    return X # Trả về nguyên cái bảng đã có thêm 3 cột