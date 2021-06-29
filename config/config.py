MODALITY_MAP = {
    "face": 0,
    "tongueTop": 1,
    "tongueBottom": 2,
    "pulse": 3,
    "query": 4
}

SUPPORT_FEATURES = {
    "face": {
        "face_color": 0, # 面部颜色
        "face_L_value": 0, # L值
        "face_A_value": 0, # A值
        "face_B_value": 0, # B值
        "lip_color": 0, # 嘴唇颜色
        "lip_L_value": 0, # L值
        "lip_A_value": 0, # A值
        "lip_B_value": 0, # B值
        "face_light_result": 0, # 光泽判断结果
        "face_light_index": 0, # 有光泽指数
        "face_less_light_index": 0, # 少光泽指数
        "face_no_light_index": 0 # 无光泽指数
    },
    "tongue_top": {
        "tongue_color": 0, # 舌色
        "tongue_L_value": 0, # L
        "tongue_A_value": 0, # A
        "tongue_B_value": 0, # B
        "tongue_width": 0, # 胖瘦指数
        "tongue_thickness": 0, # 厚薄指数
        "coating_color": 0, # 苔色
        "coating_L_value": 0, # L
        "coating_A_value": 0, # A
        "coating_B_value": 0, # B
        "crack": 0, # 裂纹
        "crack_index": 0, # 裂纹指数
        "tooth_mark": 0 # 齿痕检测
    },
    "tongue_bottom": {
        "veins_color": 0, # 脉络颜色
        "veins_L_value": 0, # L
        "veins_A_value": 0, # A
        "veins_B_value": 0, # B
        "veins_index": 0 # 脉络形状指数(宽高比)
    },
    "pulse": {
        "hemodynamics": 0, # 血液动力学参数
        "condition": 0 # 脉象
    }
}