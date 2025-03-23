import pandas as pd
import random
import pathlib
import datetime
from restaurant_database_generate import FoodCarbonCalculator

# 全局變數
restock_cache = {}

def get_weight_from_menu(restaurant, dish_name, ingredient_name, oz_override=None):
    """
    從牛排館菜單.csv 或 便當店菜單.csv 中根據 dish_name 與 ingredient_name 取得重量。
    1. 重量值從檔案中的 ingredient(1~5)_weight 欄位取得，並隨機在 ±3% 範圍內浮動。
    2. 若餐廳為牛排館且該食材的「食物分類子項」屬於 ["牛肉類", "豬肉類", "羊肉類"]，
       則隨機選擇 6, 8, 10, 12 oz，轉換成克後，再隨機在 ±3% 生成重量。
    """
    # 選擇菜單檔案
    if restaurant == "牛排館":
        menu_file = pathlib.Path("牛排館菜單.csv")
    else:
        menu_file = pathlib.Path("便當店菜單.csv")
    
    try:
        df_menu = pd.read_csv(menu_file)
    except Exception as e:
        print(f"讀取菜單檔案失敗: {menu_file}, {e}")
        return 100  # 若讀取失敗則回傳預設值

    # 讀取單品食材資料庫
    try:
        df_ingredients = pd.read_csv(pathlib.Path("單品食材資料庫.csv"))
    except Exception as e:
        print(f"讀取單品食材資料庫失敗: {e}")
        return 100  # 若讀取失敗則回傳預設值

    # 根據 dish_name 過濾對應的菜餚
    df_dish = df_menu[df_menu["dish_name"] == dish_name]
    if df_dish.empty:
        print(f"菜餚 {dish_name} 在 {menu_file} 中不存在")
        return 100  # 若無該菜餚則使用預設值

    row = df_dish.iloc[0]
    weight = None
    # 從 ingredient1~ingredient5 中尋找匹配的食材
    for i in range(1, 6):
        col_name = f"ingredient{i}_name"
        if col_name in row and row[col_name] == ingredient_name:
            weight = row.get(f"ingredient{i}_weight", None)
            # 查找該食材的食物分類子項
            ingredient_data = df_ingredients[df_ingredients["食材名稱"] == ingredient_name]
            if not ingredient_data.empty:
                food_subcategory = ingredient_data["食物分類子項"].iloc[0]
                # 如果是牛排館且食物分類子項屬於肉類，採用 oz 換算
                if (restaurant == "牛排館" and 
                    food_subcategory in ["牛肉類", "豬肉類", "羊肉類"]):
                    if oz_override is not None:
                        oz = oz_override
                    else:
                        oz = random.choice([6, 8, 10, 12])
                    weight = oz * 28.3495  # 轉換成克
            break

    if weight is None:
        print(f"食材 {ingredient_name} 在 {dish_name} 中未找到重量")
        weight = 100  # 若找不到則使用預設值

    try:
        weight = float(weight)
    except Exception:
        print(f"重量轉換失敗，使用預設值: {weight}")
        weight = 100

    # 隨機在 ±3% 範圍內調整
    weight = weight * (1 + random.uniform(-0.03, 0.03))
    return round(weight, 2)

# === 函數定義區 ===

def generate_ingredient_entry(calculator, ingredient_name, restaurant, dish_name, module, current_date, oz_override=None, week=None, day=None):
    """生成單筆食材記錄，包含碳排、倉儲以及新加入的製造時間數據"""
    # 從食材資料庫中尋找匹配的食材
    # 注意：由於不希望改動 restaurant_database_generate.py，因此僅傳入 ingredient_name
    matches = calculator.select_best_matches(ingredient_name)
    if matches is None:
        return None

    # 選取第一筆匹配記錄
    match = matches.iloc[0]

    # 設定進貨日期：以當前排程日期加 1 天，格式化為字串
    purchase_date = (current_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d")

    # 決定保存方式：優先使用資料庫值，若無則推斷
    storage = (match["保存方式"] if not pd.isna(match["保存方式"])
               else calculator.infer_storage_method(match["食物分類大項"], match["食材名稱"]))

    # 取得重量：參考 restaurant_database_generate.py 中的邏輯，從對應菜單取得
    weight = get_weight_from_menu(restaurant, dish_name, ingredient_name, oz_override=oz_override)

    # 新增欄位順序：
    # [編號, 代碼, 食材名稱, 食物分類大項, 食物分類子項,
    #  菜餚名, 菜餚類型, 重量, 數量, 製造時間, 進貨時間,
    #  生食/熟食, 烹飪方式, 保存方式, 產地]
    entry = [
        None,                                # 編號 (後續自動生成)
        match["代碼"],                       # 食材代碼
        match["食材名稱"],                   # 食材名稱
        match["食物分類大項"],                # 食物分類大項
        match["食物分類子項"],                # 食物分類子項
        dish_name,                           # 菜餚名
        module,                              # 菜餚類型 (此處以餐廳名稱+餐廳名稱作預設)
        weight,                              # 重量（由 get_weight_from_menu 取得）
        1,                                   # 數量
        "",                                  # 製造時間 (新欄位，待後續計算)
        purchase_date,                       # 進貨時間
        match["生食/熟食"],                  # 生食/熟食
        match["烹飪方式"],                   # 烹飪方式
        storage,                             # 保存方式
        random.choice(calculator.ORIGINS)      # 產地 (隨機選擇)
    ]

    # ※ 移除原本的「調整重量」區塊，因為重量已由 get_weight_from_menu 處理

    # 依據新的運輸邏輯設定額外的運輸碳排加值 (additional)
    origin = entry[14]  # 產地所在欄位
    if origin == "美國":
        if storage == "冷藏":
            additional = random.choice([9.17, 2.44])
        elif storage == "冷凍":
            additional = random.choice([9.47, 4.44])
        elif storage == "常溫":
            additional = random.choice([8.87, 0.44])
        else:
            additional = 0
    elif origin == "澳洲":
        if storage == "冷藏":
            additional = random.choice([5.84, 1.78])
        elif storage == "冷凍":
            additional = random.choice([6.04, 3.28])
        elif storage == "常溫":
            additional = random.choice([5.74, 0.28])
        else:
            additional = 0
    elif origin == "日本":
        if storage == "冷藏":
            additional = random.choice([1.71, 3.28])
        elif storage == "冷凍":
            additional = random.choice([1.81, 0.88])
        elif storage == "常溫":
            additional = random.choice([1.61, 0.08])
        else:
            additional = 0
    elif origin == "台灣":
        additional = 0
    else:
        additional = 0

    # 在此處加入生成製造時間的邏輯
    try:
        purchase_date_dt = datetime.datetime.strptime(purchase_date, "%Y-%m-%d").date()
    except Exception:
        purchase_date_dt = current_date
    delta = 0
    if origin == "美國":
        if storage == "冷藏":
            if abs(additional - 9.17) < 1e-6:
                delta = 3
            elif abs(additional - 2.44) < 1e-6:
                delta = 20
        elif storage == "冷凍":
            if abs(additional - 9.47) < 1e-6:
                delta = 3
            elif abs(additional - 4.44) < 1e-6:
                delta = 20
        elif storage == "常溫":
            if abs(additional - 8.87) < 1e-6:
                delta = 3
            elif abs(additional - 0.44) < 1e-6:
                delta = 20
    elif origin == "澳洲":
        if storage == "冷藏":
            if abs(additional - 5.84) < 1e-6:
                delta = 2
            elif abs(additional - 1.78) < 1e-6:
                delta = 15
        elif storage == "冷凍":
            if abs(additional - 6.04) < 1e-6:
                delta = 2
            elif abs(additional - 3.28) < 1e-6:
                delta = 15
        elif storage == "常溫":
            if abs(additional - 5.74) < 1e-6:
                delta = 2
            elif abs(additional - 0.28) < 1e-6:
                delta = 15
    elif origin == "日本":
        if storage == "冷藏":
            if abs(additional - 1.71) < 1e-6:
                delta = 1
            elif abs(additional - 3.28) < 1e-6:
                delta = 4
        elif storage == "冷凍":
            if abs(additional - 1.81) < 1e-6:
                delta = 1
            elif abs(additional - 0.88) < 1e-6:
                delta = 4
        elif storage == "常溫":
            if abs(additional - 1.61) < 1e-6:
                delta = 1
            elif abs(additional - 0.08) < 1e-6:
                delta = 4
    elif origin == "台灣":
        delta = 1
    manufacturing_date = purchase_date_dt - datetime.timedelta(days=delta)
    manufacturing_date_str = manufacturing_date.strftime("%Y-%m-%d")
    entry[9] = manufacturing_date_str

    # 取得原始碳排數據，不加入 additional
    base_carbon_list = [round(float(match.get(col, 0)), 2) for col in calculator.CARBON_COLUMNS]
    
    # 計算基礎碳排（不含額外運輸碳排），使用前 6 項碳排數值乘以重量
    base_carbon_sum = round(sum(base_carbon_list[:-1]) * weight, 2)
    # 計算最終總碳排：基礎碳排加上額外運輸碳排（額外的部分不乘 weight）
    total_carbon = base_carbon_sum + additional
    # 更新「全部」欄位 (index 6) 為最終總碳排
    base_carbon_list[6] = total_carbon
    
    # 將計算後的基礎碳排數據追加至記錄
    entry.extend(base_carbon_list)
     
    # 計算倉儲碳排，確保天數差不為負數
    try:
        purchase_date_dt = datetime.datetime.strptime(purchase_date, "%Y-%m-%d").date()
    except Exception:
        purchase_date_dt = current_date
    days_diff = max(0, (current_date - purchase_date_dt).days)
    factor = {"常溫": 0, "冷藏": 0.1, "冷凍": 0.2}.get(storage, 0)
    warehouse = round(days_diff * factor, 2)

    # 將 "全部" 欄位已更新於 base_carbon_list[6]
    # 計算總碳排（不含倉儲）已在 total_carbon 中
    total_carbon_with_warehouse = round(total_carbon + warehouse, 2)

    # 追加倉儲與加總碳排數據到記錄
    entry.extend([warehouse, total_carbon_with_warehouse])
    entry.append("")                       # 狀態 (索引 23，預設為空)
    # 新增補貨日期欄 (索引 24)，填入 "於{進貨時間}進貨"
    if week is not None and day is not None:
        # day 是從 0 開始，因此顯示時用 day + 1
        entry.append(f"於第{week}週第{day + 1}天進貨")
    else:
        # 若未提供 week 和 day，回退到原始日期格式
        entry.append(f"於{purchase_date}進貨")

    return entry


def restock_ingredient(ingredient_name, restaurant, dish_name, module, steakhouse_data, boxed_meal_data, 
                       restaurant_data, current_date, steakhouse_carbon_cache, boxed_meal_carbon_cache, 
                       restock_count=100, oz_override=None, ingredient_initial_stock=None, week=None, day=None):
    """
    為指定餐廳補貨，使庫存補足到100%（依據缺口數量 restock_count），並更新 Excel 檔案和緩存。
    
    參數:
      ingredient_name: 食材名稱
      restaurant: 餐廳名稱 ("牛排館" 或 "便當店")
      dish_name: 菜餚名
      module: 模組名稱
      steakhouse_data, boxed_meal_data: 各餐廳的資料字典
      current_date: 當前日期（datetime.date）
      steakhouse_carbon_cache, boxed_meal_carbon_cache: 各餐廳的碳排快取
      restock_count: 補貨缺口數量，表示需要補充多少筆記錄
      oz_override: (可選) 原為牛排館肉品指定 oz 值，此處保留但不使用
      ingredient_initial_stock: 初始庫存字典
    回傳:
      無，直接更新餐廳資料和快取，同時將資料寫回 Excel
    """
    # 初始化計算器
    calculator = FoodCarbonCalculator(pathlib.Path("單品食材資料庫.csv"))
    new_entries = []
    
    # 統一生成補貨記錄，不考慮 oz 或牛排館主餐肉類特殊處理
    for _ in range(restock_count):
        entry = generate_ingredient_entry(calculator, ingredient_name, restaurant, dish_name, module, current_date, oz_override, week, day)
        if entry:
            new_entries.append(entry)
        else:
            print(f"[除錯] 生成記錄失敗，食材: {ingredient_name}")
    
    if not new_entries:
        print(f"無法為 {ingredient_name} 生成新記錄")
        return
    
    # !!修改：這邊會讓原始庫存一直增加所以註解掉
    # 更新 ingredient_initial_stock，使用統一的 key 格式 (ingredient_name, dish_name, None)
    # if ingredient_initial_stock is not None:
    #     # new_stock = ingredient_initial_stock.copy()
    #     key = (ingredient_name, dish_name, None)
    #     old_stock = ingredient_initial_stock.get(key, 0)
    #     ingredient_initial_stock[key] = old_stock + restock_count
    
    # 根據餐廳選擇目標資料字典
    if restaurant == "牛排館":
        data = steakhouse_data
        cache = steakhouse_carbon_cache
    else:
        data = boxed_meal_data
        cache = boxed_meal_carbon_cache

    sheet_name = f"完整{module}碳排"
    if sheet_name not in data:
        data[sheet_name] = pd.DataFrame(columns=[
            "編號", "代碼", "食材名稱", "食物分類大項", "食物分類子項",
            "菜餚名", "菜餚類型", "重量", "數量", "製造時間", "進貨時間",
            "生食/熟食", "烹飪方式", "保存方式", "產地", "農業", "加工", "包裝", "超市及配送",
            "運輸碳排", "能源消耗", "全部", "倉儲", "加總碳排", "狀態", "補貨日期"
        ])
    df = data[sheet_name]
    expected_columns = [
        "編號", "代碼", "食材名稱", "食物分類大項", "食物分類子項",
        "菜餚名", "菜餚類型", "重量", "數量", "製造時間", "進貨時間",
        "生食/熟食", "烹飪方式", "保存方式", "產地", "農業", "加工", "包裝", "超市及配送",
        "運輸碳排", "能源消耗", "全部", "倉儲", "加總碳排", "狀態", "補貨日期"
    ]
    if len(df.columns) != len(expected_columns):
        df = df.reindex(columns=expected_columns, fill_value="")
    max_id = df["編號"].max() if not df.empty else 0
    for i, entry in enumerate(new_entries, start=int(max_id) + 1):
        entry[0] = i  # 分配新編號
        # 將新記錄存入 restock_cache
        new_df_entry = pd.DataFrame([entry], columns=expected_columns)
        new_index = df.index.max() + 1 if not df.empty else 0
        restock_cache[(restaurant, sheet_name, new_index)] = new_df_entry.iloc[0]

    # 更新 restaurant_data 和 data（僅在記憶體中）
    new_df = pd.DataFrame(new_entries, columns=expected_columns)
    df = pd.concat([df, new_df], ignore_index=True)
    df["狀態"] = df["狀態"].fillna("")
    data[sheet_name] = df
    restaurant_data[sheet_name] = df.copy()  # 同步更新記憶體中的資料

    # 更新碳排緩存
    from test_greedy import compute_base_storage_carbon, update_dynamic_carbon_cache
    cache[sheet_name] = compute_base_storage_carbon(df)
    update_dynamic_carbon_cache(cache[sheet_name], current_date)
    print(f"已為 {restaurant} {module} 的 {ingredient_name} 生成 {restock_count} 筆補貨記錄")

    # # 從 v6_schedule.py 導入更新快取與寫入 Excel 的函式
    # from test import compute_base_storage_carbon, update_ingredient_file, update_dynamic_carbon_cache
    # cache[sheet_name] = compute_base_storage_carbon(df)
    # update_dynamic_carbon_cache(cache[sheet_name], current_date)
    # update_ingredient_file(data, file_path)
    # print(f"已為 {restaurant} {module} 的 {ingredient_name} 補足庫存至100%，新增 {restock_count} 筆記錄")


