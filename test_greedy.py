# 五週為一個減碳週期
# 在上一餐沒有超過碳排時，使用新貪婪演算法。但是當上一餐超過碳排時，使用窮舉法。

import pandas as pd
import random
import typing
import pathlib
import datetime

from new_cooking_carbon import load_cooking_data, compute_cooking_carbon
from restock_system import restock_ingredient

# 全局變數（新增碳排緩存）
steakhouse_carbon_cache = {}
boxed_meal_carbon_cache = {}
restock_cache = {}

# ========== Persona.xlsx 相關 ==========
def load_persona(file_path: pathlib.Path) -> dict:
    xls = pd.ExcelFile(str(file_path))
    return {
        "habit": xls.parse("用餐習性"),
        "steak": xls.parse("牛排館選擇"),
        "boxed_meal": xls.parse("便當店選擇")
    }

# 根據居民姓名（name）以及日期類型（例如「平日」），從個人資料中選出符合條件的設定
def get_persona_choice(persona_df: pd.DataFrame, name: str,
                       date_type: str) -> dict:
    df = persona_df[persona_df["name"] == name]
    if "date_type" in df.columns:
        df_match = df[df["date_type"] == date_type]
        if not df_match.empty:
            return df_match.iloc[0].to_dict()
    if not df.empty:
        return df.iloc[0].to_dict()
    return {}

# 篩選餐廳
def select_restaurant(habit, res, date_type, steak_choice_df, boxed_meal_choice_df,
                      steakhouse_data, boxed_meal_data, cooking_data_steak,
                      cooking_data_boxed_meal):
    """ 根據居民的偏好選擇餐廳，並返回對應的數據 """
    steak_prob = float(habit.get("steakhouse_pro", 0.5))  # 預設0.5若無資料
    boxed_meal_prob = float(habit.get("boxed_meal_pro", 0.5))     # 預設0.5若無資料
    total_prob = steak_prob + boxed_meal_prob
    
    if total_prob == 0:  # 避免除以零
        restaurant = random.choice(["牛排館", "便當店"])
    else:
        steak_prob_normalized = steak_prob / total_prob
        if random.random() < steak_prob_normalized:
            restaurant = "牛排館"
        else:
            restaurant = "便當店"

    if restaurant == "牛排館":
        return {
            "restaurant": "牛排館",
            "modules_req": steakhouse_modules,
            "restaurant_data": steakhouse_data,
            "cooking_data": cooking_data_steak,
            "persona_choice": get_persona_choice(steak_choice_df, res, date_type)
        }
    else:
        return {
            "restaurant": "便當店",
            "modules_req": boxed_meal_modules,
            "restaurant_data": boxed_meal_data,
            "cooking_data": cooking_data_boxed_meal,
            "persona_choice": get_persona_choice(boxed_meal_choice_df, res, date_type)
        }

# 根據居民偏好與過去選擇，篩選可用的主食
def filter_available_food(df_origin, restaurant, mod, persona_choice,
                          prev_rice):
    if restaurant == "便當店" and mod == "主食":
        # 若上一餐有米飯，使用 next_rice_pro 和 next_noodle_pro
        if prev_rice == "米飯類":
            rice_prob = float(persona_choice.get("next_rice_pro", 0.5))
            noodle_prob = float(persona_choice.get("next_noodle_pro", 0.5))
        elif prev_rice == "麵食類":
            rice_prob = float(persona_choice.get("rice_pro", 0.5))
            noodle_prob = float(persona_choice.get("next_noodle_pro", 0.5))
        else:
            rice_prob = float(persona_choice.get("rice_pro", 0.5))
            noodle_prob = float(persona_choice.get("noodle_pro", 0.5))

        total = rice_prob + noodle_prob
        intended_type = random.choice(["米飯類", "麵食類"]) if total == 0 else ("米飯類" if random.random() < (rice_prob/total) else "麵食類")
        # 解決一道菜餚對應多個食材的情境
        # **步驟 1**：篩選符合 `intended_type` 的食材
        filtered_df = df_origin[df_origin["食物分類子項"] == intended_type]
        # **步驟 2**：取得符合條件的 `菜餚名`
        selected_dishes = filtered_df["菜餚名"].unique()
        # **步驟 3**：篩選出所有相同 `菜餚名` 的資料
        module_df = df_origin[df_origin["菜餚名"].isin(selected_dishes)].copy()
        return module_df, intended_type
    return df_origin.copy(), prev_rice

# 移除不喜歡或禁止的食物
def exclude_unwanted_food(module_df, habit):
    for key, column in [("dislike_food", "食材名稱"), ("ban_type", "食物分類子項")]:
        values = habit.get(key, "")
        if isinstance(values, str):
            exclude_list = [x.strip() for x in values.split(",") if x.strip()]
            module_df = module_df[~module_df[column].isin(exclude_list)]
    return module_df

# ========== 單品資料庫相關 ==========
# 讀取指定 Excel 檔案中的各個工作表，每個工作表代表一個菜餚資料庫
def load_restaurant_data(
        file_path: pathlib.Path) -> typing.Dict[str, pd.DataFrame]:
    xls = pd.ExcelFile(str(file_path))
    sheets = {}
    for sheet_name in xls.sheet_names:
        df = xls.parse(sheet_name)
        if "狀態" not in df.columns:   # 確保 df 內有 狀態 欄位
            df["狀態"] = ""
        sheets[sheet_name] = df
    return sheets

# 將更新後的餐廳菜餚資料寫回 Excel 檔案中
def update_ingredient_file(ingredients: typing.Dict[str, pd.DataFrame],
                           file_path: pathlib.Path) -> None:
    try:
        absolute_path = file_path.resolve()
        with pd.ExcelWriter(str(absolute_path), engine="xlsxwriter",
                            mode="w") as writer:
            for sheet_name, df in ingredients.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
    except PermissionError:
        print(f"錯誤：無法寫入 {absolute_path}，檔案可能被鎖定或無寫入權限。")
    except Exception as e:
        print(f"錯誤：更新 {absolute_path} 失敗，原因：{e}")

# 分別寫入牛排館跟便當店的狀態
def save_selection_results(steakhouse_data: dict, boxed_meal_data: dict,
                           steakhouse_file: pathlib.Path,
                           boxed_meal_file: pathlib.Path) -> None:
    # 應用補貨緩存中的變更
    for (restaurant, sheet_name, idx), updated_record in restock_cache.items():
        if restaurant == "牛排館":
            if sheet_name in steakhouse_data:
                steakhouse_data[sheet_name].loc[idx] = updated_record
        else:
            if sheet_name in boxed_meal_data:
                boxed_meal_data[sheet_name].loc[idx] = updated_record
    
    # 寫入 Excel
    update_ingredient_file(steakhouse_data, steakhouse_file)
    update_ingredient_file(boxed_meal_data, boxed_meal_file)
    
    # 清空緩存（可選，若希望保留緩存則移除此行）
    restock_cache.clear()

# ========== 計算食物碳排 ==========
# 根據oz以及時間篩選食材
def select_food(group, persona_choice):
    # 若屬於牛、豬、羊類，依據 {oz} 進行重量篩選
    food_category = group.iloc[0]["食物分類子項"]
    if food_category in ["牛肉類", "豬肉類", "羊肉類"]:
        oz_val = float(persona_choice.get("oz", 0))
        if oz_val > 0:
            desired_weight = oz_val * 28.3495  # oz 轉換成克
            lower_bound = desired_weight * 0.97
            upper_bound = desired_weight * 1.03
            group = group[(group["重量"] >= lower_bound) & (group["重量"] <= upper_bound)]
            ##
            # if group.empty:
            #     print(f"[除錯] 食材 '{group.iloc[0]['食材名稱'] if not group.empty else '未知'}' 在 oz={oz_val} 的重量範圍內無記錄")
            ##
            
    if group.empty:
        ##
        # print(f"[除錯] select_food: group 在過濾後為空，食物分類: {food_category}")
        ##
        return group.copy()

    group = group.copy()  # 確保是副本
    group["製造時間_dt"] = pd.to_datetime(group["製造時間"], errors='coerce')
    # 取得所有非空的製造日期
    valid_dates = group["製造時間_dt"].dropna().unique()
    # 對日期排序（日期越早表示越舊）
    valid_dates = sorted(valid_dates)
    # 取最舊的五個不同日期
    candidate_dates = valid_dates[:5]
    # 篩選出製造時間屬於這些日期的所有記錄
    candidate_records = group[group["製造時間_dt"].isin(candidate_dates)]
    # 移除臨時的 datetime 欄位
    candidate_records = candidate_records.drop(columns=["製造時間_dt"])
    
    return candidate_records.copy()

# 預計算食材的基礎碳排和進貨日期
def compute_base_storage_carbon(df):
    cache = {}
    for idx, record in df.iterrows():
        try:
            purchase_date = datetime.datetime.strptime(record["進貨時間"], "%Y-%m-%d").date()
        except Exception:
            purchase_date = datetime.datetime.now().date()
        base_carbon = record["加總碳排"] - record["倉儲"]  # 原始碳排減去舊倉儲碳排
        factor = 0
        if record["保存方式"] == "常溫":
            factor = 0
        elif record["保存方式"] == "冷藏":
            factor = 0.1
        elif record["保存方式"] == "冷凍":
            factor = 0.2
        cache[idx] = {"base_carbon": base_carbon, "purchase_date": purchase_date, "factor": factor}
    return cache

# 根據當前日期更新動態碳排緩存
def update_dynamic_carbon_cache(cache, current_date):
    for idx, info in cache.items():
        purchase_date = info.get("purchase_date", current_date)
        days_diff = (current_date - purchase_date).days
        factor = info.get("factor", 0)
        base_carbon = info.get("base_carbon", 0)
        info["dynamic_carbon"] = base_carbon + days_diff * factor
        # print(f"idx: {idx}, info: {info}")

# 計算最終的碳排放量
def compute_final_carbon(ingredient_carbon, cooking_carbon) -> float:
    if pd.isna(ingredient_carbon):
        ingredient_carbon = 0
    final_val = ingredient_carbon if pd.isna(
        cooking_carbon
    ) or cooking_carbon == 0 else ingredient_carbon + cooking_carbon
    return round(final_val, 2)

# 計算含烹煮方式的碳排放量
def compute_meal_emission(persona_choice, dish_total_carbon, cooking_data,
                          rep_dish_name, mod, restaurant):
    """ 計算碳排放量 """
    cook_rows = cooking_data[(cooking_data["dish_name"] == rep_dish_name)
                             & (cooking_data["module"] == mod)]
    cooking_carbon = (compute_cooking_carbon(cook_rows.iloc[0], rep_dish_name,
                                             mod, persona_choice, restaurant)
                      if not cook_rows.empty else 0)
    return compute_final_carbon(dish_total_carbon, cooking_carbon)

# ================= 全村動態碳排調整函式 =================
def adjust_village_emissions(village_results, week_targets, reserved_carbon):
    # 1. 找出超標者及其超出量
    overages = {}
    for res in village_results:
        if village_results[res] > week_targets[res]:
            overages[res] = village_results[res] - week_targets[res]
    total_excess = sum(overages.values())
    
    # 2. 找出非超標者作為分配對象
    non_over = [res for res in village_results if res not in overages]
    if not non_over or total_excess <= 0:
        return reserved_carbon  # 無需調整
    
    allocation_per_villager = total_excess / len(non_over)
    remaining_allocation = {res: allocation_per_villager for res in non_over}

    # 加入除錯訊息
    print(f"調整全村碳排：總超出量={total_excess}, 非超標村民數={len(non_over)}, 每人分配={allocation_per_villager}")

    # 3. 迭代扣除各村民的 reserved_carbon，加入最大迴圈次數限制
    max_iterations = 1000  # 避免無限迴圈
    iteration = 0
    while any(alloc > 0 for alloc in remaining_allocation.values()) and iteration < max_iterations:
        available = [res for res in remaining_allocation if reserved_carbon[res] > 0]
        if not available:
            print("無可扣預儲村民，結束調整")
            break
        
        total_remaining = sum(remaining_allocation[res] for res in available)
        # 加入除錯訊息
        # print(f"第 {iteration+1} 次迭代：可用村民數={len(available)}, 剩餘分配總額={total_remaining}")
        
        for res in available:
            if reserved_carbon[res] >= remaining_allocation[res]:
                reserved_carbon[res] -= remaining_allocation[res]
                remaining_allocation[res] = 0
            else:
                deficit = remaining_allocation[res] - reserved_carbon[res]
                reserved_carbon[res] = 0
                remaining_allocation[res] = deficit
        
        iteration += 1
    
    if iteration >= max_iterations:
        print("警告：達到最大迭代次數，可能未完全分配碳排")
    
    return reserved_carbon

# 第二個週期後選擇菜餚的邏輯
def select_module_meal(mod, req, avail_df, persona_choice, restaurant, restaurant_data, cooking_data, current_date,
                       ingredient_usage, ingredient_initial_stock, carbon_cache, prev_meal_exceeded=False, use_greedy=True):
    """
    從最舊的五筆製造時間中選擇菜餚，並計算其碳排。
    - 主餐和其他模組一致：
      - prev_meal_exceeded=False：使用貪婪演算法，選擇與平均目標碳排最接近的菜餚。
      - prev_meal_exceeded=True：使用窮舉法，主餐選擇最低碳排組合，其他模組選擇最接近目標的組合。
    """
    sheet_name = "完整" + mod + "碳排"

    # 輔助函式：計算單筆菜餚的最終碳排
    def calc_dish_emission(df, dish):
        dish_groups = df[df["菜餚名"] == dish].groupby("食材名稱")
        total = 0
        for food_name, group in dish_groups:
            filtered = select_food(group, persona_choice)
            if filtered.empty:
                # print(f"[DEBUG] filtered_records is empty for dish '{dish}', food '{food_name}' in module '{mod}'")
                continue
            sel_rec = filtered.sample(n=1, random_state=random.randint(1, 10000)).iloc[0]
            sel_idx = sel_rec.name
            total += carbon_cache[sheet_name][sel_idx]["dynamic_carbon"]
        return compute_meal_emission(persona_choice, total, cooking_data, dish, mod, restaurant)

    # 從最舊五筆中獲取候選菜餚
    avail_df = avail_df.copy()
    avail_df.loc[:, "製造時間_dt"] = pd.to_datetime(avail_df["製造時間"], errors='coerce')
    candidate_dates = sorted(avail_df["製造時間_dt"].dropna().unique())[:5]
    oldest_five_df = avail_df[avail_df["製造時間_dt"].isin(candidate_dates)]
    candidate_dishes = oldest_five_df["菜餚名"].unique().tolist()

    if not candidate_dishes:
        print(f"[DEBUG] No candidate dishes available for module '{mod}' in restaurant '{restaurant}'")
        return [{"菜餚名": "找不到食材", "final_carbon": 0}] * req, 0

    selected_dishes = []
    total_carbon = 0

    # 隨機選擇邏輯（第 1-5 週）
    if not use_greedy:
        selected_names = set()
        for _ in range(req):
            available_dishes = [dish for dish in candidate_dishes if dish not in selected_names]
            if not available_dishes:
                selected_dishes.append({"菜餚名": "找不到食材", "final_carbon": 0})
                continue
            rep_dish_name = random.choice(available_dishes)
            final_carbon = calc_dish_emission(oldest_five_df, rep_dish_name)
            selected_dishes.append({"菜餚名": rep_dish_name, "final_carbon": final_carbon})
            total_carbon += final_carbon
            selected_names.add(rep_dish_name)
    else:
        # 計算每道菜的碳排
        dish_carbon_map = {dish: calc_dish_emission(oldest_five_df, dish) for dish in candidate_dishes}
        remaining_dishes = candidate_dishes.copy()
        target_carbon = globals().get('module_target', 0)
        remaining_req = req

        # 若為牛排館配菜且 meat_pro > 0，先選雞蛋
        if mod == "配菜" and restaurant == "牛排館" and float(persona_choice.get("meat_pro", 0)) > 0:
            egg_dishes = [dish for dish in remaining_dishes if "雞蛋" in dish]
            if egg_dishes:
                egg_dish = random.choice(egg_dishes)
                egg_carbon = dish_carbon_map[egg_dish]
                selected_dishes.append({"菜餚名": egg_dish, "final_carbon": egg_carbon})
                total_carbon += egg_carbon
                remaining_dishes.remove(egg_dish)
                target_carbon -= egg_carbon
                remaining_req -= 1
            else:
                print(f"[警告] 牛排館配菜中無雞蛋可選，繼續選擇")

        # 主餐和其他模組的選擇邏輯
        if mod == "主餐":
            if prev_meal_exceeded:
                # 上一餐超標：使用窮舉法選擇最低碳排組合
                from itertools import combinations   # 從 itertools 模組導入 combinations 函數，用於生成所有可能的菜餚組合
                best_combination = None
                best_carbon = float('inf')  # 目標是最小化總碳排
                for combo in combinations(remaining_dishes, remaining_req):
                    combo_carbon = sum(dish_carbon_map[dish] for dish in combo)
                    if combo_carbon < best_carbon:  # 若當前組合的總碳排小於目前最佳碳排，更新最佳值。
                        best_carbon = combo_carbon
                        best_combination = combo
                if best_combination:
                    for dish in best_combination:
                        final_carbon = dish_carbon_map[dish]
                        selected_dishes.append({"菜餚名": dish, "final_carbon": final_carbon})
                        total_carbon += final_carbon
                else:
                    selected_dishes.extend([{"菜餚名": "找不到食材", "final_carbon": 0}] * remaining_req)
            else:
                # 上一餐未超標：使用貪婪演算法選擇與平均目標最接近的菜餚
                remaining_target = target_carbon
                while remaining_req > 0 and remaining_dishes:
                    if remaining_req > len(remaining_dishes):
                        selected_dishes.extend([{"菜餚名": "找不到食材", "final_carbon": 0}] * (remaining_req - len(remaining_dishes)))
                        remaining_req = len(remaining_dishes)

                    target_per_dish = remaining_target / remaining_req if remaining_req > 0 else 0
                    best_dish = min(remaining_dishes, key=lambda dish: abs(dish_carbon_map[dish] - target_per_dish))
                    final_carbon = dish_carbon_map[best_dish]
                    selected_dishes.append({"菜餚名": best_dish, "final_carbon": final_carbon})
                    total_carbon += final_carbon
                    remaining_dishes.remove(best_dish)
                    remaining_target -= final_carbon
                    remaining_req -= 1

                if remaining_req > 0:
                    selected_dishes.extend([{"菜餚名": "找不到食材", "final_carbon": 0}] * remaining_req)
        else:
            # 其他模組的選擇邏輯（與主餐一致）
            if prev_meal_exceeded:
                # 上一餐超標：使用窮舉法選擇最接近目標的組合
                from itertools import combinations
                best_combination = None
                best_diff = float('inf')
                for combo in combinations(remaining_dishes, remaining_req):
                    combo_carbon = sum(dish_carbon_map[dish] for dish in combo)
                    diff = abs(combo_carbon - target_carbon)
                    if diff < best_diff:
                        best_diff = diff
                        best_combination = combo
                if best_combination:
                    for dish in best_combination:
                        final_carbon = dish_carbon_map[dish]
                        selected_dishes.append({"菜餚名": dish, "final_carbon": final_carbon})
                        total_carbon += final_carbon
                else:
                    selected_dishes.extend([{"菜餚名": "找不到食材", "final_carbon": 0}] * remaining_req)
            else:
                # 上一餐未超標：使用貪婪演算法
                remaining_target = target_carbon
                while remaining_req > 0 and remaining_dishes:
                    if remaining_req > len(remaining_dishes):
                        selected_dishes.extend([{"菜餚名": "找不到食材", "final_carbon": 0}] * (remaining_req - len(remaining_dishes)))

                        remaining_req = len(remaining_dishes)
                    target_per_dish = remaining_target / remaining_req if remaining_req > 0 else 0
                    best_dish = min(remaining_dishes, key=lambda dish: abs(dish_carbon_map[dish] - target_per_dish))
                    final_carbon = dish_carbon_map[best_dish]
                    selected_dishes.append({"菜餚名": best_dish, "final_carbon": final_carbon})
                    total_carbon += final_carbon
                    remaining_dishes.remove(best_dish)
                    remaining_target -= final_carbon
                    remaining_req -= 1

                if remaining_req > 0:
                    selected_dishes.extend([{"菜餚名": "找不到食材", "final_carbon": 0}] * remaining_req)

    return selected_dishes, total_carbon  # 確保總是有返回值

# ========== 排程主程式 ==========
steakhouse_modules = {"主餐": 1, "主食": 1, "麵包": 1, "沙拉": 2, "配菜": 3, "湯品": 1, "飲品": 1}
boxed_meal_modules = {"主食": 1, "主餐": 1, "副餐": 3, "水果": 1, "湯品": 1, "飲品": 1}

menu_records = []

# 新增補貨檢查函數
def check_and_restock_ingredient(restaurant, mod, dish_name, ingredient_name, oz, 
                                 restaurant_data, current_date, steakhouse_carbon_cache, 
                                 boxed_meal_carbon_cache, ingredient_initial_stock, 
                                 ingredient_usage, today_restocked, 
                                 steakhouse_data, boxed_meal_data, week, day):

    is_steakhouse_main_meat = (restaurant == "牛排館" and mod == "主餐" and oz is not None)
    if is_steakhouse_main_meat:
        key = (ingredient_name, dish_name, oz)
    else:
        key = (ingredient_name, dish_name, None)
    
    initial_stock = ingredient_initial_stock.get(key, 0)
    if initial_stock == 0:
        print(f"警告：{key} 的 initial_stock 為 0，無法補貨")
        return
    
    sheet_name = f"完整{mod}碳排"
    df_sheet = restaurant_data.get(sheet_name, pd.DataFrame())
    
    if is_steakhouse_main_meat:
        target_weight = oz * 28.3495
        lower_bound = target_weight * 0.97
        upper_bound = target_weight * 1.03
        current_stock = len(df_sheet[(df_sheet["食材名稱"] == ingredient_name) &
                                     (df_sheet["菜餚名"] == dish_name) &
                                     (df_sheet["重量"] >= lower_bound) &
                                     (df_sheet["重量"] <= upper_bound) &
                                     (df_sheet["狀態"] != "已選")])
    else:
        current_stock = len(df_sheet[(df_sheet["食材名稱"] == ingredient_name) &
                                     (df_sheet["菜餚名"] == dish_name) &
                                     (df_sheet["狀態"] != "已選")])
    
    # print(f"計算 {restaurant} {mod} {dish_name} 的 {ingredient_name} oz={oz if is_steakhouse_main_meat else '無'}, current_stock={current_stock}, initial_stock={initial_stock}")
    
    if current_stock < 0.3 * initial_stock:
        deficiency = max(0, int(initial_stock - current_stock))
        print(f"補貨觸發: {restaurant} {mod} {dish_name} 的 {ingredient_name} current_stock={current_stock}, 原始庫存={initial_stock}, 缺少 {deficiency} 筆")
        restock_ingredient(ingredient_name, restaurant, dish_name, mod,
                       steakhouse_data, boxed_meal_data, restaurant_data,
                       current_date, steakhouse_carbon_cache, boxed_meal_carbon_cache,
                       restock_count=deficiency, oz_override=oz if is_steakhouse_main_meat else None,
                       ingredient_initial_stock=ingredient_initial_stock, week=week, day=day)
        # 更新 today_restocked
        restock_key = (restaurant, mod, dish_name, ingredient_name, oz if is_steakhouse_main_meat else None)
        today_restocked[restock_key] = current_date

def simulate_weekly_schedule():
    global menu_records
    menu_records = []  # 清空舊記錄
    print("CREat排程中...")

    steakhouse_file = pathlib.Path("牛排館單品食材資料庫.xlsx").resolve()
    boxed_meal_file = pathlib.Path("便當店單品食材資料庫.xlsx").resolve()
    steakhouse_data = load_restaurant_data(steakhouse_file)
    boxed_meal_data = load_restaurant_data(boxed_meal_file)

    # 計算各食材初始庫存量
    # 在 simulate_weekly_schedule 中初始化
    ingredient_initial_stock = {}  # 初始庫存：(ingredient, dish_name, oz 或 None) -> 初始數量
    ingredient_usage = {}  # 使用次數：(ingredient, dish_name, oz 或 None) -> 已使用次數

    # 初始化牛排館庫存
    for sheet, df in steakhouse_data.items():
        for dish_name in df["菜餚名"].unique():
            df_dish = df[df["菜餚名"] == dish_name]
            for ingredient in df_dish["食材名稱"].unique():
                df_ingredient = df_dish[df_dish["食材名稱"] == ingredient]
                if df_ingredient["食物分類子項"].iloc[0] in ["牛肉類", "豬肉類", "羊肉類"]:
                    for oz in [6, 8, 10, 12]:
                        target_weight = oz * 28.3495
                        lower_bound = target_weight * 0.97
                        upper_bound = target_weight * 1.03
                        count = len(df_ingredient[(df_ingredient["重量"] >= lower_bound) &
                                                  (df_ingredient["重量"] <= upper_bound)])
                        ingredient_initial_stock[(ingredient, dish_name, oz)] = count
                        ingredient_usage[(ingredient, dish_name, oz)] = 0
                else:
                    count = df_ingredient.shape[0]
                    ingredient_initial_stock[(ingredient, dish_name, None)] = count
                    ingredient_usage[(ingredient, dish_name, None)] = 0

    # 初始化便當店庫存
    for sheet, df in boxed_meal_data.items():
        for dish_name in df["菜餚名"].unique():
            df_dish = df[df["菜餚名"] == dish_name]
            for ingredient in df_dish["食材名稱"].unique():
                count = df_dish[df_dish["食材名稱"] == ingredient].shape[0]
                ingredient_initial_stock[(ingredient, dish_name, None)] = count
                ingredient_usage[(ingredient, dish_name, None)] = 0

    # 預計算所有食材的基礎碳排和進貨日期
    for sheet_name, df in steakhouse_data.items():
        steakhouse_carbon_cache[sheet_name] = compute_base_storage_carbon(df)
    for sheet_name, df in boxed_meal_data.items():
        boxed_meal_carbon_cache[sheet_name] = compute_base_storage_carbon(df)

    # 第一週的單餐排程（改為 cycle0，第 1-5 週）
    def simulate_cycle0_meal_for_resident(res, habit, steak_choice_df, boxed_meal_choice_df,
                                      steakhouse_data, boxed_meal_data, cooking_data_steak,
                                      cooking_data_boxed_meal, week, current_date, meal_no,
                                      state, steakhouse_carbon_cache, boxed_meal_carbon_cache,
                                      ingredient_usage, ingredient_initial_stock):
        daily_meal = int(habit["daily_meal"])
        day = (meal_no - 1) // daily_meal  # 計算天次（從 0 到 6）
        meal_of_day = (meal_no - 1) % daily_meal  # 計算當天第幾餐（從 0 開始）

        # 根據當前日期決定 date_type
        weekday = current_date.weekday()  # 0=週一 ... 6=週日
        dt = "平日" if weekday < 5 else "假日"

        # 選擇餐廳
        selection = select_restaurant(habit, res, dt, steak_choice_df,
                                    boxed_meal_choice_df, steakhouse_data,
                                    boxed_meal_data, cooking_data_steak,
                                    cooking_data_boxed_meal)
        restaurant, modules_req, restaurant_data, cooking_data, persona_choice = selection.values()
        modules_req = modules_req.copy()  # 確保不修改原始菜餚類型定義之菜餚數量

        # 決定是否吃肉
        total_meals = int(habit["daily_meal"]) * 7
        remaining_meals = total_meals - meal_no + 1
        required_meat = int(habit.get("week_meat_meal", 0))
        meat_meal_count = state["meat_meal_count"]
        prev_eats_meat = state["prev_eats_meat"]
        eats_meat = random.random() < (
            habit["next_meat_pro"] if prev_eats_meat else habit["meat_pro"]
        ) if meat_meal_count + remaining_meals != required_meat else True
        if eats_meat:
            state["meat_meal_count"] += 1
        state["prev_eats_meat"] = eats_meat

        meal_emission = 0
        meal_details = {}

        for mod, req in modules_req.items():
            # 主餐若不吃肉，直接設定為「從缺」
            if mod == "主餐" and restaurant in ["牛排館", "便當店"] and not eats_meat:
                meal_details[mod] = [{"菜餚名": "從缺", "final_carbon": 0}] * req
                continue

            sheet_name = "完整" + mod + "碳排"
            df_origin = restaurant_data.get(sheet_name, pd.DataFrame())

            module_df, state["prev_rice"] = filter_available_food(df_origin, restaurant, mod, persona_choice, state["prev_rice"])
            module_df = exclude_unwanted_food(module_df, habit)

            avail_df = pd.DataFrame()
            if not module_df.empty:
                avail_df = module_df[module_df["狀態"] != "已選"]   # 僅選擇沒有標已選的食材
            if avail_df.empty:
                print(f"[除錯] mod '{mod}' 的 avail_df 為空，無可用食材")
                meal_details[mod] = [{"菜餚名": "找不到食材", "final_carbon": 0}] * req
                continue

            # 從最舊五筆中選擇菜餚
            selected_dishes, mod_carbon = select_module_meal(
                mod, req, avail_df, persona_choice, restaurant, restaurant_data,
                cooking_data, current_date, ingredient_usage, ingredient_initial_stock,
                steakhouse_carbon_cache if restaurant == "牛排館" else boxed_meal_carbon_cache,
                prev_meal_exceeded=False, use_greedy=False
            )

            # 更新狀態與記錄
            for dish_info in selected_dishes:
                rep_dish_name = dish_info["菜餚名"]
                final_carbon = dish_info["final_carbon"]
                if rep_dish_name == "找不到食材":
                    # 找不到食材時，直接記錄
                    year = (week - 1) // 52 + 1
                    week_in_year = (week - 1) % 52 + 1
                    menu_records.append({
                        "姓名": res,
                        "年次": year,
                        "週次": week_in_year,
                        "天次": day + 1,
                        "餐次": meal_of_day + 1,
                        "餐廳": restaurant,
                        "菜餚類型": mod,
                        "菜餚名": rep_dish_name,
                        "食物分類子項": "",
                        "oz": "",  # 找不到食材時，oz 為空
                        "final_carbon": final_carbon
                    })
                    continue

                # 處理食材選擇和狀態更新
                dish_groups = avail_df[avail_df["菜餚名"] == rep_dish_name].groupby("食材名稱")
                selected_ingredients = []  # 用於記錄該菜餚的所有食材資訊
                for food_name, group in dish_groups:
                    filtered_records = select_food(group, persona_choice)
                    if filtered_records.empty:
                        continue
                    selected_record = filtered_records.sample(n=1, random_state=random.randint(1, 10000)).iloc[0]
                    selected_index = selected_record.name
                    restaurant_data[sheet_name].loc[selected_index, "狀態"] = "已選"

                    # 更新食材使用次數
                    ingredient_name = selected_record["食材名稱"]
                    dish_name = rep_dish_name
                    is_steakhouse_main_meat = (restaurant == "牛排館" and 
                                            mod == "主餐" and 
                                            selected_record["食物分類子項"] in ["牛肉類", "豬肉類", "羊肉類"])
                    oz = None
                    if is_steakhouse_main_meat:
                        oz = persona_choice.get("oz", 0)
                        if oz not in [6, 8, 10, 12]:
                            oz = random.choice([6, 8, 10, 12])
                            print(f"警告：{ingredient_name} 的 oz 值 {persona_choice.get('oz')} 無效，使用隨機值 {oz}")
                        key = (ingredient_name, dish_name, oz)
                    else:
                        key = (ingredient_name, dish_name, None)
                    ingredient_usage[key] = ingredient_usage.get(key, 0) + 1

                    # 檢查是否需要補貨
                    current_ingredients = set(avail_df["食材名稱"].unique())
                    if ingredient_name in current_ingredients and dish_name == rep_dish_name:
                        restock_key = (restaurant, mod, rep_dish_name, ingredient_name, oz if is_steakhouse_main_meat else None)
                        if today_restocked.get(restock_key) != current_date:
                            check_and_restock_ingredient(
                                restaurant, mod, rep_dish_name, ingredient_name, 
                                oz if is_steakhouse_main_meat else None,
                                restaurant_data, current_date, steakhouse_carbon_cache, 
                                boxed_meal_carbon_cache, ingredient_initial_stock, 
                                ingredient_usage, today_restocked,
                                steakhouse_data, boxed_meal_data, week, day
                            )

                    # 收集食材資訊
                    selected_ingredients.append({
                        "food_name": food_name,
                        "food_category": selected_record["食物分類子項"],
                        "oz": oz if is_steakhouse_main_meat else None  # 只在牛排館主餐時記錄 oz
                    })

                # 在處理完所有食材後，記錄一次該菜餚
                if selected_ingredients:  # 確保有選擇的食材
                    # 根據模組類型選擇代表性的食物分類子項
                    if mod == "主食":
                        # 優先選擇「全穀根莖類」
                        representative_food_category = next(
                            (ing["food_category"] for ing in selected_ingredients if ing["food_category"] == "全穀根莖類"),
                            selected_ingredients[0]["food_category"]  # 若無則用第一個食材的類別
                        )
                    else:
                        # 優先選擇「豆魚蛋肉類」，若無則用第一個食材的類別
                        representative_food_category = next(
                            (ing["food_category"] for ing in selected_ingredients if ing["food_category"] in ["豆類", "魚類", "蛋類", "肉類", "牛肉類", "豬肉類", "羊肉類"]),
                            selected_ingredients[0]["food_category"]
                        )
                    # 只有在牛排館主餐時才設置 oz 值，其他情況為空
                    oz_value = selected_ingredients[0]["oz"] if (restaurant == "牛排館" and mod == "主餐") else ""

                    # 計算年次
                    year = (week - 1) // 52 + 1
                    week_in_year = (week - 1) % 52 + 1

                    # 記錄該菜餚（只記錄一次）
                    menu_records.append({
                        "姓名": res,
                        "年次": year,
                        "週次": week_in_year,
                        "天次": day + 1,
                        "餐次": meal_of_day + 1,
                        "餐廳": restaurant,
                        "菜餚類型": mod,
                        "菜餚名": rep_dish_name,
                        "食物分類子項": representative_food_category,
                        "oz": oz_value,
                        "final_carbon": final_carbon
                    })

            meal_details[mod] = selected_dishes
            meal_emission += mod_carbon

        return meal_emission

    # 後續週的單餐排程（改為 cycle，第 6 週開始）
    def simulate_cycle_meal_for_resident(res, week_target, habit, steak_choice_df,
                                     boxed_meal_choice_df, steakhouse_data, boxed_meal_data,
                                     cooking_data_steak, cooking_data_boxed_meal, week,
                                     current_date, meal_idx, simulation_state, reserved_carbon,
                                     prev_week_exceeded, steakhouse_carbon_cache,
                                     boxed_meal_carbon_cache, ingredient_usage,
                                     ingredient_initial_stock):
        daily_meal = int(habit["daily_meal"])
        num_meals = daily_meal * 7
        meal_target = week_target / num_meals  # 初始每餐目標碳排

        # 從 meal_idx 反推 day 和 meal_no 用於打印和記錄
        day = (meal_idx - 1) // daily_meal  # 計算天次（從 0 到 6）
        meal_no = (meal_idx - 1) % daily_meal  # 計算當天第幾餐（從 0 開始）

        # 獲取居民狀態
        state = simulation_state[res]
        accumulated_excess = state["accumulated_excess"]
        prev_meal_excess = state["prev_meal_excess"]

        # 根據當前日期決定 date_type
        weekday = current_date.weekday()
        dt = "平日" if weekday < 5 else "假日"

        # 選擇餐廳
        selection = select_restaurant(habit, res, dt, steak_choice_df,
                                    boxed_meal_choice_df, steakhouse_data,
                                    boxed_meal_data, cooking_data_steak,
                                    cooking_data_boxed_meal)
        restaurant, modules_req, restaurant_data, cooking_data, persona_choice = selection.values()
        modules_req = modules_req.copy()  # 確保不修改原始菜餚類型定義之菜餚數量

        # 決定是否吃肉
        remaining_meals = num_meals - meal_idx
        required_meat = int(habit.get("week_meat_meal", 0))
        if state["meat_meal_count"] + remaining_meals == required_meat:
            eats_meat = True
        else:
            prob = habit["next_meat_pro"] if state["prev_eats_meat"] else habit["meat_pro"]
            eats_meat = random.random() < prob
        if eats_meat:
            state["meat_meal_count"] += 1
        state["prev_eats_meat"] = eats_meat

        # 動態調整本餐目標碳排
        adjusted_meal_target = meal_target - (accumulated_excess / max(remaining_meals, 1))
        if adjusted_meal_target < 0:
            adjusted_meal_target = 0

        meal_emission = 0
        meal_details = {}

        # 先選擇主餐
        if "主餐" in modules_req:
            mod = "主餐"
            req = modules_req[mod]
            if not eats_meat and restaurant in ["牛排館", "便當店"]:
                meal_details[mod] = [{"菜餚名": "從缺", "final_carbon": 0}] * req
            else:
                sheet_name = "完整" + mod + "碳排"
                df_origin = restaurant_data.get(sheet_name, pd.DataFrame())
                module_df, state["prev_rice"] = filter_available_food(df_origin, restaurant, mod, persona_choice, state["prev_rice"])
                module_df = exclude_unwanted_food(module_df, habit)
                avail_df = module_df[module_df["狀態"] != "已選"]
                if avail_df.empty:
                    meal_details[mod] = [{"菜餚名": "找不到食材", "final_carbon": 0}] * req
                else:
                    # 主餐根據persona機率從最舊五筆中選擇
                    selected_dishes, main_carbon = select_module_meal(
                        mod, req, avail_df, persona_choice, restaurant, restaurant_data,
                        cooking_data, current_date, ingredient_usage, ingredient_initial_stock,
                        steakhouse_carbon_cache if restaurant == "牛排館" else boxed_meal_carbon_cache
                    )
                    meal_details[mod] = selected_dishes
                    meal_emission += main_carbon

                    # 更新狀態與記錄
                    for dish_info in selected_dishes:
                        rep_dish_name = dish_info["菜餚名"]
                        final_carbon = dish_info["final_carbon"]
                        if rep_dish_name == "找不到食材":
                            # 找不到食材時，直接記錄
                            year = (week - 1) // 52 + 1
                            week_in_year = (week - 1) % 52 + 1
                            menu_records.append({
                                "姓名": res,
                                "年次": year,
                                "週次": week_in_year,
                                "天次": day + 1,
                                "餐次": meal_no + 1,
                                "餐廳": restaurant,
                                "菜餚類型": mod,
                                "菜餚名": rep_dish_name,
                                "食物分類子項": "",
                                "oz": "",  # 找不到食材時，oz 為空
                                "final_carbon": final_carbon
                            })
                            continue

                        # 處理食材選擇和狀態更新
                        dish_groups = avail_df[avail_df["菜餚名"] == rep_dish_name].groupby("食材名稱")
                        selected_ingredients = []  # 收集該菜餚的所有食材資訊
                        for food_name, group in dish_groups:
                            filtered_records = select_food(group, persona_choice)
                            if filtered_records.empty:
                                continue
                            selected_record = filtered_records.sample(n=1, random_state=random.randint(1, 10000)).iloc[0]
                            selected_index = selected_record.name
                            restaurant_data[sheet_name].loc[selected_index, "狀態"] = "已選"

                            # 更新食材使用次數
                            ingredient_name = selected_record["食材名稱"]
                            dish_name = rep_dish_name
                            is_steakhouse_main_meat = (restaurant == "牛排館" and 
                                                    mod == "主餐" and 
                                                    selected_record["食物分類子項"] in ["牛肉類", "豬肉類", "羊肉類"])
                            oz = None
                            if is_steakhouse_main_meat:
                                oz = persona_choice.get("oz", 0)
                                if oz not in [6, 8, 10, 12]:
                                    oz = random.choice([6, 8, 10, 12])
                                    print(f"警告：{ingredient_name} 的 oz 值 {persona_choice.get('oz')} 無效，使用隨機值 {oz}")
                                key = (ingredient_name, dish_name, oz)
                            else:
                                key = (ingredient_name, dish_name, None)
                            ingredient_usage[key] = ingredient_usage.get(key, 0) + 1

                            # 檢查是否需要補貨
                            current_ingredients = set(avail_df["食材名稱"].unique())
                            if ingredient_name in current_ingredients and dish_name == rep_dish_name:
                                restock_key = (restaurant, mod, rep_dish_name, ingredient_name, oz if is_steakhouse_main_meat else None)
                                if today_restocked.get(restock_key) != current_date:
                                    day = (meal_idx - 1) // int(habit["daily_meal"])  # 計算當前天數（從 0 到 6）
                                    check_and_restock_ingredient(
                                        restaurant, mod, rep_dish_name, ingredient_name, 
                                        oz if is_steakhouse_main_meat else None,
                                        restaurant_data, current_date, steakhouse_carbon_cache, 
                                        boxed_meal_carbon_cache, ingredient_initial_stock, 
                                        ingredient_usage, today_restocked,
                                        steakhouse_data, boxed_meal_data, week, day
                                    )

                            # 收集食材資訊
                            selected_ingredients.append({
                                "food_name": food_name,
                                "food_category": selected_record["食物分類子項"],
                                "oz": oz if is_steakhouse_main_meat else None  # 只在牛排館主餐時記錄 oz
                            })

                        # 在處理完所有食材後，記錄一次該菜餚
                        if selected_ingredients:  # 確保有選擇的食材
                            # 主餐優先選擇「豆魚蛋肉類」
                            representative_food_category = next(
                                (ing["food_category"] for ing in selected_ingredients if ing["food_category"] in ["豆類", "魚類", "蛋類", "肉類", "牛肉類", "豬肉類", "羊肉類"]),
                                selected_ingredients[0]["food_category"]
                            )
                            oz_value = selected_ingredients[0]["oz"] if (restaurant == "牛排館" and mod == "主餐") else ""
                            year = (week - 1) // 52 + 1
                            week_in_year = (week - 1) % 52 + 1
                            menu_records.append({
                                "姓名": res,
                                "年次": year,
                                "週次": week_in_year,
                                "天次": day + 1,
                                "餐次": meal_no + 1,
                                "餐廳": restaurant,
                                "菜餚類型": mod,
                                "菜餚名": rep_dish_name,
                                "食物分類子項": representative_food_category,
                                "oz": oz_value,
                                "final_carbon": final_carbon
                            })

        # 其餘菜餚類型的目標碳排
        remaining_target = adjusted_meal_target - meal_emission
        remaining_modules = {k: v for k, v in modules_req.items() if k != "主餐"}
        total_remaining_req = sum(remaining_modules.values())

        # 定義 Excel 檔案路徑
        excel_files = {
            "牛排館": pathlib.Path("牛排館單品食材資料庫.xlsx").resolve(),
            "便當店": pathlib.Path("便當店單品食材資料庫.xlsx").resolve()
        }

        # 選擇菜餚的邏輯
        for mod, req in modules_req.items():
            sheet_name = "完整" + mod + "碳排"
            df_origin = restaurant_data.get(sheet_name, pd.DataFrame())
            module_df, state["prev_rice"] = filter_available_food(df_origin, restaurant, mod, persona_choice, state["prev_rice"])
            module_df = exclude_unwanted_food(module_df, habit)
            avail_df = module_df[module_df["狀態"] != "已選"]

            # 若不吃肉且為主餐，直接從缺
            if mod == "主餐" and restaurant in ["牛排館", "便當店"] and not eats_meat:
                meal_details[mod] = [{"菜餚名": "從缺", "final_carbon": 0}] * req
                continue

            # 若 avail_df 為空，更新庫存並補貨
            if avail_df.empty:
                print(f"[警告] {restaurant} 的 {mod} 模組無可用食材，更新庫存並補貨")
                
                # 步驟 1：讀取 Excel 檔案更新庫存記憶
                excel_file = excel_files.get(restaurant)
                try:
                    if restaurant == "牛排館":
                        steakhouse_data = load_restaurant_data(excel_file)
                        restaurant_data = steakhouse_data
                    elif restaurant == "便當店":
                        boxed_meal_data = load_restaurant_data(excel_file)
                        restaurant_data = boxed_meal_data
                    print(f"從 {excel_file} 更新 {restaurant} 的庫存資料")
                except Exception as e:
                    print(f"讀取 Excel 檔案失敗: {excel_file}, {e}")
                    meal_details[mod] = [{"菜餚名": "找不到食材", "final_carbon": 0}] * req
                    continue

                # 重新獲取更新後的 df_origin
                df_origin = restaurant_data.get(sheet_name, pd.DataFrame())
                if df_origin.empty:
                    print(f"[錯誤] 更新後 {sheet_name} 仍為空")
                    meal_details[mod] = [{"菜餚名": "找不到食材", "final_carbon": 0}] * req
                    continue

                # 步驟 2：檢查並補充低於模組需求的食材
                for dish_name in df_origin["菜餚名"].unique():
                    df_dish = df_origin[df_origin["菜餚名"] == dish_name]
                    for ingredient_name in df_dish["食材名稱"].unique():
                        # 計算當前可用庫存
                        current_stock = len(df_dish[(df_dish["食材名稱"] == ingredient_name) & 
                                                (df_dish["狀態"] != "已選")])
                        initial_stock = ingredient_initial_stock.get((ingredient_name, dish_name, None), 0)

                        # 若庫存低於模組需求 (req)，則補貨
                        if current_stock < req:
                            print(f"補貨 {restaurant} {mod} 的 {dish_name} - {ingredient_name}，當前庫存 {current_stock} < 需求 {req}")
                            is_steakhouse_main_meat = (restaurant == "牛排館" and 
                                                    mod == "主餐" and 
                                                    df_dish["食物分類子項"].iloc[0] in ["牛肉類", "豬肉類", "羊肉類"])
                            oz = persona_choice.get("oz", None) if is_steakhouse_main_meat else None
                            if oz and oz not in [6, 8, 10, 12]:
                                oz = random.choice([6, 8, 10, 12])
                                print(f"警告：{ingredient_name} 的 oz 值 {persona_choice.get('oz')} 無效，使用隨機值 {oz}")

                            day = (meal_idx - 1) // int(habit["daily_meal"])
                            check_and_restock_ingredient(
                                restaurant=restaurant,
                                mod=mod,
                                dish_name=dish_name,
                                ingredient_name=ingredient_name,
                                oz=oz,
                                restaurant_data=restaurant_data,
                                current_date=current_date,
                                steakhouse_carbon_cache=steakhouse_carbon_cache,
                                boxed_meal_carbon_cache=boxed_meal_carbon_cache,
                                ingredient_initial_stock=ingredient_initial_stock,
                                ingredient_usage=ingredient_usage,
                                today_restocked=today_restocked,
                                steakhouse_data=steakhouse_data,
                                boxed_meal_data=boxed_meal_data,
                                week=week,
                                day=day
                            )

                # 補貨後重新生成 avail_df
                df_origin = restaurant_data.get(sheet_name, pd.DataFrame())
                module_df, state["prev_rice"] = filter_available_food(df_origin, restaurant, mod, persona_choice, state["prev_rice"])
                module_df = exclude_unwanted_food(module_df, habit)
                avail_df = module_df[module_df["狀態"] != "已選"]

                # 若補貨後仍無可用食材，則從缺
                if avail_df.empty:
                    print(f"[錯誤] {restaurant} 的 {mod} 模組補貨後仍無可用食材")
                    meal_details[mod] = [{"菜餚名": "找不到食材", "final_carbon": 0}] * req
                    continue

            # 從最舊五筆中選擇，檢查上一餐是否超標
            module_target = remaining_target * (req / total_remaining_req) if total_remaining_req > 0 else 0
            selected_dishes, mod_carbon = select_module_meal(
                mod, req, avail_df, persona_choice, restaurant, restaurant_data,
                cooking_data, current_date, ingredient_usage, ingredient_initial_stock,
                steakhouse_carbon_cache if restaurant == "牛排館" else boxed_meal_carbon_cache,
                prev_meal_exceeded=(state["prev_meal_excess"] > 0), use_greedy=True
            )
            # 在全局範圍內設置 module_target 以供 select_module_meal 使用
            globals()['module_target'] = module_target

            # 更新狀態與記錄
            for dish_info in selected_dishes:
                rep_dish_name = dish_info["菜餚名"]
                final_carbon = dish_info["final_carbon"]
                if rep_dish_name == "找不到食材":
                    # 找不到食材時，直接記錄
                    year = (week - 1) // 52 + 1
                    week_in_year = (week - 1) % 52 + 1
                    menu_records.append({
                        "姓名": res,
                        "年次": year,
                        "週次": week_in_year,
                        "天次": day + 1,
                        "餐次": meal_no + 1,
                        "餐廳": restaurant,
                        "菜餚類型": mod,
                        "菜餚名": rep_dish_name,
                        "食物分類子項": "",
                        "oz": "",  # 找不到食材時，oz 為空
                        "final_carbon": final_carbon
                    })
                    continue

                # 處理食材選擇和狀態更新
                dish_groups = avail_df[avail_df["菜餚名"] == rep_dish_name].groupby("食材名稱")
                selected_ingredients = []  # 收集該菜餚的所有食材資訊
                for food_name, group in dish_groups:
                    filtered_records = select_food(group, persona_choice)
                    if filtered_records.empty:
                        continue
                    selected_record = filtered_records.sample(n=1, random_state=random.randint(1, 10000)).iloc[0]
                    selected_index = selected_record.name
                    restaurant_data[sheet_name].loc[selected_index, "狀態"] = "已選"

                    # 更新食材使用次數
                    ingredient_name = selected_record["食材名稱"]
                    dish_name = rep_dish_name
                    is_steakhouse_main_meat = (restaurant == "牛排館" and 
                                            mod == "主餐" and 
                                            selected_record["食物分類子項"] in ["牛肉類", "豬肉類", "羊肉類"])
                    oz = None
                    if is_steakhouse_main_meat:
                        oz = persona_choice.get("oz", 0)
                        if oz not in [6, 8, 10, 12]:
                            oz = random.choice([6, 8, 10, 12])
                            print(f"警告：{ingredient_name} 的 oz 值 {persona_choice.get('oz')} 無效，使用隨機值 {oz}")
                        key = (ingredient_name, dish_name, oz)
                    else:
                        key = (ingredient_name, dish_name, None)
                    ingredient_usage[key] = ingredient_usage.get(key, 0) + 1

                    # 檢查是否需要補貨
                    current_ingredients = set(avail_df["食材名稱"].unique())
                    if ingredient_name in current_ingredients and dish_name == rep_dish_name:
                        restock_key = (restaurant, mod, rep_dish_name, ingredient_name, oz if is_steakhouse_main_meat else None)
                        if today_restocked.get(restock_key) != current_date:
                            check_and_restock_ingredient(
                                restaurant, mod, rep_dish_name, ingredient_name, 
                                oz if is_steakhouse_main_meat else None,
                                restaurant_data, current_date, steakhouse_carbon_cache, 
                                boxed_meal_carbon_cache, ingredient_initial_stock, 
                                ingredient_usage, today_restocked,
                                steakhouse_data, boxed_meal_data, week, day
                            )

                    # 收集食材資訊
                    selected_ingredients.append({
                        "food_name": food_name,
                        "food_category": selected_record["食物分類子項"],
                        "oz": oz if is_steakhouse_main_meat else None  # 只在牛排館主餐時記錄 oz
                    })

                # 在處理完所有食材後，記錄一次該菜餚
                if selected_ingredients:  # 確保有選擇的食材
                    # 根據模組類型選擇代表性的食物分類子項
                    if mod == "主食":
                        # 優先選擇「全穀根莖類」
                        representative_food_category = next(
                            (ing["food_category"] for ing in selected_ingredients if ing["food_category"] == "全穀根莖類"),
                            selected_ingredients[0]["food_category"]
                        )
                    else:
                        # 優先選擇「豆魚蛋肉類」，若無則用第一個食材的類別
                        representative_food_category = next(
                            (ing["food_category"] for ing in selected_ingredients if ing["food_category"] in ["豆類", "魚類", "蛋類", "肉類", "牛肉類", "豬肉類", "羊肉類"]),
                            selected_ingredients[0]["food_category"]
                        )
                    oz_value = selected_ingredients[0]["oz"] if (restaurant == "牛排館" and mod == "主餐") else ""
                    year = (week - 1) // 52 + 1
                    week_in_year = (week - 1) % 52 + 1
                    menu_records.append({
                        "姓名": res,
                        "年次": year,
                        "週次": week_in_year,
                        "天次": day + 1,
                        "餐次": meal_no + 1,
                        "餐廳": restaurant,
                        "菜餚類型": mod,
                        "菜餚名": rep_dish_name,
                        "食物分類子項": representative_food_category,
                        "oz": oz_value,
                        "final_carbon": final_carbon
                    })

            meal_details[mod] = selected_dishes
            meal_emission += mod_carbon

        # 更新超標狀態
        excess = meal_emission - adjusted_meal_target
        if excess > 2000 and reserved_carbon[res] > 0:
            release = min(excess, reserved_carbon[res])
            reserved_carbon[res] -= release
            print(f"{res} 第{week}週第 {day+1} 天第 {meal_no+1} 餐釋放預儲碳排：{round(release, 2)} gCO2e")
        elif excess > 0:
            accumulated_excess += excess

        state["accumulated_excess"] = accumulated_excess
        state["prev_meal_excess"] = excess
        return meal_emission

    # 模擬多週排程（改為按餐排程）
    def simulate_multi_week_schedule(steakhouse_data, boxed_meal_data, steakhouse_file, boxed_meal_file,
                                 ingredient_usage, ingredient_initial_stock):
        print("開始多週排程模擬...")
        persona = load_persona(pathlib.Path("persona.xlsx"))
        habit_df = persona["habit"]
        steak_choice_df = persona["steak"]
        boxed_meal_choice_df = persona["boxed_meal"]
        residents = habit_df["name"].tolist()

        cooking_data_steak = load_cooking_data(pathlib.Path("烹飪碳排.xlsx"), "牛排館")
        cooking_data_boxed_meal = load_cooking_data(pathlib.Path("烹飪碳排.xlsx"), "便當店")

        # 每天開始前清空當天的補貨記錄
        global today_restocked
        if 'today_restocked' not in globals() or today_restocked is None:
            today_restocked = {}

        simulation_state = {}
        reserved_carbon = {}
        prev_week_exceeded = {}
        week_totals = {}
        for res in residents:
            habit = habit_df[habit_df["name"] == res].iloc[0].to_dict()
            simulation_state[res] = {
                "habit": habit,
                "baseline": None,
                "rollover": 0,
                "rollover_weeks": 0,
                "finished": False,
                "meat_meal_count": 0,
                "prev_eats_meat": False,
                "prev_beef": False,
                "prev_rice": None,
                "accumulated_excess": 0,
                "prev_meal_excess": 0,
                "week_total": 0,
                "cycle0_total": 0  # 新增：用於記錄 cycle0 的總碳排
            }
            reserved_carbon[res] = 0
            prev_week_exceeded[res] = False
            week_totals[res] = 0

        week = 1
        start_date_week = datetime.date.today()
        active = True
        max_daily_meals = max(int(habit_df["daily_meal"].max()), 1)

        while active:  # 移除 week <= max_weeks 條件，無限排程
            print(f"=== 模擬第 {week} 週 ===")
            # 每週開始時清空當前週的使用記錄
            ingredient_usage.clear()
            weekly_targets = {}
            active = False
            weekly_targets = {}

            for res in residents:
                if simulation_state[res]["finished"]:
                    continue
                active = True
                habit = simulation_state[res]["habit"]
                if week <= 5:  # cycle0（第1-5週）
                    weekly_targets[res] = None
                else:
                    baseline = simulation_state[res]["baseline"]
                    rollover = simulation_state[res]["rollover"]
                    cycle = (week - 1) // 5  # 第1-5週 cycle=0，第6-10週 cycle=1，第11-15週 cycle=2
                    T = baseline * (0.9 ** cycle) + rollover  # 5週總和目標
                    r = 0.95  # 公比，可根據需求調整
                    denominator = 1 + r + r**2 + r**3 + r**4
                    C = T / denominator
                    week_in_cycle = (week - 1) % 5  # 週期內第幾週（0到4）
                    weekly_targets[res] = C * (r ** week_in_cycle)  # 每週目標

            for day in range(7):
                current_date = start_date_week + datetime.timedelta(days=day)
                # 每天開始前清空當天的補貨記錄
                today_restocked.clear()
                for cache in [steakhouse_carbon_cache, boxed_meal_carbon_cache]:
                    for sheet_cache in cache.values():
                        update_dynamic_carbon_cache(sheet_cache, current_date)

                for meal_no in range(max_daily_meals):
                    for res in residents:
                        if simulation_state[res]["finished"]:
                            continue
                        habit = simulation_state[res]["habit"]
                        daily_meal = int(habit["daily_meal"])
                        if meal_no >= daily_meal:
                            continue

                        total_meal_idx = (day * daily_meal + meal_no) + 1
                        print(f"為 {res} 排第 {week} 週第 {day+1} 天第 {meal_no+1} 餐")  # 新增除錯

                        if week <= 5:  # cycle0（第 1-5 週）
                            meal_emission = simulate_cycle0_meal_for_resident(
                                res, habit, steak_choice_df, boxed_meal_choice_df,
                                steakhouse_data, boxed_meal_data, cooking_data_steak,
                                cooking_data_boxed_meal, week, current_date, total_meal_idx,
                                simulation_state[res], steakhouse_carbon_cache,
                                boxed_meal_carbon_cache, ingredient_usage, ingredient_initial_stock
                            )
                            simulation_state[res]["week_total"] += meal_emission
                            simulation_state[res]["cycle0_total"] += meal_emission
                        else:
                            meal_emission = simulate_cycle_meal_for_resident(
                                res, weekly_targets[res], habit, steak_choice_df,
                                boxed_meal_choice_df, steakhouse_data, boxed_meal_data,
                                cooking_data_steak, cooking_data_boxed_meal, week,
                                current_date, total_meal_idx, simulation_state,
                                reserved_carbon, prev_week_exceeded, steakhouse_carbon_cache,
                                boxed_meal_carbon_cache, ingredient_usage, ingredient_initial_stock
                            )
                            simulation_state[res]["week_total"] += meal_emission

            for res in residents:
                if simulation_state[res]["finished"]:
                    continue
                week_total = simulation_state[res]["week_total"]
                if week == 5:  # 在第 5 週結束時設定 baseline
                    simulation_state[res]["baseline"] = simulation_state[res]["cycle0_total"]
                    reserved_carbon[res] = 0
                    print(f"{res} 第1-5週總碳排（基準）：{round(simulation_state[res]['cycle0_total'], 2)} gCO2e 預儲碳排：{round(reserved_carbon[res], 2)} gCO2e")
                elif week > 5:
                    week_target = weekly_targets[res]
                    status = "符合碳排限制" if week_total <= week_target else "未符合碳排限制"
                    print(f"{res} 第{week}週目標碳排：{round(week_target, 2)} gCO2e  第{week}週總碳排：{round(week_total, 2)} gCO2e {status} 預儲碳排：{round(reserved_carbon[res], 2)} gCO2e")
                    prev_week_exceeded[res] = week_total > week_target

                    # 單人碳排動態調整
                    if week_total < week_target:
                        excess_carbon = week_target - week_total
                        reserved_carbon[res] += excess_carbon
                        if reserved_carbon[res] > 90000:
                            spillover = reserved_carbon[res] - 90000  # 計算超出的碳空間
                            reserved_carbon[res] = 90000
                            if not prev_week_exceeded[res]:
                                simulation_state[res]["rollover"] = spillover / 7  # 設置7週分攤消耗
                                simulation_state[res]["rollover_weeks"] = 7
                            else:
                                simulation_state[res]["rollover"] = 0
                                simulation_state[res]["rollover_weeks"] = 0
                    elif week_total > week_target + 15000 and reserved_carbon[res] > 0:   # 週結算超出目標15,000，釋放預儲
                        release = min(week_total - week_target, reserved_carbon[res])
                        reserved_carbon[res] -= release
                        print(f"{res} 第{week}週超出目標釋放預儲碳排：{round(release, 2)} gCO2e")
                        # 若預儲碳排降至90,000以下，重置 rollover，後續不能加上去
                        if reserved_carbon[res] < 90000:
                            simulation_state[res]["rollover"] = 0
                            simulation_state[res]["rollover_weeks"] = 0

                week_totals[res] = week_total
                simulation_state[res]["week_total"] = 0
            
            # if week % 5 == 0:  # 每五週寫入一次 Excel
            if week > 5:  # 從第 6 週開始，每週更新
                save_selection_results(steakhouse_data, boxed_meal_data, steakhouse_file, boxed_meal_file)
                print(f"第 {week} 週：已將餐廳資料更新至 Excel 檔案")

            # if (week > 5) and (week % 5 == 0):  # 每五週檢查一次
            if (week > 5):  # 改為五週以後每一週檢查一次
                reserved_carbon = adjust_village_emissions(week_totals, weekly_targets, reserved_carbon)
                print("全村調整後各村民的預儲碳排：")
                for res in residents:
                    print(f"{res} 預儲碳排：{round(reserved_carbon[res], 2)} gCO2e")

                total_village_emission = round(sum(week_totals[res] for res in week_totals if not simulation_state[res]["finished"]), 2)
                total_village_target = round(sum(weekly_targets[res] for res in weekly_targets if not simulation_state[res]["finished"]), 2)
                if total_village_emission > total_village_target + 10000 and all(reserved_carbon[res] == 0 for res in reserved_carbon if not simulation_state[res]["finished"]):
                    print(f"全村第 {week} 週總碳排 {total_village_emission:.2f} gCO2e 超出全村目標 {total_village_target:.2f} gCO2e 超過 10,000 gCO2e，且所有村民的預儲碳排皆為0，故全村停止排程")
                    
                    # 在停止條件觸發時，先進行全村調整
                    reserved_carbon = adjust_village_emissions(week_totals, weekly_targets, reserved_carbon)
                    print("因全村超出碳排限制，再次調整後各村民的預儲碳排：")
                    for res in residents:
                        print(f"{res} 預儲碳排：{round(reserved_carbon[res], 2)} gCO2e")
                    
                    # 更新 Excel
                    save_selection_results(steakhouse_data, boxed_meal_data, steakhouse_file, boxed_meal_file)
                    print(f"停止排程：已將餐廳資料更新至 Excel 檔案")
                    active = False
                    break  # 跳出迴圈，避免繼續執行

            week += 1
            start_date_week += datetime.timedelta(days=7)

        # 模擬結束後的最後一次更新與匯出
        save_selection_results(steakhouse_data, boxed_meal_data, steakhouse_file, boxed_meal_file)
        print("模擬結束：已將最終餐廳資料更新至 Excel 檔案")

        # 統一匯出 villager_menu.xlsx
        menu_df = pd.DataFrame(menu_records)
        if not menu_df.empty:
            # 確保唯一性（雖然理論上已經在生成時處理好）
            menu_summary = menu_df.drop_duplicates(
                subset=["姓名", "年次", "週次", "天次", "餐次", "餐廳", "菜餚類型", "菜餚名"]
            )
            # 選擇輸出的欄位
            menu_summary = menu_summary[["姓名", "年次", "週次", "天次", "餐次", "餐廳", "菜餚類型", "菜餚名", "食物分類子項", "oz", "final_carbon"]]
            
            output_file = pathlib.Path("villager_menu.xlsx").resolve()
            try:
                menu_summary.to_excel(output_file, index=False)
                print(f"模擬結束：村民菜單已匯出至 {output_file}")
            except PermissionError:
                print(f"錯誤：無法寫入 {output_file}，檔案可能被鎖定")
            except Exception as e:
                print(f"錯誤：匯出失敗，原因：{e}")
        else:
            print("[DEBUG] menu_records 為空，無法生成菜單")

        return steakhouse_data, boxed_meal_data

    steakhouse_data, boxed_meal_data = simulate_multi_week_schedule(
        steakhouse_data, boxed_meal_data, steakhouse_file, boxed_meal_file,
        ingredient_usage, ingredient_initial_stock
    )
    print("所有餐廳資料已更新到 Excel 檔案。")

if __name__ == "__main__":
    simulate_weekly_schedule()