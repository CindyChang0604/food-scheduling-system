import pandas
import random
import typing
import pathlib
import datetime
import dataclasses

@dataclasses.dataclass
class FoodMatch:
    code: str
    name: str
    category: str
    is_raw: bool
    cooking_method: str
    storage_method: str
    carbon_values: typing.Dict[str, float]

class FoodCarbonCalculator:
    # CSV 中碳排放相關欄位
    CARBON_COLUMNS = ["農業", "加工", "包裝", "運輸", "超市及配送", "能源消耗", "全部"]
    # 移除全穀雜糧類預設，由 infer_storage_method 處理
    STORAGE_RULES = {"乳品類": "冷藏", "水果類": "冷藏"}
    ORIGINS = ["台灣", "日本", "澳洲", "美國"]

    def __init__(self, ingredient_db_path: pathlib.Path):
        self.ingredient_db = self._load_ingredient_database(ingredient_db_path)
        self.purchase_dates = self._generate_purchase_dates()
        self.today = datetime.datetime.now().date()  # 僅計算一次
        # 注意：menu_file_name 由 process_menu 設定

    def _load_ingredient_database(self, path: pathlib.Path) -> pandas.DataFrame:
        df = pandas.read_csv(path)
        for col in FoodCarbonCalculator.CARBON_COLUMNS:
            df[col] = pandas.to_numeric(df[col].replace("-", 0), errors="coerce").fillna(0)
        return df

    def _generate_purchase_dates(self) -> typing.List[str]:
        today = datetime.datetime.now()
        start_date = today - datetime.timedelta(days=14)
        dates = []
        for _ in range(5):
            dt = start_date + datetime.timedelta(days=random.randint(0, 13))
            if dt <= today:
                dates.append(dt.strftime("%Y-%m-%d"))
        return dates

    def select_best_matches(self, food_name: str) -> typing.Optional[pandas.DataFrame]:
        matches = self.ingredient_db[self.ingredient_db["食材名稱"] == food_name]
        if matches.empty:
            return None
        raw = matches[matches["生食/熟食"] == "生食"]
        if not raw.empty:
            return raw
        cooked = matches[matches["生食/熟食"] == "熟食"]
        if not cooked.empty:
            return cooked.loc[[cooked["全部"].idxmin()]]
        return None

    def _process_ingredient(self, row: pandas.Series, i: int, dish_name: str, module: str) -> typing.Optional[typing.List]:
        # 正確使用 ingredient 序號 i
        ing_name = row.get("ingredient" + str(i) + "_name")
        if pandas.isna(ing_name):
            return None
        ing_type = row["ingredient" + str(i) + "_type"]
        ing_weight = row["ingredient" + str(i) + "_weight"]
        matches = self.select_best_matches(ing_name)
        if matches is None:
            return self._create_missing_ingredient_entry(dish_name, module, ing_type, ing_name, ing_weight)
        else:
            return self._create_ingredient_entry(matches.iloc[0], dish_name, module, ing_weight)

    def _create_missing_ingredient_entry(self, dish_name: str, module: str, ing_type: str, ing_name: str, ing_weight: float) -> typing.List:
        # 固定前 9 個欄位不變，接著新加「製造時間」與「進貨時間」
        entry = [
            None,                           # 編號 (0)
            None,                           # 代碼 (1)
            ing_name,                       # 食材名稱 (2)
            ing_type,                       # 食物分類大項 (3)
            "",                             # 食物分類子項 (4)
            dish_name,                      # 菜餚名 (5)
            module,                         # 菜餚類型 (6)
            ing_weight,                     # 重量 (7)
            1,                              # 數量 (8)
            None,                           # 製造時間 (新欄位, 9)
            "",                             # 進貨時間 (原本位置, 現在 index 10)
            None,                           # 生食/熟食 (11)
            None,                           # 烹飪方式 (12)
            self.infer_storage_method(ing_type, ing_name)  # 保存方式 (13)
        ]
        entry.append(None)                  # 產地 (14)
        entry.extend([None] * 7)            # 7 個碳排欄位（原本從 index 14 到 20，現依序放到 index 15~21）
        entry.extend([0, 0])                # 倉儲與加總碳排 (分別位於 index 22 與 23)
        return entry

    def _create_ingredient_entry(self, match: pandas.Series, dish_name: str, module: str, ing_weight: float) -> typing.List:
        storage = match["保存方式"] if not pandas.isna(match["保存方式"]) else self.infer_storage_method(match["食物分類大項"], match["食材名稱"])
        # 調整順序：最終順序為
        # [農業, 加工, 包裝, 超市及配送, 運輸, 能源消耗, 全部]
        CARBON_COLUMNS_OUTPUT = ["農業", "加工", "包裝", "超市及配送", "運輸", "能源消耗", "全部"]
        carbon_list = [float(match[col]) for col in FoodCarbonCalculator.CARBON_COLUMNS]
        carbon_mapping = {col: carbon_list[FoodCarbonCalculator.CARBON_COLUMNS.index(col)] for col in FoodCarbonCalculator.CARBON_COLUMNS}
        carbon_list = [carbon_mapping[col] for col in CARBON_COLUMNS_OUTPUT]
        # 計算基礎碳排（不含額外運輸碳排），使用前 6 項碳排數值乘以重量
        base_carbon_sum = (carbon_list[0] + carbon_list[1] + carbon_list[2] +
                           carbon_list[3] + carbon_list[4] + carbon_list[5]) * ing_weight
        # 取得產地，如果資料庫中沒有，則隨機指定一個
        origin = match.get("產地", None)
        if origin is None or pandas.isna(origin):
            origin = random.choice(FoodCarbonCalculator.ORIGINS)
        # 根據 origin 與 storage 計算額外運輸碳排（不乘以重量）
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

        # 計算最終總碳排：基礎碳排加上額外運輸碳排
        total_carbon = base_carbon_sum + additional
        # 更新「全部」欄位為 total_carbon
        carbon_list[6] = total_carbon

        entry = [
            None,                           # 編號 (0)
            match["代碼"],                  # 代碼 (1)
            match["食材名稱"],              # 食材名稱 (2)
            match["食物分類大項"],           # 食物分類大項 (3)
            match["食物分類子項"],           # 食物分類子項 (4)
            dish_name,                      # 菜餚名 (5)
            module,                         # 菜餚類型 (6)
            ing_weight,                     # 重量 (7)
            1,                              # 數量 (8)
            None,                           # 製造時間 (新欄位, 9)
            "",                             # 進貨時間 (新欄位, 10)
            match["生食/熟食"],             # 生食/熟食 (11)
            match["烹飪方式"],              # 烹飪方式 (12)
            storage                         # 保存方式 (13)
        ]
        entry.append(origin)                # 產地 (14)
        entry.extend(carbon_list)           # 碳排欄位：放於 index 15~21
        entry.extend([0, total_carbon])       # 倉儲與加總碳排 (分別位於 index 22 與 23)
        return entry

    def calculate_carbon_emissions(self, weight: float, carbon_values: typing.Dict[str, float]) -> float:
        return float(weight) * sum(float(carbon_values[col]) for col in ["農業", "加工", "包裝", "運輸", "超市及配送", "能源消耗"])

    def infer_storage_method(self, category: str, food_name: str) -> str:
        if category == "全穀雜糧類":
            if any(x in food_name for x in ["油麵", "烏龍麵", "豌豆"]):
                return "冷藏"
            elif any(x in food_name for x in ["奶油餐包", "香蒜麵包"]):
                return "冷凍"
            return "常溫"
        if category in FoodCarbonCalculator.STORAGE_RULES:
            return FoodCarbonCalculator.STORAGE_RULES[category]
        if category == "豆魚蛋肉類":
            if "黃豆" in food_name:
                return "常溫"
            if any(x in food_name for x in ["豆腐", "豆乾", "豆乾絲", "蛋"]):
                return "冷藏"
            return "冷凍"
        if category == "蔬菜類":
            if any(x in food_name for x in ["薑", "大蒜", "胡椒", "洋蔥"]):
                return "常溫"
            return "冷藏"
        if category == "水果類":
            if any(x in food_name for x in ["鳳梨罐頭", "柑橘", "西瓜"]):
                return "常溫"
            return "冷藏"
        if category == "油脂與堅果種子類":
            if "奶油" in food_name:
                return "冷凍"
            if "大蒜" in food_name:
                return "冷藏"
        return "常溫"

    def _generate_random_entries(self, entry: typing.List, num_entries: int) -> typing.List:
        # 調整函式：修改運輸碳排的計算方式，同時新增製造時間的計算邏輯
        def modify(e: typing.List) -> typing.List:
            try:
                orig_weight = float(e[7])
                new_weight = orig_weight * (1 + random.uniform(-0.03, 0.03))
                e[7] = round(new_weight, 2)
            except Exception:
                pass
            # 設定進貨時間 (改至 index 10)
            e[10] = random.choice(self.purchase_dates)
            # 設定製造時間預設為空，稍後計算
            e[9] = ""
            # 若產地不存在，則隨機指定產地；產地現位於 index 14
            if not e[14]:
                e[14] = random.choice(FoodCarbonCalculator.ORIGINS)
            origin = e[14]
            # 取得保存方式 (現位於 index 13)
            storage = e[13]
            additional = 0
            if origin == "美國":
                if storage == "冷藏":
                    additional = random.choice([9.17, 2.44])
                elif storage == "冷凍":
                    additional = random.choice([9.47, 4.44])
                elif storage == "常溫":
                    additional = random.choice([8.87, 0.44])
            elif origin == "澳洲":
                if storage == "冷藏":
                    additional = random.choice([5.84, 1.78])
                elif storage == "冷凍":
                    additional = random.choice([6.04, 3.28])
                elif storage == "常溫":
                    additional = random.choice([5.74, 0.28])
            elif origin == "日本":
                if storage == "冷藏":
                    additional = random.choice([1.71, 0.48])
                elif storage == "冷凍":
                    additional = random.choice([1.81, 0.88])
                elif storage == "常溫":
                    additional = random.choice([1.61, 0.08])
            else:
                additional = 0
            try:
                original_transport = float(e[19]) if e[19] is not None else 0
            except Exception:
                original_transport = 0
            e[19] = original_transport + additional
            try:
                purchase_date = datetime.datetime.strptime(e[10], "%Y-%m-%d").date()
            except Exception:
                purchase_date = datetime.datetime.now().date()
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
            manufacturing_date = purchase_date - datetime.timedelta(days=delta)
            e[9] = manufacturing_date.strftime("%Y-%m-%d")
            try:
                today = datetime.datetime.now().date()
                days_diff = (today - purchase_date).days
            except Exception:
                days_diff = 0
            factor = 0
            if storage == "常溫":
                factor = 0
            elif storage == "冷藏":
                factor = 0.1
            elif storage == "冷凍":
                factor = 0.2
            e[22] = days_diff * factor
            try:
                weight = float(e[7]) if e[7] is not None else 0
            except Exception:
                weight = 0
            sum_carbon = 0
            for i in [15, 16, 17, 18, 19, 20]:
                try:
                    sum_carbon += float(e[i]) if e[i] is not None else 0
                except Exception:
                    sum_carbon += 0
            warehouse = float(e[22]) if e[22] is not None else 0
            e[23] = round(sum_carbon * weight + warehouse, 2)
            return e

        def modify_special(e: typing.List, base: float) -> typing.List:
            e[7] = round((base * 28.3495) * (1 + random.uniform(-0.03, 0.03)), 2)
            e[10] = random.choice(self.purchase_dates)
            e[9] = ""
            if not e[14]:
                e[14] = random.choice(FoodCarbonCalculator.ORIGINS)
            origin = e[14]
            storage = e[13]
            additional = 0
            if origin == "美國":
                if storage == "冷藏":
                    additional = random.choice([9.17, 2.44])
                elif storage == "冷凍":
                    additional = random.choice([9.47, 4.44])
                elif storage == "常溫":
                    additional = random.choice([8.87, 0.44])
            elif origin == "澳洲":
                if storage == "冷藏":
                    additional = random.choice([5.84, 1.78])
                elif storage == "冷凍":
                    additional = random.choice([6.04, 3.28])
                elif storage == "常溫":
                    additional = random.choice([5.74, 0.28])
            elif origin == "日本":
                if storage == "冷藏":
                    additional = random.choice([1.71, 0.48])
                elif storage == "冷凍":
                    additional = random.choice([1.81, 0.88])
                elif storage == "常溫":
                    additional = random.choice([1.61, 0.08])
            else:
                additional = 0
            try:
                original_transport = float(e[19]) if e[19] is not None else 0
            except Exception:
                original_transport = 0
            e[19] = original_transport + additional
            try:
                purchase_date = datetime.datetime.strptime(e[10], "%Y-%m-%d").date()
            except Exception:
                purchase_date = datetime.datetime.now().date()
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
            manufacturing_date = purchase_date - datetime.timedelta(days=delta)
            e[9] = manufacturing_date.strftime("%Y-%m-%d")
            try:
                today = datetime.datetime.now().date()
                days_diff = (today - purchase_date).days
            except Exception:
                days_diff = 0
            factor = 0
            if storage == "常溫":
                factor = 0
            elif storage == "冷藏":
                factor = 0.1
            elif storage == "冷凍":
                factor = 0.2
            e[22] = days_diff * factor
            try:
                weight = float(e[7]) if e[7] is not None else 0
            except Exception:
                weight = 0
            sum_carbon = 0
            for i in [15, 16, 17, 18, 19, 20]:
                try:
                    sum_carbon += float(e[i]) if e[i] is not None else 0
                except Exception:
                    sum_carbon += 0
            warehouse = float(e[22]) if e[22] is not None else 0
            e[23] = round(sum_carbon * weight + warehouse, 2)
            return e

        if hasattr(self, "menu_file_name") and self.menu_file_name == "牛排館菜單.csv" and entry[4] in ["牛肉類", "豬肉類", "羊肉類"]:
            new_entries = []
            for base in [6, 8, 10, 12]:
                total_entries = 400  # 設定總資料數量
                count = total_entries // 4  # 平均分配給 4 個重量類別
                for _ in range(count):
                    new_entries.append(modify_special(entry.copy(), base))
            return new_entries
        else:
            return [modify(entry.copy()) for _ in range(num_entries)]

    def _add_to_consolidated_data(self, sheet_data: typing.List[typing.List], entry: typing.List, dish_name: str) -> None:
        sheet_data.append(entry)

    def process_menu(self, menu_path: pathlib.Path, output_path: pathlib.Path) -> None:
        # 記錄菜單檔案名稱，以便後續判斷
        self.menu_file_name = menu_path.name
        menu_df = pandas.read_csv(menu_path)
        result_data = {}
        for index, row in menu_df.iterrows():
            dish_name, module = row["dish_name"], row["module"]
            sheet_name = "完整" + module + "碳排"
            if sheet_name not in result_data:
                result_data[sheet_name] = []
            for i in range(1, 6):
                ing = self._process_ingredient(row, i, dish_name, module)
                if ing:
                    self._add_to_consolidated_data(result_data[sheet_name], ing, dish_name)
        expanded_data = {}
        for sheet, sheet_data in result_data.items():
            expanded_data[sheet] = []
            for entry in sheet_data:
                expanded_data[sheet].extend(self._generate_random_entries(entry, random.randint(100, 150)))
        self._save_to_excel(expanded_data, output_path)

    def _save_to_excel(self, result_data: typing.Dict[str, typing.List[typing.List]], output_path: pathlib.Path) -> None:
        # 調整欄位標題，新欄位「製造時間」置於「進貨時間」左側
        columns = [
            "編號", "代碼", "食材名稱", "食物分類大項", "食物分類子項", "菜餚名", "菜餚類型",
            "重量", "數量", "製造時間", "進貨時間", "生食/熟食", "烹飪方式", "保存方式",
            "產地", "農業", "加工", "包裝", "超市及配送", "運輸碳排", "能源消耗", "全部", "倉儲", "加總碳排"
        ]
        writer = pandas.ExcelWriter(str(output_path), engine="xlsxwriter")
        for sheet, data in result_data.items():
            for i, entry in enumerate(data, 1):
                entry[0] = i
            df = pandas.DataFrame(data, columns=columns)
            df.to_excel(writer, sheet_name=sheet, index=False)
        writer.close()

def main():
    calculator = FoodCarbonCalculator(pathlib.Path("單品食材資料庫.csv"))
    menu_files = [
        (pathlib.Path("牛排館菜單.csv"), pathlib.Path("牛排館單品食材資料庫.xlsx")),
        (pathlib.Path("便當店菜單.csv"), pathlib.Path("便當店單品食材資料庫.xlsx")),
    ]
    for input_file, output_file in menu_files:
        calculator.process_menu(input_file, output_file)
        print(f"{output_file.name} 已生成完畢")

if __name__ == "__main__":
    main()
