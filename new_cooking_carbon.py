import pandas as pd
import pathlib

def load_cooking_data(file_path: pathlib.Path, restaurant_type: str) -> pd.DataFrame:
    """
    讀取烹飪碳排資料，依據餐廳類型（例如 "牛排館" 或 "自助餐"）讀取對應工作表。
    """
    xls = pd.ExcelFile(str(file_path))
    sheet_name = restaurant_type + "烹飪碳排"
    return xls.parse(sheet_name)

def compute_cooking_carbon(cooking_info: pd.Series, dish_name: str, module: str, persona_choice: dict, restaurant_type: str) -> float:
    """
    計算烹飪碳排：
    
    條件1：當餐廳為牛排館、菜餚模組為主餐、且菜餚名稱中含有「牛」時，
           {cooking_sec}欄位按照 persona_choice 字典中的 {steak_rare} 值取得，對應關係為：
             {steak_rare}=3  =>  60秒
             {steak_rare}=5  =>  180秒
             {steak_rare}=7  =>  300秒
             {steak_rare}=10 =>  420秒
           若未正確設定則預設為300秒。
    
    條件2：當餐廳為牛排館、菜餚模組為主餐、且菜餚名稱中含有「牛」、「豬」或「羊」時，
           {dish_weight}欄位會依據 persona_choice 字典中的 {oz} 值轉換成克（1 oz = 28.3495 克），
           接著乘上 {cooking_sec} 及 cooking_info 中的 {carbon_per_sec}，得到 {total_cooking_carbon}：
           
             total_cooking_carbon = dish_weight (克) * cooking_sec (秒) * carbon_per_sec
             
    否則，返回 cooking_info["total_cooking_carbon"] 的預設值。
    """
    if restaurant_type == "牛排館" and module == "主餐" and any(x in dish_name for x in ["牛", "豬", "羊"]):
        # 條件1：取得 cooking_sec
        if "牛" in dish_name:
            steak_rare = persona_choice.get("steak_rare", None)
            steak_mapping = {3: 60, 5: 180, 7: 300, 10: 420}
            if steak_rare in steak_mapping:
                cooking_sec = steak_mapping[steak_rare]
            else:
                cooking_sec = 300
        else:
            cooking_sec = cooking_info["cooking_sec"]
        
        # 條件2：取得 dish_weight，由 persona_choice 中的 {oz} 值轉換成克
        oz_value = persona_choice.get("oz", None)
        if oz_value is not None:
            dish_weight = oz_value * 28.3495
        else:
            dish_weight = 0  # 或設定其他預設值
        
        carbon_per_sec = cooking_info.get("carbon_per_sec", 0)
        total_cooking_carbon = dish_weight * cooking_sec * carbon_per_sec
        return total_cooking_carbon
    else:
        return cooking_info["total_cooking_carbon"]

if __name__ == "__main__":
    file = pathlib.Path("烹飪碳排.xlsx")
    df = load_cooking_data(file, "牛排館")
    print(df.head())
