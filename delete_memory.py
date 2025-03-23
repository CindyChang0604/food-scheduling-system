import pandas
import pathlib

def clear_status(file_path: pathlib.Path) -> None:
    # 讀取 Excel 檔案中的所有工作表
    xls = pandas.ExcelFile(str(file_path))
    sheets = {}
    for sheet_name in xls.sheet_names:
        df = xls.parse(sheet_name)

        if "補貨日期" in df.columns:
            # 過濾掉「補貨日期」有值的列，只保留補貨日期為空（NaN 或空字串）的列
            df = df[df["補貨日期"].isna() | (df["補貨日期"] == "")]

        # 如果已經有「狀態」欄，則將其全部清空；若無則新增此欄並設為空
        if "狀態" in df.columns:
            df["狀態"] = ""
        else:
            df["狀態"] = ""
        sheets[sheet_name] = df
    # 將更新後的工作表寫回原 Excel 檔案，並移除「狀態」欄（表頭也一併移除）
    with pandas.ExcelWriter(str(file_path), engine="xlsxwriter") as writer:
        for sheet_name, df in sheets.items():
            if "狀態" in df.columns:
                df = df.drop(columns=["狀態"])
            if "補貨日期" in df.columns:
                df = df.drop(columns=["補貨日期"])
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"{file_path} 的狀態已清空。")

def main():
    # 定義要清空狀態的檔案
    files = ["牛排館單品食材資料庫.xlsx", "便當店單品食材資料庫.xlsx"]
    for file in files:
        clear_status(pathlib.Path(file))

if __name__ == "__main__":
    main()

