# food-scheduling-system

程式碼：<br>
<b>1.test_greedy.py：排程系統主程式碼</b><br>
<b>2.restock_system.py：補貨的程式碼，當食材庫存量小於30%進行補貨</b><br>
3.restaurant_database_generate.py：透過讀取「牛排館菜單.csv」以及「便當店菜單.csv」生成初始的資料庫「牛排館單品食材資料庫.xlsx」、「便當店單品食材資料庫.xlsx」。<br>
4.delete_memory.py：將更新過後的單品食材資料庫恢復初始狀態。<br>
5.new_cooking_carbon.py：計算烹飪碳排<br><br>

檔案：<br>
1.單品食材資料庫.csv：初始食材資料庫，餐廳的單品食材資料庫的碳排數值都是從這裡抓的<br>
2.persona.xlsx：養生村村民的飲食習慣變數<br>
3.「牛排館菜單.csv」「便當店菜單.csv」：兩間餐廳的菜單<br>
4.「牛排館單品食材資料庫.xlsx」「便當店單品食材資料庫.xlsx」：兩間餐廳的食材庫存<br>
5.烹飪碳排.xlsx：紀錄每道菜的烹飪時間和碳排<br>
6.<b>villager_menu.xlsx 是test_greedy.py跑完之後會產出的檔案，目前會跑出找不到食材的錯誤</b><br><br>
