# Что было сделано
[Github репозиторий проекта](https://github.com/EugeneRumynskiy/Students-Alchohol)


### Задача 1: положить данные из CSV в любую реляционную БД (MySQL)  

Залил таблицу на PostgresSQL на [Heroku](heroku.com). [Ссылка на нотбук](https://github.com/EugeneRumynskiy/Students-Alchohol/blob/master/prototyping/SQLdatabase.ipynb)

  
  
### Задача 2: на основе данных выявить поведенчиские шаблоны пьющих и непьющих студентов, можно делать в Jupyter notebook  

[Тут смотрел на фичи и корреляции.](https://github.com/EugeneRumynskiy/Students-Alchohol/blob/master/prototyping/Dataset_description.ipynb)
  
  
### Задача 3: предсказать успеваемость студента по его данным (см. колонки G1, G2, G3).  Прототипирование можно делать в jupyter, итоговый результат надо   оформить в виде питоновского модуля 

[Здесь тюнил и учил модельки](https://github.com/EugeneRumynskiy/Students-Alchohol/blob/master/prototyping/Tuning_Predictions_v2.ipynb)

1 Смотрел две метрики **MSE, MAE**
2 Итоговая модель -  **RandomForestRegressor**
3 Итоговую модель обучал на датасете **students-mat**
4 Скрипт, создающий и сохраняющий модель [create_model.py](https://github.com/EugeneRumynskiy/Students-Alchohol/blob/master/flask_api/create_model.py)
 
   
### Задача 4: сделать REST API для предсказания и код для загрузки натренированной модели  
1 Сервер на Flask + gunicorn [server.py](https://github.com/EugeneRumynskiy/Students-Alchohol/blob/master/flask_api/server.py)
2 Сервер с моделями, сохраненными через **pickle** поднят на **Heroku**
3 **API** сделан в виде такого запроса "https://students-and-alcohol.herokuapp.com/predict"
4 Скрипт, который позволяет проверить API [api_request_predictions.py](https://github.com/EugeneRumynskiy/Students-Alchohol/blob/master/flask_api/api_request_predictions.py) Он отсылает датасет **students-por** на (https://students-and-alcohol.herokuapp.com/predict) , получает **JSON** с предсказаниями и считает **MAE**
