# Лабораторная работа №3
Выполнил: Антонов Илья, 24 МАГ ИАД

## Датасет
Используется датасет `nyc_yellow_taxi.csv`. 
Он состоит из первых 250.000 строчек датасет [NYC Yellow Taxi Trip Data](https://www.kaggle.com/datasets/elemento/nyc-yellow-taxi-trip-data) за 2015 год.
## ML задача
Решается задача предсказания стоимости поездки в такси с помощью линейной регрессии.
## Запуск
- Соберите docker образ:
```
docker build -t lakehouse .
```
- Запустите контейнер:
```
docker compose up -d
```
## Выключение:
- Выполните команду:
```
docker compose down
```
## Полезная информация:
- Чтобы посмотреть MLFlow зайдите на http://localhost:5000/
- Логи можно найти в `logs/output.log`