import streamlit as st
import wheel
import pandas as pd
import json
import plotly.graph_objs as go
import pydeck as pdk
import numpy as np
from math import radians as rad
from math import sin, cos, acos
from lr import LinearRegression as LR


with st.echo():
    def geo_dist(lat1, lon1, lat2, lon2):
        l = 111.1
        return acos(sin(rad(lat1)) * sin(rad(lat2)) + cos(rad(lat1)) * cos(rad(lat2)) * cos(rad(lon2) - rad(lon1))) * l
        #return (lat1 - lat2)**2 + (lon1 - lon2)**2

    EPSG_MOSCOW = 3352

    MOSCOW_REGION = 3
    N_SAMPLE = 7

    MIN_LAT = 55.5735
    MAX_LAT = 55.9104
    MIN_LON = 37.37
    MAX_LON = 37.8597
    MIN_R = 0.01
    MAX_R = 0.07

    st.title('Поиск квартир')
    st.write('Информацию о проекте вы сможете увидеть ниже, когда загрузится карта, а здесь вы можете выставить параметры квартиры')

    lat_sl = st.slider(label="Ширина", min_value=MIN_LAT, max_value=MAX_LAT, value=(MIN_LAT + MAX_LAT) / 2)
    lon_sl = st.slider(label="Долгота", min_value=MIN_LON, max_value=MAX_LON, value=(MIN_LON + MAX_LON) / 2)
    r = st.slider(label="radius", min_value=MIN_R, max_value=MAX_R, value=(MIN_R + MAX_R) / 2, step = 0.001)

    data = pd.read_csv('moscow.csv')

    rooms = st.slider(label="Кол-во комнат", min_value=int(data['rooms'].min()), max_value=int(data['rooms'].max()), value=int(data['rooms'].min()), step=1)

    data = data.loc[(data['lat'] - lat_sl <= r) & (-r <= data['lat'] - lat_sl) & (data['lon'] - lon_sl <= r) & (-r <= data['lon'] - lon_sl) & (data['rooms'] == rooms)]
    data = data.loc[(data['rooms'] == rooms)]

    price = st.slider(label="Цена", min_value=int(data['price'].min()), max_value=int(data['price'].max()), value=[int(data['price'].min()), max(int(data['price'].max() + data['price'].min()) // 9, int(data['price'].min()))])

    data = data.loc[(price[0] < data['price']) & (data['price'] < price[1])]

    map_data = data.reset_index()

    #st.map(map_data.head(n=10))
    my_choice = {"lon": [lon_sl], "lat": [lat_sl]}
    my_choice = pd.DataFrame(data=my_choice)

    layers = [ 
        pdk.Layer(
            'ScatterplotLayer',
            data=my_choice,
            get_position='[lon, lat]',
            get_color='[0, 255, 0, 255]',
            get_radius=200,
            pickable=True,
            filled=True
        ),
        pdk.Layer(
            'ScatterplotLayer',
            data=map_data,
            get_position='[lon, lat]',
            get_color='[255, 0, 0, 160]',
            get_radius=40,
        )]

    with open('weights.txt') as r:
        r = r.read().split()
        w = list(map(float, r))

    clf = LR()
    clf._W = np.array(w)

    if len(map_data.index) != 0:
        apart = st.selectbox(label='Выберите одну из предложенные квартир для дополнительной информации', options=map_data.index.tolist())
        st.write(map_data.iloc[apart])
        x = map_data.iloc[apart][['lat', 'lon', 'area', 'kitchen_area']]
        house_lat = x['lat']
        house_lon = x['lon']
        with open('mean.txt') as r:
            m = r.read().split()
            m = list(map(float, m))
        m = np.array(m)
        
        with open('std.txt') as r:
            s = r.read().split()
            s = list(map(float, m))
        s = np.array(m)
        x = np.array(x)
        x = (np.array(x) - m) / s
        x = np.insert(x, 0, 1)
        st.write("Реальная цена {}, переплата составляет {}".format(clf.predict(x), int(map_data.iloc[apart]['price'] - clf.predict(x))))
        dominos = pd.read_csv('dominos.csv')
        lonm, latm = dominos.iloc[0]['lon'], dominos.iloc[0]['lat']
        for i in dominos.index:
            lonn, latn = dominos.iloc[i]['lon'], dominos.iloc[i]['lat']
            if geo_dist(house_lat, house_lon, latm, lonm) > geo_dist(house_lat, house_lon, latn, lonn):
                latm, lonm = latn, lonn
        e_dom = {'lon': [lonm], 'lat': [latm]}
        e_dom = pd.DataFrame(e_dom)
        st.write('updated')
        st.write(e_dom)
        layers.append(
            pdk.Layer(
                'ScatterplotLayer',
                data=e_dom,
                get_position='[lon, lat]',
                get_color='[0, 0, 255, 160]',
                get_radius=200,
            )
        )
        house = {'lon': [house_lon], 'lat': [house_lat]}
        house = pd.DataFrame(house)
        layers.append(
            pdk.Layer(
                'ScatterplotLayer',
                data=house,
                get_position='[lon, lat]',
                get_color='[90, 0, 157, 160]',
                get_radius=200,
            )
        )
        st.write('Ниже представлена ссылка на гугл-карты. Снизу справа расположен человечек, которого, если зажать, можно попробовать "кинуть" прям к дому квартиры, чтобы посмотреть окрестности')
        st.write("https://www.google.com/maps/search/maps/@{},{},17.25z".format(house_lat, house_lon))

    else:
        st.write("Не найдено квартир с такими параметрами")

    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v8',
        initial_view_state=pdk.ViewState(
            latitude=lat_sl,
            longitude=lon_sl,
            zoom=11,
            pitch=50,
            tooltip={"html": "<b>Point ID: </b> {price} <br /> "}
        ),
        layers=layers
    ))

    mes = """
    \tЭто приложение позволяет вам найти квартиру с нужной ценой и количеством комнат. Для того, чтобы указать точку, вокруг которой смотреть квартиры - надо установить слайдеры широты, долготы и "радиуса" поиск в подходящие позиции. Установленная точка для поиска будет отображаться зелёным цветом.\n
    \tЕсли хоть одна квартира будет найдена - вам будет доступен выбор квартир для дальнейшего анализа. Этот анализ показывает информацию о квартире: цену, широту, долготу, этаж, кол-во комнат, площадь квартиры и площадь кухни. Анализ происходит на основе манипуляций с Pandas DataFrame.\n
    \tТакже написаны методы машинного обучения в файле https://github.com/dogedatascience/hse_house/blob/main/lr.py для оценки реальной стоимости квартиры. Метод машинного обучения - линейная регрессия, которая считается с помощью математических операций над NumPy матрицами. Так, например, можно видеть, что есть квартиры, которые чересчур завышены в цене. Выбранная квартира будет отоборажаться фиолетовым цветом. Также можно проследовать по ссылке и посмотреть на район, в котором находится квартира, через google-maps.\n
    \tТак как автор работы является большим поклонником пиццы Доминос, ещё на карте синим цветом отображается ближайшая точка Доминос пиццы. Рассчёт ближайшей Доминос пиццы происходит с помощью математической формулы https://en.wikipedia.org/wiki/Great_circle_distance. Для рассчёта испоьзуются математические функции из библотеки math. Получение данных о расположении Доминос пицц происходит с помощью веб-скреппинга. Это делает скрипт https://github.com/dogedatascience/hse_house/blob/main/dominos.py. Так как Доминос пытается блокировать чистый скреппинг, нужно выставить user-agent. Ближе к концу документа есть информация о Доминос пиццах и она вычленяется с помощью алгоритма, которых похож на правильные скобочные последовательности: сначала устанавливается, что открылась одна фигурная скобка, а дальше происходит анализ каждого символа. Если фигурная скобка открывающаяся - мы добавляем единицу к кол-ву открытых фигурных скобок. Если закрывающаяся - убавляем кол-во открытых скобок. Как только их становится 0 - мы дошли до конца json о Доминос Пиццахш. Вся эта информация достаётся из XML страницы (которая указана в скрипте).\n
    \tИнформация о квартире берётся из датасета с kaggle с помощью Kaggle API. Вот этот скрипт это делает: https://github.com/dogedatascience/hse_house/blob/main/main.py. Для этого должны быть выполнены остальные условия выполнения (должен быть сгенерирован API ключ, положен в нужное место итд.).\n
    Визуализация сначала планировалась с помощью geopandas, но в последствии была переписана на просто pandas + Selenium, т.к. нужные функции выполняют гугл-карты и сам Selenium - у него есть отличный способ визуализации, который тоже предоставила гугл, который работает точно также, как geopandas, но лучше встраивается в сам Streamlit
    """

    for i in mes.split('\n'):
        st.write(i)
    st.echo()
