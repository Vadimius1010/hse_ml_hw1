import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle

def safe_log(x):
    return np.log(np.where(x > 0, x, 1))  # заменяем неположительные значения на 1

# настройки страницы
st.set_page_config(page_title="Предсказание стоимости авто", page_icon="🚗", layout="wide")

@st.cache_resource
def load_model(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model

@st.cache_data 
def load_data(path):
    return pd.read_csv(path, index_col=0)

X_train = load_data('X_train_final.csv')
X_test = load_data('X_test_final.csv')
y_train = load_data('y_train.csv')
y_test = load_data('y_test.csv')
coef_df = load_data('coef_df.csv')

model = load_model('pipeline_inference.pkl')

# объединю для удобного EDA
df_train = pd.concat([X_train, y_train], axis=1)
df_test = pd.concat([X_test, y_test],  axis=1)

st.title("🚗 Предсказание стоимости авто")

# EDA
st.header("📊 EDA-дашборд")

# Фильтрация данных
col1, col2, col3, col4 = st.columns(4)

with col1:
    min_year = int(df_train['year'].min())
    max_year = int(df_train['year'].max())
    selected_years = st.slider("Год выпуска", min_year, max_year, (min_year, max_year))
    
    min_km = int(df_train['km_driven'].min())
    max_km = int(df_train['km_driven'].max())
    selected_km = st.slider("Пробег", min_km, max_km, (min_km, max_km))

with col2:
    brand_types = df_train['brand'].unique()
    selected_brand = st.multiselect("Марка", brand_types, default=brand_types)
    
    # динамический фильтр на модель, зависящий от выбранной марки
    model_types = df_train[df_train['brand'].isin(selected_brand)]['model'].unique()
    selected_model = st.multiselect("Модель", model_types, default=model_types)

with col3:
    fuel_types = df_train['fuel'].unique()
    selected_fuels = st.multiselect("Тип топлива", fuel_types, default=fuel_types)
    transmission_types = df_train['transmission'].unique()
    selected_transmission = st.multiselect("Трансмиссия", transmission_types, default=transmission_types)
    min_engine, max_engine = int(df_train['engine'].min()), int(df_train['engine'].max())
    selected_engine = st.slider("Объем двигателя (см^3)", min_engine, max_engine, (min_engine, max_engine))
    min_power, max_power = int(df_train['max_power'].min()), int(df_train['max_power'].max())
    selected_power = st.slider("Макс. мощность", min_power, max_power, (min_power, max_power))

with col4:
    seller_types = df_train['seller_type'].unique()
    selected_seller = st.multiselect("Кто продаёт", seller_types, default=seller_types)
    owner_types = df_train['owner'].unique()
    selected_owner = st.multiselect("Владелец", owner_types, default=owner_types)

filtered_df_train = df_train[
    (df_train['year'] >= selected_years[0]) & (df_train['year'] <= selected_years[1]) &
    (df_train['km_driven'] >= selected_km[0]) & (df_train['km_driven'] <= selected_km[1]) &
    (df_train['engine'] >= selected_engine[0]) & (df_train['engine'] <= selected_engine[1]) &
    (df_train['max_power'] >= selected_power[0]) & (df_train['max_power'] <= selected_power[1]) &

    (df_train['brand'].isin(selected_brand)) &
    (df_train['model'].isin(selected_model)) &
    (df_train['fuel'].isin(selected_fuels)) &
    (df_train['seller_type'].isin(selected_seller)) &
    (df_train['owner'].isin(selected_owner))
]

filtered_df_test = df_test[
    (df_test['year'] >= selected_years[0]) & (df_test['year'] <= selected_years[1]) &
    (df_test['km_driven'] >= selected_km[0]) & (df_test['km_driven'] <= selected_km[1]) &
    (df_test['engine'] >= selected_engine[0]) & (df_test['engine'] <= selected_engine[1]) &
    (df_test['max_power'] >= selected_power[0]) & (df_test['max_power'] <= selected_power[1]) &

    (df_test['brand'].isin(selected_brand)) &
    (df_test['model'].isin(selected_model)) &
    (df_test['fuel'].isin(selected_fuels)) &
    (df_test['seller_type'].isin(selected_seller)) &
    (df_test['owner'].isin(selected_owner))
]

st.write(f"**Тренировочные данные, найдено автомобилей: {len(filtered_df_train)}**")
st.dataframe(filtered_df_train, use_container_width=True)

def eda_plots(df, title):
    # гистограмма цен на авто
    fig_price_dist = px.histogram(df, x='selling_price', 
                                  nbins=50,
                                  title='Распределение цен на автомобили',
                                  labels={'selling_price': 'Цена'})
    fig_price_dist.update_layout(xaxis_title="Цена", yaxis_title="Количество")
    st.plotly_chart(fig_price_dist, use_container_width=True, key=f'price_{title}')

    # скрипичный график
    fig_violin = px.violin(df, 
        x="selling_price", 
        box=True, 
        points="all", 
        hover_data=df.columns)
    st.plotly_chart(fig_violin, use_container_width=True, key=f'violin_{title}')

    # матрица корреляций
    сorr = df.select_dtypes(include=['number']).corr()
    fig_corr = go.Figure(data=go.Heatmap(
        z=сorr.values,
        x=сorr.columns,
        y=сorr.columns,
        zmin=-1, zmax=1,
        text=сorr.round(2).values,
        texttemplate='%{text}',
        textfont={"size": 10}
    ))
    fig_corr.update_layout(
        title='Матрица корреляции числовых признаков',
        xaxis_title="Признаки",
        yaxis_title="Признаки"
    )
    st.plotly_chart(fig_corr, use_container_width=True, key=f'corr_{title}')

    # pairplot мощности и цены
    fig_pairplot = px.scatter(df, 
                              title='Зависимость цены от мощности',
                              x='max_power', 
                              y = 'selling_price')
    st.plotly_chart(fig_pairplot, use_container_width=True, key=f'pair_{title}')


# строим графики для train и test датасетов
st.header("Сравнение train, test")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Train") 
    eda_plots(filtered_df_train, 'train')

with col2:
    st.subheader("Test") 
    eda_plots(filtered_df_test, 'test')

st.header("Регрессионная модель")
col1, col2 = st.columns(2)
with col1:
    st.write("Веса обученной модели")
    # проверить достав из пайплайна
    st.dataframe(coef_df.sort_values(by='coefficient', key=abs, ascending=False)) 

with col2:
    st.write("**Прогнозирование**")
    st.download_button("Скачать тестовые данные", X_test.to_csv(), file_name='test_dataset.csv',)   

    uploaded_file = st.file_uploader("Загрузка файла", type=["csv"])
    if uploaded_file is None:
        st.info("**Загрузите CSV файл для начала работы**")
        st.stop()

    uploaded_df = load_data(uploaded_file)
    st.write(f"**Файл {uploaded_file.name} сохранен, делаем предсказание:**")

try:
    pred_test = np.exp(model.predict(uploaded_df)) # делал логарифмирование при обучении
except:
    st.header("Что-то пошло не так!")
    st.write('Загрузите подходящий файл')
    st.stop()

st.header("Результаты предсказания по файлу")

st.dataframe(pd.concat([uploaded_df, pd.Series(pred_test, name='prediction')],  axis=1), use_container_width=True)