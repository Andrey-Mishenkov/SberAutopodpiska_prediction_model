import pandas as pd
import numpy as np
import dill
import sys
import time
from datetime import datetime

# -----------------------------------------------------------------------------------------------------------------
from sklearnex import patch_sklearn        # патч - ускорение библиотеки scikit-learn
patch_sklearn()

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
# from sklearn.compose import make_column_selector
from sklearn.preprocessing import TargetEncoder, OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, roc_curve, auc
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.utils.class_weight import compute_class_weight

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# -----------------------------------------------------------------------------------------------------------------
# import matplotlib_inline
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import cycle

# -----------------------------------------------------------------------------------------------------------------
# доп пакеты для установки
# -----------------------------------------------------------------------------------------------------------------
# pip install geopy
# pip install xgboost
# pip install catboost
# pip install scikit-learn-intelex
# ----------------------------------------------------------------------
# from geopy.geocoders import Nominatim
# from geopy.distance import geodesic
# from geopy.extra.rate_limiter import RateLimiter


# -----------------------------------------------------------------------------------------------------------------
# Техническая часть (переменные, функции)
# -----------------------------------------------------------------------------------------------------------------
# Глобальные параметры (переменные)
# Названия файлов с данными
filename_sessions   = 'data/ga_sessions.pkl'
filename_hits       = 'data/ga_hits.pkl'

dt_start = datetime.now()

RANDOM_STATE = 42
TEST_SIZE = 0.2
MODEL_TUNING_TIME_LIMIT = 1800  # 30 минут в секундах

# -----------------------------------------------------------------------------------------------------------------
# Вывод продолжительности выполнения кода
def get_duration_time(dt_start):
    dt_finish = datetime.now()
    dt_duration = dt_finish - dt_start

    text = (f'Начало = {dt_start.strftime("%Y-%m-%d (%H:%M:%S)")};   ' +
            f'Текущее время = {dt_finish.strftime("%Y-%m-%d (%H:%M:%S)")};   ' +
            f'Общая продолжительность = ({str(dt_duration).split('.')[0]})')
    print(text)


# -----------------------------------------------------------------------------------------------------------------
def print_debug_info(a_text):
    print('-' * 100)
    get_duration_time(dt_start)
    print(f'{'-' * 100}\nВыполняется функция: {a_text}...')


# -----------------------------------------------------------------------------------------------------------------
# вывод информации о % заполнения датасета
def print_useful_rows_info(df):
    print(f'Количество полностью заполненных объектов из всей выборки: {len(df.dropna())}')
    print(
        f'Процент полностью заполненных объектов из всей выборки:    {round(len(df.dropna()) / len(df) * 100, 2)}')


# Вывод списка незаполненных значений столбцов и % заполнения
def print_list_missing_values(df):
    missing_values = ((df.isna().sum() / len(df)) * 100).sort_values(ascending=False)

    print('\nПроцент пропущенных значений:')
    # missing_values
    print(missing_values[missing_values != 0].index.to_list())
    print()
    print(missing_values[missing_values != 0])


# Вывод информации о незаполненных столбцах и % пропусках в данных
def print_info_about_missing(a_df):
    print('-' * 100)
    print_useful_rows_info(a_df)
    print_list_missing_values(a_df)

# -----------------------------------------------------------------------------------------------------------------
# Основные функции
# -----------------------------------------------------------------------------------------------------------------
# Загрузка исходных данных; получение основного датасета для работы
def read_source_data(is_debug=True):

    if is_debug:
        text = sys._getframe().f_code.co_name        # text = 'read_source_data'
        print(f'{'-' * 100}\nВыполняется функция: {text}...')

    # -----------------------------------------------------------------------------------------------------------------
    # 1. df_sessions
    print('-' * 100)
    print(f'Чтение файла {filename_sessions}')
    df_sessions = pd.read_pickle(filename_sessions)
    get_duration_time(dt_start)
    print('-' * 100)
    print(f'Чтение файла {filename_hits}')
    df_hits     = pd.read_pickle(filename_hits)
    get_duration_time(dt_start)

    # размеры датасетов и списки столбцов
    print('-' * 100)
    print(f'df_sessions; {df_sessions.shape}\n')
    print(df_sessions.columns)
    print()
    print(f'df_hits; {df_hits.shape}\n')
    print(df_hits.columns)

    # -----------------------------------------------------------------------------------------------------------------
    # 2. df_hits
    # Удаляю лишние столбцы, которые по условию задачи будут не нужны. Оставляю только 'session_id' и 'event_action'.
    columns_to_drop = [col for col in df_hits.columns if col not in ['session_id', 'event_action']]
    df_hits.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    # Созадаю новый столбец is_target_action - числовая целевая переменная (значения 1 и 0)
    # Удаляю столбец event_action, оставляю только session_id и target_event_action
    list_target_event_actions = ['sub_car_claim_click',
                                'sub_car_claim_submit_click',
                                'sub_open_dialog_click',
                                'sub_custom_question_submit_click',
                                'sub_call_number_click',
                                'sub_callback_submit_click',
                                'sub_submit_success',
                                'sub_car_request_submit_click']

    df_hits['target_event_action'] = df_hits['event_action'].isin(list_target_event_actions).astype(int)
    df_hits.drop(columns=['event_action'], inplace=True, errors='ignore')

    # Группирую данные в df_hits по session_id + функция агрегирования MAX(target_event_action)
    df_hits_grouped = df_hits.groupby('session_id')['target_event_action'].max().reset_index()

    # -----------------------------------------------------------------------------------------------------------------
    # 4. Соединяю два датасета df_sessions и df_hits_grouped в один основной
    df_sessions = df_sessions.merge(
                                        df_hits_grouped,
                                        on='session_id',
                                        how='left',
                                        indicator=False
                                    )
    df_sessions['target_event_action'] = df_sessions['target_event_action'].fillna(0).astype(int)

    # -----------------------------------------------------------------------------------------------------------------
    x = df_sessions.drop('target_event_action', axis=1)
    y = df_sessions['target_event_action']

    print(f'\n{'-' * 100}')
    print(f'Итоговый датасет df_sessions; {df_sessions.shape}\n')
    get_duration_time(dt_start)

    return df_sessions
    # return x, y


# -----------------------------------------------------------------------------------------------------------------
# Удаление выбранных столбцов из датасета (в начале обработки)
def drop_columns_first(a_df, is_debug=False):
    import sys

    text = sys._getframe().f_code.co_name  # text = 'prepare_data_set_types'
    if is_debug:
        print(f'{'-' * 100}\nВыполняется функция: {text} ...')
    else:
        print(f'{'-' * 100}\nВыполняется функция: {text} ...')

    # Вывод информации о незаполненных столбцах и % пропусков в данных
    # print_info_about_missing(a_df)

    columns_to_drop = ['device_model']
    return a_df.drop(columns_to_drop, axis=1, errors='ignore')


# -----------------------------------------------------------------------------------------------------------------
# Преобразования типов данных
def prepare_data_set_types(a_df, is_debug=False):
    import pandas as pd
    import sys

    text = sys._getframe().f_code.co_name  # text = 'prepare_data_set_types'
    if is_debug:
        print(f'{'-' * 100}\nВыполняется функция: {text} ...')
    else:
        print(f'{'-' * 100}\nВыполняется функция: {text} ...')

    # дата и время - тип даtetime
    a_df['visit_date'] = pd.to_datetime(a_df['visit_date'], errors='coerce')
    a_df['visit_time'] = pd.to_datetime(a_df['visit_time'], format='%H:%M:%S', errors='coerce')

    return a_df


# -----------------------------------------------------------------------------------------------------------------
# Заполнение "проблемных" значений признаков
class ValueReplacer(BaseEstimator, TransformerMixin):

    def __init__(self, is_debug=False):
        self.brand_mode_ = None
        self.browser_mode_ = None
        self.brand_replace_values_ = ['', '(not set)', 0]
        self.columns_ = None  # Будет установлено в fit
        self.is_debug = is_debug
        self._sklearn_output_config = {"transform": "default"}

    def set_debug(self, is_debug):
        self.is_debug = is_debug
        return self

    def set_output(self, *, transform=None):
        self._sklearn_output_config = {"transform": transform}
        return self

    def fit(self, X, y=None):
        # Сохраняем имена столбцов, если X - DataFrame

        X_df = X.copy()

        self.brand_mode_ = X_df['device_brand'].mode()[0]
        self.browser_mode_ = X_df['device_browser'].mode()[0]

        X['device_brand'] = X['device_os'].fillna(self.brand_mode_)
        X['device_browser'] = X['device_browser'].fillna(self.brand_mode_)

        return self

    def transform(self, X):
        if self.is_debug:
            print('-' * 100 + '\nВыполняется ValueReplacer ...')
            print(f"На входе = {X.shape}; cтолбцы:\n", X.columns.tolist())
        else:
            print('-' * 100 + '\nВыполняется ValueReplacer ...')

        X = X.copy()

        X['device_brand'] = X['device_os'].fillna(self.brand_mode_)
        X['device_browser'] = X['device_browser'].fillna(self.brand_mode_)

        # 1. Замена проблемных значений в device_brand
        if 'device_brand' in X.columns:
            X.loc[X['device_brand'].isin(self.brand_replace_values_), 'device_brand'] = 'other'

        # 2. Замена (not set) в device_browser
        if 'device_browser' in X.columns:
            X.loc[X['device_browser'] == '(not set)', 'device_browser'] = self.browser_mode_

        # 3. Замена utm_medium
        if 'utm_medium' in X.columns:
            X.loc[(X['utm_medium'] == '(none)') | (X['utm_medium'] == '(not set)'), 'utm_medium'] = 'other'

        if self.is_debug:
            print(f"На выходе = {X.shape}; cтолбцы:\n", X.columns.tolist())

        return X


# -----------------------------------------------------------------------------------------------------------------
# Заполнение "проблемных" значений для device_os
class DeviceOSImputer(BaseEstimator, TransformerMixin):

    def __init__(self, is_debug=False):
        self.brand_os_map_ = {}  # Будет хранить топ-1 OS для каждого бренда
        self.global_mode_ = None  # Глобальная мода для резервной замены
        self.is_debug = is_debug
        self._sklearn_output_config = {"transform": "default"}

    def set_debug(self, is_debug):
        self.is_debug = is_debug
        return self

    def set_output(self, *, transform=None):
        self._sklearn_output_config = {"transform": transform}
        return self

    def fit(self, X, y=None):

        X = X.copy()

        # Заменяем пропуски на маркер
        X['device_os'] = X['device_os'].fillna('(not set)')

        # Создаем таблицу бренд -> топ-1 OS (исключая маркер '(not set)')
        valid_os = X[X['device_os'] != '(not set)']

        # Группируем и находим топ-1 OS для каждого бренда
        self.brand_os_map_ = (
            valid_os.groupby('device_brand')['device_os']
            .apply(lambda x: x.value_counts().index[0])
            .to_dict()
        )

        # Запоминаем глобальную моду
        self.global_mode_ = valid_os['device_os'].mode()[0] if not valid_os['device_os'].empty else 'other'

        return self

    def transform(self, X):

        if self.is_debug:
            print('-' * 100 + '\nВыполняется DeviceOSImputer ...')
            print(f"На входе = {X.shape}; cтолбцы:\n", X.columns.tolist())
        else:
            print('-' * 100 + '\nВыполняется DeviceOSImputer ...')

        X = X.copy()

        # 1. Первичная замена пропусков на маркер
        X['device_os'] = X['device_os'].fillna('(not set)')

        # 2. Замена маркера на топ-1 OS для бренда
        mask = X['device_os'] == '(not set)'
        X.loc[mask, 'device_os'] = X.loc[mask, 'device_brand'].map(self.brand_os_map_)

        # 3. Замена оставшихся пропусков на глобальную моду
        X['device_os'] = X['device_os'].fillna(self.global_mode_)

        # 4. Замена 0 и прочих невалидных значений
        X.loc[X['device_os'].isin([0, '', '(not set)']), 'device_os'] = self.global_mode_

        if self.is_debug:
            print(f"На выходе = {X.shape}; cтолбцы:\n", X.columns.tolist())
        return X


# -----------------------------------------------------------------------------------------------------------------
# Заполнение "проблемных" значений для гео данных
class GeoDataImputer(BaseEstimator, TransformerMixin):

    def __init__(self, is_debug=False):
        self.top_country_ = None
        self.country_city_map_ = {}
        self.city_mode_ = None
        self.is_debug = is_debug
        self._sklearn_output_config = {"transform": "default"}

    def set_debug(self, is_debug):
        self.is_debug = is_debug
        return self

    def set_output(self, *, transform=None):
        self._sklearn_output_config = {"transform": transform}
        return self

    def fit(self, X, y=None):
        X = X.copy()

        # 1. Определяем топ-1 страну
        country_counts = X.groupby('geo_country')['session_id'].count()
        self.top_country_ = country_counts.idxmax()

        # 2. Создаем маппинг страна -> топ-1 город
        valid_cities = X[X['geo_city'] != '(not set)']
        self.country_city_map_ = (
            valid_cities.groupby('geo_country')['geo_city']
            .apply(lambda x: x.value_counts().index[0])
            .to_dict()
        )

        # 3. Запоминаем глобальную моду для городов
        self.city_mode_ = valid_cities['geo_city'].mode()[0] if not valid_cities['geo_city'].empty else 'other'

        return self

    def transform(self, X):
        if (self.is_debug):
            print('-' * 100 + '\nВыполняется GeoDataImputer ...')
            print(f"На входе = {X.shape}; cтолбцы:\n", X.columns.tolist())
        else:
            print('-' * 100 + '\nВыполняется GeoDataImputer ...')

        X = X.copy()

        # 1. Обработка geo_country
        X.loc[X['geo_country'] == '(not set)', 'geo_country'] = (
            self.top_country_ if self.top_country_ != '(not set)' else 'other')

        # 2. Создаем временный ключ для merge
        X['key_country_city'] = X['geo_country'] + '_(not set)'

        # 3. Замена geo_city
        # Сначала заменяем '(not set)' на топ-1 город для страны
        mask = X['geo_city'] == '(not set)'
        X.loc[mask, 'geo_city'] = X.loc[mask, 'geo_country'].map(self.country_city_map_)

        # Затем заполняем оставшиеся пропуски
        X['geo_city'] = X['geo_city'].fillna(self.city_mode_)

        # 4. Удаляем вспомогательные колонки
        X.drop(columns=['key_country_city'], inplace=True, errors='ignore')

        # 5. Обновляем ключ (если нужно сохранить)
        X['key_country_city'] = X['geo_country'] + '_' + X['geo_city']

        if self.is_debug:
            print(f"На выходе = {X.shape}; cтолбцы:\n", X.columns.tolist())

        # X = X.reset_index(drop=True)
        return X


# -----------------------------------------------------------------------------------------------------------------
# Создание новых признаков
class FeatureGenerator(BaseEstimator, TransformerMixin):

    def __init__(self, is_debug=False):
        self.is_debug = is_debug
        self.screen_res_mode_ = None
        self.aspect_ratio_mode_ = None
        self.client_first_visit_ = None
        self._sklearn_output_config = {"transform": "default"}

    def set_debug(self, is_debug):
        self.is_debug = is_debug
        return self

    def set_output(self, *, transform=None):
        self._sklearn_output_config = {"transform": transform}
        return self

    def fit(self, X, y=None):
        # Запоминаем значения для transform
        self.screen_res_mode_ = X['device_screen_resolution'].mode()[0]
        self.aspect_ratio_mode_ = X['device_screen_aspect_ratio'].mode()[
            0] if 'device_screen_aspect_ratio' in X.columns else 1.0
        self.client_first_visit_ = X.groupby('client_id')['visit_date'].min()
        return self

    def transform(self, X):
        import pandas as pd

        if self.is_debug:
            print('-' * 100 + '\nВыполняется FeatureGenerator ...')
            print(f"На входе = {X.shape}; cтолбцы:\n", X.columns.tolist())
        else:
            print('-' * 100 + '\nВыполняется FeatureGenerator ...')

        X = X.copy()  # .reset_index(drop=True)

        if self.is_debug:
            print(f"\nИндексы на входе: {X.index[:5]}")

        # 1. Временные признаки
        X['visit_day_of_month'] = X['visit_date'].dt.day
        X['visit_day_of_week'] = X['visit_date'].dt.dayofweek + 1
        X['visit_hour'] = X['visit_time'].dt.hour

        # 2. Время суток
        time_bins = [0, 5, 9, 12, 16, 20, 24]
        time_labels = ['late_night', 'early_morning', 'morning', 'afternoon', 'evening', 'night']
        X['visit_time_of_day'] = pd.cut(X['visit_hour'], bins=time_bins, labels=time_labels, right=False)

        # 3. Пиковые часы (оптимизированная версия)
        peak_hours = {11, 12, 13, 14, 15, 16}
        X['visit_is_peak_hour'] = X['visit_hour'].apply(lambda x: 1 if x in peak_hours else 0)

        # 4. Обработка разрешения экрана
        X['device_screen_resolution'] = X['device_screen_resolution'].replace('(not set)', self.screen_res_mode_)

        # 5. Размеры экрана (векторизированная версия)
        screen_size = X['device_screen_resolution'].str.extract(r'(\d+)x(\d+)').astype(float)
        X[['device_screen_width', 'device_screen_height']] = (screen_size / 100).round() * 100

        # 6. Соотношение сторон (с обработкой крайних значений)
        X['device_screen_aspect_ratio'] = (X['device_screen_width'] / X['device_screen_height']).clip(0.5, 3)
        X['device_screen_aspect_ratio'] = X['device_screen_aspect_ratio'].fillna(self.aspect_ratio_mode_)

        # 7. Дни с первого визита (оптимизированная версия)
        original_index = X.index  # Сохраняем исходные индексы

        X = X.merge(
            self.client_first_visit_.rename('visit_date_min').reset_index(),
            on='client_id',
            how='left'
        )

        X.index = original_index  # Восстанавливаем исходные индексы

        X['visit_days_since_first_visit'] = (X['visit_date'] - X['visit_date_min']).dt.days
        X['visit_days_since_first_visit'] = X['visit_days_since_first_visit'].fillna(0)

        X.drop(columns=['visit_date_min'], inplace=True)

        # 8. Признаки трафика
        organic_sources = {'(none)', 'organic', 'referral'}
        social_sources = {
            'QxAxdyPLuQMEcrdZWdWb', 'MvfHsxITijuriZxsqZqt', 'ISrKoXQCxqqYvAZICvjs',
            'IZEXUFLARCUMynmHNBGo', 'PlbkrSYoHuZBWfYjYnfw', 'gVRrcxiDQubJiljoTbGm'
        }

        X['utm_is_organic_traffic'] = X['utm_medium'].isin(organic_sources).astype(int)
        X['utm_is_advertising_in_social_networks'] = X['utm_source'].isin(social_sources).astype(int)

        if self.is_debug:
            print(f"На выходе = {X.shape}; cтолбцы:\n", X.columns.tolist())
            print(f"\nИндексы на выходе: {X.index[:5]}")

        # X = X.reset_index(drop=True)  # Сброс индексов
        return X


# -----------------------------------------------------------------------------------------------------------------
# Создание новых признаков
# Новые признаки - Географические координаты места - geo_latitude (Широта) и geo_longitude (Долгота)
# определяются по стране и городу
# координаты были получены с помощьюю библиотеки geopy
# полученные координаты были сохранены в готовом виде в csv-файле locations.csv
# здесь по умолчанию задан режим source_location = 'csv_file', т.е. координаты брать из файла (для экономии времени)
class GeoFeaturesGenerator(BaseEstimator, TransformerMixin):
    import pandas as pd

    def __init__(self, locations_file='models/locations.csv', target_city_coords=(55.7558, 37.6176), is_debug=False):
        self.locations_file = locations_file
        self.target_city_coords = target_city_coords
        self.is_debug = is_debug
        self.geo_data_ = None
        self.median_cache_ = {}
        self._sklearn_output_config = {"transform": "default"}

        # Проверка доступности geopy при инициализации
        try:
            from geopy.distance import geodesic
            self._geodesic = geodesic
        except ImportError:
            raise ImportError("Для работы GeoFeaturesGenerator требуется установить geopy: pip install geopy")

    def set_debug(self, is_debug):
        self.is_debug = is_debug
        return self

    def set_output(self, *, transform=None):
        self._sklearn_output_config = {"transform": transform}
        return self

    def _load_geo_data(self):
        import pandas as pd

        """Загрузка геоданных из CSV"""
        geo_data = pd.read_csv(self.locations_file)
        geo_data.rename(columns={
            'latitude': 'geo_latitude',
            'longitude': 'geo_longitude'
        }, inplace=True)
        return geo_data

    def _calculate_distances(self, df):
        from geopy.distance import geodesic

        """Расчет расстояний до целевого города"""
        mask = df['geo_latitude'].notna()
        df.loc[mask, 'geo_distance_to_city_km'] = df[mask].apply(
            # lambda row: self._geodesic((row['geo_latitude'], row['geo_longitude']), self.target_city_coords).km, axis=1)
            lambda row: geodesic((row['geo_latitude'], row['geo_longitude']), self.target_city_coords).km, axis=1)
        return df

    def _fill_geo_missing(self, df):
        # Заполнение пропущенных геоданных
        geo_cols = ['geo_latitude', 'geo_longitude', 'geo_distance_to_city_km']

        # Шаг 1: По странам и городам
        country_city_medians = df.groupby(['geo_country', 'geo_city'])[geo_cols].transform('median')
        df[geo_cols] = df[geo_cols].fillna(country_city_medians)

        # Шаг 2: По городам
        city_medians = df.groupby('geo_city')[geo_cols].transform('median')
        df[geo_cols] = df[geo_cols].fillna(city_medians)

        # Шаг 3: По странам
        country_medians = df.groupby('geo_country')[geo_cols].transform('median')
        df[geo_cols] = df[geo_cols].fillna(country_medians)

        # Шаг 4: Глобальные медианы
        global_medians = df[geo_cols].median()
        df[geo_cols] = df[geo_cols].fillna(global_medians)

        return df

    def fit(self, X, y=None):
        # Загружаем и подготавливаем справочник геоданных
        self.geo_data_ = self._load_geo_data()
        self.geo_data_ = self._calculate_distances(self.geo_data_)

        # Кэшируем медианы для быстрого заполнения
        numeric_cols = ['geo_latitude', 'geo_longitude', 'geo_distance_to_city_km']
        self.median_cache_ = {
            'country_city': self.geo_data_.groupby(['geo_country', 'geo_city'])[numeric_cols].median(),
            'city': self.geo_data_.groupby('geo_city')[numeric_cols].median(),
            'country': self.geo_data_.groupby('geo_country')[numeric_cols].median(),
            'global': self.geo_data_[numeric_cols].median()
        }

        return self

    def transform(self, X):
        if self.is_debug:
            print('-' * 100 + '\nВыполняется GeoFeaturesGenerator ...')
            print(f"На входе = {X.shape}; cтолбцы:\n", X.columns.tolist())
        else:
            print('-' * 100 + '\nВыполняется GeoFeaturesGenerator ...')

        X = X.copy()  # .reset_index(drop=True)

        if self.is_debug:
            print(f"\nИндексы на входе: {X.index[:5]}")

        # Объединяем с геоданными
        original_index = X.index  # Сохраняем исходные индексы

        X = X.merge(
            self.geo_data_[['geo_country', 'geo_city', 'geo_latitude', 'geo_longitude', 'geo_distance_to_city_km']],
            on=['geo_country', 'geo_city'],
            how='left'
        )  # .reset_index(drop=True)

        X.index = original_index  # Восстанавливаем исходные индексы

        # Заполняем пропуски
        X = self._fill_geo_missing(X)

        if self.is_debug:
            print(f"На выходе = {X.shape}; cтолбцы:\n", X.columns.tolist())
            print(f"\nИндексы на выходе: {X.index[:5]}")

        # X = X.reset_index(drop=True)  # Сброс индексов
        return X


# -----------------------------------------------------------------------------------------------------------------
# Замена редких значений признаков
class ExactRareLabelEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, threshold_prc=1, rare_label='rare_value', is_debug=False):
        self.threshold_prc = threshold_prc  # процент для определения редких значений (1%)
        self.rare_label = rare_label
        self.rare_categories_ = {}  # будет хранить редкие категории для каждого столбца
        self.is_debug = is_debug
        self._sklearn_output_config = {"transform": "default"}

    def set_debug(self, is_debug):
        self.is_debug = is_debug
        return self

    def set_output(self, *, transform=None):
        self._sklearn_output_config = {"transform": transform}
        return self

    def fit(self, X, y):
        self.rare_categories_ = {}

        categorical_features = ['utm_source', 'utm_medium', 'utm_campaign', 'utm_adcontent', 'utm_keyword',
                                'device_category', 'device_os', 'device_brand', 'device_browser']

        # Работаем только с колонками, которые есть в данных
        cols_to_process = [col for col in categorical_features if col in X.columns]

        for col in cols_to_process:
            # Ваш оригинальный алгоритм:
            value_counts = X[col].value_counts()
            cnt_unique = X[col].nunique()
            threshold = max(1, int(cnt_unique * (self.threshold_prc / 100)))

            # Категории, где target_event_action всегда 0
            zero_target_mask = y.groupby(X[col]).sum() == 0

            # Редкие категории (по частоте И по target)
            rare_values = value_counts[value_counts.index.isin(zero_target_mask[zero_target_mask].index) &
                                       (value_counts < threshold)
                                       ].index

            self.rare_categories_[col] = rare_values.tolist()

        return self

    def transform(self, X):
        if self.is_debug:
            print('-' * 100 + '\nВыполняется ExactRareLabelEncoder ...')
            print(f"На входе = {X.shape}; cтолбцы:\n", X.columns.tolist())
            print(f"\nИндексы на входе: {X.index[:5]}")
        else:
            print('-' * 100 + '\nВыполняется ExactRareLabelEncoder ...')

        X = X.copy()
        for col, rare_values in self.rare_categories_.items():
            if col in X.columns:
                X[col] = X[col].apply(lambda x: self.rare_label if x in rare_values else x)

        if self.is_debug:
            print(f"На выходе = {X.shape}; cтолбцы:\n", X.columns.tolist())
            print(f"\nИндексы на выходе: {X.index[:5]}")

        return X


# -----------------------------------------------------------------------------------------------------------------
# Категориальные признаки
# Признаки со средней кардинальностью (10-100 уникальных значений)
# метод Частотное ординальное кодирование (Frequency encoding)
# Значение каждой категории заменяется на частоту встречаемости этой категории в наборе данных
class FrequencyEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, suffix='_encoded', is_debug=False):
        self.suffix = suffix
        self.is_debug = is_debug
        self.frequency_maps_ = {}
        self.feature_names_in_ = None
        self.feature_names_out_ = None
        self._sklearn_output_config = {"transform": "default"}

    def set_debug(self, is_debug):
        self.is_debug = is_debug
        return self

    def set_output(self, *, transform=None):
        self._sklearn_output_config = {"transform": transform}
        return self

    def fit(self, X, y=None):
        import pandas as pd

        # Преобразуем в DataFrame, если это numpy array
        if not hasattr(X, 'columns'):
            X = pd.DataFrame(X)

        # Сохраняем имена фичей для get_feature_names_out
        self.feature_names_in_ = X.columns.tolist()

        for col in X.columns:
            self.frequency_maps_[col] = X[col].value_counts(normalize=True).to_dict()

            # if self.is_debug:
            #     print(f"Frequencies for {col}:")
            #     print(self.frequency_maps_[col])

        # Генерируем имена выходных фичей
        self.feature_names_out_ = [f"{col}{self.suffix}" for col in X.columns]

        return self

    def transform(self, X):
        import pandas as pd

        if self.is_debug:
            print('-' * 100 + '\nВыполняется FrequencyEncoder ...')
            print(f"На входе = {X.shape}; cтолбцы:\n", X.columns.tolist())
        else:
            print('-' * 100 + '\nВыполняется FrequencyEncoder ...')

        # Преобразуем в DataFrame, если это numpy array
        if not hasattr(X, 'columns'):
            X = pd.DataFrame(X, columns=self.feature_names_in_)

        X = X.copy()
        result = pd.DataFrame()

        for col in X.columns:
            new_col = f"{col}{self.suffix}"
            # result[new_col] = X[col].map(self.frequency_maps_[col])
            result[new_col] = X[col].map(self.frequency_maps_[col]).fillna(0)

        if self.is_debug:
            print(f"На выходе = {result.shape}; cтолбцы:\n", result.columns.tolist())

        return result

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_


# -----------------------------------------------------------------------------------------------------------------
# Удаление основных и вспомогательных столбцов после всех преобразований
def drop_unnecessary_columns(a_df, is_debug=False):
    # import pandas as pd
    from datetime import datetime
    import sys

    text = sys._getframe().f_code.co_name  # text = 'drop_unnecessary_columns'
    if is_debug:
        print(f'{'-' * 100}\nВыполняется функция: {text} ...')
        print('Кол-во столбцов =', len(a_df.columns), '\nСписок столбцов =', a_df.columns)
        print(f"На входе = {a_df.shape}; cтолбцы:\n", a_df.columns.tolist())
    else:
        print(f'{'-' * 100}\nВыполняется функция: {text} ...')

    # оригинальные и вспомогательные столбцы, ненужные после создания новых признаков
    list_columns_original = ['visit_date',
                             'visit_time',
                             'session_id',
                             'client_id',
                             'geo_country',
                             'geo_city',
                             'key_country_city',
                             'visit_date_min',
                             'device_model',
                             'device_screen_resolution']

    list_columns_medium_values = ['device_browser',
                                  'utm_medium']

    columns_to_drop = list_columns_original + list_columns_medium_values
    a_df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    # Вывод информации о незаполненных столбцах и % пропусков в данных
    # print_info_about_missing(a_df)

    if is_debug:
        print(' ...Это была последняя функция в pipeline из списка FunctionTransformer')
        print(f"На выходе = {a_df.shape}; cтолбцы:\n", a_df.columns.tolist())

    return a_df


# -----------------------------------------------------------------------------------------------------------------
# Последний пустой трансформер
def last_step_transformer(a_df, is_debug=False):
    import sys

    text = sys._getframe().f_code.co_name  # text = 'last_transformer'
    if is_debug:
        print(f'{'-' * 100}\nВыполняется функция: {text} ...')
        print('Кол-во столбцов =', len(a_df.columns), '\nСписок столбцов =', a_df.columns)
        print(f"На входе = {a_df.shape}; cтолбцы:\n", a_df.columns.tolist())
    else:
        print(f'{'-' * 100}\nВыполняется функция: {text} ...')

    return a_df


# ---------------------------------------------------------------------------------------------------
# Подготовка и разделение данных
def prepare_and_split_data(a_df, coef_classes=2, is_debug=True):
    # import pandas as pd

    if is_debug:
        text = sys._getframe().f_code.co_name        # text = 'prepare_and_split_data'
        print(f'{'-' * 100}\nВыполняется функция: {text}...')
        # print('Кол-во столбцов =', len(a_df.columns), '\nСписок столбцов =', a_df.columns)

    x_all_data = a_df.drop(['target_event_action'], axis=1)
    y_all_data = a_df['target_event_action']
    # ---------------------------------------------------------------------------------------------------
    # Разделение датасета на train и test
    if coef_classes != 0:  # начальный вариант - используется весь датасет

        # x = a_df.drop(['target_event_action'], axis=1)
        # y = a_df['target_event_action']
        x = x_all_data.copy()
        y = y_all_data.copy()
    else:  # уменьшение размера класса 0 (undersampling)

        df_class_0 = a_df[a_df['target_event_action'] == 0]
        df_class_1 = a_df[a_df['target_event_action'] == 1]

        # Undersampling: оставляю запсией в {coef_size_class} раз больше, чем класс 1; соотношение (1 : coef_size_class)
        n_class_1 = len(df_class_1)
        coef_size_class = coef_classes
        df_class_0_sampled = df_class_0.sample(n=coef_size_class * n_class_1, random_state=42)

        # Собираю сбалансированный датасет
        df_balanced = pd.concat([df_class_0_sampled, df_class_1])
        print()
        print('Размерности классов ДО    уменьшения класса 0:\t df_class_0  =', df_class_0.shape, '\tdf_class_1 =',
              df_class_1.shape)
        print('Размерности классов ПОСЛЕ уменьшения класса 0:\t df_class_0  =', df_class_0_sampled.shape,
              '\tdf_class_1 =', df_class_1.shape)
        print('Размерности сбалансированного датасета:\t\t df_balanced =', df_balanced.shape)

        x = df_balanced.drop('target_event_action', axis=1)
        y = df_balanced['target_event_action']

    # ---------------------------------------------------------------------------------------------------
    # train_test_split - Разделение датасета на train и test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    print('Размеры датасетов:')
    print(f'\t x       = {x.shape};      \t   y       = {x.shape}' +
          f'\n\t x_train = {x_train.shape};\t   y_train = {y_train.shape}' +
          f'\n\t x_test  = {x_test.shape}; \t   y_test  = {y_test.shape}')

    return x_all_data, y_all_data, x, y, x_train, x_test, y_train, y_test


# -----------------------------------------------------------------------------------------------------------------
# 8. Обучение финальной модели на всех данных
# def train_final_model(best_model_name, best_params, x, y, preprocessor, is_debug=True):
#
#     if is_debug:
#         print_debug_info(sys._getframe().f_code.co_name)
#
#     models = initialize_models(y)
#     model  = models[best_model_name]['model']
#
#     # Обновление параметров модели
#     if best_params:
#         model.set_params(**best_params)
#
#     # Создание pipeline
#     pipeline = Pipeline(steps=[
#                                 ('preprocessor', preprocessor),
#                                 ('classifier', model)
#                             ])
#     # Обучение
#     pipeline.fit(x, y)
#
#     return pipeline

# -----------------------------------------------------------------------------------------------------------------
# 9. Сохранение модели
def save_model(pipe_model, filename='model_prediction_sber.pkl', roc_auc=0.0, is_debug=True):

    if is_debug:
        print_debug_info(sys._getframe().f_code.co_name)

    s_folder = 'models/'
    filename = f'{s_folder}{filename}'

    with open(filename, 'wb') as dill_file:
        dill.dump({
                    'model': pipe_model,
                    'metadata': {
                                    'name':    'SberAutoPodpiska: prediction model',
                                    'author':  'Andrei Mishenkov',
                                    'version': 1.0,
                                    'date':    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                    'type':    type(pipe_model.named_steps["classifier"]).__name__,
                                    'roc_auc': roc_auc
                                }
                    }, dill_file)

    print(f"Model saved as file: {filename}")


# ------------------------------------------------------------------------------------------------------------
# Создание pipeline
def create_preprocessor():
    # ------------------------------------------------------------------------------------------------------------
    columns_other = ['utm_keyword', 'utm_adcontent', 'utm_campaign']
    columns_mode = ['utm_source', 'device_brand', 'device_browser', 'geo_city']

    list_columns_many_values = ['utm_keyword',
                                'utm_campaign',
                                'utm_source',
                                'utm_adcontent',
                                'device_brand']

    list_columns_few_values = ['device_os',
                               'device_category',
                               'visit_time_of_day']

    list_columns_for_rare_values = [
        'utm_source', 'utm_medium', 'utm_campaign', 'utm_adcontent', 'utm_keyword',
        'device_category', 'device_os', 'device_brand', 'device_browser'
    ]

    # список исходных числовых столбцов
    list_columns_numeric = ['visit_number',
                            'visit_day_of_month',
                            'visit_day_of_week',
                            'visit_hour',
                            'visit_days_since_first_visit',
                            'device_screen_width',
                            'device_screen_height',
                            'device_screen_aspect_ratio',
                            'geo_latitude',
                            'geo_longitude',
                            'geo_distance_to_city_km']

    # ---------------------------------------------------------------------------------------------------------
    # замена значений (простые)
    column_transformer_imputers_simple = ColumnTransformer([
        ('imputer_const', SimpleImputer(strategy='constant', fill_value='other'),
         ['utm_keyword', 'utm_adcontent', 'utm_campaign']),
        ('imputer_mode', SimpleImputer(strategy='most_frequent'), ['utm_source', 'geo_city']),
        ('value_replacer', ValueReplacer(is_debug=False),
         ['device_brand', 'device_browser', 'device_os', 'utm_medium']),
    ],
        remainder='passthrough',
        verbose_feature_names_out=False
    ).set_output(transform="pandas")

    # ---------------------------------------------------------------------------------------------------------
    # замена значений (сложные)
    column_transformer_imputers_composite = ColumnTransformer([
        ('device_os_imputer', DeviceOSImputer(is_debug=False), ['device_os', 'device_brand']),
        ('geo_imputer', GeoDataImputer(is_debug=False), ['geo_country', 'geo_city', 'session_id'])
    ],
        remainder='passthrough',
        verbose_feature_names_out=False
    ).set_output(transform="pandas")

    # ---------------------------------------------------------------------------------------------------------
    # создание новых признаков
    column_transformer_generaters_features = ColumnTransformer([
        ('feature_generator', FeatureGenerator(is_debug=False),
         ['device_screen_resolution', 'visit_date', 'visit_time', 'utm_medium', 'utm_source', 'client_id']),
        ('geo_features', GeoFeaturesGenerator(locations_file='models/locations.csv', is_debug=False),
         ['geo_country', 'geo_city'])
    ],
        remainder='passthrough',
        verbose_feature_names_out=False
    ).set_output(transform="pandas")

    # ---------------------------------------------------------------------------------------------------------
    # преобразование редких значенй признаков
    column_transformer_rare_values = ColumnTransformer([
        ('rare_values', ExactRareLabelEncoder(threshold_prc=1.0, is_debug=False), list_columns_for_rare_values)
    ],
        remainder='passthrough',
        verbose_feature_names_out=False
    ).set_output(transform="pandas")

    # ---------------------------------------------------------------------------------------------------------
    # преобразование частонное кодирование
    column_transformer_frequency_encoder = ColumnTransformer([
        ('frequency_encoder', FrequencyEncoder(suffix='_freq', is_debug=False), ['device_browser', 'utm_medium'])
    ],
        remainder='passthrough',
        verbose_feature_names_out=False
    ).set_output(transform="pandas")

    # преобразование столбцов
    column_transformer_target_onehot_scaler = ColumnTransformer(
        transformers=[
            ('target_encoder_1', TargetEncoder(), list_columns_many_values),
            ('onehot_encoder_2', OneHotEncoder(handle_unknown='infrequent_if_exist', sparse_output=False),
             list_columns_few_values),
            ('standard_scaler_3', StandardScaler(), list_columns_numeric)
        ],
        remainder='passthrough',
        verbose_feature_names_out=False
    ).set_output(transform="pandas")

    # основной pipeline
    preprocessor = Pipeline(steps=[
        ('drop_columns_first', FunctionTransformer(drop_columns_first, validate=False)),
        ('prepare_types', FunctionTransformer(prepare_data_set_types, validate=False)),

        ('column_transformer_imputers_simple', column_transformer_imputers_simple),
        ('column_transformer_imputers_composite', column_transformer_imputers_composite),

        ('column_transformer_generaters_features', column_transformer_generaters_features),
        ('column_transformer_rare_values', column_transformer_rare_values),
        ('column_transformer_frequency_encoder', column_transformer_frequency_encoder),

        ('drop_columns', FunctionTransformer(drop_unnecessary_columns, validate=False)),
        ('column_transformer_target_onehot_scaler', column_transformer_target_onehot_scaler),

        ('last_step_transformer', FunctionTransformer(last_step_transformer, validate=False)),
    ],
        verbose=False)

    return preprocessor

# ------------------------------------------------------------------------------------------------------------
# Инициализация моделей
def init_models(a_is_debug=False):
    if not a_is_debug:
        models_list = [
                        ('LogisticRegression', LogisticRegression(
                                                                    # class_weight=class_weights,
                                                                    class_weight='balanced',
                                                                    random_state=RANDOM_STATE,
                                                                    max_iter=1000
                                                                )),
                        ('RandomForest', RandomForestClassifier(
                                                                    class_weight='balanced',
                                                                    n_estimators=100,
                                                                    max_depth=10,
                                                                    min_samples_split=10,
                                                                    min_samples_leaf=5,
                                                                    max_features="sqrt",
                                                                    random_state=RANDOM_STATE,
                                                                    n_jobs=-1
                                                                )),
                        ('MLPClassifier', MLPClassifier(
                                                            random_state=RANDOM_STATE,
                                                            early_stopping=True
                                                        )),
                        ('XGBoost', XGBClassifier(
                                                    scale_pos_weight=37,  # так как дисбаланс 37:1
                                                    random_state=RANDOM_STATE,
                                                    n_jobs=-1,
                                                    eval_metric='auc'
                                                )),
                        ('LightGBM', LGBMClassifier(
                                                        boosting_type='gbdt',
                                                        class_weight='balanced',
                                                        colsample_bytree=1.0,
                                                        importance_type='split',
                                                        learning_rate=0.1,
                                                        max_depth=-1,
                                                        min_child_samples=20,
                                                        min_child_weight=0.001,
                                                        min_split_gain=0.0,
                                                        n_estimators=100,
                                                        num_leaves=50,
                                                        objective=None,
                                                        reg_alpha=0.0,
                                                        reg_lambda=0.0,
                                                        subsample=1.0,
                                                        subsample_for_bin=200000,
                                                        subsample_freq=0,
                                                        scale_pos_weight=37,
                                                        feature_fraction=0.9,
                                                        n_jobs=-1,
                                                        verbose=-1,
                                                        random_state=RANDOM_STATE
                                                    )),
                        ('CatBoost', CatBoostClassifier(
                                                            scale_pos_weight=37,
                                                            random_state=RANDOM_STATE,
                                                            verbose=0,
                                                            thread_count=-1
                                                        ))
                    ]
    else:
        models_list = [
                        ('LightGBM', LGBMClassifier(
                                                        force_row_wise=True,
                                                        scale_pos_weight=37,
                                                        # class_weight='balanced',
                                                        random_state=RANDOM_STATE,
                                                        verbose=-1,
                                                        n_jobs=-1))
                                                ]
    return models_list

# -----------------------------------------------------------------------------------------------------------------
# Основной блок
def main():

    print('-' * 100)
    dt_finish = datetime.now()
    dt_duration = dt_finish - dt_start

    get_duration_time(dt_start)
    print('-' * 100)
    print('creating PipeLine...')

    # ------------------------------------------------------------------------------------------------------------
    # загрузка исходных данных
    # x, y = read_source_data()
    df_sessions = read_source_data()

    # ------------------------------------------------------------------------------------------------------------
    # undersampling; разделение данных на train и test

    # часть датасета (класс 0 в соотношении 2:1 с классом 1)
    x_all_data, y_all_data, x, y, x_train, x_test, y_train, y_test = prepare_and_split_data(df_sessions, coef_classes=2)

    # весь датасет (без сэмплирования)
    # x_all_data, y_all_data, x, y, x_train, x_test, y_train, y_test = prepare_and_split_data(df_sessions, coef_classes=0)

    # Создание препроцессора pipeline
    preprocessor = create_preprocessor()

    # ------------------------------------------------------------------------------------------------------------
    # Список моделей для сравнения
    models = init_models(a_is_debug=True)

    pipe_best       = None
    model_best      = None
    model_name_best = ''
    score_best      = 0
    start_time_global = time.time()
    model_best_time = 0

    # перебор моделей в цикле (поиск лучшей модели)
    for model_name, model in models:

        print('-' * 100)
        print(f'model = {model_name} training...')
        start_time = time.time()

        pipe_model = Pipeline( steps=[
            ('preprocessor', preprocessor),
            ('classifier',   model)
        ])

        # кроссвалидация - оценка моделей
        scores = cross_val_score(pipe_model, x, y, cv=2, scoring='roc_auc', n_jobs=-1, error_score='raise')
        # scores = cross_val_score(pipe_model, x_train, y_train, cv=2, scoring='roc_auc', n_jobs=-1, error_score='raise')

        score_mean = scores.mean()
        score_std  = scores.std()
        training_time = round(time.time() - start_time, 2)

        print('-' * 100)
        print(f'model = {model_name};   roc_auc_mean = {score_mean:.4f};   roc_auc_std: {score_std:.4f};   training_time: {training_time:.2f}s')

        # обновление метрик лучшей модели
        if score_mean > score_best:
            pipe_best       = pipe_model
            model_best      = model
            model_name_best = model_name
            score_best      = score_mean
            model_best_time = training_time

    # ------------------------------------------------------------------------------------------------
    # Лучшая модель
    # training_time_global = round(time.time() - start_time_global, 2)

    print('-' * 100)
    print(  f'best model: {type(pipe_best.named_steps["classifier"]).__name__};{' ' * 5}' +
            f'roc_auc best = {score_best:.4f};{' ' * 5}' +
            f'training_time: {model_best_time:.2f}s')

    # ------------------------------------------------------------------------------------------------
    # запись лучшей модели в файл
    # save_model(pipe_best, 'model_prediction_sber_best.pkl', round(score_best, 4))

    #------------------------------------------------------------------------------------------------
    # Финальная модель - Обучение лучшей модели на всех данных
    print(f'final model = {model_name} training...')

    pipe_final = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model_best)
    ])

    # все данные
    pipe_final.fit(x_all_data, y_all_data)
    score_final = roc_auc_score(y, pipe_final.predict_proba(x)[:, 1])
    training_time_global = round(time.time() - start_time_global, 2)

    print('-' * 100)
    print(f'best model: {type(pipe_final.named_steps["classifier"]).__name__};{' ' * 5}' +
          f'roc_auc best = {score_best:.4f};{' ' * 5}' +
          f'roc_auc final = {score_final:.4f}{' ' * 5}' +
          f'training_time_all_models: {training_time_global:.2f}s')

    # ------------------------------------------------------------------------------------------------
    # Запись финальной модели в файл
    # save_model(pipe_final, 'model_prediction_sber_final.pkl', round(score_final, 4))
    save_model(pipe_final, 'model_prediction_sber.pkl', round(score_final, 4))

    print('-' * 100)
    get_duration_time(dt_start)
    print('-' * 100)


# -----------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # main()
    main()