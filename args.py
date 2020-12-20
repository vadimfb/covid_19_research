import os

# Cколько дней мы будем использовать в качестве признаков
LIMIT_DAYS = 10

# Константы, контролирующие условия, для создания признаков
USE_LOG = 1
USE_DIFF = 1
USE_DIV = 1

# Используем GPU
USE_GPU = 1

# Количество итераций для бустинга
REQUIRED_ITERATIONS = 5

# Дней предсказания в будущем
DAYS_TO_PREDICT = 7

# Использовать если нужно посмотреть на реальную валидацию
STEP_BACK = None
# STEP_BACK = 7

# Использовать последние данные для стран и регионов
USE_LATEST_DATA_COUNTRY = False
USE_LATEST_DATA_RUS = True

# Дополнительные данные
USE_YANDEX_MOBILITY_DATA = True
USE_SIMPLE_LINEAR_FEATURES = True
USE_INTERPOLATION_FEATURES = True
USE_WEEKDAY_FEATURES = True

# Переменные путей
ROOT_PATH = './'
INPUT_PATH = ROOT_PATH + 'input/'
OUTPUT_PATH = ROOT_PATH + 'modified_data/'
if not os.path.isdir(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)
MODELS_PATH = ROOT_PATH + 'models/'
if not os.path.isdir(MODELS_PATH):
    os.mkdir(MODELS_PATH)
CACHE_PATH = ROOT_PATH + 'cache/'
if not os.path.isdir(CACHE_PATH):
    os.mkdir(CACHE_PATH)
FEATURES_PATH = ROOT_PATH + 'features/'
if not os.path.isdir(FEATURES_PATH):
    os.mkdir(FEATURES_PATH)
HISTORY_FOLDER_PATH = MODELS_PATH + "history/"
if not os.path.isdir(HISTORY_FOLDER_PATH):
    os.mkdir(HISTORY_FOLDER_PATH)
SUBM_PATH = ROOT_PATH + 'subm/'
if not os.path.isdir(SUBM_PATH):
    os.mkdir(SUBM_PATH)