from catboost.datasets import titanic
from pathlib import Path
import pandas as pd
pd.set_option('display.max_columns', None)

from ucimlrepo import fetch_ucirepo, list_available_datasets

# загружаем данные
adult = fetch_ucirepo(name='Adult')

# объеденяем признаки и функцию для удобства хранения
data_train_src = pd.concat([adult.data.features, adult.data.targets], sort=False, axis=1)

# обработка данных
# датасет №1 содержит поля income, age, occupation, sex
data_train_1 = data_train_src[['income','age','sex','occupation']].copy()
pd.set_option('display.max_columns', None)
print(data_train_src.info())

# анализ парметров
print(data_train_1.groupby('income').agg({'sex':'count'}))
print(data_train_1.groupby('age').agg({'sex':'count'}))
print(data_train_1.groupby('occupation').agg({'age':'count'}))
print(data_train_1.groupby('sex').agg({'age':'count'}))
print(data_train_1.head())

# сохранение в csv-файл
filepath_train1 = Path('datasets/data_train_1.csv')
filepath_train1.parent.mkdir(parents=True, exist_ok=True)
data_train_1.to_csv(filepath_train1, index=False)

# датасет №2 содержит поля income, age, occupation, sex
# заменим пропуски и установим значение функции 0 - если доход <= 50000 и 1 если > 50000
data_train_2 = data_train_src[['income','age','sex','occupation']].copy()
data_train_2['income'] = data_train_src['income'].apply(lambda x: 0 if x == '<=50K' else (0 if x == '<=50K.' else 1))

# заменим текстовые значения ('male' или 'female') на (0 или 1)
data_train_2['sex'] = data_train_src['sex'].apply(lambda x: 0 if x == 'Male' else 1)

# в признаке "Возраст" пропуски заполним средним значением
data_train_2['age'] = data_train_src['age'].fillna(data_train_src.age.mean())

# в признаке "Occupation" пропуски заменим  "Other"
data_train_2['occupation'] = data_train_src['occupation'].fillna('other')

# сохранение в csv-файл
filepath_train = Path('datasets/data_train_2.csv')
filepath_train.parent.mkdir(parents=True, exist_ok=True)
data_train_2.to_csv(filepath_train, index=False)

# датасет №3 содержит поля income, age, occupation, sex
# селаем датасет с OneHotEncoder параметра Sex
from sklearn.preprocessing import OneHotEncoder
data_train_3 = data_train_src[['income','age','sex','occupation']].copy()
encoder = OneHotEncoder(handle_unknown='ignore')
enc_df = pd.DataFrame(encoder.fit_transform(data_train_3[['sex']]).toarray ())
data_train_3_OneHotEnc = data_train_3.join(enc_df)
print(data_train_3_OneHotEnc.columns)
data_train_3_OneHotEnc = data_train_3_OneHotEnc.rename(columns= {0 : 'Sex_is_male', 1 : 'Sex_is_female'})
print(data_train_3_OneHotEnc.columns)

# сохранение в csv-файл
filepath_train3 = Path('datasets/data_train_3_OneHotEnc.csv')
filepath_train3.parent.mkdir(parents=True, exist_ok=True)
data_train_3_OneHotEnc.to_csv(filepath_train3, index=False)

print('Данные обработаны и сохранены')