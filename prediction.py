# Import yang dibutuhkan
import pandas as pd
import pickle
from google.colab import files

# Upload file CSV
uploaded = files.upload()

# Membaca file CSV yang diupload
data_pred = pd.read_csv(next(iter(uploaded)))

data_pred = data_pred.drop(columns=['Attrition'], errors='ignore')  # Drop kolom 'Attrition' jika ada

data_pred_cleaned = data_pred.drop(columns=['EmployeeId'], errors='ignore')  # Drop kolom 'EmployeeId' jika ada

# Enccoding Fitur Kategori
encode_pred_df = pd.concat([data_pred_cleaned, pd.get_dummies(data_pred_cleaned['BusinessTravel'], prefix='BusinessTravel')],axis=1)
encode_pred_df = pd.concat([encode_pred_df , pd.get_dummies(encode_pred_df ['Department'], prefix='Department')],axis=1)
encode_pred_df = pd.concat([encode_pred_df , pd.get_dummies(encode_pred_df ['EducationField'], prefix='EducationField')],axis=1)
encode_pred_df = pd.concat([encode_pred_df , pd.get_dummies(encode_pred_df ['Gender'], prefix='Gender')],axis=1)
encode_pred_df = pd.concat([encode_pred_df , pd.get_dummies(encode_pred_df ['JobRole'], prefix='JobRole')],axis=1)
encode_pred_df = pd.concat([encode_pred_df , pd.get_dummies(encode_pred_df ['MaritalStatus'], prefix='MaritalStatus')],axis=1)
encode_pred_df = pd.concat([encode_pred_df , pd.get_dummies(encode_pred_df ['Over18'], prefix='Over18')],axis=1)
encode_pred_df = pd.concat([encode_pred_df , pd.get_dummies(encode_pred_df ['OverTime'], prefix='OverTime')],axis=1)
encode_pred_df.drop(['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime'], axis=1, inplace=True)

# Memastikan data yang diunggah tampak
print(encode_pred_df.head())

# Memuat model XGBoost yang telah dilatih sebelumnya
model_pred = pickle.load(open('/model/xgboost_model.pkl', 'rb'))

# Lakukan prediksi
predictions = model_pred.predict(encode_pred_df)

# Menambahkan kolom 'Predicted Attrition' yang berisi 'Yes' atau 'No'
data_pred['Predicted Attrition'] = ['Yes' if pred == 1 else 'No' for pred in predictions]

# Menampilkan hasil prediksi
print(data_pred[['EmployeeId', 'Predicted Attrition']])  # Tampilkan ID karyawan dan prediksi

# Menyimpan hasil prediksi ke dalam file CSV baru
data_pred.to_csv('/content/predicted_attrition.csv', index=False)

# # Memberikan link untuk mendownload hasil prediksi
# files.download('/content/predicted_attrition.csv')

count = data_pred['Predicted Attrition'].value_counts()
percent = 100*data_pred['Predicted Attrition'].value_counts(normalize=True)
category_column_df = pd.DataFrame({'Jumlah Data':count, 'Persentase':percent.round(1)})
print(category_column_df)
count.plot(kind='bar', title='Parameter Predicted Attrition');