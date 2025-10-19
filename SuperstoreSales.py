
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


# CSV dosyasını oku
df = pd.read_csv("Superstore.csv", encoding='latin1')

# İlk 5 satırı göster
print("İlk 5 satır:")
print(df.head())

# Veri hakkında bilgi
print(df.info())

# Sayısal özet
print("Sayısal özet:")
print(df.describe())

print("Eksik Veri Sayıları:")
print(df.isnull().sum()) 


print("\nKategorik Değişken Dağılımları:")
for col in ['Category', 'Region', 'Segment']:
    print(f"\n{col} sütunu dağılımı:\n{df[col].value_counts()}")


numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
print("\nSayısal sütunlar:", list(numeric_cols))

# Her sayısal sütun için aykırı değerleri bul
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)   # 1. çeyrek (Q1)
    Q3 = df[col].quantile(0.75)   # 3. çeyrek (Q3)
    IQR = Q3 - Q1                 # IQR hesapla

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    # Aykırı değerleri tespit etme
    outliers = df[(df[col] < lower_limit) | (df[col] > upper_limit)]

    print(f"{col} sütununda {outliers.shape[0]} aykırı değer var.")

# Aykırı Değerlerin Görselleştirilmesi
# Profit sütunundaki aykırı değerler
sns.boxplot(x=df["Profit"])
plt.title("\nProfit Değişkeninde Aykırı Değerler")
plt.show()

sns.barplot(x='Category', y='Sales', data=df, estimator=sum)
plt.title("\nKategoriye Göre Toplam Satış")
plt.show()

# tek değişkenli görselleştirme 
sns.histplot(df['Sales'], bins=30, kde=True)
plt.title("Satış Dağılımı")
plt.show()

# çift değişkenli görselleştirme 
sns.scatterplot(x='Sales', y='Profit', data=df)
plt.title("Satış vs Kâr")
plt.show()

#Korelasyon Isı Haritası
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Sayısal Değişkenler Arası Korelasyon ")
plt.show()


num_cols = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df.select_dtypes(include=['object']).columns


# Sayısal sütunlarda ortalama ile doldurdum
for col in num_cols:
    df[col] = df[col].fillna(df[col].mean())

# Kategorik sütunları en sık görülen değerle doldurdum
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])


print("Eksik veriler dolduruldu.")

# Kategorik değişkenleri OneHotEncoder ile dönüştürdüm

print("\nKategorik sütunlar:", list(cat_cols))
print("\nSayısal sütunlar:", list(num_cols))

ohe = OneHotEncoder(drop='first', sparse_output=False)
encoded_data = ohe.fit_transform(df[cat_cols])
encoded_cols = ohe.get_feature_names_out(cat_cols)
encoded_df = pd.DataFrame(encoded_data, columns=encoded_cols)
df_encoded = pd.concat([df[num_cols].reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

print("\nOneHotEncoder ile tüm kategorik sütunlar dönüştürüldü.")

# Sayısal değişkenleri ölçeklendirme(normalizasyon)
scaler = MinMaxScaler()
df_encoded[num_cols] = scaler.fit_transform(df_encoded[num_cols])

print("\nSayısal sütunlar 0-1 aralığına ölçeklendirildi.")
print("\nYeni Veri Seti Bilgileri:")
print(df_encoded.info())

print("\nİlk 5 satır (işlenmiş veri):")
print(df_encoded.head())


# Modelleme (RandomForest)
X = df_encoded.drop(columns=[col for col in df_encoded.columns if "Category" in col])
y = df['Category'] 

# Train-test split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, stratify=y)

# RandomForest modeli
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

# Başarı metrikleri
print("RandomForest Model Sonuçları:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred, average='weighted', zero_division=0):.3f}")
print(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.3f}")
print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.3f}")

print("\nDetaylı rapor:")
print(classification_report(y_test, y_pred))

#Sonuç
print("""\nRandomForest modelinin genel doğruluğu (accuracy) %81 ve oldukça başarılı.
Office Supplies ürünlerini %99 recall değeri ile yüksek doğrulukla tahmin edebiliyor, ancak Furniture ve Technology kategorilerinde recall değerleri sırasıyla %59 ve %46 yani model bazı ürünleri kaçırıyor. 
Precision değerleri ise tüm sınıflarda nispeten yüksek yani modelin yanlış sınıflandırmaları minimum seviyede. Bu durumlar modelin daha çok veri içeren sınıflarda daha iyi performans sergilediğini, küçük sınıflarda ise eksik kaldığını gösteriyor. """)

#Bu çalışmada Superstore veri setini kullanarak veri analizi ve sınıflandırma modeli kurdum. Eksik verileri doldurup, kategorik değişkenleri One-Hot Encoding ile dönüştürdüm ve sayısal değişkenleri ölçeklendirdim. RandomForest modeli ile tahminler yaparak accuracy, precision, recall ve f1-score değerlerini inceledim. Bu süreç, veri ön işlemenin ve uygun model seçiminin sonuçlar üzerindeki etkisini anlamamı sağladı.