# Data_Analysis_Task
Kaggle Superstore Sales veri seti kullanılarak yapılan mini veri analizi projesi
 
# Veri Seti Hakkında
"Superstore Sales" veri seti, farklı bölgeler ve müşteri segmentlerini kapsayan satış ve ürün performansı verilerini içerir. Bu veriler, satış trendlerini ve müşteri tercihlerini analiz ederek perakende satış süreçlerini incelemek ve iş kararlarını desteklemek için toplanmıştır.

Toplam Kayıt: 9,994

Toplam Sütun: 21

Sayısal Değişkenler:Row ID, Postal Code, Sales, Quantity, Discount, Profit

Kategorik Değişkenler:Order ID, Order Date, Ship Date, Ship Mode, Customer ID, Customer Name, Segment, Country, City, State, Region, Product ID, Category, Sub-Category, Product Name

# Sonuç
RandomForest modeli, genel doğruluk (accuracy) açısından %81 gibi yüksek bir performans sergilemiştir. Office Supplies kategorisindeki ürünler neredeyse tamamen doğru tahmin edilirken, Furniture ve Technology kategorilerinde recall değerleri daha düşüktür yani model bazı ürünleri doğru sınıflandıramamıştır. Bu durum, modelin daha çok veri içeren sınıflarda daha iyi performans sergilediğini, küçük sınıflarda ise eksik kaldığını gösteriyor.

Bu süreçte öğrenilenler:

-Veri ön işleme (eksik verilerin doldurulması, kategorik değişkenlerin kodlanması, sayısal verilerin ölçeklendirilmesi) model performansını doğrudan etkiler.

-Veri dengesizliği, modelin bazı sınıflarda daha zayıf tahmin yapmasına neden olabilir.

-Uygun model seçimi ve hiperparametre ayarlamaları, tahmin başarısını artırmada kritik rol oynar.
