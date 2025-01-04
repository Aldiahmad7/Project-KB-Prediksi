import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('simulasi_kelulusan.csv')

def categorize_ipk(ipk):
    if ipk < 2.5:
        return 0 
    elif ipk < 3.5:
        return 1
    else:
        return 2

def categorize_kehadiran(kehadiran):
    if kehadiran <= 65:
        return 0  
    elif kehadiran <= 85:
        return 1
    else:
        return 2 

def categorize_lama_studi(lama_studi):
    if lama_studi == 1:
        return 0 
    elif lama_studi == 2:
        return 1
    elif lama_studi == 3:
        return 2
    elif lama_studi == 4:
        return 3
    else:
        return 4

df['IPK'] = df['IPK'].apply(categorize_ipk)
df['Kehadiran'] = df['Kehadiran'].apply(categorize_kehadiran)
df['Lama_Studi'] = df['Lama_Studi'].apply(categorize_lama_studi)

X = df[['IPK', 'Kehadiran', 'Partisipasi_Akademik', 'Lama_Studi']]
y = df['Kelulusan']

scaler = StandardScaler()
X[['IPK', 'Kehadiran', 'Partisipasi_Akademik', 'Lama_Studi']] = scaler.fit_transform(X[['IPK', 'Kehadiran', 'Partisipasi_Akademik', 'Lama_Studi']])

# Mengurangi pengaruh Partisipasi Akademik
X['Partisipasi_Akademik'] = X['Partisipasi_Akademik'] * 0.1

# Membagi dataset menjadi data latih 80% dan uji 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Menyesuaikan Bobot Fitur
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# Membuat dan Melatih Model Decision Tree
model = DecisionTreeClassifier(criterion='entropy', max_depth=4)
model.fit(X_train, y_train, sample_weight=sample_weights)

# Evaluasi Model
y_pred = model.predict(X_test)

# F1-Score
f1 = f1_score(y_test, y_pred, average='weighted')

# Simulasi Input Pengguna
def user_input_prediction():
    print("\n--- Prediksi Kelulusan ---")
    
    print("Pilih IPK Anda:")
    print("1: Dengan Pujian (>=3.5)")
    print("2: Memuaskan (2.5 - 3.49)")
    print("3: Perlu Perbaikan (<2.5)")
    ipk = int(input("Masukkan pilihan (1/2/3): ")) - 1
    
    print("\nPilih Kehadiran Anda:")
    print("1: Rendah (<=65%)")
    print("2: Sedang (66%-85%)")
    print("3: Tinggi (>85%)")
    kehadiran = int(input("Masukkan pilihan (1/2/3): ")) - 1

    partisipasi = int(input("\nMasukkan Partisipasi Akademik Anda (Persentase): "))

    lama_studi = int(input("\nBerapa lama studi yang sudah Anda tempuh (dalam tahun): "))
    lama_studi_cat = categorize_lama_studi(lama_studi)

    # Membuat data baru untuk prediksi
    input_data = pd.DataFrame([[ipk, kehadiran, partisipasi, lama_studi_cat]], 
                              columns=['IPK', 'Kehadiran', 'Partisipasi_Akademik', 'Lama_Studi'])

    # Normalisasi input pengguna
    input_data[['IPK', 'Kehadiran', 'Partisipasi_Akademik', 'Lama_Studi']] = scaler.transform(input_data[['IPK', 'Kehadiran', 'Partisipasi_Akademik', 'Lama_Studi']])

    # Mengurangi pengaruh Partisipasi Akademik
    input_data['Partisipasi_Akademik'] = input_data['Partisipasi_Akademik'] * 0.1

    # Prediksi kelulusan
    pred = model.predict(input_data)
    
    decision_path = model.apply(input_data)
    print(f"Jalur yang diambil oleh model: {decision_path[0]}")
    
    hasil = "Lulus Tepat Waktu" if pred[0] == 1 else "Tidak Lulus Tepat Waktu"
    print(f"\nPrediksi Anda: {hasil}")

    # Visualisasi pohon keputusan dengan nomor node
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_tree(model, feature_names=['IPK', 'Kehadiran', 'Partisipasi Akademik', 'Lama Studi'], 
              class_names=['Tidak Lulus', 'Lulus'], filled=True, node_ids=True, ax=ax)

    plt.title('Decision Tree dengan Node ID')
    plt.show()

    # Menampilkan Confusion Matrix dan F1-Score
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nF1 Score (Weighted):", f1)

user_input_prediction()
