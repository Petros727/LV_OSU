#4
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers
from keras.models import load_model
 
# Učitavanje podataka
titanic_df = pd.read_csv('titanic.csv')
 
# a) Broj žena u skupu podataka
broj_zena = titanic_df[titanic_df['Sex'] == 'female'].shape[0]
print(f'a) Broj žena u skupu podataka: {broj_zena}')
 
# b) Postotak osoba koje nisu preživjele potonuće broda
postotak_ne_prezivjelih = (1 - titanic_df['Survived'].mean()) * 100
print(f'b) Postotak osoba koje nisu preživjele potonuće broda: {postotak_ne_prezivjelih:.2f}%')
 
# c) Stupčasti dijagram postotka preživjelih muškaraca i žena
prezivjeli_po_spolu = titanic_df.groupby('Sex')['Survived'].mean() * 100
 
plt.bar(prezivjeli_po_spolu.index, prezivjeli_po_spolu.values, color=['yellow', 'green'])
plt.xlabel('Spol')
plt.ylabel('Postotak preživjelih')
plt.title('Postotak preživjelih po spolu')
plt.show()
 
# d) Prosječna dob preživjelih žena i muškaraca
prosjecna_dob_zena = titanic_df[titanic_df['Sex'] == 'female']['Age'].mean()
prosjecna_dob_muskaraca = titanic_df[titanic_df['Sex'] == 'male']['Age'].mean()
 
print(f'd) Prosječna dob preživjelih žena: {prosjecna_dob_zena:.2f} godina')
print(f'   Prosječna dob preživjelih muškaraca: {prosjecna_dob_muskaraca:.2f} godina')
 
# e) Najstariji preživjeli muškarac po klasi
najstariji_muškarac_po_klasi = titanic_df[titanic_df['Sex'] == 'male'].groupby('Pclass')['Age'].max()
print('e) Najstariji preživjeli muškarac po klasi:')
print(najstariji_muškarac_po_klasi)
 
 
#6

 
# Učitavanje podataka
data_df = pd.read_csv("titanic.csv")
data_df = data_df.dropna()
 
# Podjela podataka na ulazne i izlazne varijable
X = data_df.drop(columns=['Survived']).to_numpy()
y = data_df['Survived'].to_numpy()
 
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)
 
# Skaliranje podataka
sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
 
# a) Izgradnja neuronske mreže
model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(units=12, activation="relu"),
    layers.Dense(units=8, activation="relu"),
    layers.Dense(units=4, activation="relu"),
    layers.Dense(units=1, activation="sigmoid")
])
model.summary()
 
# b) Podešavanje procesa treniranja
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
 
# c) Treniranje mreže
history = model.fit(X_train, y_train, batch_size=5, epochs=100, validation_split=0.1)
 
# d) Pohrana modela
model.save('Model/')
 
# e) Evaluacija mreže na testnom skupu podataka
model = load_model('Model/')
score = model.evaluate(X_test, y_test, verbose=0)
print("Evaluacija mreže na testnom skupu podataka:")
for i in range(len(model.metrics_names)):
    print(f'{model.metrics_names[i]} = {score[i]}')
 
# f) Predikcija mreže i prikaz matrice zabune
y_predictions = model.predict(X_test)
y_predictions = np.around(y_predictions).astype(np.int32)
cm = confusion_matrix(y_test, y_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title('Confusion Matrix for Test Data')
plt.show()