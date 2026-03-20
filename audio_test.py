import librosa
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load audio
audio, sr = librosa.load(r"C:\Users\rakpa\OneDrive\Desktop\Gargi Projects\voice_basics\training_audios\hello_1.wav", sr=16000)

print("Sample rate:", sr)
print("Audio length (seconds):", len(audio) / sr)

# Plot waveform
plt.ion()
plt.figure(figsize=(10, 3))
plt.plot(audio)
plt.title("Waveform of hello")
plt.show()

# Extract MFCC features
mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
print("MFCC shape:", mfcc.shape)

# Visualize MFCCs
plt.figure(figsize=(10, 3))
plt.imshow(mfcc, aspect='auto', origin='lower')
plt.title("MFCC Features")
plt.colorbar()
plt.show()

# Build dataset
X = []
y = []
folder = r"C:\Users\rakpa\OneDrive\Desktop\Gargi Projects\voice_basics\training_audios"
for file in os.listdir(folder):
    if file.endswith(".wav"):
        label = file.split("_")[0]   # "hello_1.wav" goes to "hello"
        audio, sr = librosa.load(os.path.join(folder, file), sr=16000)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)  # compress to 13 numbers

        X.append(mfcc_mean)
        y.append(label)

X = np.array(X)
y = np.array(y)

print("Feature matrix shape:", X.shape)
print("Labels:", y)

# Train classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))
