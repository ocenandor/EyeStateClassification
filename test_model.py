import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
import time
from tqdm import tqdm

from open_eyes_classificator import OpenEyesClassificator

def calculate_eer(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return eer

model = OpenEyesClassificator()
df = pd.read_csv('image_labels.csv')
paths = df['Image_Path']
labels = df['Label']
N = 50
start = time.perf_counter()
for _ in tqdm(range(N)):    
    predictions = np.array(list(map(model.predict, paths)))
end = time.perf_counter()
inference_time = len(predictions) * N / (end - start) 
print('speed, fps', inference_time)

plt.plot(*roc_curve(labels, predictions)[:2])
eer = calculate_eer(labels, predictions)
plt.vlines([eer], 0, 1, color='red')
plt.title(eer)
plt.show()
plt.close()