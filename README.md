# Deep Learning 

# Exécution du projet 
- créer un notebook 
- insérer les lignes de codes ci-dessous

```python
import ai
model=ai.ai_plantes(BATCH_SIZE=32)
model.preporcess(isMacOs=True,out_put="sigmoid",min_sample=False )
model.create_model(balance=True)
his=model.fit(epochs=50)

model.print_multilabel_confusion_matrix()
model.get_roc_curve()
```
<img width="801" alt="Capture d’écran 2023-03-05 à 14 59 10" src="https://user-images.githubusercontent.com/25811960/222964936-a61cd0f1-065d-4e4b-93e8-6cfd68faebe4.png">
