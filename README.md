# Facial Emotion Recognition
A small CNN model based on VGGNet to recognize facial emotion realtime, the model can run without support from GPU with approximately 14 - 16 fps.

### Overview
Facial emotion plays an important role in communication to present the all the meanings, attitudes want to transmit. Besides, facial emotion recognition is also bring other interesting benefits so it becomes getting more and more attention nowadays. Therefore, we propose a small CNN model can run on devices with or without GPU.
This project references the project [FER using VGGnet](https://github.com/usef-kh/fer)
Our model got **89.09%** accuracy on train data and **71.4%** accuracy on test data.

### Dataset
Here we use Facial Emotion Recognition 2013 (FER2013), AffectNet Sample we found on Kaggle and The Extended Cohn-Kanade (CK+) to train our model.
You can download these datasets here:
 - **FER-2013**: https://www.kaggle.com/datasets/msambare/fer2013
 - **AffectNet Sample**: https://www.kaggle.com/datasets/mouadriali/affectnetsample?select=train_class
 - **CK+**: https://www.kaggle.com/datasets/shawon10/ckplus

### 5-Block CNN model
You can see the model's architecture in image below:
:------------------------------------------------------:
![Model architecture here](images/modelx.png)

### Training
To install the required packages, run `pip install -r requirements.txt`.
After that, you can run the code below to train the model:
```
python train.py network=fer_mini name=my-fer-mini
```
Losses           |  Accs
:-------------------------:|:-------------------------:
![Optimizers](images/loss.png)  |  ![Schedulers](images/accuracy.png)

Confusion Matrix

![Confusion Matrix](images/confuse_matrix.png)
### Demo
You can go to `demo/` and use `demo.ipynb` run on colab or use the code below:
```
python demo.py
```
![Gif demo](demo/demo.gif)
