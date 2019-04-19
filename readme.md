# Hackathon 2019 
## Suggesting Categories & Measures in reporting designer 



### Setup Environment
Create base folder ML and change directory to it
```
mkdir ML
chdir ML
```
git clone [DeepLearning-Utility](https://github.com/Hayk-Martirosyan/DeepLearning-Utility) project
	

git clone [DeepLearning-ReportingSuggestions](https://github.com/Hayk-Martirosyan/DeepLearning-ReportingSuggestions) project 
	
### Train Neural Network
Go to DeepLearning-ReportingSuggestions and run Docker image with configured AI tools (Keras, Tensorflow, Python)
```
cd DeepLearning-ReportingSuggestions 
./keras.sh
```

Start Tensorboard in background mode (note & at the end of command). It is accessable  via browser at [Tensorboard](http://localhost:6006/#)
```
./tboard.sh &
```
Start training with modelid = 9 (see nnModel.py for model descriptions)
```
./nn.py -t train -id 9
```
Continue training with modelid = 9
```
./nn.py -t train -id 9 -c 1
```

### Run Prediction
Start prediction in command line
```
./nn.py -t predict -id 9
```
Start prediction as flask rest service [Sample](http://localhost:5000/prediction/locati)

```
./nn.py -t predict-service -id 9
```

### Other
Source project for creating DeepLearning Docker image [DeepLearning-docker](https://github.com/Hayk-Martirosyan/DeepLearning-docker) 


