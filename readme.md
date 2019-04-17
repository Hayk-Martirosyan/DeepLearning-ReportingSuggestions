# Hackathon 2019 
## Suggesting Categories & Measures in reporting designer 



### Setup Environment
Create base folder ML and change directory to it
```
mkdir ML
chdir ML
```
git clone [DeepLearning-Utility](http://gerrit.synisys.com/#/admin/projects/DeepLearning-Utility) project
	

git clone [DeepLearning-ReportingSuggestions](http://gerrit.synisys.com/#/admin/projects/DeepLearning-ReportingSuggestions) project 
	
### Train Neural Network
Go to DeepLearning-ReportingSuggestions and run Docker image with configured AI tools (Keras, Tensorflow, Python)
```
cd DeepLearning-ReportingSuggestions 
./keras.sh
```

Start Tensorboard in background mode (note & at the end of command)
```
./tboard.sh &
```
Start training with nn.py
```
nn.py -t train -id %modelid%
```


