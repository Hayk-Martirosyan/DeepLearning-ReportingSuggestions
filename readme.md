# Hackathon 2019 
# Suggesting Categories & Measures in reporting designer 



# Setup Enviroment
Create base folder ML

1. git clone DeepLearning-Utility project into ML
	http://gerrit.synisys.com/#/admin/projects/DeepLearning-Utility

2. git clone DeepLearning-ReportingSuggestions project into ML
	http://gerrit.synisys.com/#/admin/projects/DeepLearning-ReportingSuggestions

3. Go to DeepLearning-ReportingSuggestions and run ./keras.sh

This will run docker image with configured AI tools(Keras, Tensorflow, Python)


4. run ./tboard.sh &
	This will run tensorboard at 6006 port. Note & at the end of command, it will run in the background.
5. start training with nn.py -t train -id %modelid%

