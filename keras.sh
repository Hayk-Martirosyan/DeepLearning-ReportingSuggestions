docker run -it  -v $(pwd)/:/ml -v $(pwd)/../DeepLearning-Utility:/pl -p 6006:6006 -p 5000:5000 regdb.synisys.com/keras:4.0 /bin/bash
