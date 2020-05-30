# task3

JOB: I have made deeplearning model which already have some layers added for classification ,uses keras for predicting from the mnist dataset. --Load the image from dockers that have keras installed using docker. --Create a pipeline of job1,job2,job3,job4,job5.

Job1: Load the image that have keras installed in dockerfile.

Job2: Pull the mnist-keras.py model from github repo directly using jenkins.

Job3: Jnekins will automatically start doing the job of training the model and dockerfile installed keras file will aumotically start training the code .

Job4: Train the model and accuracy of model will be predicted.

Job5: will automatically start the job of adding the convolutinal layers for judging the incerese in accuracy till model didnot reach the accuracy greater than 80%. If container where job is running fails due to any reason , will automatically starts another container.

