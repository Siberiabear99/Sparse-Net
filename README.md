# Sparse-Net
Sparse Decomposition for deep learning

First of all, in order to obtain the sparsely decomposed dataset, 
put your image sequence in the 'raw_data' folder and place one of the frames in the ‘test’ folder to view the training effect during network training

After that, run the 'Scripts_createdatasets.py' file to generate the dataset, and your dataset will appear in the 'datasets' folder
Run the 'Scripts_train' file to train the U-net network. The test images and models output by the network are all in the img folder, where you can see the effect of sparse decomposition

Now that you have trained the appropriate model, copy it to the Model folder and run the 'test.py' file. Your image sequence will be sparsely decomposed and saved to the 'predictions' folder
