## Cat and dogs detection

*In this repository you can find pytorch implementation of a simple classifier.*

* **Dataset** 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; We have dataset with aproximately 3000 pictures of cats and dogs with 7 to 3 dogs to cats ratio. Every image has a corresponding text file containing five numbers:
1. Label: 1 for cats and 2 for dogs
2. Coordinates of a bounding box enclosing the animal's face: x_min y_min x_max y_max

Every picture contains always only one cat or dog. 

![data1](https://user-images.githubusercontent.com/74068173/121778250-d2646c00-cb9e-11eb-97e0-a9f689801d03.png)
![data2](https://user-images.githubusercontent.com/74068173/121778275-eb6d1d00-cb9e-11eb-836b-bf19524a49bc.png)
![data3](https://user-images.githubusercontent.com/74068173/121778288-f3c55800-cb9e-11eb-9b0d-f63cf9d400e4.png)*<p align="center">_Images after resizing_</p>*


* **Model**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; We are sure that there is only one animal on the picture. Therefore instead of a general-purpose object detector we can build a more simple custom network.
Unlike common object detectors, this network won't generate any anchor boxes. 
Instead, it always gives 5 numbers for every image: class label and predicted coordinates of a bounding box.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; For transfer learning I used a pre-trained Inception V3 neural network, where the last layer was implaced with a series of fully-connected layers. The next-to-last linear layer has 16 outputs, which are independently fed to a classifier with one neuron and to a regressor with four neurons.
As for loss functions, it was used binary cross entropy for classification and mean intersection over union over a batch for detection.
Finally, it was computed the gradient of the difference of classification and detection losses.
