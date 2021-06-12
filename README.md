## Cat and dogs detection

* **Dataset** 

We have dataset with aproximately 3000 pictures of cats and dogs. Every image has a corresponding text file containing five numbers:
1. Label: 1 for cats and 2 for dogs
2. Coordinates of a bounding box enclosing the animal's face: x_min y_min x_max y_max

Each picture contains always only one cat or dog. 

![data1](https://user-images.githubusercontent.com/74068173/121778250-d2646c00-cb9e-11eb-97e0-a9f689801d03.png)
![data2](https://user-images.githubusercontent.com/74068173/121778275-eb6d1d00-cb9e-11eb-836b-bf19524a49bc.png)
![data3](https://user-images.githubusercontent.com/74068173/121778288-f3c55800-cb9e-11eb-9b0d-f63cf9d400e4.png)*<p align="center">_Images after resizing_</p>*

Therefore instead of a general-purpose object detector we can write a more simple custom network.
Unlike common object detectors, this network won't generate any anchor boxes. 
Instead, it always gives 5 numbers for every image: class label and predicted coordinates of a bounding box.


* **Model**

For transfer learning I used Inception v3 network. The feature map is fed to fully connected layers.
