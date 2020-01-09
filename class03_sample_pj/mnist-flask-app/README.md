# MNIST Flask demo 

For education purposes only.
The model is the simple 2-hidden layer neural network, created using vectorized numpy network implementation
[MLP MNIST](https://github.com/alm4z/mlp-mnist). 


## Live demo
Live demo located [here](http://almaz.social/flaskmnist). 

![Demo screenshot](/static/imgs/demo.png?raw=true "Demo")


## Requirements

  * Python 3.>
  * Flask
  * OpenCV
  * Numpy


To install required libraries, use commands below:
```
pip install flask
pip install opencv-python
pip install numpy

```

## Running
Clone repository, go to the specified folder and run
```
python application.py
```
```
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
```
Open your browser and access the address http://127.0.0.1:5000/ .

## Training
If you want to try another MLP architecture, edit *train.py* and run
```
python train.py
```
## Credits
* Signature pad (https://github.com/szimek/signature_pad).
* MNIST pad (https://github.com/brloureiro/mnist-pad)

## Contact
If you have any further questions or suggestions, please, do not hesitate to contact by email at a.sadenov@gmail.com.

