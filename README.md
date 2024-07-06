# piksAI

# About Dataset 

For the training, testing and validation we used MIT Places Dataset URL: http://places.csail.mit.edu/downloadData.html.
We sampled the dataset to 10 classes to acomplish the use case of predicting the class label and use the class label to provide the feedback and advertisment to the user.
For the use case we used the following classes:

1) Airport Terminal
2) Auditorium
3) Bedroom
4) Book Store
5) Bus Station and Buses
6) Cloths Store
7) Computer and Gadgets Room
8) Food Court
9) Jewelry Shop
10) Railways

Sampled dataset URL: 

# Project Application
We created the end user application using streamlit in python. Used Pytorch for creating the instance of the model and load the saved weights from the pth file.
Application facilitates the user to upload the images of their choice, the images can be uploaded one after the other, and the application transform the image and pass the image through Resnet, Densenet, VGG13 and Alexnet model. The most predicted label will be chosen to provide the feedback and advertisment to the user.

Note: For running the application user required to install.

1) Streamlit
# pip install streamlit

2) Torch, Torchvision
# https://pytorch.org/get-started/locally/

3) pillow
# pip install pillow

4) OpenCV
# pip install opencv-python


Steps to run the application:
1) Nagivate to directory where master.py is located and open terminal in this directory.
2) If not installed the required packages, run installation command provided above.
3) Run the command 'streamlit run master.py'
4) Program will automatically redirect to your default browser, else open 'http://localhost:8501/' on your browser.
5) User can login with credentials
# User: paragsha
# Password: paragsha
6) User can upload the image of their choice, should be related to the categories defined above (10 classes).
7) Wait and see the results.
