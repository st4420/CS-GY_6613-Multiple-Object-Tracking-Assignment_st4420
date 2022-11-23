# Multiple Object Tracking

## Task 1: Understand the problem and setup environment

## Task 2: Object Detector

### UNet Environment installation 

```
# conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch  
opencv-python  
pillow  
pyqt5  
tqdm  
pillow  
```


### UNet Implement 

train -- Implement network training   
Among the run_data, there are Test_Images, Test_Labels, Training_Images and Training_Labels.    
```
Training_Images  /Store the training image  
Training_Labels  /Store the training labels  
Test_Images  /Store the testing picture  
Test_Labels  /Store the testing labels  
```
#### Modify _train.py_
```
data_path = "C:/Users/Administrator/Desktop/unet/run_data" # todo  /Modify to your local dataset location  
```

#### Modify _test.py_
```
def cal_miou(test_dir="C:/Users/Administrator/Desktop/unet/Test_Images",# todo  /Modify to your local dataset location  
                pred_dir="C:/Users/Administrator/Desktop/unet/results",  
             gt_dir="C:/Users/Administrator/Desktop/unet/Test_Labels"):  
```
             
#### Main files in the directory:  
_predict.py_ -- Prediction file   
_run_predict.py_ -- Video prediction file  
_test.py_ -- Testing prediction file    
_train.py_ -- Training training file    

### UNet operating steps 

1.First make the dataset’s training image and label file  

2.Divide the dataset into training and testing  

3.Training the dataset _train.py_  

4.Test the dataset _test.py_  

5._run_predict.py_ -- Run the ball video to see the recognition result

### Demo 



## Task 3: Tracker


### Kalman Environment installation 
VS 2017, or above  

OpenCV 3.4.1  



### Kalman filtering Implement

In Kalman class, input is the input state.  

In Kalman class, output is the predicted state.  

In Kalman class, init() is the initialization function.  

In Kalman class, update() is a function that calculates the prediction result.  

process1(Mat img) A function that processes ball’s white paper occlusion  

process2(Mat img) Two balls prediction function  

### Implementation logic

#### Ball’s white paper occlusion

1.Kalman coefficients initialization  
2.Ball image’s gray processing  
3.Ball image’s binaryzation  
4.Remove fine noise  
5.Extract the center coordinates of the ball  
6.Enter the center coordinates of the ball into the input of Kalman  
7.Kalman update  
8.Kalman output calculation  
9.output display  

Among them,  
If the ball is not present，the input in the step 5 = the output of the coordinates predicted last time + the output of the last predicted speed.  
If the ball is present, execute step 5.  

#### Double balls

1.kalman1 Coefficients initialization, kalman2 Coefficients initialization  
2.Ball image’s gray processing  
3.Ball image’s binaryzation  
4.Remove fine noise  
5.Extract rectangle coordinates of the ball (rect1，rect2)  
6.Enter the center coordinates of the ball into the input of Kalman1 and kalman2, respectively.  
7.Kalman update  
8.Kalman output calculation  
9.output display  

Among them,  
rect3，rect4 are the predicted rectangle  
rect1，rect2 are the destination rectangle  
rect[0],rect[1] are the rectangle detected by the image  
The updates of rect1 and rect2’s recognition depend on rect[0] and rect[1] at the beginning. Later updates depend on the IoU of rectangles between rect3, rect4 and rect[0], rect[1]. The higher loU one will update to rect1, rect2.  


