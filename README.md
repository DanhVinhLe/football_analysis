# Football analysis

## Introduction

I have implemented a model using YOLOv8 to detect and track football players, referees and ball in a short video of a football match, based on a tutorial video on Youtube. I also trained YOLOv8 model on football player detection dataset from Roboflow in the [link](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/dataset/1), so that model can better detect objects in the video. Additionally, I have also marked the players based on their uniforms color and identified the player who is currently holding the ball.

## Results
I have finetuned model YOLOv8l on the football player detection dataset, and here is my result:
### Results on additional dataset
| Class | mAP50 | mAP50-95 |
| --- | --- | --- |
|all|0.817|0.589|
|ball|0.394|0.167|
|goalkeeper|0.937|0.753|
|player|0.986|0.789|
|referee|0.951|0.646|

### Output video
[The output video](https://drive.google.com/file/d/1cX4eboaiP4-BthlYeAoZvCf7K1vjzfBJ/view?usp=drive_link)

### Best model
[Weight of the best model](https://drive.google.com/file/d/1M0IRcldDmehb-coHKz6iQbz2POaNbKiu/view?usp=drive_link)
