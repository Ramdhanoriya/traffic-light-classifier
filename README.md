This is a traffic light image classifier, written for the final project of my Udacity Intro to Self Driving Cars Nanodegree. The dataset is sourced from [MIT](https://selfdrivingcars.mit.edu/).

Project requirements:
1. **Greater than 90% accuracy**
2. ***Never* classify red lights as green**

I achieved 97.97% accuracy. The stipulation to never classify a red light as green is reflected in the design, as I essentially designed the algorithm to place a 'burden of proof' on the image to prove it was green. If the image fails to convince the algorithm that it is, in fact, green, it is assumed to be red. You can see my associated blog post [here](https://danielhunter.io/traffic-light-classifier/).

This was an excellent exercise in computer vision and image classification, and I had a lot of fun playing around with RGB and HSV values. Definitely will be looking for similar projects in the future.
