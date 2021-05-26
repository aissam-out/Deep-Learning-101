# Tensorboard : Visualize your models

Tensorboard is a web app that will make your life easier if you are dealing with Tensorflow based programs. It is a visualization software that comes with any standard TensorFlow installation in order to make it easier to understand, debug, and optimize TensorFlow programs.


To store data from a computed result, say softmax weights, predications, loss, etc. Call :
```tf.summary.histogram("your_variable_name", your_variable)``` 


To visualize the program with TensorBoard, you have to write log files of the program. To do that, we first need to create a writer for those logs:
```writer = tf.summary.FileWriter([logdir], [graph])```


to visualize the graph, run the command line: 
```# tensorboard --logdir=[tensorflow_logs]```.
Then navigate your web browser to 
```http://localhost:6006``` to view the TensorBoard
