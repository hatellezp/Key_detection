* be sure the old code works
    * finish convert.py analysis
* define your own structure with .cfg file
* how to redefine number of anchors ?
    * hard code the number ?
    * or retrain to with kmeans to redo the clusters?
* create a quick model
* create a more developed model
* train both with colab or floydhub
* maybe do a notebook ?
* write rapport 



### Order of files (and mentioned modifications)
_Observation: we didn't rewrite the whole https://github.com/belarbi2733/keras_yolov3. We made a restructuring of the 
directory, adding modules when need it and modifications to the specified files. It does not seems necessary to us to 
create a whole new library, more interesting is to customize to our expectations._

* **setup** : encapsulates the creation of mixed images from keys and  background images
    * keys_with_background: here we propose the idea of several bounding boxes and multiple keys in the same image, 
      also, cropping the key images produce an identification from a partial match
* **model_creation** : this module comprehend the transformation from darknet weights configuration files to keras 
  syntax and structure,
    * converty.py was already present, here no modifications were made
* **train** : here we make the most of the configuration of _**yolo**_, with our own definition of anchors and newly 
  annotated images in _annotations.csv_ is possible to accelerate both training and detection of images with our models
* **detect_key** : use the specified model to detect keys in an image or video