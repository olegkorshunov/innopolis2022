# innopolis2022

Earth remote sensing data from space allow solving a large number of production tasks. For example, to determine plant cultures based on the analysis of time series of vegetation index $(NDVI \in (0,1))$ values obtained during the growing season (the period of growth and development of plants).  

NDVI is an index determined by absorption and reflection plants of the rays of the red and near-infrared zone of the spectrum when analyzing satellite images. By the value of this index, one can judge the development of the green mass of plants during the growing season. The more green biomass in the fields, the higher the NDVI value.

The aim of this competition is to build a model for the classification of crops based on the change in the indicator of vegetation indices in time sequence.


### Dataset

* id – identifier of the object (field)
* area – field area in ha
* nd mean YYYY-MM-DD – median value of the NDVI vegetation index for the given field on the specified date
* geo – coordinates of field boundaries
* crop – prediction column, contains the type of growing crop according to the data of agricultural producers  
  0 - sunflower  
  1 - potatoes  
  2 - winter wheat  
  3 - buckwheat  
  4 - corn  
  5 - spring wheat  
  6 - sugar beet  

### Metric
$$Recall=\frac{TP}{TP+FN}$$

# Solution:

My main mistake it's that I don't use geodata :disappointed:. 

The data contains a lot of missing values, I tried to restore them both with the help of interpolation and with the help of neural networks, I used different smoothing, but this did not improve the score. there were too many missing values and it can probably be considered a leak, and this main reason why my lightGBM overfitting on public lb.

Here, I public my some result. 

* cnn1d_train_val_.ipynb - CNN with conv1d, I suppose this my best model and I don't chose her(( because I'm belived in lightGBM and this was a mistake. 
* missing_values - try to recover missing values with KNN and CNN with conv2d, as and other methods this is don't improved  my cv.
* lightgbm - here I in first time tryed to use optuna instead of gridsearchCV.
* cb - outliers in dataset with [object importance](https://catboost.ai/en/docs/features/object-importances-calcution)
