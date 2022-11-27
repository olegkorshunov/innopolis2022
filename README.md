# innopolis2022

Earth remote sensing data from space allow solving a large number of production tasks. For example, to determine plant cultures based on the analysis of time series of vegetation index (NDVI) values ​​obtained during the growing season (the period of growth and development of plants).  

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
$$Recall={TP}{TP+FN}$$
