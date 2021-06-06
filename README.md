## Reusable-MLCodes-DifferentPurposes
### During my professional data science experiences, have tried to consolidate some boiler-plate code blocks using in any phases of a typical ML Project.  

* **RunTestMLTest:** Just for picking the best ML algorithm and the optimum hyperparemeters

* **OptimalBinning:** For avoiding overfit issue and generate mode general ML models, it should be better to create BINs for numerical features. Without checking target value, such stuff might be done by splitting the numerical data with percentiles. However such approach has a significant issue that it does not take target into account and can not cover discrimative power. With **OptimalBinning** logic, just univariate analysis wrt target value is done and create smart bins represent the **most discriminative power**. One can apply the logic into all numerical features and use the resulted BIN values while modelling.

* **FeatureExplore:** Just before modelling, a data scientist should perform such **exploratory data analysis** to get better understanding of the data. It includes some basic feature explore functions in it. 

* **TimeSeries:** Just some basic functions and best practices while dealing with time series data and time series prediction cases.

* **XAI**: Sometimes you need to explain your model's results to project stakeholders (think about your model was about credit risk scoring, and business users are curious about why your model is rejecting any specific user's application), such folder includes some best practices about **XAI (Explainable AI)**

* **Generic_Functions**: Some generic functions often needs to be reused like "oracle db connection", "finding IQR limits", "chi-square test resulst", "Confusion matrix results for customized threshold probabilities", "Finding Similar Instance in multidimensional space" etc.
