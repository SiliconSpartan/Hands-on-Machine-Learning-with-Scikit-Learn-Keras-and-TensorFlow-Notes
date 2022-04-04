# Working with Real Data
- Some place to find real open source data
    - Popular open data repositories:
        - UC Irvine Machine Learning Repository — Kaggle datasets
        - Amazon’s AWS datasets
    - Meta portals (they list open data repositories): — http://dataportals.org/
        - http://opendatamonitor.eu/
        - http://quandl.com/
    - Other pages listing many popular open data repositories: — Wikipedia’s list of Machine Learning datasets
        - Quora.com question
        - Datasets subreddit

# Look at the Big Picture
- We are going to build a model that can predict the median housing price for a specific block district

## Frame the Problem
- How do the users expect to use the model?
    - Can help determine which algorithms to use, how much tweaking should be done, etc
- For this project the results of our model will be sent to another ML system
- "Signals" are pieces of information fed to a ML system
- Pipelines
    - A sequence of data processing components is a data pipeline
    - Each component runs asynchronously and usually stores the data to a different data store to be processed by another component
- Ask if there is a current solution so you have a reference point to compare performance
- For predicting house prices
    - Supervised Learning since we have labelelled data
    - Regression task since we are predicting a value
    - Since we are working with finite data and not a continuous flow then it is Batch Learning

## Select a Performance Measure

