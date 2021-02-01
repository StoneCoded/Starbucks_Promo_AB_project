# Starbucks Promotion Model
A test project set by Udacity/Starbucks to build a model for finding Customers
who will spend $10 given a Promotion. This project was originally a take home
assignment for prospective Starbucks employees.

##Contents
1. Files and Implementation
2. Project Overview
3. Results and Observations
4. Licensing, Authors, and Acknowledgements

##1. Files and Implementation
1. `training.csv`
- CSV containing training data for building model
2. `Test.csv`
- CSV containing test data for model evaluating
3. `Starbucks.py`
- Data Wrangling and Model Creation. The main bulk of my working.
4. `test_results.py`
- Contains functions for evaluation IRR and NIR of model

The standard Anaconda distribution of Python 3 is all you will need to run this.

##2. Project Overview

A randomized experiment was conducted and the results are in `Training.csv`
`Treatment` – Indicates if the customer was part of treatment or control
`Purchase` – Indicates if the customer purchased the product
`ID` – Customer ID
`V1` to `V7` – features of the customer
Cost of sending a Promotion: $0.15
Revenue from purchase of product: $10 (There is only one product)

Questions:
1.	Analyze the results of the experiment and identify the effect of the
  Treatment on product purchase and Net Incremental Revenue
2.	Build a model to select the best customers to target that maximizes the
  Incremental Response Rate and Net Incremental Revenue.
Deliverables:
1.	Score the ‘Test.csv’ using the model and select the best customers and share
  the customer ID’s as  csv file
2.	Explain briefly the approach used in a separate document and also share the
  code that can be executed to reproduce results.

Incremental Response Rate:
 (# of Purchasers In Treated) / Total # of customers in Treated  
 - (# of Purchasers In Control) / Total # of customers in Control

Net Incremental Revenue:
 [(# of Purchasers in Treated *$10) – (# of Treated Customers *$0.15)]  
 -  [# of Purchasers in Control * $10]

##3. Results and Observations

Model 1 in `Starbucks.py` performed suprisingly well for being a somewhat simple
model. With an IRR of 0.0183 and NIR of 290.50, this a much more optimistic model
than the given example. Whilst working I noticed a much higher NIR (above 400)
could be achieved by pure chance of the random sample choice. Setting the
random_state to 42 has given us the 290.50.

Model 2 was a little more complex with an attempt at upsampling but either the
method or the shear scale of the upsampling gave a rather lackluster result of
an IRR of 0 and NIR of -0.75.

There are many many more ways to play with this data and I no doubt will return
to this at some point to see if I can improve and refactor (which I certainly can)

##4.Licensing, Authors, and Acknowledgements
All data was provided by Udacity and Starbuck for the purpose of this project.

Udacity:[Here](https://www.Udacity.com)
Starbucks:[Here](https://www.Starbucks.com)

Copyright 2021 Ben Stone

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
