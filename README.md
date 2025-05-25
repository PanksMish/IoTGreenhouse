# IoTGreenhouse
This project implements the methodology described in the research paper:

Title: IoT-based Smart Greenhouses: Sustainable Farming
Authors: V. Venkataramanan* (Corresponding Author), Mrunalini Ingle, Vijay. Kapure, Pankaj Mishra, Aarya Rokade, Tulsi Bhushan, Jitendra Singh

Overview
This project implements a machine learning-based smart greenhouse monitoring system that analyses environmental data to optimise irrigation, predict crop yield, and demonstrate sustainability improvements over traditional rule-based systems. The system uses linear regression for water consumption prediction and LSTM neural networks for crop yield forecasting, as proposed in the research paper.

Requirements
Python 3.x
pandas
numpy
matplotlib
scikit-learn
TensorFlow
Google Colab (optional, for file upload functionality)
Installation

pip install pandas numpy matplotlib scikit-learn tensorflow


Dataset Description
The greenhouse monitoring system uses a time-series dataset located in the dataset folder:
File: greenhouse_data.csv
This dataset contains environmental measurements and crop metrics collected from a smart greenhouse environment:

Collection Details:

Period: Data recorded for approximately 2 hours (10:00 AM to 12:00 PM)
Sampling Rate: New entries are recorded every 30 seconds
Total Samples: Approximately 240 data points 
