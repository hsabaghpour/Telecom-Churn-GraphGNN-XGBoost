# Telecom-Churn-GraphGNN-XGBoost

Telecom Customer Churn Prediction using Graph Neural Networks (GNNs) and XGBoost

This project demonstrates how to predict customer churn in the telecom industry using graph-based solutions with Graph Neural Networks (GNNs) for community detection and XGBoost for classification.

Overview

Telecom companies often face customer churn, where customers leave the service for competitors. Predicting customer churn is crucial for improving retention strategies. In this project, we use Graph Neural Networks to detect communities of customers based on their relationships (e.g., shared network usage or proximity). These community embeddings are then combined with customer features to train an XGBoost classifier, which predicts whether a customer is likely to churn.

Key Features:

	•	Graph Neural Network (GNN): Detects customer communities based on their interactions in a telecom network.
	•	XGBoost: Predicts customer churn using both customer features and community embeddings from the GNN.
	•	Graph-Based Solution: Represents telecom customers and their relationships as a graph for improved prediction accuracy.

Project Structure

	•	telecom_churn_gnn_xgboost.py: Main code for training the GNN, extracting community embeddings, and using XGBoost for customer churn prediction.
	•	data/: (Optional) Directory to store the sample dataset or any custom datasets.
	•	README.md: This file, describing the project, how to run it, and dependencies.

Dependencies

To run this project, install the following Python libraries:


pip install spektral xgboost tensorflow numpy scikit-learn

Libraries Used:

	•	Spektral: For building the Graph Neural Network (GNN).
	•	XGBoost: For the final customer churn classification.
	•	TensorFlow: To support neural network training.
	•	Numpy: For numerical operations.
	•	Scikit-Learn: For data preprocessing and evaluation metrics.

How It Works

	1.	Data Representation:
	•	Each customer is represented as a node with features (e.g., usage patterns, demographics).
	•	Edges between customers represent relationships, such as shared network usage or proximity.
	2.	Graph Neural Network (GNN):
	•	We use a Graph Convolutional Network (GCN) from the Spektral library to learn customer embeddings that capture their community structure.
	•	This community structure helps to group customers into clusters, which is useful for detecting churn-prone groups.
	3.	XGBoost Classifier:
	•	The GNN embeddings are combined with the original customer features to train an XGBoost classifier.
	•	XGBoost predicts customer churn based on the combined features (original features + community embeddings).

How to Run

	1.	Prepare Your Data:
	•	You can replace the randomly generated sample data with real customer data. Make sure to have:
	•	Node features (customer features like usage, signal strength, etc.).
	•	Graph adjacency matrix (connectivity between customers).
	2.	Run the Code:
	•	After installing the dependencies, run the main code:


 
