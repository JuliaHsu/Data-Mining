# Anomaly Detection

CS 584 Data Mining Project 4. (April, 3, 2019)<br>

## Overview and Goals
* Implement the StrOUD algorithm from scratch, using LOF as the strangeness function
* Handle Signal data collected from a device that controls a centrifuge
* Find anomalous and normal data in the test set

## Data
The data consist of electromagnetic signals captured by a sensor attached to a control unit for a centrifuge. The centrifuge has four normal modes of operation (A,B,C, and D), captured respectively in folders ModeA, ModeB, ModeC, and ModeD of the base data below. Each data file is composed by a sampled signal (20,000 samples, each a floating point number). The data corresponding to maliciously infected or M mode is captured in the folder ModeM. 

## Implementation

1. Read training data and split them for private testing: read normal data and reserve 100 normal data from different modes for private testing. Use the remaining normal data create a baseline.     
2.	Extract features via Fast Fourier transform
3.	Implement LOF:
	a. Compute and store the distance among each point. And get the distances of the Kth-nearest neighbor of each point (k_dist).
	b.	Store the sets of k nearest neighbors of each point into a 2D array, k_neighbors.
	c.	Compute local reachability distance for each point, which is the inverse of the reachability distance of each point from its neighbors (through k_dist and k_neighbors).
	d.	Compute local density factor for each point with its lrd and the lrd of its k nearest neighbors.
4.	Compute and store LOF for each point in baseline.
5.	Implement StrOUD: 
	a.	Compute the LOF of each data point in test data with respect to the baseline.
	b.	Calculate the p value for each data point in test data by comparing the LOF of each of them with the LOF of baseline. ((b+1)/ (N+1))
6.	Compute the area of roc curve to determine the best K.



## Rank and roc score on leaderboard
9<br>
1.0