# CQLwithDDPG
1.First,I uses the OnlineDDPG algorithm for training and obtains the training results. 
  The model parameters are stored in a file named "/DDPG_weight" in the directory.
2.I carried out large-scale sampling according to the weight obtained by OnlineDDPG and storage them to npydata.
  the storage sample size is 100,000 pairs of samples
  (users can set the sample size for sampling, please run the "Sampling_data.py" file and modify the iteration rounds for sampling)
3.I used the sampling data to write offline DDPG and offline CQLDDPG. 
  The main models are in the "OffLineDDPG.py" and "OffLineCQLDDPG.py" files respectively.
4.You can directly run the two files "OFFLineT_CQL.py" and "OFFLineT_QL.py" to correspond to the DDPG algorithm with CQL 
  and the DDPG algorithm without CQL respectively.

required pkg:
1.torch
2.gym


