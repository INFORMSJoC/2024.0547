[![INFORMS Journal on Computing Logo](https://INFORMSJoC.github.io/logos/INFORMS_Journal_on_Computing_Header.jpg)](https://pubsonline.informs.org/journal/ijoc)

# Code of the refined WDRO for contract pricing
This archive is distributed in association with the [INFORMS Journal on Computing](https://pubsonline.informs.org/journal/ijoc) under the [MIT License](LICENSE.txt)﻿.

The data in this repository is a snapshot of the data that was used in the research reported on in the paper [Refined Wasserstein Distributionally Robust Optimization for Contract Pricing: The Value of Optimality Conditions in Transactions](https://doi.org/10.1287/ijoc.2024.0547) by Guodong Yu, Pengcheng Dong, and Huiping Sun.

This paper considers a contract pricing problem in a two-tier supply chain with information asymmetry. To ensure decision reliability with small data, a Wasserstein-based data-driven distributionally robust pricing model using a dual-source data set is developed to maximize the seller’s worst-case profit. Numerical experiments demonstrate that proposed solution methods have higher computational efficiency compared to traditional methods, and derived optimal decisions exhibit superior out-of-sample performance compared to classical data-driven decisions. The codes in this repository can be used to replicate the results of the numerical experiments.

## Cite 
To cite the contents of this repository, please cite both the paper and this repo, using their respective DOIs.

https://doi.org/10.1287/ijoc.2024.0547

https://doi.org/10.1287/ijoc.2024.0547.cd


## Description and Replicating
All codes are programmed in Python language.

### minimax_tilting_sampler.py 
The file defines the rules for generating historical data. As an external package, this file is imported into other xxx.py files and used to generate historical demand realization data under multivariate truncated normal distribution.

### Computational efficiency center.py
This file is used to solve the augmented center of the Wasserstein ambiguity set by calling the Gurobi solver. Specifically, two solving methods are programmed: the LD method (benchmark) and the LRDS method proposed in this paper. The solution results and solution times are recorded to illustrate the computational effectiveness of the LRDS method.

The results are stated in Section 5.2.1, and running the code can replicate the results in Table 1.

### GCP dep product.py
The file uses the GCP method to solve the seller's worst-case profit in dependent multi-product. And Table 2 records the computational efficiency of this program.

### value of order data n=1.py
This is an integrated code for the single-product case, which means that all results related to single-product in the paper can be obtained by running this program (input parameters need to be adjusted for different experimental purposes). 

This file can output: the seller's optimal pricing decision, out-of-sample performance under the empirical distribution, out-of-sample performance under the classic Wasserstein DRO, and out-of-sample performance under the proposed refined WDRO. These results are used to demonstrate the value of historical contract data and the effectiveness of our proposed model. 

Running this program can contribute to replicating the results shown in Figure 2(a), Figure 4, and Figure 5.

### value of order data indep.py
This code is programmed for the purpose similar to the file "value of order data n=1.py", with the distinction that this code is set for independent multi-product case. Therefore, the program uses the proposed SOCP method to solve the seller's worst-case profit.

Recording the running time of this program can replicate the results shown in Table 3. And recording the outputs of this program can replicate the results shown in Figure 2(b).

### value of order data dep.py
This code is programmed for the purpose similar to the files "value of order data n=1.py" and "value of order data indep.py", with the distinction that this code is set for dependent multi-product case. Therefore, the program uses the proposed PCP method to solve the seller's worst-case profit.

Recording the running time of this program can replicate the results shown in Table 2 and Table 3. And recording the outputs of this program can replicate the results shown in Figure 2(c).

### only historical contract data.py
When the seller only has access to historical contract data, he can make pricing decisions with only historical contract data, as discussed in section 5.4. 

Running this program can contribute to replicating the results shown in Figure 3.

### ACD  vs ED wasserstein.py
This file is used to calculate the Wasserstein distance from the empirical distribution to the true distribution and the Wasserstein distance from the augmented center distribution to the true distribution. The results are used to illustrate that our proposed augmented center provides better predictions of the true distribution.

Running this program can contribute to replicating the results shown in Figure 1.

### results.xlsx
This file is a brief summary of the output experimental results in this paper. It contains the solution results under different data settings and different model settings. They are used to form the tabular results and picture results in the section Numerical Experiment. 

