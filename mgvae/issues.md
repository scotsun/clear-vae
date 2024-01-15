### pinned issue:
1. numerical results of the two estimator contradict with the inequality derivation
2. after adding ewc regularization, loss keeps increasing in the fine-tuning stage
	even having a large $\lambda$ doesn't help

### trivially good stuff:
1. GM-AM inequality and discrete jensen's inequality (log is concave) arrive at the same result


### things can be improved from the MGVAE:
1. The current objective can be extended to **IWAE** 
2. EWC is okay; however, *Nguyen et al. (2017) proposed* **VCL** framework more related to the variational framework (which is cited but not implemented in MGVAE)
3. The current objective function is computationally expensive

### current reults for $\rho=100$
|![Original](./res/100original.png)|
|:---:|
|Original "majority" images|

|![With pre-training](./res/100with_pt.png)|
|:---:|
|Transformed images (with pre-training)|

|![Without pre-training](./res/100wout_pt.png)|
|:---:|
|Transformed images (without pre-training)|
