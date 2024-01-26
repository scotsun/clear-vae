### pinned issue:
1. <s>numerical results of the two estimator contradict with the inequality derivation (the issue should be caused by `estimated` approach)</s> (Solved! I exponentiated `logvar` twice...)
2. <s>gradient explosion on `logvar_major` and some encoding layers by `ExpBackward`</s> (Solved! circumvent this issue by using the <u>*Log-sum-exp*</u> trick)
3. how to prove there is continual learning? after adding ewc regularization, loss keeps increasing in the fine-tuning stage; a large $\lambda$ doesn't help

### things can be improved from the MGVAE:
1. Disentangled representation:  
	a. [beta-VAE](https://arxiv.org/pdf/1804.03599.pdf)  
	b. [beta-TCVAE](https://proceedings.neurips.cc/paper_files/paper/2018/file/1ee3dfcd8a0645a25a35977997223d22-Paper.pdf) (Isolating Sources of Disentanglement in VAEs)
2. EWC is okay; however, *Nguyen et al. (2017) proposed* **VCL** framework more related to the variational framework (which is cited but not implemented in MGVAE)
3. The current objective can be extended to **VQ-VAE**
4. Optimal transportation theory
