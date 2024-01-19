### pinned issue:
1. <s>numerical results of the two estimator contradict with the inequality derivation (the issue should be caused by `estimated` approach)</s> (Solved! I exponentiated `logvar` twice...)
2. <s>gradient explosion on `logvar_major` and some encoding layers by `ExpBackward`</s> (Solved! circumvent this issue by using the <u>*Log-sum-exp*</u> trick)
3. after adding ewc regularization, loss keeps increasing in the fine-tuning stage
	even having a large $\lambda$ doesn't help

### things can be improved from the MGVAE:
1. Disentangled representation and <u>*$\beta$-VAE*</u>
2. EWC is okay; however, *Nguyen et al. (2017) proposed* **VCL** framework more related to the variational framework (which is cited but not implemented in MGVAE)
3. The current objective can be extended to **IWAE or VQ-VAE**
