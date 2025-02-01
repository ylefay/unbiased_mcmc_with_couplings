# Unbiased Markov Chain Monte Carlo (MCMC) with Couplings

This project implements methods from the paper *"Unbiased Markov Chain Monte Carlo Estimators with Couplings"* by Pierre E. Jacob, John O'Leary, and Yves F. Atchadé. It was developed as part of the Bayesian Machine Learning course taught by Rémi Bardenet and Julyan Arbel.

The project includes implementations of:
- A **coupled Metropolis-Hastings algorithm**,
- The **maximal coupling** method,
- A generalization of the **Thorisson's Algorithm**.

These methods are designed to produce unbiased estimators in Markov Chain Monte Carlo (MCMC) simulations with chain of finite but random length.

The original R code accompanying the paper is available at [Pierre Jacob's GitHub repository](https://github.com/pierrejacob/unbiasedmcmc).

---

## Why is this useful?

Markov Chain Monte Carlo (MCMC) methods are powerful tools for sampling from complex probability distributions, which is essential in many fields. However, traditional MCMC methods can produce biased results, especially when the simulation is stopped early. This issue makes standard MCMC methods hard to parallelise: a parallel estimate would capture the bias. This project addresses this issue by implementing **unbiased MCMC estimators using couplings**, which ensure that the Monte Carlo estimate over the finite length chain is unbiased, allowing for parallel estimate. 
However it comes at the cost of an increased variance and a random stopping time.

---

## Applications in Industry

### Finance:
- **Risk Assessment:** Estimate the risk of complex financial portfolios.
- **Option Pricing:** Accurately price financial derivatives using unbiased simulations.
- **Algorithmic Trading:** Optimize trading strategies by simulating market conditions.

### Pharmaceuticals:
- **Drug Efficacy:** Simulate the effects of a drug across different patient populations.
- **Clinical Trials:** Model the outcomes of clinical trials to optimize drug development.
- **Pharmacokinetics:** Predict how drugs are absorbed, distributed, and metabolized in the body.

### Energy:
- **Reservoir Simulation:** Model oil and gas reservoirs to optimize extraction strategies.
- **Renewable Energy:** Simulate energy production from renewable sources like wind or solar farms.

### Technology:
- **Machine Learning:** Improve Bayesian inference in large-scale machine learning models.
- **Computer Vision:** Simulate complex image generation processes for training AI models.

### Supply Chain and Logistics:
- **Inventory Optimization:** Simulate demand and supply scenarios to optimize inventory levels.
- **Route Planning:** Model transportation networks to minimize costs and delivery times.

---

## How to Use This Package

### Installation:
This package requires `jax` and `numpy`. You can install it using:
```bash
pip install -e .
```

### Examples
See the `examples` folder for example usage of the package.

## References

- Pierre E. Jacob, John O'Leary, Yves F. Atchadé,  
  **"Unbiased Markov-Chain Monte Carlo estimators with couplings"**,  
  *Journal of the Royal Statistical Society: Series B*, 2020.  
  [https://academic.oup.com/jrsssb/article/82/3/543/7056129](https://academic.oup.com/jrsssb/article/82/3/543/7056129)

- Yves F. Atchadé, Pierre E. Jacob,  
  **"Unbiased Markov Chain Monte Carlo: what, why and how"**,  
  *Technical Report*, 2020.  
  [https://math.bu.edu/people/atchade/umcmc_rev.pdf](https://math.bu.edu/people/atchade/umcmc_rev.pdf)