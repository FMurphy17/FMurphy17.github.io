---
layout: post
title:  "Gaussian Process Ordinal Regression using GPyTorch"
date: 2026-06-28
categories: gaussian-processes
---


Ordinal regression is a regression framework where the target variable has an inherent ranking structure. This appears in contexts such as rating scales, where, for example, possible responses may fall into discrete classes $$y = \{1,2,3,4,5\}$$, and each successive class corresponds to a higher rating than the previous class. 

{% cite chu_gaussian_2005 %} introduce a probabilistic approach to ordinal regression by employing Gaussian processes (GP). In this post, I show how to implement a GP ordinal regression model in GPyTorch, a powerful library for GP modelling. I will define here custom likelihood classes for both the Ordered Logistic likelihood and Ordered Probit likelihood, and walk through the ordinal likelihood derivation as well as some other key aspects of the GP ordinal regression model presented in {% cite chu_gaussian_2005 %}. In this way, I attempt to clearly illuminate the theory behind GPyTorch's handy abstractions for GP ordinal regression modelling.

## GP Ordinal Regression Model

The ordinal regression model consists of a response variable $$y$$ which may take on ordered, discrete values, i.e. $$y=1,\dots,K$$ for $$K$$ possible response classes. A response $$y$$ is assumed to be a noisy representation of the latent continuous variable $$f(\mathbf{x}) \in \mathbb{R}$$, a function of the predictors $$\mathbf{x} \in \mathbb{R}^d$$. Using a GP for ordinal regression is very similar to standard GP regression, in that we place a GP prior on the function $$f(\mathbf{x})$$,  

$$
f(\mathbf{x}) \sim GP(\mu(\mathbf{x}), k(\mathbf{x}, \mathbf{x}')),
\tag{1}
$$

where we may set the mean function $$\mu(\mathbf{x})$$ to $$\mathbf{0}$$, as is commonly done (see {% cite rasmussen2006gp %} for a comprehensive background on GPs), and we can use our kernel of choice, such as the Squared Exponential (SE) kernel, 


$$
k(\mathbf{x}, \mathbf{x}') = \tau^2\exp\Bigg(\frac{(\mathbf{x} - \mathbf{x}')^\top(\mathbf{x} - \mathbf{x}')}{2\ell^2}\Bigg),
\tag{2}
$$

where the hyperparameters of the SE kernel are the kernel variance $$\tau^2$$ and the lengthscale $$\ell$$. For $$n$$ inputs, $$X \in \mathbb{R}^{n \times d}$$, the kernel matrix is defined as $$K_{nn} = k(X, X)$$.

The key to adapting the GP model for ordinal regression lies in the specification of the likelihood. 

#### <ins>Likelihood</ins>

Depending on the noise assumptions, there are two possible approaches for the likelihood: under a logistic noise assumption we use an ordered logistic likelihood, while under Gaussian noise we use an ordered probit likelihood. In practice, these likelihoods have similar performance, but the ordered probit incurs some analytical calculations when used with Gaussian processes. As such, I will discuss the ordered probit likelihood derivations going forward, but will include an implementation of the ordered logistic likelihood at the end. 

{% cite chu_gaussian_2005 %} first present the noise-free likelihood function as 

$$
p_{ideal}(y=i|f(\mathbf{x})) = \begin{cases}
1, b_{y-1} < f(\mathbf{x}) < b_y\\
0, \text{otherwise}\\
\end{cases}.
\tag{3}
$$

Here, $$b_i$$ for $$i=1,\dots, K-1$$ are cutpoints, or the thresholds on the continuous latent variable $$f(\mathbf{x})$$, defining which discrete class the associated $$y$$ belongs in. Implicitly, the exterior cutpoints, $$b_0$$ and $$b_K$$ are equivalent to $$-\infty$$ and $$+\infty$$, while the internal cutpoints are defined by $$b_i = b_1 + \sum_{j=2}^i\Delta_j$$ with $$\Delta_j > 0$$ to ensure that categories remaind ordered. The first step for defining the ordered probit likelihood class is thus to define the cutpoint parameters,

{% highlight python %}
class OrderedProbitLikelihood(gpytorch.likelihoods._OneDimensionalLikelihood):
    def __init__(self, K):
        super().__init__()
        self.normal = torch.distributions.Normal(0.0, 1.0)
        self.K = K
        self.nu_c = nn.Parameter(torch.zeros(1))
        self.delta_unconstrained = nn.Parameter(torch.zeros(K-1))

    @property
    def cutpoints(self):
        delta = F.softplus(self.delta_unconstrained)
        return self.nu_c[:, None] + torch.cumsum(delta, dim=-1)
{% endhighlight %}

Here, the $K-1$ cutpoints are estimated via the sum of an estimated baseline location $\nu_c$ and cumulative sum of estimated $\Delta$s, which are constrained to be positive with the softplus function.

In the case where we assume the responses are distorted by additive Gaussian noise, the likelihood which accounts for this noise recovers a difference of probits,

$$
\begin{aligned}
p(y\mid f(\mathbf{x})) &= \int p_{ideal}(y\mid f(\mathbf{x}) + \varepsilon)\mathcal{N}(\varepsilon; 0, \sigma^2)d\varepsilon = \Phi\Bigg(\frac{b_y - f(\mathbf{x})}{\sigma}\Bigg) - \Phi\Bigg(\frac{b_{y-1} - f(\mathbf{x})}{\sigma}\Bigg).\\
\end{aligned}
\tag{4}
$$

The derivation for (4) is included in the Appendix [A1](#a1-difference-of-probit-likelihood). In the case where $$K=2$$, we recover exactly the probit likelihood used in binary classification. Similarly, assuming logistic noise, $$\varepsilon \sim \text{Logistic}(0, s)$$, the likelihood resolves to a difference of inverse logit functions, or standard logistic regression in the case of 2 response classes. This can be verified using the logistic distributed latent variable $$\varepsilon$$ and following the derivation for the probit model, where the normal CDF is replaced by the inverse logit function, yielding

$$
p(y\mid f(\mathbf{x})) = \text{logit}^{-1}\Bigg(\frac{b_y - f(\mathbf{x})}{s}\Bigg) - \text{logit}^{-1}\Bigg(\frac{b_{y-1} - f(\mathbf{x})}{s}\Bigg).
\tag{5}
$$

For identifiability, the standard deviation in the probit likelihood and the scale in the logistic likelihood are typically set as $$\sigma, s =1$$. In GPyTorch, we define the likelihood distribution in the ```forward``` method,

{% highlight python %}
def forward(self, function_dist, *args, **kwargs):
    f = function_dist.unsqueeze(-1)  
    cutpoints = self.cutpoints.squeeze()
    cutpoints = cutpoints.view(*([1] * (f.dim() - 1)), -1)
    pa = self.normal.cdf(cutpoints - f) 
    p = torch.empty(*pa.shape[:-1], self.K, device=f.device, dtype=f.dtype)
    p[..., 0] = pa[..., 0]
    p[..., 1:-1] = pa[..., 1:] - pa[..., :-1]
    p[..., -1] = 1 - pa[..., -1]
    return torch.distributions.Categorical(probs=p)
{% endhighlight %}

#### <ins>Inference</ins>

Due to the use of a non-Gaussian likelihood, we rely on approximate inference methods in the GP model. {% cite chu_gaussian_2005 %} specifically discuss using either the Laplace approximation around the mode of the posterior distribution (MAP) or Expecation Propagation. However, GPyTorch has built-in implementations for approximate inference including ELBO-based variational inference {% cite hensman_scalable_2015 %} or variational inference with an optimization objective based on the log predictive likelihood for an emphasis on predictive performance {% cite jankowiak_parametric_2020 %}. 

First, we will take a look at the method discussed in {% cite hensman_scalable_2015 %}, who show that a single variational lower bound on the marginal likelihood can be derived in the form

$$
\log p({\mathbf y}) \geq \mathbb{E}_{q({\mathbf f})}[\log p(\mathbf y \mid \mathbf{f})] - KL[q({\mathbf u}) || p({\mathbf u})].
\tag{6}
$$

Here, $$\mathbf y$$ is an $$n$$-dimensional vector of observations, and $$\mathbf f = [f(\mathbf x_1), f(\mathbf x_2), \dots, f(\mathbf x_n)]$$. The vector of inducing points $$\mathbf u$$ is $$m$$-dimensional ($$m<<n$$) with prior distribution $$p({\mathbf u}) = \mathcal{N}(0, K_{mm})$$. The inducing points are a function of inducing inputs, $$\mathbf{u} = f(Z), Z \in \mathbb{R}^{m \times d}$$, whose locations are learned during training, s.t. $$K_{mm} = k(Z, Z)$$. The objective is to optimize the parameters of the variational distributional family $$q({\mathbf u})$$, which induces $$q({\mathbf f}) := \int p(\mathbf{f} \mid \mathbf{u})q(\mathbf{u})d\mathbf{u}$$, the variational posterior in the function space. The derivation for this lower bound is included in the Appendix [A2](#a2-variational-lower-bound). The use of sparse inducing point methods improves scalability for large datasets.

 As we are using a Gaussian prior in the GP, we will use a multivariate normal variational distribution family $$q({\mathbf u}) = \mathcal{N}({\mathbf u} \mid {\boldsymbol\mu}, \Sigma)$$. Because the conditional distribution $$p(\mathbf{f} \mid \mathbf{u})$$ is also multivariate normal, the variational posterior $$q(\mathbf{f})$$ can be computed analytically,

$$
    q(\mathbf f) = \mathcal{N}(\mathbf{f} \mid A{\boldsymbol \mu}, K_{nn} + A(\Sigma - K_{mm})A^\top),
    \tag{7}
$$

where $$A = K_{nm}K_{mm}^{-1}$$ (see [A3](#a3-analytical-form-of-posterior) for derivation). The two terms in the lower bound (6) can now be understood as the difference between the expectation of the log likelihood w.r.t the multivariate normal $$q(\mathbf{f})$$ and a KL divergence between two Gaussian distributions, which is analytically tractable. In GPyTorch, we can explicitly define a function for computing the first term in our custom likelihood class:

{% highlight python %}
def expected_log_prob(self, observations, function_dist, *args, **kwargs):    
    log_prob_lambda = lambda function_samples: (
        self.forward(function_samples, *args, **kwargs)
        .log_prob(observations)
    )
    return self.quadrature(log_prob_lambda, function_dist) 
{% endhighlight %}

Because this is a one-dimensional likelihood, the expectation of the log-likelihood w.r.t. the multivariate normal $$q(\mathbf{f})$$ may be efficiently approximated with Gauss-Hermite quadrature. 

The KL divergence term will be handled internally by GPyTorch by defining the marginal likelihood with the ```VariationalElbo``` class. Under the hood, GPyTorch handles defining the derivatives and optimizing the parameters $$\boldsymbol{\mu}, \Sigma$$ of the distribution $$q(\mathbf{u})$$. Additionally, we will use the Cholesky factorization of $$\Sigma = LL^\top$$ via the ```CholeskyVariationalDistribution``` class, ensuring that $$\Sigma$$ remains positive semi-definite during optimization of $$L$$.

Alternatively, the marginal log-likelihood may be approximated using the ```PredictiveLogLikelihood``` class, which instead computes the objective proposed in {% cite jankowiak_parametric_2020 %},

$$
\begin{aligned}
    \sum_{i=1}^N \log \mathbb{E}_{q({f_i})}[ p(y_i \mid f_i)] - KL[q({\mathbf u}) || p({\mathbf u})].
\end{aligned}
\tag{8}
$$

This objective is no longer a conventional ELBO objective, but was shown to reduce predictive variance in comparison to the ```VariationalELBO``` method. A derivation for this objective is included in Appendix [A4](#a4-log-predictive-likelihood-objective). To use this approximate inference approach, the ```log_marginal``` method must be implemented in the custom likelihood class. For the ordered probit, the marginal $$\mathbb{E}_{q(f_i)}[p(y_i \mid f_i)]$$ has an analytical form, which may be implemented as 

{% highlight python %}
def marginal(self, function_dist, *args, **kwargs):
    mu = function_dist.mean.unsqueeze(-1)
    var = function_dist.variance.unsqueeze(-1)
    cutpoints = self.cutpoints.squeeze()
    cutpoints = cutpoints.view(*([1] * (mu.dim() - 1)), -1)
    total_sd = torch.sqrt(var + 1.0)
    pa = self.normal.cdf((cutpoints - mu).div(total_sd)) 
    p = torch.zeros(*mu.squeeze().shape, self.K, device=mu.device, dtype=mu.dtype)
    p[..., 0] = pa[..., 0]
    p[..., 1:-1] = pa[..., 1:] - pa[..., :-1]
    p[..., -1] = 1 - pa[..., -1]
    return torch.distributions.Categorical(probs=p)

def log_marginal(self, observations, function_dist, *args, **kwargs):
    marginal_dist = self.marginal(function_dist, *args, **kwargs)
    logp = marginal_dist.log_prob(observations)
    return logp
{% endhighlight %}

#### <ins>Prediction</ins>

The variational posterior predictive for the latent $$f_*$$ at a new input $$\mathbf{x}_*$$ uses the tuned parameters $$\theta_* = \{\boldsymbol\mu, \Sigma\}$$ of $$q(\mathbf{u})$$. It is evaluated as,

$$
\begin{aligned}
    &q(f_* \mid \mathbf{x}_*,  D, \theta_*) = \int p(f_* \mid \mathbf{u}, \mathbf{x}_*)q(\mathbf{u} \mid  D, \theta_*)d\mathbf{u},\\
    &= \mathcal{N}(f_* \mid \mathbf{a}_*{\boldsymbol \mu},~ k(\mathbf{x}_*, \mathbf{x}_*) + \mathbf{a}_*(\Sigma - K_{mm})\mathbf{a}_*^\top),
\end{aligned}
    \tag{9}
$$

where $$\mathbf{a}_* =  k(\mathbf{x}_*, Z)K_{mm}^{-1}$$. With the ordered probit likelihood, the posterior predictive distribution for the observed responses has the analytical form (derived in [A5](#a5-analytical-form-of-posterior-predictive))

$$
    \begin{aligned}
    p(y \mid \mathbf{x_*}, D, \theta_*) &= \int p(y \mid f_*, \theta_*)q(f_* \mid \mathbf{x}_*,  D, \theta_*)df_*\\
    &= \Phi\Bigg(\frac{b_y - \mu_p}{\sqrt{\sigma^2 + \sigma^2_p}}\Bigg) - \Phi\Bigg(\frac{b_{y-1} - \mu_p}{\sqrt{\sigma^2 + \sigma^2_p}}\Bigg)
    \end{aligned}
$$

where the posterior mean and variance given input $$\mathbf{x}_*$$ are defined as in (9),

$$
\begin{aligned}
    \mu_p &= \mathbf{a}_*{\boldsymbol \mu} \\
    \sigma^2_p &= k(\mathbf{x}_*, \mathbf{x}_*) + \mathbf{a}_*(\Sigma - K_{mm})\mathbf{a}_*^\top.
\end{aligned}
$$

## GPyTorch Implementation

The full custom likelihood class for the ordered probit likelihood is, 

<details class="code-panel">
<summary>Ordered Probit Likelihood</summary>

{% highlight python %}
class OrderedProbitLikelihood(gpytorch.likelihoods._OneDimensionalLikelihood):
    def __init__(self, K):
        super().__init__()
        self.normal = torch.distributions.Normal(0.0, 1.0)
        self.K = K
        self.nu_c = nn.Parameter(torch.zeros(1))
        self.delta_unconstrained = nn.Parameter(torch.zeros(K-1))

    @property
    def cutpoints(self):
        delta = F.softplus(self.delta_unconstrained)
        return self.nu_c[:, None] + torch.cumsum(delta, dim=-1) 

    def forward(self, function_dist, *args, **kwargs):
        f = function_dist.unsqueeze(-1)  
        cutpoints = self.cutpoints.squeeze()
        cutpoints = cutpoints.view(*([1] * (f.dim() - 1)), -1)
        pa = self.normal.cdf(cutpoints - f) 
        p = torch.empty(*pa.shape[:-1], self.K, device=f.device, dtype=f.dtype)
        p[..., 0] = pa[..., 0]
        p[..., 1:-1] = pa[..., 1:] - pa[..., :-1]
        p[..., -1] = 1 - pa[..., -1]
        return torch.distributions.Categorical(probs=p)

    def marginal(self, function_dist, *args, **kwargs):
        mu = function_dist.mean.unsqueeze(-1)
        var = function_dist.variance.unsqueeze(-1)
        cutpoints = self.cutpoints.squeeze()
        cutpoints = cutpoints.view(*([1] * (mu.dim() - 1)), -1)
        total_sd = torch.sqrt(var + 1.0)
        pa = self.normal.cdf((cutpoints - mu).div(total_sd)) 
        p = torch.zeros(*mu.squeeze().shape, self.K, device=mu.device, dtype=mu.dtype)
        p[..., 0] = pa[..., 0]
        p[..., 1:-1] = pa[..., 1:] - pa[..., :-1]
        p[..., -1] = 1 - pa[..., -1]
        return torch.distributions.Categorical(probs=p)

    def log_marginal(self, observations, function_dist, *args, **kwargs):
        marginal_dist = self.marginal(function_dist, *args, **kwargs)
        logp = marginal_dist.log_prob(observations)
        return logp

    def expected_log_prob(self, observations, function_dist, *args, **kwargs):    
        log_prob_lambda = lambda function_samples: (
            self.forward(function_samples, *args, **kwargs)
            .log_prob(observations)
        )
        return self.quadrature(log_prob_lambda, function_dist)
{% endhighlight %}

</details>
\\
For the ordered logistic likelihood, the computations that may not be done analytically can be done using quadrature or MC sampling. The full implementation is,

<details class="code-panel">
<summary>Ordered Logistic Likelihood</summary>

{% highlight python %}
class OrderedLogisticLikelihood(gpytorch.likelihoods._OneDimensionalLikelihood):
    def __init__(self, K):
        super().__init__()
        self.K = K
        self.nu_c = nn.Parameter(torch.zeros(1))
        self.delta_unconstrained = nn.Parameter(torch.zeros(K-1))

    @property
    def cutpoints(self):
        delta = F.softplus(self.delta_unconstrained)
        return self.nu_c[:, None] + torch.cumsum(delta, dim=-1)

    def forward(self, function_dist, *args, **kwargs):
        f = function_dist.unsqueeze(-1)  
        cutpoints = self.cutpoints.squeeze()
        cutpoints = cutpoints.view(*([1] * (f.dim() - 1)), -1)
        pa = torch.sigmoid(cutpoints - f) 
        p = torch.empty(*pa.shape[:-1], self.K, device=f.device, dtype=f.dtype)
        p[..., 0] = pa[..., 0]
        p[..., 1:-1] = pa[..., 1:] - pa[..., :-1]
        p[..., -1] = 1 - pa[..., -1]
        return torch.distributions.Categorical(probs=p)

    def marginal(self, function_dist, *args, **kwargs):
        prob_lambda = lambda function_samples: (
            self.forward(function_samples, *args, **kwargs).probs
        )
    
        marginal_probs = self.quadrature(prob_lambda, function_dist)
        return torch.distributions.Categorical(probs=marginal_probs)
        
    def log_marginal(self, observations, function_dist, *args, **kwargs):
        marginal_dist = self.marginal(function_dist, *args, **kwargs)
        logp = marginal_dist.log_prob(observations)
        return logp
    
    def expected_log_prob(self, observations, function_dist, *args, **kwargs):    
        log_prob_lambda = lambda function_samples: (
            self.forward(function_samples, *args, **kwargs)
            .log_prob(observations)
        )
        return self.quadrature(log_prob_lambda, function_dist)
{% endhighlight %}

</details>

The full implementation of the ordinal regression model in GPyTorch is available at [FMurphy17/gp_ord_regression_gpytorch](https://github.com/FMurphy17/gp_ord_regression_gpytorch), including running the model on the diabetes dataset as done in {% cite chu_gaussian_2005 %}.

## Appendix

#### **(A1) Difference of Probit Likelihood**
$$
\begin{aligned}
p(y|f(\mathbf{x})) &= p(b_{y-1} < f(\mathbf{x}) + \varepsilon < b_y),\\
&= p(b_{y-1} - f(\mathbf{x})< \varepsilon < b_y - f(\mathbf{x})),\\
&= \int^{b_{y} - f(\mathbf{x})}_{b_{y-1} - f(\mathbf{x})}\mathcal{N}(\varepsilon; 0, \sigma^2)d\varepsilon,\\
&= \int^{b_{y} - f(\mathbf{x})}_{-\infty}\mathcal{N}(\varepsilon; 0, \sigma^2)d\varepsilon - \int^{b_{y-1} - f(\mathbf{x})}_{-\infty}\mathcal{N}(\varepsilon; 0, \sigma^2)d\varepsilon,\\
&\stackrel{\star}{=} \int^{\frac{b_{y} - f(\mathbf{x})}{\sigma}}_{-\infty}\mathcal{N}(\varepsilon; 0, 1)d\varepsilon - \int^{\frac{b_{y-1} - f(\mathbf{x})}{\sigma}}_{-\infty}\mathcal{N}(\varepsilon; 0, 1)d\varepsilon,\\
&=\Phi\Bigg(\frac{b_y - f(\mathbf{x})}{\sigma}\Bigg) - \Phi\Bigg(\frac{b_{y-1} - f(\mathbf{x})}{\sigma}\Bigg),
\end{aligned}
$$

where we normalized the bounds by the standard deviation in step $$\star$$ to recover the unit-variance probit function.

#### **(A2) Variational Lower Bound**
We begin with a lower bound on the log-likelihood using inducing points $$\mathbf{u}$$,

$$
\begin{aligned}
\log p(\mathbf{y} \mid \mathbf{u}) \geq \mathbb{E}_{p(\mathbf{f} \mid \mathbf{u})}[\log p(\mathbf{y} \mid \mathbf{f})].
\end{aligned}
\tag{A2.1}
$$

This follows because we can define the likelihood using inducing points as 

$$
\begin{aligned}
p(\mathbf{y} \mid \mathbf{u}) &= \int p(\mathbf{y} \mid \mathbf{f})p(\mathbf{f} \mid \mathbf{u})d \mathbf{f},\\
&= \mathbb{E}_{p(\mathbf{f} \mid \mathbf{u})}[ p(\mathbf{y} \mid \mathbf{f})],
\end{aligned}
\tag{A2.2}
$$

and Jensen's inequality states that $$\log \mathbb{E}_{p(\mathbf{f} \mid \mathbf{u})}[ p(\mathbf{y} \mid \mathbf{f})] \geq  \mathbb{E}_{p(\mathbf{f} \mid \mathbf{u})}[\log p(\mathbf{y} \mid \mathbf{f})]$$. The standard form of the Evidence Lower Bound (ELBO) using inducing points is 

$$
\begin{aligned}
\log p({\mathbf y}) \geq \mathbb{E}_{q(\mathbf{u})}[\log p(\mathbf{y} \mid \mathbf{u})] - KL[q(\mathbf{u}) || p(\mathbf{u})].
\end{aligned}
\tag{A2.3}
$$

Substituting (A2.1) into (A2.3), we obtain the new lower bound 

$$
\begin{aligned}
\log p({\mathbf y}) \geq \mathbb{E}_{q(\mathbf{u})}[\mathbb{E}_{p(\mathbf{f} \mid \mathbf{u})}[\log p(\mathbf{y} \mid \mathbf{f})]] - KL[q(\mathbf{u}) || p(\mathbf{u})],
\end{aligned}
\tag{A2.4}
$$

where we see the first term is equivalent to 

$$
\begin{aligned}
\mathbb{E}_{q(\mathbf{u})}[\mathbb{E}_{p(\mathbf{f} \mid \mathbf{u})}[\log p(\mathbf{y} \mid \mathbf{f})]] &= \int q(\mathbf{u})\int p(\mathbf{f} \mid \mathbf{u}) \log p(\mathbf{y} \mid \mathbf{f})d\mathbf{f}d\mathbf{u},\\
&= \int\log p(\mathbf{y} \mid \mathbf{f}) \int p(\mathbf{f} \mid \mathbf{u})q(\mathbf{u})d\mathbf{u}d\mathbf{f},
\end{aligned}
\tag{A2.5}
$$

such that defining $$q(\mathbf{f}) := \int  p(\mathbf{f} \mid \mathbf{u})q(\mathbf{u})d\mathbf{u}$$ yields the variational bound in the function space

$$
\begin{aligned}
    \log p({\mathbf y}) \geq \mathbb{E}_{q(\mathbf{f})}[\log p(\mathbf{y} \mid \mathbf{f})] - KL[q(\mathbf{u}) || p(\mathbf{u})].
\end{aligned}
\tag{A2.6}
$$

#### **(A3) Analytical form of posterior**

If $$p(\mathbf{f}) = \mathcal{N}(0, K_{nn})$$, $$p(\mathbf{u}) = \mathcal{N}(0, K_{mm})$$, then $$p(\mathbf{f})$$ and $$p(\mathbf{u})$$ are jointly Gaussian s.t. the conditional distribution using inducing points is the multivariate Gaussian $p(\mathbf{f} \mid \mathbf{u}) = \mathcal{N}(\mathbf{f} \mid K_{nm}K_{mm}^{-1}\mathbf{u}, K_{nn} - K_{nm}K_{mm}^{-1}K_{nm}^\top)$. With $$q(\mathbf{u}) = \mathcal{N}(\mathbf{u} \mid \boldsymbol{\mu}, \Sigma)$$, the variational posterior is equivalent to

$$
\begin{aligned}
    q(\mathbf{f}) &= \int p(\mathbf{f} \mid \mathbf{u})q(\mathbf{u})d\mathbf{u},\\
    &= \int \mathcal{N}(\mathbf{f} \mid K_{nm}K_{mm}^{-1}\mathbf{u}, K_{nn} - K_{nm}K_{mm}^{-1}K_{nm}^\top) \mathcal{N}(\mathbf{u} \mid \boldsymbol{\mu}, \Sigma)d\mathbf{u}.
\end{aligned}
$$

The posterior mean and covariance can be derived using the Laws of Total Expectation and Total Covariance. The posterior mean simplifies via 

$$
\begin{aligned}
\mathbb{E}[\mathbf{f}] &= \mathbb{E}_{\mathbf{u}}[\mathbb{E}[\mathbf{f}\mid\mathbf{u}]],\\
&= \mathbb{E}_{\mathbf{\mathbf{u}}}[A\mathbf{u}],\\
&= A\boldsymbol{\mu}.
\end{aligned}
$$

where $A = K_{nm}K_{mm}^{-1}$. The posterior covariance is derived via 

$$
\begin{aligned}
    Cov(\mathbf{f}) &= \mathbb{E}_\mathbf{u}[Cov(\mathbf{f} \mid \mathbf{u})] + Cov(\mathbb{E}[\mathbf{f}\mid\mathbf{u}]),\\
    &= \mathbb{E}_\mathbf{u}[K_{nn} - K_{nm}K_{mm}^{-1}K_{nm}^\top] + Cov(A\mathbf{u}),\\
    &= K_{nn} - K_{nm}K_{mm}^{-1}K_{nm}^\top + ACov(\mathbf{u})A^\top,\\
    &= K_{nn} - K_{nm}K_{mm}^{-1}K_{nm}^\top + A\Sigma A^\top,\\
    &= K_{nn} + A(\Sigma - K_{mm})A^\top.
\end{aligned}
$$

#### **(A4) Log-predictive likelihood objective**

The objective presented in {% cite jankowiak_parametric_2020 %} targets the predictive likelihood, aiming to minimize the KL divergence (maximize negative KL) between the actual data-generating conditional distribution $$p_{data}(y \mid \mathbf x)$$ and the predictive conditional $$p(y \mid \mathbf x)$$,

$$
\begin{aligned}
    -\mathbb{E}_{p_{data}(\mathbf x)}[KL(p_{data}(y \mid \mathbf x) || p(y \mid \mathbf x))].
\end{aligned}
\tag{A4.1}
$$

As the conditional is dependent on possible inputs $$\mathbf x$$, this is an expectation w.r.t the data-distribution of inputs. Expanding (A4.1) results in, 

$$
\begin{aligned}
&= -\mathbb{E}_{p_{data}(\mathbf x)}\mathbb{E}_{p_{data}(y \mid \mathbf{x})}[\log p_{data}(y \mid \mathbf x) - \log p(y \mid \mathbf x)],\\
&\stackrel{\star}{=} \mathbb{E}_{p_{data}(\mathbf{x}, y)}[\log p(y \mid \mathbf x)] - \mathbb{E}_{p_{data}(\mathbf{x}, y)}[\log p_{data}(y \mid \mathbf x)],
\end{aligned}
\tag{A4.2}
$$

where $$\star$$ follows because $$p_{data}(\mathbf x)p_{data}(y \mid \mathbf x) = p_{data}(\mathbf x, y)$$. The second term $$\mathbb{E}_{p_{data}(\mathbf{x}, y)}[\log p_{data}(y \mid \mathbf x)]$$ is the conditional entropy for the data-generating distribution, which is independent of the predictive model, and is thus removed from the maximization objective. 

As the true data-generating distribution is unknown, it is approximated via empirical data samples, 

$$
\begin{aligned}
\mathbb{E}_{p_{data}(\mathbf{x}, y)}[\log p(y \mid \mathbf x)] \approx \frac{1}{N}\sum_{i=1}^N\log p (y_i \mid \mathbf{x}_i),
\end{aligned}
$$

yielding the Maximum Likelihood Estimation objective. 

With the use of a GP, the posterior-predictive $$p(y_i \mid \mathbf{x}_i)$$ requires integrating out the latent $$f_i$$, such that

$$
\begin{aligned}
    p(y_i \mid \mathbf{x}_i) = \int p(y_i \mid f_i)p(f_i \mid \mathbf{x}_i, D)df_i,
\end{aligned}
$$

which is approximately 

$$
\begin{aligned}
    \mathbb{E}_{q(f_i)}[p(y_i \mid f_i)],
\end{aligned}
$$

in the variational setting. This results in the final objective,

$$
\begin{aligned}
 \sum_{i=1}^N\log\mathbb{E}_{q(f_i)}[p(y_i \mid f_i)] - KL(q(\mathbf u) || p(\mathbf u)),
\end{aligned}
$$

to which the inducing point KL is added as a regularization term.

#### **(A5) Analytical form of posterior predictive**

This derivation closely follows $$\underline{A3}$$, making use of the Law of Total Expectation and Law of Total Variance. For the ordered probit likelihood, we can write the posterior predictive as 

$$
\begin{aligned}
p(y \mid \mathbf{x_*}, D, \theta_*) &= \int p(y \mid f_*, \theta_*)q(f_* \mid \mathbf{x}_*, D, \theta_*)df_*,\\
&= \int \Bigg(\Phi\Bigg(\frac{b_y - f_*}{\sigma}\Bigg) - \Phi\Bigg(\frac{b_{y-1} - f_*}{\sigma}\Bigg)\Bigg)\mathcal{N}(f_* \mid \mathbf{a}_*{\boldsymbol \mu}, k(\mathbf{x_*}, \mathbf{x_*}) + \mathbf{a}_*(\Sigma - K_{mm})\mathbf{a}_*^\top)df_*,
\end{aligned}
\tag{A5.1}
$$

using the definitons in (4) and (9), where $$\mathbf{a}_* =  k(\mathbf{x}_*, Z)K_{mm}^{-1}$$ and we can define the posterior mean and covariance

$$
\begin{aligned}
&\mu_p = \mathbf{a}_*{\boldsymbol \mu},\\
&\sigma^2_p = k(\mathbf{x_*}, \mathbf{x_*}) + \mathbf{a}_*(\Sigma - K_{mm})\mathbf{a}_*^\top.
\end{aligned}
$$

The difference in (A5.1) can be written as two separate integrals

$$
\begin{aligned}
p(y \mid \mathbf{x_*}, D, \theta_*) &= \int \Phi\Bigg(\frac{b_y - f_*}{\sigma}\Bigg)\mathcal{N}(f_* \mid \mu_p, \sigma^2_p)df_*\\
&- \int \Phi\Bigg(\frac{b_{y-1} - f_*}{\sigma}\Bigg)\mathcal{N}(f_* \mid \mu_p, \sigma^2_p)df_*.
\end{aligned}
$$

For each integral, we can derive the expectation of $$y$$ as 

$$
\begin{aligned}
\mathbb{E}[y] &= \mathbb{E}_{f_*}[\mathbb{E}[y \mid f_*]],\\
&= \mathbb{E}_{f_*}[f_*],\\
&= \mu_p
\end{aligned}
$$

and the variance of $$y$$ as

$$
\begin{aligned}
Var(y) &= \mathbb{E}_{f_*}[Var(y \mid f_*)] + Var[\mathbb{E}(y \mid f_*)],\\
&= \mathbb{E}_{f_*}(\sigma^2) + Var(f_*),\\
&= \sigma^2 + \sigma_p^2.
\end{aligned}
$$

For the difference of probits, this yields the analytical posterior predictive distribution over the ordinal targets

$$
\begin{aligned}
\Phi\Bigg(\frac{b_y - \mu_p}{\sqrt{\sigma^2 + \sigma^2_p}}\Bigg) - \Phi\Bigg(\frac{b_{y-1} - \mu_p}{\sqrt{\sigma^2 + \sigma^2_p}}\Bigg).
\end{aligned}
$$

#### References

{% bibliography --cited %}
