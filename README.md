Domain Explorer: sampling a domain

In this library we look at sampling a "hard" domain and testing that the sample has indeed covered the domain of interest.

Motivation: 
In biological applications (such as DSGRN) parameters of interest are found in domains defined by semi-algebraic conditions. We are therefore interested in searching
the domains and create samples that approach the uniform distribution over the domain.
We explore two approaches for this goal: the Metropolis-Hasting/Brownian motion algorithm and the biliard algorithm.

Once the sample is computed, the following problem is to test the vicinity to the uniform distribution. Tools are provided to this effect (in development).
