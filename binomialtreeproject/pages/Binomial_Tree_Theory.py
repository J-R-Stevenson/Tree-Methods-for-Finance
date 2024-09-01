import streamlit as st
import Binomial_Tree_Calculator as btc
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import graphviz
import pandas as pd
from PIL import Image

st.set_page_config(
    layout="centered",
    page_title="Binomial Tree Theory",
)


st.sidebar.success("Select a page above.")

def GBM_simulate():
	st.image("binomialtreeproject/images/GBMpaths.png")
	st.image("binomialtreeproject/images/GBMdistribution.png")
	return "*1,000 sample GBM paths with a plot of the sample distribution*"

def one_step_tree():
	dot = graphviz.Digraph()
	dot.attr('node', shape='box')

	dot.node("1", label="100", style='rounded', color='white', fontcolor='white')
	dot.node("2", label="120", style='rounded', color='white', fontcolor='white')
	dot.node("3", label="80", style='rounded', color='white', fontcolor='white')

	dot.edge('1', '2', color='white')
	dot.edge('1', '3', color='white')

	dot.attr(rankdir='LR')
	dot.attr(bgcolor='transparent')

	st.graphviz_chart(dot, use_container_width=True)

	return "*One-step binomial tree*"

def multi_step_tree():
	dot = graphviz.Digraph()
	dot.attr('node', shape='box')

	dot.node("1", label="100", style='rounded', color='white', fontcolor='white')
	dot.node("2", label="120", style='rounded', color='white', fontcolor='white')
	dot.node("3", label="80", style='rounded', color='white', fontcolor='white')
	dot.node("4", label="144", style='rounded', color='white', fontcolor='white')
	dot.node("5", label="100", style='rounded', color='white', fontcolor='white')
	dot.node("6", label="64", style='rounded', color='white', fontcolor='white')

	dot.edge('1', '2', color='white')
	dot.edge('1', '3', color='white')
	dot.edge('2', '4', color='white')
	dot.edge('2', '5', color='white')
	dot.edge('3', '5', color='white')
	dot.edge('3', '6', color='white')


	dot.attr(rankdir='LR')
	dot.attr(bgcolor='transparent')

	st.graphviz_chart(dot, use_container_width=True)

	return "*Multi-step binomial tree*"
    
def plotTerminalPrices():
	up = 1.2
	down = 0.8
	ns = [10, 25, 50, 100]
	fig, ax = plt.subplots()
	for n in ns:
		p = (np.exp(0.05 * (1 / n)) - down) / (up - down)
		S_T = np.array([100 * np.power(up, n - i) * np.power(down, i) for i in range(n)])
		probs = np.array([stats.binom.pmf(k, n, p) for k in range(n)])
		ax.plot(S_T, probs)
	plt.xlim(0, 600)
	plt.xlabel(f"Terminal stock prices (S(T))", fontname='times new roman')
	plt.ylabel(f"Probability", fontname='times new roman')
	plt.legend(['n=10', 'n=25','n=50','n=100'])

	st.pyplot(fig)

	return "*Distribution of terminal stock prices converging in distribution to the lognormal*"

def compareBinomialConvergence():
	st.image("binomialtreeproject/images/CRRJRTianConvergence.png")
	return "*Comparison of convergence of different parameter selections to analytic price*"

def compareBinomialConvergenceWithLR():
    prices = []
    sigma = 0.3
    r = 0.07
    ns = [25, 50, 75, 100, 200]

    # CRR
    CRR_prices = []
    for n in ns:
        up, down, prob = btc.CRRParams(sigma, 0.5 / n, r)
        BT = btc.EuropeanTree(n, up, down, 100, r, 0.5 / n, "European call", prob, 100)
        CRR_prices.append(BT.nodes[(0,0)].option_price)
    prices.append(CRR_prices)

    # JR
    JR_prices = []
    for n in ns:
        up, down, prob = btc.JRParams(sigma, 0.5 / n, r)
        BT = btc.EuropeanTree(n, up, down, 100, r, 0.5 / n, "European call", prob, 100)
        JR_prices.append(BT.nodes[(0,0)].option_price)
    prices.append(JR_prices)

    # Tian
    Tian_prices = []
    for n in ns:
        up, down, prob = btc.TianParams(sigma, 0.5 / n, r)
        BT = btc.EuropeanTree(n, up, down, 100, r, 0.5 / n, "European call", prob, 100)
        Tian_prices.append(BT.nodes[(0,0)].option_price)
    prices.append(Tian_prices)  

    # LR
    LR_prices = []
    for n in ns:
        up, down, prob = btc.LRParams(sigma, 0.5 / n, r, n)
        BT = btc.EuropeanTree(n, up, down, 100, r, 0.5 / n, "European call", prob, 100)
        LR_prices.append(BT.nodes[(0,0)].option_price)

    prices.append(LR_prices)        
    
    df = pd.DataFrame(prices)
    df.columns = ['n=25', 'n=50', 'n=75', 'n=100', 'n=200']
    df.index = ['CRR', 'Jarrow-Rudd', 'Tian', 'Leisen-Reimer']
    st.table(df)
    return "*Comparison of convergence of CRR, JR, Tian, and LR parameter selection methods to analytic price (\\$10.13377)*"



def drawTrinomialTree():
	dot = graphviz.Digraph()
	dot.attr('node', shape='box')

	dot.node("1", label="100", style='rounded', color='white', fontcolor='white')
	dot.node("2", label="120", style='rounded', color='white', fontcolor='white')
	dot.node("3", label="100", style='rounded', color='white', fontcolor='white')
	dot.node("4", label="80", style='rounded', color='white', fontcolor='white')

	dot.edge('1', '2', color='white')
	dot.edge('1', '3', color='white')
	dot.edge('1', '4', color='white')

	dot.attr(rankdir='LR')
	dot.attr(bgcolor='transparent')

	st.graphviz_chart(dot, use_container_width=True)

	return "*Trinomial stock price tree*"

def naiveBarrierConvergence():
	st.image("binomialtreeproject/images/NaiveBarrierConvergence.png")
	return "*Convergence of naive binomial tree to analytic price of a down-and-out call*"
 
def drawBarrierTreeConvergence():
	st.image("binomialtreeproject/images/TrinomialNaiveConvergenceComparison.png")
	return "*Comparison of convergence of naive binomial tree and Ritchken trinomial tree to analytic price of a down-and-out call*"

def drawAmericanTree():
	btc.AmericanTree(6, 1.2, 0.8, 100, 0.06, 1 / 6, 'American put', (np.exp(0.06/6) - 0.8) / (1.2 - 0.8), 100).draw_tree()
	return "*American put tree*"

st.write(f"""On this page, we will provide an overview of binomial tree methods in finance as a supplement to the calculator and visualizer on
	the other page. Readers are encouraged to have some familiarity with probability theory (at least a working understanding of random variables
	of distributions that can be obtained in an introductory course) and a basic understanding of options (knowing the definitions of calls
	puts will suffice). 


__Introducing binomial trees__


Binomial trees are perhaps the simplest method of option pricing. This simplicity makes them ubiquitous in introductory textbooks on
derivative securities, but we will see that their uses are not limited to the pedagogical: tree methods and their variants can be modified to
price a wide variety of derivative securities with quite complicated payoff functions; hey can easily handle the early-exercise property of
American-style derivatives; and though the convergence of the original binomial tree can be relatively slow, subsequent research has developed techniques 
that significantly improved the speed of convergence. 

The idea of pricing options using binomial tree is said to have first been suggested by William Sharpe, but the foundational model still
widely cited in the literature comes from a 1979 paper by Cox, Ross, and Rubinstein (hereafter abbreviated to CRR). The
initial motivation for binomial trees was to present a mathematically simple alternative to the mathematically sophisticated Black-Scholes (or Black-Scholes-Merton)
option pricing model. The trees would make clear the economics of option pricing, particularly the no-arbitrage and static hedging
arguments, while also being usable by finance practioners who did not possess the necessary mathematical background to understand
the arguments of Black-Scholes. Since then, trees has been used for far more than simplification, and on this page we will see still only a sample
of the uses of trees.  



__Option prices from a binomial tree__

We will now consider how to actually calculate option prices using binomial trees. We begin by considering a stock with price ${{S_0}}$ = 100 at time 
t = 0. At the next time, t = T, assume two possible outcomes for the stock price: it increases to 120 or it decreases to 80.""")

st.write(f"""{one_step_tree()}

The original CRR paper introduced the binomial tree using a portflio consisting of calls, stocks, and a bond. Consider a portfolio consisting of 
*x* shares of the stock and an amount *y* of a risk-free asset (e.g. a bond). The key point mathematically is that the stock price at time 
T is not known for certain, though we can assign probabilities to it moving up to 120 or down to 80, and the bond price at time T is known for 
certain. 

Suppose that we wrote (i.e. sold) one European call option on the stock. A call option gives the holder the right (but not the obligation) to purchase 
the stock at aspecificed time (T in this instance) for a specified price (known as the strike price and denoted by K). The payoff of the call option at 
maturity (time T) is the maximum of ${{S_T - K}}$ or 0. The latter occurs when the stock price is at or below the strike price, in which case
the holder would not exercise the option since doing so would lead to a loss. That the call option is European just means that we can only exercise
it at maturity. Other styles of options, namely American and Bermudan, allow for the holder to exercise both before and at maturity. 

We want to construct a riskless portfolio consisting of some combination of the stock
and bond, that is to say a portfolio that will replicate the payoff of the call option regardless of the movement of the stock price. Thus if
the stock moves up to 120 and we must, as writer of the call option, deliver a stock worth \\$120 in exchange for \\$110, then we will be
compensated by the portfolio so that end up with no loss or gain. If the stock moves down to \\$80, the same must occur. 

The result is a replicating portfolio, a combination of stock and bond that will replicate the payoff of the call option regardless of the
movement of the stock price. The value of this replicating portfolio must then be the value of the call option. This last point is true only
if we assume the principle of no arbitrage: it is not possible for an investor to make a net initial investment of $0 that (1) has a non-negative 
probability of a positive profit and (2) zero probability of a loss. The idea is that in the market, any such "arbitrages" would be quickly identified 
and disappear. Regardless of the fact that real markets do not perfectly adhere to this principle, it suffices for our model to assume that no 
arbitrage is possible. 

Let us now see how to construct a replicating portfolio in general. We purchase *x* amount of the stock and *y* amount of the bond. We determine *x* 
and *y* by the following system of equations:

${{xSu + ye^{{r\\Delta t}} = C_u}}$

${{xSd + ye^{{r\\Delta t}} = C_d}}$

where *Su* and *Sd* are the prices of the stock at time *T* for an up and down movement respectively and ${{C_u}}$ and ${{C_d}}$ are the payoffs of 
the option at time *T*. We solve this system to get the following values for *x* and *y*:

${{x = \\frac{{C_u - C_d}}{{S(u - d)}} }}$

${{y = e^{{-r{{\\Delta}} t}}(C_u - (C_d))}}$

Then if we plug in *x* and *y* to the original portfolio, we get the value of the call option:

${{C = xS + y}}$

${{C = e^{{-rT}}[ \\frac{{e^{{rT}} - d}}{{u - d}}C_u + \\frac{{u - e^{{rT}} }}{{u - d}}C_d ]}}$

We mentioned earlier that there were a pair of probabilities associated with the movement of the stock price, but note that nowhere did these
probabilities enter into our formulas. For the purposes of no-arbitrage pricing we do not need them. Insofar as the actual probability of the
stock price moving to some particular price over the lifetime of the option affects the option price, it does so through other variables in 
our model. 

However, we can look a bit closer at the coefficient ${{C_u}}$ in the above formula. Let us denote this coefficient by the variable *p*:

${{p = \\frac{{e^{{rT}} - d}}{{u - d}}}}$

Under the principle of no-arbitrage, p cannot be greater than or equal to 1. It also cannot be less than or equal to 0. In order
to understand these restrictions, we need to clarify some conditions on *u* and *d*. The no-arbitrage principle requires that 
$${{0 < d < e^{{rT}} < u}}$$. Suppose that ${{e^{{rT}}  {{\\geq}}  u}}$. Then borrow the stock at time 0. Immediately sell it and
reinvest the proceeds in a bond. At time T, buy the stock and return it using proceeds from the redemption of the bond. There will
be a profit of ${{S_0e^{{rT}} - S_T > 0}}$ regardless of the movement of the stock price. We therefore must have ${{e^{{rT}} > u}}$
to prevent arbitrage opportunities. A similar argument can be made to show that ${{e^{{rT}} < d}}$ must hold. 

The condition that ${{e^{{rT}} > d}}$ is equivalent to requiring that *p* is strictly positive since *u* > *d* by definition (if u = d then
our stock is just a risk-free asset and if u < d then we can just relabel them). The condition that ${{e^{{rT}} > u}}$ is equivalent to
requiring that *p* is strictly less than 1 (otherwise the numerator of *p* is greater than or equal to the denominator). 

All of this might be simplified by saying that *p* must be in the interval (0,1). We can then think of p as a probability, but then we should wonder 
of what is p the probability? Consider that we can rewrite our option pricing formula to obtain

${{C = e^{{-rT}}[pC_u + (1 - p)C_d]}}$

The value of the call option is the discounted expected value of the payoff *if *p* were the probability of the stock price
moving up.* We can determine the precise circumstances under which p is just that probability. Let the size of the movements
and the lifetime of the option be fixed. Then *p* is a function of the risk-free rate *r*. Moreover, suppose that p was the probability
associated with stock price movements. Then the expected growth rate of the stock would be

${{S_0e^{{rT}} = pSu + (1 - p)Sd}}$

In other words, *p* is the probability in the world in which our risky assets grows at a risk-free rate. In such a world, investors
would then be indifferent toward risk since risky and risk-free assets promise the same returns. 

We summarize the above findings in the following slogan: *the price of a derivative security is the discounted expected value of its
payoff under the risk-neutral measure.* A risk-neutral measure is one in which p is the probability associated with the stock price
moving up (for our purposes, a probability measure is just an assignment of probabilities to stock price movements). 

Risk-neutral pricing was one of the major innovations of the Black-Scholes model in the 70s, and it is still foundational
to option pricing today. We don't need to know the returns and the demands of investors. We price in risk-neutrality, where investors
are neither averse to risk (and thus require higher returns to undertake risk) nor risk-seeking (and thus require lower returns on a risky
investment before they would consider taking it). If risk is unimportant to the risk-neutral investor, then their expected return would be
the same as the return on a risk-free asset such as our bond. The actual probability of the stock price movement does not appear in our
formula for the price of an option. CRR write that

*This means, surprisingly, that even if different investors have different subjective probabilities about an upward or downward movement in the 
stock, they could still agree on the relationship of C to S, u, d, and r...[T]he value of the call does not depend on investors' attitude toward 
risk. In constructing the formula, the only assumption we made about an individual's behavior was that he prefers more wealth to less wealth and 
therefore has an incentive to take advantage of profitable riskless arbitrage opportunities...[T]he formula is only a relative pricing relationship 
giving C in terms of S, u, d, and r. Investors' attitudes toward risk and the characteristics of other assets may indeed influence call values 
indirectly, through their effect on these variables, but they will not be separate determinants of call value.*

__Multi-step trees__

It is natural to extend our one-step tree to a multi-step tree, one which would allow a greater range of outcomes. We start at the second-to-last
time-step and proceed as if each node and its two child nodes are one-step trees. We move backwards through the tree from the terminal nodes to
the initial node, which will contain the desired option price. 

The nature of the original binomial tree is such that we only add one more node at each time step. This is because the tree
recombines. This is nice fact which reduces computation time, although some textbooks present first a non-recombining
tree and then only later introduce the recombining tree. In the literature, non-recombining trees are generally avoided
if possible because of their poor computation speed (2^n grows much faster than n), but sometimes the virtues of 
the non-recombining trees, namely their ease of representing path-dependent options (to be discussed later), outweigh
the computational problems.

""") 

st.write(f"""{multi_step_tree()}

While the calculuator on the other page calculates the option price of every node in a tree, we only need to consider the stock prices of the
terminal nodes.* The terminal nodes together with the respective probabilities jointly suffice to price the option at time 0. The probability
distribution of the terminal stock prices is a binomial distribution, whose density function is 

${{P(X = k) = {{n \\choose k}}p^k(1 - p)^{{n - k}}}}$

where n is the number of terminal stock prices, p is the probability of an up movement, and k is the number of up movements for a particular
stock price path.. 

*This is not the case for all option types. For example, we need to check the option values of intermediate nodes when pricing American puts. 

The general formula for the value of a European call option is then

${{C = {{e^{{-rT}}}}\\sum_{{j = 0}}^{{n}} {{n \\choose j}}p^k(1 - p)^{{n - j}} \\text{{max}}{{\\{{ 0, S_0u^{{j}}d^{{n - j}} - K \\}}}}}}$

where K is the strike price. 

We can further simplify this formula in a way that will make clear its relation with Black-Scholes. Let *a* denote the number of up jumps above which
the stock price is greater than the stock price. For ${{j \\geq a}}$, the payoff of the call option is always ${{S_T - K}}$  (since the payoff is a
monotonically non-decreasing function of the number of up movements). 

${{C = S[\\sum_{{j \\geq a}}^{{n}} {{n \\choose j}}p^j(1 - p)^{{n-j}} \\frac{{u^jd^{{n - j}}}}{{e^{{rT}}}}] - Ke^{{-rT}}[\\sum_{{j \\geq a}}^{{n}} {{n \\choose j}}p^j(1 - p)^{{n-j}}]  }}$

Let *p'* = ${{p\\frac{{u}}{{e^{{rT}}}} }}$, which is also a probability. Let F(a; n, p) and F(a; n, p') be the cumulative distribution function sof the
binomial distribution function with parameters (*n*, *p*) and (*n*, *p'*) respectively. We can rewrite the option pricing formula to be

${{C = S_0F(a; n, p) - Ke^{{-rT}}F(a; n, p')}}$ (Binomial option pricing formula for a European call)

Readers familiar with the Black-Scholes formula for a European call will immediately notice the similarities:

${{C = S_0N(d_1) - Ke^{{-rT}}N(d_2)}}$ (Black-Scholes option pricing formula for a European call option)

${{d_1 = \\frac{{ ln(\\frac{{S_0}}{{K}}) + (r + \\frac{{\\sigma^2}}{{2}})T }}{{ \\sigma \\sqrt{{T}} }}  }}$

${{d_2 = d_1 - \\sigma\\sqrt{{T}} }}$

where ${{N(x)}}$ is the cumulative distribution function of a standard normal (or Gaussian) random variable. 

In order for the binomial option prices to converge to the Black-Scholes prices, it is sufficient to ensure that the two distribution terms,
F(a; n, p) and F(a; n, p'), converge to the Black-Scholes terms, N(d1) and N(d2). In the next section, we will explore further why the discrete binomial
process is able to do this. 


__Binomial Trees and Brownian Motion__

We now consider how the relatively simple binomial stock price process is able to approximate the more complicated continuous-time 
process assumed by Black-Scholes. 

The purpose of a binomial tree is to provide a numerical approximation to the price of a derivative security. By numerical approximation, we mainly
mean something that we can do on a computer that outputs something close to the analytic option price. The analytic price is in a sense the “true” 
option price. It is true in the sense that it is the output of the solution to the Black-Scholes partial differential equation for that option. 
	
This is an important point to which we will continually return: the binomial tree was originally developed to provide a numerical approximation to 
the Black-Scholes option price.* But for some options there were no solutions to the Black-Scholes, namely the American put, and thus binomial 
trees could provide both the theoretical backing of Black-Scholes as well as an actual price for the option. 

*Though CRR do note that the binomial tree can approximate the prices of other continuous-time models

The fact that the binomial tree is designed with Black-Scholes prices in mind will motivate much of the modifications in terms of parameter-selection. 
We want some discrete process that, as the possible outcomes (or states) become closer and closer, will approximately exhibitthe continuous-time behavior 
of a stock whose price is assumed to follow a geometric Brownian motion in the Black-Scholes framework. Geometric Brownian motion (GBM) can be defined 
as a particular kind of stochastic process. A stochastic process is simply a sequence of random variables, usually indexed by time. Suppose that you
are playing a coin-flipping game. If the coin lands on heads, you get a point and if it lands on tails you lose a point. The coin is flipped several
times (possibly infinitely many). We may represent your score as a stochastic process. Although we cannot say for certain what your score will be 
after any given flip of the coin, there is still a structure to the randomness of the game that we can study. 

The coin-flipping game is an example of a *random walk* and Brownian motion is its' continuous analogue. Suppose that ${{S_t}}$ is your score in the
game at time *t*. Define $W^{{(n)}}(t) = \\frac{{1}}{{\\sqrt{{n}}}}S_t$ to be a scaled random walk, where *n* is some integer. Fix *t* and send *n* 
to infinity. By the Central Limit Theorem, $W^{{n}}(t)$ will converge to the normal distribution with mean zero and variance *t*. The scaled random
walk itself is still a stochastic process, but if we look at a particular interval of time, the distribution of values of the stochastic process
at the end of that interval will have a normal distribution. Likewise, if we look at the last layer of nodes of our binomial tree, we will find 
that these terminal nodes have a binomial distribution.

We make some further modifications to our scaled random walk to get to GBM. The Brownian motion as stated so far will allow for the possibility of
negative prices and usually we do not want our models of the stock price to allow this. Denote the scaled random walk with n sent to infinity by
*W(t)*. Then the stock price follows a GBM of the following form:

${{S_t = S_0e^{{\\sigma W(t) - \\frac{{1}}{{2}}\\sigma^2t}}}}$

where ${{\\sigma}}$ is the standard deviation of the stock price (also known as *volatility*). A key difference between GBM and the earlier Brownian
motion is that, for a fixed *t*, the distribution of ${{S_t}}$ is *lognormal* rather than normal. An important consequence is that stock prices
are always positive in the model. Below we run some simulations and plot the distribution of terminal stock prices. 

""")
st.write(f"""{GBM_simulate()}

The binomial stock price price is well suited for GBM. Shreve noted that "the binomial model is a discrete-time version of the geometric Brownian 
motion model, which is the basis for the Black-Scholes-Merton option-pricing formula."As the number of possible stock prices increase (by increasing 
the number of time-steps), the distribution of terminal stock prices in the binomial tree will converge in distribution to the lognormal.

Let us consider the distribution of terminal stock prices for a binomial tree. Here we plot some trees with varying numbers of time-steps but 
otherwise equal parameters. Note that as the number of time-steps increases, the distribution of terminal stock prices more closely resembles a 
lognormal distribution. 

""")
st.write(f"""{plotTerminalPrices()}

Note that we plotted the distribution of terminal stock prices. If we had instead considered the distribution of ${{log({{\\frac{{u^{{n-i}}}}{{d^i}}}})}}$, 
which counts the number of up movements a stock has taken during the lifetime of the option, or the distribution of the log returns of the stock,
${{ log({{\\frac{{S_T}}{{S_0}}}}) }}$, then we would see a convergence in distribution to the normal as the number of time-steps increased. 

But is the convergence of the distribution of terminal stock prices to the distribution underlying the Black-Scholes sufficient to guarantee
that our binomial tree will give us the relevant Black-Scholes price? After all, there is more to an option than the behavior of its underlying
asset. 

The (mathematical) essence of Black-Scholes is lopping off the lognormal curve of stock prices* according to the payoff function of the option.
The essence of the binomial tree is also lopping off a terminal distribution of stock prices according to a nonlinear payoff function. The 
discreteness of the distribution allows us, for trees of relatively few steps, to understand this fact more clearly than in the case of Black-Scholes 
solutions, whose relationship to the distribution of terminal stock prices in GBM may not be immediately obvious. 

*More specifically, lopping off a curve and then discounting the expected value of the payoff (under the risk-neutral measure) weighted by this 
lopped-off stock price curve.

So far we have discussed two of the great strengths of the binomial tree method: (1) it really is a useful pedagogical tool, making clear the relationship 
between stock price behavior and option prices; and (2) a suitably large tree can approximate reasonably well the behavior of the continuous-time
stochastic process assumed by the Black-Scholes model. One might suggest that if the goal is to model some particular continuous distribution of 
terminal stock prices by a discrete distribution, then we need not limit ourselves to the binomial and lognormal. Cox and Rubinstein explored this
possibility further in the seventh chapter of their 1985 book *Options Markets*. Interested readers are directed there.

__Choosing tree parameters and improving convergence__

What are the parameters of a tree? In the literature, the tree parameters usually refer to u, d, and p, although to price an option we also need 
the initial price of the underlying asset, the risk-free rate, and at least two of the following: number of time steps, time-to-maturity,
and/or length of time steps. We also typically need to input volatility (annualized). The actual changes in the tree model, however,
come from the three parameters u, d, and p, and this is what we mean when refer to a tree's parameters. (Note: the parameters need not be
constant throughout the tree). There seem to be as many methods for parameter selection as there are researchers, but we will focus on the
class of four methods implemented in the calculator. All of these methods select parameters so as to match the first two moments of the
binomial process to GBM. Though they were initially tested for European (and American) vanilla options, we let users implement the methods
for all options priced by binomial trees in the calculator with the warning that they may not behave as expected. 

It is a bit misleading to speak of the "convergence" of the binomial tree method. We are interested in how fast the binomial tree price
converges to the true option price. By fast, we mean how many time steps are needed for the binomial price to be reasonably close to the
true price. But we only have the true option price in one case* - the European option (call and put) - and in that case the "true option price"
refers to the Black-Scholes price. But if we can easily calculate the price-to-be-converged-to, the Black-Scholes price, then it seems rather 
useless to study the convergence of a numerical method that only converges to this price. If binomial trees were only used to price European 
option, then they would have a solely pedagogical use. 

*Technically there are other derivative securities, such as a forward contract, for which we have a true, or analytic, price to which
a numerical method may converge.

Yet binomial trees are not used solely to reinvent the Black-Scholes wheel. As we will see, binomial trees can be applied to a wide variety
of derivative securities, many of which lack an analytic solution to their respective Black-Scholes partial differential equation. Studying
the convergence in the case of European options help us understand convergence in the case of more complicated derivative securities. Don
Chance puts it so: 

*But it is difficult to consider the binomial model as a method for deriving the values of more complex options without knowing how well it works
for the one scenario in which the true continuous limit is known. An unequivocal benchmark is rare in finance.*

How to choose u, d, and p? CRR and most other methods make p a function of u and d in some way. For example, CRR has the martingale probability
p as ${{\\frac{{e^{{rT}} - d}}{{u - d}}}}$. Then CRR sets ${{u = e^{{\\sigma \\Delta t}}}}$ and ${{d = \\frac{1}{{u}}}}$, where ${{\\sigma}}$ is the
annualized volatility of the stock. This may be determined through historical data or some other methods that we will not discuss here. 
Volatility is one way to measure the spread of possible stock prices. Higher volatility will mean larger jumps between time steps. 

Where did CRR get these values of u and d? They are the solutions of the system of equations that one gets when you match the first two moments
of the log returns in the binomial tree with geometric Brownian motion. By the Central Limit Theorem, the log returns of the discrete binomial
distribution will thereby converge to the normal distribution. 

Moment matching is not necessary to ensure convergence. Hsia (1983) showed that for convergence all that is required is
that 0 < p < 1 (which we saw earlier was enforced by the principle of no arbitrage). In fact, moment matching may even be counterproductive; 
Chance (2011) argues that moment matching defeats the purpose of no-arbitrage pricing by requiring that the expected return be known. In any case, 
moment matching is widely done in the literature, including in the methods presented here. 

We are not just concerned with convergence but with fast convergence. We want our binomial trees to price options with as few time
steps as possible, because more time steps means slower computation times. How to improve convergence? Once we have matched the
first two moments, there are still two degrees of freedom left. We use one of these to set p + q = 1, where q is the probability of a
down movement, or alternatively to have also p > 0 and q < 1. We then set up a system of equations with these three restrictions:

(1) ${{p + q = 1 (p > 0, q < 1)}}$

(2) ${{pu + qd = M}}$ 

(3) ${{pu^2 + qd^2 = M^2 V}}$

where M is the mean of the continuous-time lognormal process and ${{M^2V}}$ the second moment (and V the variance?). (1) ensures that p and q are 
probabilties. (2) and (3) are the moment matching equations. These are already sufficient to guarantee convergence ((1) alone is sufficient
according to Hsia).

What to do with the fourth conditon? Here are the choices of CRR, Jarrow-Rudd, and Tian:

(CRR) ${{ud = 1}}$

(Jarrow-Rudd) ${{p = 1/2}}$

(Tian) ${{pu^3 + qd^3 = M^3V^3}}$

Tian's condition is matching the third moment. Numerical studies have shown that Jarrow-Rudd and Tian converge faster than CRR. We now plot
their convergence speeds for a particular European call option:
""")

st.write(f"""{compareBinomialConvergence()}

Note the oscillations of the convergence. What is the source of these oscillations and can we fix them? This is the starting-point of the
Leisen-Reimer method introduced in 1996. They identified the source of the oscillations in the change of the relative position of the strike
price among terminal nodes as an additional time-step is added. The number of terminal time-steps naturally alternates between odd and even.
The result is that the tree alternates between overshooting and the undershooting the analytic price. Leisen and Reimer chose the final
degree of freedom to fix the relative position of the strike position within the tree regardless of the number of time-steps. For even number of
steps, the strike is always located on some terminal node; for an odd number of time-steps, the strike is always located exactly halfway between
two terminal nodes. This is achieved by the condition ${{ ud = e^{{ \\frac{2}{{n}} ln(\\frac{{K}}{{S}})}} }}$. The result is that even or odd refinements converge montonically.

This smoothed convergence for odd numbers of time-steps, but it did not improve convergence speed. For this, Leisen and Reimer turned to the 
literature on normal approximations to the binomial. They write:

*Generally speaking, despite the simplicity, the calculation of binomial probabilities is cumber-
some because it might involve factorials of large integers or the summation of a large number of
individual terms. Therefore, normal approximations to the binomial distribution were derived in
the mathematical literature. In particular, the method by Camp (1951) and Paulson (1942) and the
approximations of Peizer and Pratt (1968) reveal a remarkable precision. With all these
approaches, basically a binomially calculated true probability P is approximated with the standard
normal function N(z), where the input is determined by some adjustment function z = h(a; n,p)
where, in our setting, a is the number of up-movements of the asset price to exceed the strike price
in an n-step binomial tree with martingale measure equal to p...*

*However, our option pricing problem represents the opposite direction: computation of
binomial option prices involves the fact that normal components are approximated by binomial
components. Thus, for a given binomial tree refinement inverting the adjustment function,
h(a; n,p) above specifies the distribution parameter ${{h^{{-1}}(z)}}$ = p to approximate P = N(z)...*

Our option pricing calculator uses the Peizer-Pratt method 2 (PP2) of inversion:


${{ h^{{-1}}(z) = 0.5}}$ +/- ${{[0.25 - 0.25e^{{ -(\\frac{{z}}{{n + \\frac{1}{3} + \\frac{{0.1}}{{n+1}} }})^2(n + \\frac{1}{6})  }}]^{{\\frac{1}{2}}}  }}$


where the sign of the second expression is chosen so as to agree with the sign of ${{d_i}}$.

Then the parameters of the binomial tree are

(1) ${{p' = h^{{-1}}(d_1)}}$

(2) ${{p = h^{{-1}}(d_2)}}$

(3) ${{u = e^{{r\\Delta t}} \\frac{{p}}{{p'}}}}$

(4) ${{d = \\frac{{e^{{r\\Delta t}} - pu}}{{1 - p}}}}$

where ${{d_1}}$ and ${{d_2}}$ are from the Black-Scholes solutions (*p'* was introduced earlier in the section on multi-step trees).

The Leisen-Reimer method significantly improves convergence. Here we replicate (as in the case of the graph above) the results of
Leisen and Reimer concerning the speed of convergence to the Black-Scholes price. Note that for an odd number of steps, the Leisen-Reimer
price monotonically converges to the analytic price. 

""")
st.write(f"""{compareBinomialConvergenceWithLR()}

__American options__

A great strength of tree methods are their ability to model the early exercise feature of American options. Whereas the holder of a European
call option can only exercise their right to buy the stock at the specificed maturity date, the holder of an American call can do so at any time
before and *at* maturity. 

It seems obvious that this early exercise feature will change the value of an American option. As it turns out, the pricing of the American call
option is unchanged from that of the European call. It is never optimal to exercise an American option before maturity (though in cases where it
seem it should be, the recommended course of action is to sell the call instead of exercising and paying the strike). However, it is sometimes
optimal to exercise an American put option early. 

The early-exercise feature of the American put option has led to there being no solution to its respective Black-Scholes partial differential
equation. By contrast, the binomial tree can easily handle the early-exercise feature. At each node, when backwards calculating the option value, we 
also calculate the payoff we would receive from exercising at that node. If the payoff is greater than the backwards calculation, then we set the 
value of the option at that node equal to the payoff. That is all that is required. In the visualizer, we highlight nodes in red if early exercise 
is optimal.

""")

st.write(f"""{drawAmericanTree()}

__Asian options__

An Asian option is one in which the payoff depends on the average price of the asset over the lifetime of the option. For a *fixed-strike* Asian
option, the payoff is ${{(S_{{avg}} - K)^+}}$ for a call and ${{(K - S_{{avg}})^+}}$ for a put, where ${{(S_{{avg}}}}$ is the average asset price.
For a *floating-strike* Asian option, the payoff is ${{(S_T - S_{{avg}})^+}}$ for a call and ${{(S_{{avg}} - S_T)^+}}$ for a put.

Notice that the payoff depends on the particular price path that the asset took. There are often multiple ways for an asset price to reach a particular
terminal price, and therefore we lose some information with our standard binomial tree. This is an example of *path-dependency*, which will often require
us to make modifications to our binomial tree in order to capture the path-dependent information otherwise lost in recombination. In general, we want
to avoid non-recombining trees because they require 2^n nodes at the nth time step instead of just n, which leads to slow computation times. We can
approximate the path-dependency of the Asian option by adding information at each node concerning the range of possible average prices that the asset
might have if it reached that node. 

This is the essence of the Hull-White method (which is applicable to path-dependent options in general). Whereas option prices are calculated
by traversing backward through the tree, the average prices are calculated by traversing forward through the tree. Calculating averages is not
so complicated. At node (i, j), we simply consider the set of parent node averages and then adjust for the stock price at (i, j). In the visualizor
on the other page, each node contains three numbers. From top to bottom, they are (1) the stock price, (2) the option price under the lowest
possible average stock price path that could reach the node, and (3) the option price under the highest possible average price path. 


__Barrier options__

Barrier options are options whose payoff depends on the asset price reaching a certain level, called the barrier, sometime during the lifetime of the
option. The calculator considers one kind of barrier option: the *knock-out option*, an option which become worthless if the price of the
underlying asset violates some predetermined barrier price. There are two types of knock-out options. A *down-and-out* option is one whose payoff is 
the same as a European option so long as the asset price does not fall below the barrier price. An *up-and-out* option is one whose payoff is the 
same as a European option so long as they asset price does not go above the barrier price. 

The payoff of a barrier option depends on the asset price’s relation to the barrier price during the life of the option, and therefore we are dealing 
with a path-dependent option. As it turns out, path-dependency is not so difficult in this case. At each node, we multiply the calculated intrinsic 
value of the option by an indicator function that will return 0 if the node's asset price violates the barrier and 1 otherwise. 

The main difficulty in constructing binomial trees to price barrier options is rather in the relative position of the barrier price with
repect to the three nodes. Note that for the standard CRR tree, asset prices line up so that a layer of nodes will posses the same 
asset price. If the barrier price does not fall on one such layer of nodes then the tree is using two barriers to price the option: (1) an inner 
barrier consisting of the nodal price between the initial asset price and the barrier and (2) an outer barrier consisting of the nodal price 
outside ofthe barrier price. 

Perhaps if the vertical space between nodes was small, then this would not be a problem? We know that the vertical space will
decrease as we increase the number of time steps for a fixed time period. Let us examine how the option price estimation changes
as we increase the number of time steps:

""")
st.write(f"""{naiveBarrierConvergence()}

A pattern emerges: the option price converges to the analytic price and then jumps away. This is because the nodes get closer and closer to the 
barrier price until the refinement passes the barrier and the barrier is back off of a layer of nodes again. We see that the jump gets smaller 
each time, but it would require a lot more time steps to get the jump size reduced to irrelevancy. 

Perhaps we could simply only construct trees with a number of time-steps just before the jump? This is the method of 
Boyle and Lau (1994), but if instead we set aside binomial trees, we can get a convergence that is faster and is more flexible for
the types of barrier used in the option. This alternative route is to make use of trinomial trees and then construct the tree so that 
the barrier price will always lie on some set of nodes. This is the method of Ritchken (1999). In a trinomial tree, at each step
there are three possible movements - up, middle, or down, where up > middle > down. 

""")
st.write(f"""{drawTrinomialTree()}

In a trinomial tree, once there is a node representing some particular asset price, then at all future time steps that particular 
asset price will be represented by some node. Now it is only a matter of making sure that the barrier price is represented by some 
node. 

One of the first papers to use a trinomial tree is Boyle (1988), and we follow this paper in constructing a general trinomial tree. 
Suppose that the current stock price is ${{S_t}}$. In a trinomial tree, there are three possible stock prices at time *t + 1*: 
${{S_tu, S_t, S_td}}$. Each possible movement has a probability associated with it. As in the binomial tree, we impose some
conditions to construct a system of equations and then solve for the probabilities and the size of the jumps, *u* and *d*. 
The conditions are essentially the same as those of CRR: the probabilities must sum to 1; the first two moments of the
discrete distribution of stock prices are matched to the continuous lognormal; and *ud* = 1. 

Note that unlike in the binomial case, this will leave us with an additional degree of freedom. Boyle notes that there are a variety of values of *u* that
will satisfy the resulting system. We set *u* = ${{e^{{\\lambda \\sigma \\sqrt{{\\Delta t}}}}}}$ with the condition that ${{\\lambda}}$ > 1.
The parameter ${{\\lambda}}$ is referred to as the "stretch" parameter. Higher values of ${{\\lambda}}$ means a larger gap in between tree layers.  We 
can tune ${{\\lambda}}$ by testing a variety of values and selecting the one which produces the option price closest to the analytic one if available. 
In any case, we must choose so as to ensure the probabilities are all non-negative. In the option price calculator on the other page, we use a stretch 
parameter of ${{\\sqrt{3}}}$ as in Hull's textbook. 

We can choose the stretch parameter so that the barrier lies on a node layer. Now we arrive at the method of Ritchken (1999). First we consider a tree 
with the stretch parameter equal to 1, so that up and down movements are the same size as in the CRR binomial tree. We are interested in how many steps 
it takes to reach a barrier. Ritchken illustrates this using a down-and-out option, so we will use an up-and-out option. Assuming that the barrier, *H*, 
is above the initial asset price, ${{S_0}}$ (otherwise the option would be worthless from the start), we want to know how many up jumps are required to 
break the barrier. We want to find ${{\\eta}}$ such that 

${{\\frac{{H}}{{S_0}} = e^{{ {{\\eta}} {{\\sigma}} {{\\Delta}} t}}}}$

Note that ${{\\eta}}$ might not be an integer. If it is, then the stretch parameter of 1 already suffices for a tree in which the
barrier lies on a layer of nodes. If ${{\\eta}}$ is not an integer, then we round it down. Denote this integer by ${{n_0}}$. We
then increase the stretch parameter so that ${{ {{\\lambda}} {{\\eta}} = n_0}}$. Thus we set ${{ {{\\lambda}} = \\frac{{n_0}}{{ {{\\eta}} }}}}$. We
reconstruct the final tree using the new stretch parameter. The barrier will now lie on a layer of tree nodes. The result is a smoother and faster
convergence. 

""")

st.write(f"""{drawBarrierTreeConvergence()}

Problems occur when the option is very close to the barrier or  when the initial asset price is very close to the barrier. Hull
recommends using other numerical methods such as a adaptive grids in such cases.  


__Other exotic options__


An exotic option is just any option beyond the so-called vanilla European and American call options introduced in preceding sections. 
Asian and barrier options are examples of exotic options. We will discuss the other exotic options included in the option 
pricing calculator in one section because, unlike Asian and barrier options, only small modifications of the binomial tree for 
vanilla European options are required to price these exotic options.


__Binary options__ 


A *binary option* (or *digital option*) is one with a payoff of zero or X at maturity. For a call option, this means if the stock price 
is below the strike, the payoff is zero, but if the stock price is at or above the strike price then the payoff is X. For a 
*cash-or-nothing* option, X is some fixed value that holds regardless of how far above the stock price is than the strike price. 
For an *asset-or-nothing* option, X is just the stock price at maturity, ${{S_T}}$. For a binomial tree, we simply adjust the 
terminal payoffs and backwardize as normal. 


__Compound options__


The idea of a *compound option* is simple: it is an option on an option. Theoretically we might have an infinite nest of options, 
but in practice (and in the calculator) there is just the option on an option. For example, we might have a call option on a put 
option. We have two strike prices: one for exercising the call to purchase the put and another for exercising the put - and two 
exercise dates as well. For a binomial tree for an X-on-Y option, we backward induct through the tree as if it were a Y option until 
we get to the compound exercise date. We take the maximum of the discounted price less the strike (or the strike less the discounted 
price if X is a put) and then resume backwardization. 


__Chooser options__


Suppose that an investor knew they wanted an option on an asset some time in the future, but they were not sure whether or not to 
take a put or a call. A *chooser option* delays the choice of put or call until some time ${{t_c < T}}$. Pricing in a binomial tree 
involves computing call and put values for nodes greater than ${{t_c}}$. For the nodes at ${{t_c}}$, we determine whether or not to 
choose a call or put by comparing the values of the call and put at that node. 

In the tree calculator, nodes after time ${{t_c}}$ contain three numbers. On top is the asset price. In the middle is the value of a 
call option and at the bottom is the value of a put option. At time ${{t_c}}$, nodes are colored orange if it is optimal to choose a 
call option and green if it is optimal to choose a put option. If it is equally optimal to choose a put as it is to choose a call, 
the node is left as white. 


__Bibliography__

Baxter, Martin and Rennie, Andrew. *Financial Calculus: an introduction to derivative pricing*. Cambridge, UK: Cambridge University Press, 1996. 

Boyle, Phelim. “A lattice framework for option pricing with two state variables.” In *Journal of Financial and Quantitative Analysis* 23, no. 1
(Mar., 1988), 1-12. 

Boyle, Phelim and Lau, Sok Hoon. “Bumping up against the barrier with the binomial method.” In *The Journal of Derivatives* 1, no. 4
(Summer 1994): 6-14.  

Chance, Don M. “A synthesis of binomial option pricing models for lognormally distributed assets.” In *Journal of Applied Finance* 18, no. 1
(2008). 

Cox, John C., and Rubinstein, Mark. *Options Markets*. Eaglewood Cliffs, NJ: Prentice Hall, 1985. 

Cox, John C., Ross, Stephan A., and and Rubinstein, Mark. “Option pricing: a simplified approach.” In *Journal of Financial Economics* 7 (1979): 229-263.
 
Hull, John C. *Options, Futures, and Other Derivatives*, 11th ed. Hoboken, NJ: Pearson, 2022.

Hull, John C. and White, Alan. “Efficient procedures for valuing European and American path-dependent options.” In *The Journal of Derivatives* 
1, no. 1 (Fall 1993): 21-31.

Hsia, “On binomial option pricing.” In *The Journal of Financial Research* 6, no. 1 (Spring 1983): 41-46. 

Jarrow, Robert and Rudd, Andrew. *Option Pricing*. Homewood, IL: Richard D. Irwin, 1993.

Leisen, D.P., and Reimer, M. “Binomial models for option valuation - examining and improving convergence.” In *Applied Mathematical Finance* 
3, no. 4 (1996): 319-346. 

Ritchken, Peter. “On pricing barrier options.” In *The Journal of Derivatives* 3, no. 2 (Winter 1995): 19-28. 

Shreve, Steve. *Stochastic Calculus for Finance I: the Binomial Model*. New York: Springer-Verlag, 2004. 

Shreve, Steve. *Stochastic Calculus for Finance II: Continuous-Time Models*. New York: Springer-Verlag, 2004. 

Wilmott, Paul, Dewynne, Jeff and Howison, Sam. *Option Pricing: Mathematical Models and Computation*. Oxford, UK: Oxford Financial Press, 1993. 

Tian, Y. “A modified lattice approach to option pricing.” In *Journal of Futures Markets* 13, no. 5 (1993): 563-577. 

Van der Hoek, John and Elliot, Robert J. *Binomial Tree Methods*. New York: Springer Finance, 2006. 


	""")