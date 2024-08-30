import streamlit as st
import math
import graphviz
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import abc

st.set_page_config(
    layout="wide",
    page_title="Binomial Tree Calculator and Visualizer",
)

st.sidebar.success("Select a page above.")


st.write("""This page includes an app that will calculate and visualize a binomial tree. In general, each node contains two 
numbers. On top is the asset price. On bottom is the option price. 

For American puts, nodes that are colored red indicate situations in which it is optimal to exercise the option early.

Asian option nodes contain three numbers. The top number is the asset price. The middle number is the value of the option under
the lowest possible average stock price that could have reached that node. The bottom number is the value of the option under the
highest possible average stock price that could have reached that node. The bottom two numbers will end up being equal at the
initial node and this is the calculated option price. 

For chooser options, nodes after the choosing time contain three numbers. From top to bottom, they are (1) the asset price, (2)
the value of the call option, and (3) the value of the put option. At the choosing time, nodes are colored orange if it is optimal
to choose a call option, green if it is optimal to choose a put option, and white if it is equally optimal to choose a put as it is
to choose a call.

Visualization becomes quite intensive very quickly, and therefore only trees of twenty-five time-steps or less can be visualized. 
Turning off visualization removes the limit of the number of time-steps, though the user should keep in mind that trees with more
than 1,000 time-steps might take a while to return an option price.

With regard to parameter selection, I recommend selecting CRR (Cox-Ross-Rubinstein) unless you are familiar with the alternatives
listed. Please refer to the theory page to learn more.""")


class Tree(abc.ABC):
    
    def draw_tree(self):

        dot = graphviz.Digraph()
        dot.attr('node', shape='box')

        i = 0
        for node in self.nodes.values():
            dot.node(str(i), label=f"{node.stock_price:.2f} \n {node.option_price:.2f}", style='rounded', color='white', fontcolor='white')
            i += 1

        for node in self.nodes.values():
            for child in node.childs:
                dot.edge(str(node.number), str(child.number), color='white')

        dot.attr(rankdir='LR') 
        dot.attr(bgcolor='transparent')

        st.graphviz_chart(dot, use_container_width=True)

    def dividends(self, time_step):
        """Modified indicator function for dividends."""
        D = 1
        if self.div_times:
            D = 1
            for t in self.div_times:
                if int(t) <= time_step:
                    D *= (1 - self.div_amt)
        return D
        

class BinomialTree(Tree):

    def __init__(self, time_steps: int, up: float, down: float, 
                initial_price: float, r: float, 
                del_t: float, option_type: str, prob: float,
                div_amt: float = 0, div_times: list = []):
        
        self.up = up
        self.down = down
        self.initial_price = initial_price
        self.r = r # risk-free rate for discounting
        self.del_t = del_t # length of each time step
        self.option_type = option_type
        self.div_amt = div_amt # amount of dividend (percentage yield)
        self.div_times = div_times # time steps at which dividends are issued
        self.nodes = {} # empty dictionary of nodes
        self.time_steps = time_steps # number of time steps
        self.prob = prob # martingale probability of up movement

        self.nodes = self.calcStockPrices()        
        # Children creation
        for n in self.nodes:
            if self.nodes[n].time_step < self.time_steps:
                self.nodes[n].childs.append(self.nodes[(self.nodes[n].time_step + 1, self.nodes[n].state)])
                self.nodes[n].childs.append(self.nodes[(self.nodes[n].time_step + 1, self.nodes[n].state + 1)])

    def calcStockPrices(self):

        nodes = {}
        nodes[(0,0)] = (Node(self.initial_price, 0, 0, [], 0, self.option_type))
        
        node_number = 1
        for i in range(1, self.time_steps + 1):
            s = 0 # state within time step, highest possible stock price at s = 0
            
            while s < i + 1: 
                stock = self.initial_price * self.dividends(i)
                nodes[(i,s)] = Node(stock * np.power(self.up, i - s) * np.power(self.down, s), 
                                           node_number, i, [], s,option_type=self.option_type)
                node_number += 1
                s += 1               
            i += 1
        
        return nodes
    
    def vanillaPayoff(self, n):
        return np.exp(-self.r * self.del_t) * (self.prob * n.childs[0].option_price 
                    + (1 - self.prob) * n.childs[1].option_price)
    

    
class EuropeanTree(BinomialTree):
    
    def __init__(self, time_steps: int, up: float, down: float, 
                initial_price: float, r: float, 
                del_t: float, option_type: str, 
                prob: float, strike: float,
                div_amt: float = 0, div_times: list = []):
        super().__init__(time_steps, up, down, 
                initial_price, r, 
                del_t, option_type, prob, 
                div_amt, div_times)
        self.calcOptions(strike)

    def calcOptions(self, strike):
        # Calculate terminal payoffs first
        for n in reversed(self.nodes.values()):
            if n.childs:
                break
            n.option_price = n.calcPayoff(strike)    
            
        # Backwardize through the tree
        for n in reversed(self.nodes.values()):
            if n.childs:
                n.option_price = self.vanillaPayoff(n)
                    
        
class AmericanTree(BinomialTree):
    
    def __init__(self, time_steps: int, up: float, down: float, 
                initial_price: float, r: float, 
                del_t: float, option_type: str, 
                prob: float, strike: float,
                div_amt: float = 0, div_times: list = []):
        super().__init__(time_steps, up, down, 
                initial_price, r, 
                del_t, option_type, prob,
                div_amt, div_times)
        self.calcOptions(strike)
    
    def calcOptions(self, strike):
        # Calculate terminal payoffs first
        for n in reversed(self.nodes.values()):
            if n.childs:
                break
            n.option_price = n.calcPayoff(strike)    
            
        # Backwardize through the tree
        for n in reversed(self.nodes.values()):
            if n.childs:
                time_value = self.vanillaPayoff(n)
                if self.option_type == "American call":
                    intrinsic_value = max(0, n.stock_price - strike)
                else: # then it is an American put
                    intrinsic_value = max(0, strike - n.stock_price)
                n.option_price = max(time_value, intrinsic_value)
                
                if n.option_price > time_value:
                        n.early_exercise = True     

    def draw_tree(self):
            
        dot = graphviz.Digraph()
        dot.attr('node', shape='box')

        i = 0
        for node in self.nodes.values():
            if node.early_exercise is True:
                dot.node(str(i), label=f"{node.stock_price:.2f} \n {node.option_price:.2f}", style='rounded', color='red', fontcolor='white')
            else:
                dot.node(str(i), label=f"{node.stock_price:.2f} \n {node.option_price:.2f}", style='rounded', color='white', fontcolor='white')
            i += 1

        for node in self.nodes.values():
            for child in node.childs:
                dot.edge(str(node.number), str(child.number), color='white')

        dot.attr(rankdir='LR') 
        dot.attr(bgcolor='transparent')

        st.graphviz_chart(dot, use_container_width=True)
        


class AsianTree(BinomialTree):
    
    def __init__(self, time_steps: int, up: float, down: float, 
                initial_price: float, r: float, 
                del_t: float, option_type: str,
                prob: float, strike: float,
                div_amt: float = 0, div_times: list = []):
        super().__init__(time_steps, up, down, 
                initial_price, r, 
                del_t, option_type, prob,
                div_amt, div_times)
        self.calcOptions(strike)
            
    def calcOptions(self, strike):
        # Step 1: Compute average prices for each node (highest and lowest)

        for n in self.nodes.values():
            n.calcAverages(self)
            
        n.option_prices = np.zeros(4)
        if "average price" in self.option_type:
            for n in self.nodes.values():
                if not n.childs:
                    for i in range(4):
                        n.option_prices[i] = n.calcPayoff(strike, n.average[i])
                    
        elif "average strike" in self.option_type:
            for n in self.nodes.values():
                if not n.childs:
                    for i in range(4):
                        n.option_prices[i] = n.calcPayoff(n.average[i])
                                        
        for n in reversed(self.nodes.values()):
            if n.childs:

                lambdas = np.zeros(4) # linear combination weights
                C_i = np.array([[0, 0], [0, 0], [0, 0] ,[0, 0]]) # array of arrays of two elements, 1st is up 2nd is down 
                if n.state != 0: 
                    for i in range(4):
                        # calculate upward movement M_u
                        M_up = (1 / (n.time_step + 2)) * ((n.time_step + 1) * n.average[i] + self.nodes[(n.time_step + 1, n.state)].stock_price) 
                        
                        # Find which two average prices for the child node this will bring the average to
                        k1, k2 = self.findInterval(M_up, self.nodes[(n.time_step + 1, n.state)].average)

                        # calculate lambdas using average price and upward movement
                        lambdas[i] = (self.nodes[(n.time_step + 1, n.state)].average[k2] - M_up) / (self.nodes[(n.time_step + 1, n.state)].average[k2] - self.nodes[(n.time_step + 1, n.state)].average[k1])

                        # calculate option price for upward movement
                        C_i[i][0] = lambdas[i] * self.nodes[(n.time_step + 1, n.state)].option_prices[k1] + (1 - lambdas[i]) * self.nodes[(n.time_step + 1, n.state)].option_prices[k2]

                else: # top node
                    for i in range(4): 
                        C_i[i][0] = self.nodes[(n.time_step + 1, n.state)].option_prices[0]
                
                if n.state != n.time_step:
                    # For downward movement
                    for i in range(4):
                        # calculate upward movement M_u
                        M_down = (1 / (n.time_step + 2)) * ((n.time_step + 1) * n.average[i] + self.nodes[(n.time_step + 1, n.state + 1)].stock_price) 

                        # Find which two average prices for the child node this will bring the average to
          
                        k1, k2 = self.findInterval(M_down, self.nodes[(n.time_step + 1, n.state + 1)].average)

                        # calculate lambdas using average price and downard movement
                        lambdas[i] = (self.nodes[(n.time_step + 1, n.state + 1)].average[k2] - M_down) / (self.nodes[(n.time_step + 1, n.state + 1)].average[k2] - self.nodes[(n.time_step + 1, n.state + 1)].average[k1])

                        # calculate price for downward movement
                        C_i[i][1] = lambdas[i] * self.nodes[(n.time_step + 1, n.state + 1)].option_prices[k1] + (1 - lambdas[i]) * self.nodes[(n.time_step + 1, n.state + 1)].option_prices[k2]
                        
                else: # bottom node
                    for i in range(4):
                        C_i[i][1] = self.nodes[(n.time_step + 1, n.state + 1)].option_prices[0]

                # Finally calculate option_i value by discounting expected option values
                if "American" in self.option_type:   
                    if "average price" in self.option_type:
                        for i in range(4):
                            n.option_prices[i] = max(self.asianPayoff(C_i[i][0], C_i[i][1]), 
                                                     n.calcPayoff(strike, n.average[i]))                         
                    elif "average strike" in self.option_type:
                        for i in range(4):
                            n.option_prices[i] = max(self/asianPayoff(C_i[i][0], C_i[i][1]), 
                                                     n.calcPayoff(n.average[i]))                                           
                else: # European-style
                    for i in range(4):
                        n.option_prices[i] = self.asianPayoff(C_i[i][0], C_i[i][1]) 


    def asianPayoff(self, C_i_up, C_i_down):
        return np.exp(-self.r * self.del_t) * (self.prob * C_i_up + (1 - self.prob) * C_i_down)
    
    def findInterval(self, M: float, averages):
        if M >= averages[0] and M <= averages[1]:
            return 0, 1
        elif M >= averages[1] and M <= averages[2]:
            return 1, 2
        elif M >= averages[2] and M <= averages[3]:
            return 2, 3

    def draw_tree(self):
        
        dot = graphviz.Digraph()
        dot.attr('node', shape='box')

        i = 0
        for node in BT.nodes.values():
            dot.node(str(i), label=f"{node.stock_price:.2f} \n {node.option_prices[0]:.2f} \n {node.option_prices[3]:.2f}", style='rounded', color='white', fontcolor='white')
            i += 1

        for node in self.nodes.values():
            for child in node.childs:
                dot.edge(str(node.number), str(child.number), color='white')

        dot.attr(rankdir='LR') 
        dot.attr(bgcolor='transparent')

        st.graphviz_chart(dot, use_container_width=True)


class BinaryTree(BinomialTree):
    
    def __init__(self, time_steps: int, up: float, down: float, 
                initial_price: float, r: float, 
                del_t: float, option_type: str,
                prob: float, strike: float,
                binary_payoff: float,
                div_amt: float = 0, div_times: list = [],):
        super().__init__(time_steps, up, down, 
                initial_price, r, 
                del_t, option_type, prob,
                div_amt, div_times)
        self.binary_payoff = binary_payoff
        self.calcOptions(strike)

    def calcOptions(self, strike):
        # Calculate terminal payoffs first
        for n in reversed(self.nodes.values()):
            if n.childs:
                break
            n.option_price = n.calcPayoff(strike, binary_payoff=self.binary_payoff)
                    
        # Backwardize through the tree
        for n in reversed(self.nodes.values()):
            if n.childs:
                n.option_price = self.vanillaPayoff(n)

class CompoundTree(BinomialTree):
    
    def __init__(self, time_steps: int, up: float, down: float, 
                initial_price: float, r: float, 
                del_t: float, option_type: str,
                prob: float, strike: float, 
                t_c: int, compound_strike: float,
                div_amt: float = 0, div_times: list = []):
        super().__init__(time_steps, up, down, 
                initial_price, r, 
                del_t, option_type, prob,
                div_amt, div_times)
        self.t_c = t_c # compounding time
        self.compound_strike = compound_strike
        self.calcOptions(strike)
        
    def calcOptions(self, strike):
        
        # calculate terminal node payoffs
        for n in reversed(self.nodes.values()):
            if n.time_step < self.t_c:
                break
            if n.childs:
                n.option_price = n.calcPayoff(strike)

        if "Call on" in self.option_type: # Call-on-call and Call-on-put
            for n in reversed(self.nodes.values()): 
                if n.time_step == self.t_c:
                    n.option_price = max(n.option_price - self.compound_strike, 0)
        elif "Put on" in self.option_type: # Put-on-put and Put-on-call
            for n in reversed(self.nodes.values()):
                if n.time_step == self.t_c:
                    n.option_price = max(self.compound_strike - n.option_price, 0)

        # complete the backwardization
        for n in reversed(self.nodes.values()):
            if n.time_step < self.t_c:
                n.option_price = self.vanillaPayoff(n)

class ChooserTree(BinomialTree):

    def __init__(self, time_steps: int, up: float, down: float, 
                initial_price: float, r: float, 
                del_t: float, option_type: str, 
                prob: float, strike: float, 
                t_c: int,
                div_amt: float = 0, div_times: list = []):
        super().__init__(time_steps, up, down, 
                initial_price, r, 
                del_t, option_type, prob,
                div_amt, div_times)
        self.t_c = t_c # choosing time
        self.calcOptions(strike)
    
    def calcOptions(self, strike):

        # calculate European put and call prices for each node at time t_c, the time of choosing
        for n in self.nodes.values():
            if not n.childs:
                n.option_prices[0] = max(n.stock_price - strike, 0)
                n.option_prices[1] = max(strike - n.stock_price, 0)

        # Chooses put or call at time t_c and then finishes the backwardization
        for n in reversed(self.nodes.values()):
            if n.time_step < self.t_c: 
                # Finish the rest of the backwardization
                n.option_price = self.vanillaPayoff(n)

            if n.childs:
                n.option_prices[0] = np.exp(-self.r * self.del_t) * (self.prob * n.childs[0].option_prices[0] 
                + (1 - self.prob) * n.childs[1].option_prices[0]) 
                n.option_prices[1] = np.exp(-self.r * self.del_t) * (self.prob * n.childs[0].option_prices[1] 
                + (1 - self.prob) * n.childs[1].option_prices[1]) 

            # At time t_c, set the option value equal to the max(Put(t_c, j), Call(t_c, j))    
            if n.time_step == self.t_c:
                    n.option_price = max(n.option_prices[0], n.option_prices[1])


    def draw_tree(BT):
    
        dot = graphviz.Digraph()
        dot.attr('node', shape='box')

        i = 0
        for node in BT.nodes.values():
            if node.time_step > BT.t_c:
                dot.node(str(i), label=f"{node.stock_price:.2f} \n {node.option_prices[0]:.2f} \n {node.option_prices[1]:.2f}", 
                         style='rounded', color='white', fontcolor='white')
                i += 1
            elif node.time_step == BT.t_c:
                if node.option_prices[0] > node.option_prices[1]: # choosing the call
                    dot.node(str(i), label=f"{node.stock_price:.2f} \n {node.option_price:.2f}", style='rounded', color='orange', fontcolor='white')
                    i += 1
                elif node.option_prices[1] > node.option_prices[0]: # choosing the put
                    dot.node(str(i), label=f"{node.stock_price:.2f} \n {node.option_price:.2f}" , style='rounded', color='green', fontcolor='white')
                    i += 1
                elif node.option_prices[1] == node.option_prices[0]: # call and put are equal
                    dot.node(str(i), label=f"{node.stock_price:.2f} \n {node.option_price:.2f}", style='rounded', color='white', fontcolor='white')
                    i += 1
            else:
                dot.node(str(i), label=f"{node.stock_price:.2f}  \n {node.option_price:.2f}", style='rounded', color='white', fontcolor='white')
                i += 1

        for node in BT.nodes.values():
            for child in node.childs:
                dot.edge(str(node.number), str(child.number), color='white')

        dot.attr(rankdir='LR') 
        dot.attr(bgcolor='transparent')

        st.graphviz_chart(dot, use_container_width=True)
    

class NaiveBarrierTree(BinomialTree):
    """Used only in the theory page for comparison with the trinomial barrier tree."""

    def __init__(self, time_steps: int, up: float, down: float, 
                initial_price: float, r: float, 
                del_t: float, option_type: str, 
                prob: float, strike: float, barrier: float,
                div_amt: float = 0, div_times: list = []):
        super().__init__(time_steps, up, down, 
                initial_price, r, 
                del_t, option_type, prob,
                div_amt, div_times)
        self.barrier = barrier
        self.calcOptions(strike)

    def calcOptions(self, strike):
        # Calculate terminal payoffs first
        for n in reversed(self.nodes.values()):
            if n.childs:
                break
            n.option_price = n.calcPayoff(strike, barrier=self.barrier)    
            
        # Backwardize through the tree
        for n in reversed(self.nodes.values()):
            if n.childs:
                n.option_price = self.vanillaPayoff(n)

        # Finish the backwardization depending on the type of barrier option
        if "down-and-out" in self.option_type:
            for n in reversed(self.nodes.values()):
                if n.stock_price < self.barrier:
                    n.option_price = 0
                elif n.childs:
                    n.option_price = np.exp(-self.r * self.del_t) * (self.prob * n.childs[0].option_price 
                        + (1 - self.prob) * n.childs[1].option_price)
            
        elif "up-and-out" in self.option_type:
            for n in reversed(self.nodes.values()):
                if n.stock_price > self.barrier:
                    n.option_price = 0
                elif n.childs:
                    n.option_price = np.exp(-self.r * self.del_t) * (self.prob * n.childs[0].option_price 
                        + (1 - self.prob) * n.childs[1].option_price)

class TrinomialTree(Tree):
    
    def __init__(self, time_steps: int, up: float, down: float, 
                initial_price: float, r: float, 
                del_t: float, option_type: str, strike: float, sigma: float, 
                div_amt: float = 0, div_times: list = []):        
        self.time_steps = time_steps
        self.initial_price = initial_price
        self.r = r
        self.del_t = del_t
        self.option_type = option_type
        self.sigma = sigma
        self.div_amt = div_amt # amount of dividend (percentage yield)
        self.div_times = div_times # time steps at which dividends are issued

                                        
    def setProbs(self):
        self.prob_up = 1 / (2 * np.power(self.stretch, 2)) + ((self.r - (np.power(self.sigma, 2) / 2)) * np.sqrt(self.del_t)) / (2 * self.stretch * self.sigma)
        self.prob_mid = 1 - (1 / np.power(self.stretch, 2))
        self.prob_down = 1 / (2 * np.power(self.stretch, 2)) - ((self.r - (np.power(self.sigma, 2) / 2)) * np.sqrt(self.del_t)) / (2 * self.stretch * self.sigma)

    def calcStockPrices(self):
        self.stretch = self.calcStretchParameter() # handled by subclasses 
        self.setProbs() 
        
        nodes = {} # dictionary of nodes        
        nodes[(0,0)] = (Node(self.initial_price, 0, 0, [], 0, self.option_type))
        node_number = 1

        for i in range(1, self.time_steps + 1):
            s = 0 # state within time step, highest possible stock price s = 0

            for s in range(i):
                # Start from Su^i and add nodes to Su
                stock = self.initial_price * self.dividends(i)
                nodes[(i,s)] = Node(stock * np.power(self.up, i - s),
                                        node_number, i, [], s, option_type=self.option_type)
                node_number += 1

            for j in range(i+1):
                # Pick up from S and add nodes to Sd^i
                s += 1
                stock = self.initial_price * self.dividends(i)
                nodes[(i,s)] = Node(stock * np.power(self.down, j),
                                        node_number, i, [], s, option_type=self.option_type)
                node_number += 1

            i += 1
        
        return nodes

        
class EuropeanTrinomialTree(TrinomialTree):
    
    def __init__(self, time_steps: int, up: float, down: float, 
                initial_price: float, r: float, 
                del_t: float, option_type: str, 
                strike: float, sigma: float, 
                div_amt: float = 0, div_times: list = []):
    
        super().__init__(time_steps, up, down, initial_price, r, del_t, option_type, strike, sigma, div_amt, div_times)
        self.nodes = self.calcStockPrices() 
        # Children creation
        for n in self.nodes:
            if self.nodes[n].time_step < self.time_steps:
                self.nodes[n].childs.append(self.nodes[(self.nodes[n].time_step + 1, self.nodes[n].state)])
                self.nodes[n].childs.append(self.nodes[(self.nodes[n].time_step + 1, self.nodes[n].state + 1)]) 
                self.nodes[n].childs.append(self.nodes[(self.nodes[n].time_step + 1, self.nodes[n].state + 2)])

        self.calcOptions(strike)

    def calcStretchParameter(self):
        self.up = np.exp(sigma * np.sqrt(3 * self.del_t))
        self.down = 1 / self.up
        return np.sqrt(3)
    
    def calcOptions(self, strike):
        # Calculate terminal payoffs
        for n in self.nodes.values():
            if not n.childs:
                n.option_price = n.calcPayoff(strike)

        # Finish the backwardization
        for n in reversed(self.nodes.values()):
            if n.childs:
                n.option_price = np.exp(-self.r * self.del_t) * (self.prob_up * n.childs[0].option_price 
                + (self.prob_mid) * n.childs[1].option_price + (self.prob_down * n.childs[2].option_price))
                
    
class TrinomialBarrierTree(TrinomialTree):
    
    def __init__(self, time_steps: int, up: float, down: float, 
            initial_price: float, r: float, 
            del_t: float, option_type: str, 
            strike: float, sigma: float, barrier: float,
            div_amt: float = 0, div_times: list = []):
        super().__init__(time_steps, up, down, 
                initial_price, r, 
                del_t, option_type, strike, sigma,  
                div_amt, div_times)
        self.barrier = barrier
        self.nodes = self.calcStockPrices() 
        # Children creation
        for n in self.nodes:
            if self.nodes[n].time_step < self.time_steps:
                self.nodes[n].childs.append(self.nodes[(self.nodes[n].time_step + 1, self.nodes[n].state)])
                self.nodes[n].childs.append(self.nodes[(self.nodes[n].time_step + 1, self.nodes[n].state + 1)]) 
                self.nodes[n].childs.append(self.nodes[(self.nodes[n].time_step + 1, self.nodes[n].state + 2)])
        self.calcOptions(strike)

    def calcStretchParameter(self):
        if "down" in self.option_type:
            eta = np.log(self.initial_price / self.barrier) / (self.sigma * np.sqrt(self.del_t))
        elif "up" in self.option_type:
            eta = np.log(self.barrier / self.initial_price) / (self.sigma * np.sqrt(self.del_t))
        n_0 = np.floor(eta)
        if n_0 != 0:
            self.up = np.exp((eta / n_0) * sigma * np.sqrt(self.del_t))
            self.down = 1 / self.up
            return eta / n_0
        else:
            self.up = np.exp(sigma * np.sqrt(self.del_t))
            self.down = 1 / self.up
            return 1


    def calcOptions(self, strike):
        # Calculate terminal payoffs
        for n in reversed(self.nodes.values()):
            if n.childs:
                break
            else:
                n.option_price = n.calcPayoff(strike, barrier=self.barrier)
        # Finish the backwardization depending on the type of barrier option
        if "down-and-out" in self.option_type:
            for n in reversed(self.nodes.values()):
                if n.stock_price < self.barrier or math.isclose(n.stock_price, self.barrier, rel_tol=1e-2):
                    n.option_price = 0
                elif n.childs:
                    n.option_price = np.exp(-self.r * self.del_t) * ((self.prob_up * n.childs[0].option_price) 
                    + (self.prob_mid * n.childs[1].option_price) + (self.prob_down * n.childs[2].option_price))
            
        elif "up-and-out" in self.option_type:
            for n in reversed(self.nodes.values()):
                if n.stock_price > self.barrier or math.isclose(n.stock_price, self.barrier, rel_tol=1e-2):
                    n.option_price = 0
                elif n.childs:
                    n.option_price = np.exp(-self.r * self.del_t) * ((self.prob_up * n.childs[0].option_price) 
                    + (self.prob_mid * n.childs[1].option_price) + (self.prob_down * n.childs[2].option_price))           
                    
class Node:
    
    def __init__(self, stock_price: float, number: int,
                 time_step: tuple, childs: list, state: int,
                option_type: str, option_price: float = 0.0): 
        self.stock_price = stock_price
        self.number = number # Used for graphviz
        self.time_step = time_step
        self.state = state # position within time step (0 is top node)
        self.option_price = option_price 
        self.option_type = option_type
        self.childs = childs

        if "chooser" in self.option_type or "Asian" in self.option_type:
            self.option_prices = np.zeros(4)
        if "American" in self.option_type:
            self.early_exercise = False
    
    def calcPayoff(self, strike, stock_avg=None, binary_payoff=None, barrier=None):
        
        # Asian option payoffs
        if "Asian call (average price)" in self.option_type:
            return max(stock_avg - strike, 0)
        elif "Asian put (average price)" in self.option_type:
            return max(strike - stock_avg, 0)
        elif "Asian call (average strike)" in self.option_type:
            return max(self.stock_price - strike, 0)
        elif "Asian put (average strike)" in self.option_type:
            return max(strike - self.stock_price, 0)
                
        # binary option payoffs
        elif self.option_type == "binary call (cash-or-nothing)":  
            if self.stock_price > strike:
                return binary_payoff 
            else:
                return 0
        elif self.option_type == "binary put (cash-or-nothing)":
            if self.stock_price < strike:
                return binary_payoff
            else:
                return 0
        elif self.option_type == "binary call (asset-or-nothing)":
            if self.stock_price >= strike:
                return self.stock_price
            else:
                return 0
        elif self.option_type == "binary put (asset-or-nothing)":
            if self.stock_price < strike:
                return self.stock_price
            else:
                return 0
        
        # barrier option terminal payoffs
        elif "barrier" in self.option_type:            
            if self.option_type == "down-and-out call (barrier)" or self.option_type == "up-and-in call (barrier)":
                return max(self.stock_price - strike, 0) * self.isAbove(barrier)
            elif self.option_type == "down-and-out put (barrier)" or self.option_type == "up-and-in put (barrier)":
                return max(strike - self.stock_price, 0) * self.isAbove(barrier)
            elif self.option_type == "down-and-in call (barrier)" or self.option_type == "up-and-out call (barrier)":
                return max(self.stock_price - strike, 0) * self.isBelow(barrier)
            elif self.option_type == "down-and-in put (barrier)" or self.option_type == "up-and-out put (barrier)":
                return max(strike - self.stock_price, 0) * self.isBelow(barrier)
            
        # American and European vanilla option payoffs and compound option terminal payoffs
        elif "call" in self.option_type:
            return max(self.stock_price - strike, 0)
        elif "put" in self.option_type:
            return max(strike - self.stock_price, 0) 
    
    def isBelow(self, barrier):
        """Modified indicator function for barrier options. Returns 1 if the option is at or below the 
        barrier and 0 otherwise. """
        if self.stock_price > barrier:
            return 0
        else:
            return 1

    def isAbove(self, barrier):
        """Modified indicator function for barrier options. Returns 1 if the option is at or above the barrier
        and 0 otherwise. """
        if self.stock_price < barrier:
            return 0
        else:
            return 1
    
    def calcAverages(self, tree):
        """Calculates stock price averages for Asian options."""
        self.average = np.zeros(4)
        if self.number == 0:
            # initial node
            self.average[0] = self.average[1] = self.average[2] = self.average[3] = self.stock_price
        elif self.state == 0:
            # nodes at the top edge
            self.average[3] = (1 / (self.time_step + 1)) * (tree.nodes[(self.time_step - 1, self.state)].average[0] * self.time_step + self.stock_price)
            self.average[0] = self.average[1] = self.average[2] = self.average[3]
        elif self.state == self.time_step:
            # nodes at the bottom edge
            self.average[0] = (1 / (self.time_step + 1)) * (tree.nodes[(self.time_step - 1, self.state - 1)].average[3] * self.time_step + self.stock_price)
            self.average[1] = self.average[2] = self.average[3] = self.average[0]
        else: 
            # nodes located vertically between two other nodes
            self.average[0] = (1 / (self.time_step + 1)) * (tree.nodes[(self.time_step - 1, self.state)].average[0] * self.time_step + self.stock_price)
            self.average[3] = (1 / (self.time_step + 1)) * (tree.nodes[(self.time_step - 1, self.state - 1)].average[3] * self.time_step + self.stock_price)

            # Interpolate to get average2 and average3
            self.average[1] = (2 * self.average[0] + self.average[3]) / 3
            self.average[2] = (self.average[0] + 2 * self.average[3]) / 3


def CRRParams(sigma, delta_t, risk_free_rate):
    up = np.exp(sigma * np.sqrt(delta_t))
    down = 1 / up
    prob = (np.exp(risk_free_rate * delta_t) - down) / (up - down)

    return up, down, prob

def JRParams(sigma, delta_t, risk_free_rate):
    up = np.exp((risk_free_rate - (np.power(sigma, 2) / 2)) * delta_t + sigma * np.sqrt(delta_t))
    down = np.exp((risk_free_rate - (np.power(sigma, 2) / 2)) * delta_t - sigma * np.sqrt(delta_t))
    return up, down, 0.5

def TianParams(sigma, delta_t, risk_free_rate):
    M = np.exp(risk_free_rate * delta_t)
    V = np.exp(np.power(sigma, 2) * delta_t)
    up = ((M * V) / 2) * (V + 1 + np.sqrt(np.power(V, 2) + 2 * V - 3))
    down = ((M * V) / 2) * (V + 1 - np.sqrt(np.power(V, 2) + 2 * V - 3))
    prob = (M - down) / (up - down)
    return up, down, prob

def LRParams(sigma, delta_t, risk_free_rate, n):
    T =  delta_t * n # time-to-maturity
    d1 = (np.log(100 / 100) + T * (0.07 + (np.power(sigma, 2) / 2))) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    inner_d2 = np.sqrt(0.25 - 0.25 * np.exp(-1 * np.power(d2 / (n + (1/3) + (0.1 / (n + 1) ) ), 2) * (n + (1 / 6))) )
    prob = 0.5 + math.copysign(inner_d2, d2)
    inner_d1 = np.sqrt(0.25 - 0.25 * np.exp(-1 * np.power(d1 / (n + (1/3) + (0.1 / (n + 1) ) ), 2) * (n + (1 / 6))) )
    q = 0.5 + math.copysign(inner_d1, d1)
    up = np.exp(0.07 * delta_t) * (q / prob)
    down = (np.exp(0.07 * delta_t) - prob * up) / (1 - prob)
    return up, down, prob

# Streamlit interface #

def instantiateTree(class_name, tree_vars):
    return class_name(*tree_vars)

tree_vars = []
B, compound_time, choosing_time, K_c, sigma, H = None, None, None, None, None, None

tree_objects = {
                "European call": [instantiateTree, EuropeanTree, tree_vars],                
                "European put": [instantiateTree, EuropeanTree, tree_vars],
                "American call": [instantiateTree, AmericanTree, tree_vars],
                "American put": [instantiateTree, AmericanTree, tree_vars],
                "Asian call (average price)": [instantiateTree, AsianTree, tree_vars], 
                "Asian put (average price)": [instantiateTree, AsianTree, tree_vars], 
                "Asian call (average strike)": [instantiateTree, AsianTree, tree_vars],  
                "Asian put (average strike)": [instantiateTree, AsianTree, tree_vars], 
                "Asian call (average price) American-style": [instantiateTree, AsianTree, tree_vars], 
                "Asian put (average price) American-style": [instantiateTree, AsianTree, tree_vars], 
                "Asian call (average strike) American-style": [instantiateTree, AsianTree, tree_vars],  
                "Asian put (average strike) American-style": [instantiateTree, AsianTree, tree_vars], 
                "binary call (cash-or-nothing)": [instantiateTree, BinaryTree, tree_vars], 
                "binary put (cash-or-nothing)": [instantiateTree, BinaryTree, tree_vars], 
                "binary call (asset-or-nothing)": [instantiateTree, BinaryTree, tree_vars],  
                "binary put (asset-or-nothing)": [instantiateTree, BinaryTree, tree_vars], 
                "Call on call": [instantiateTree, CompoundTree, tree_vars], 
                "Call on put": [instantiateTree, CompoundTree, tree_vars],  
                "Put on put": [instantiateTree, CompoundTree, tree_vars],  
                "Put on call": [instantiateTree, CompoundTree, tree_vars], 
                "chooser": [instantiateTree, ChooserTree, tree_vars], 
                "European call (trinomial)": [instantiateTree, EuropeanTrinomialTree, tree_vars], 
                "European put (trinomial)": [instantiateTree, EuropeanTrinomialTree, tree_vars], 
                "down-and-out call (barrier)": [instantiateTree, TrinomialBarrierTree, tree_vars], 
                "down-and-out put (barrier)": [instantiateTree, TrinomialBarrierTree, tree_vars], 
                "up-and-out call (barrier)": [instantiateTree, TrinomialBarrierTree, tree_vars], 
                "up-and-out put (barrier)": [instantiateTree, TrinomialBarrierTree, tree_vars], 
            }

# Visualization?
visualization = st.checkbox("visualization?", value=True)

# Select menu for option type
o_type = st.selectbox("Option type", options = tree_objects.keys())

# Text input for initial stock price
S0 = st.empty()
S0 = float(st.text_input("Initial stock price: ", value=95))

# Text input for strke price
K = st.empty()
K = float(st.text_input("Strike price: ", value=100))

if visualization:
    # Slider for number of time steps (1-25)
    num_time_steps = int(st.slider(label="\# of time steps: ", min_value=1, max_value=25))
else:
    # Input for number of time steps
    num_time_steps = int(st.text_input("Number of time steps", value = 5))

# Slider for time-to-maturity
T = float(st.text_input(label="time-to-maturity (years)", value=1))
delta_t = T / num_time_steps

# Input for annualized volatility
sigma = float(st.text_input("Annualized volatility (0.xx)", value=0.25))

# Input for risk-free rate
risk_free_rate = float(st.text_input("Risk-free rate (0.xx): ", value=0.10))

# Input for tree parameter method
tree_methods = ("Cox-Ross-Rubinstein (1979)", "Jarrow-Rudd (1983)", "Tian (1993)", "Leisen-Reimer (1996)")
tree_method = st.selectbox(label="Method of parameter selection:", options = tree_methods)

# Input for dividends
div_amt = float(st.text_input("Dividend amount (0.xx)", value = 0.00))
div_times = st.multiselect("Ex-dividend dates", options=[i for i in range(0, num_time_steps)])
    
if tree_method == tree_methods[0]: 
    # CRR
    up, down, prob = CRRParams(sigma, delta_t, risk_free_rate)
elif tree_method == tree_methods[1]: 
    # Jarrow-Rudd
    up, down, prob = JRParams(sigma, delta_t, risk_free_rate)
elif tree_method == tree_methods[2]:
    # Tian
    up, down, prob = TianParams(sigma, delta_t, risk_free_rate)
elif tree_method == tree_methods[3]:
    # Leisen-Reimer
    up, down, prob = LRParams(sigma, delta_t, risk_free_rate, num_time_steps)

# Tree parameter methods
if "trinomial" in o_type or "barrier" in o_type:
    # Trinomial trees cook up three martingale probabilities inside the class
    tree_vars = [num_time_steps, up, down, S0, risk_free_rate, delta_t, o_type, K, sigma]
else:
    # Need this line - difference is the inclusion of prob
    tree_vars = [num_time_steps, up, down, S0, risk_free_rate, delta_t, o_type, prob, K]

# barrier 
if 'barrier' in o_type:
    H = float(st.text_input("barrier: ", value=90))
    tree_vars.append(H)
else:
    H = None   

# chooser parameters
if o_type == 'chooser':
    choosing_time = int(st.slider(label="choosing time", min_value=0, max_value=num_time_steps))
    tree_vars.append(choosing_time)
else: 
    choosing_time = None

# compound parameters
if 'on' in o_type:
    compound_time = int(st.slider(label="compound time", min_value=0, max_value=num_time_steps))
    K_c = float(st.text_input("Compound strike price: ", value=20))
    tree_vars.append(compound_time)
    tree_vars.append(K_c)
else:
    compound_time = None
    K_c = None

# binary parameters
if 'binary' in o_type:
    B = float(st.text_input(label="binary payoff", value=50))
    tree_vars.append(B)
else:
    B = None

if div_amt != 0 and div_times:
    tree_vars.append(div_amt)
    tree_vars.append(div_times)

# Update with a for loop
for key in tree_objects.keys():
    tree_objects[key][2] = tree_vars

class_components = tree_objects[o_type]
BT = class_components[0](class_components[1], class_components[2])

if visualization:
    BT.draw_tree()
else:
    st.write(f"""Option price is {BT.nodes[(0,0)].option_price}""")
