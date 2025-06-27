from langchain_groq import ChatGroq
from dotenv import load_dotenv 
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.1,
)

model2 = ChatGroq(
    model="deepseek-r1-distill-llama-70b"
)

prompt1 = PromptTemplate(
    template= 'generate short and simple notes from the following prompt \n {text}',
    input_variables= ['text']
)

prompt2 = PromptTemplate(
    template= 'generate 5 short question answers from the following text \n {text}',
    input_variables= ['text']
)

prompt3 = PromptTemplate(
    template= ' Merge the provide notes and quiz into a single document \n notes -> {notes} and quiz-> {quiz}',
    input_variables= ['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain =RunnableParallel( {
    'notes': prompt1| model | parser,
    'quiz': prompt2 | model2 |parser
})

merge_chain = prompt3 | model | parser

chain = parallel_chain | merge_chain 

text = """ First Order Differential Equations - In this chapter we will look at several of the standard solution methods for first order differential equations including linear, separable, exact and Bernoulli differential equations. We also take a look at intervals of validity, equilibrium solutions and Euler’s Method. In addition we model some physical situations with first order differential equations.
Linear Equations – In this section we solve linear first order differential equations, i.e. differential equations in the form 
y
′
+
p
(
t
)
y
=
g
(
t
)
. We give an in depth overview of the process used to solve this type of differential equation as well as a derivation of the formula needed for the integrating factor used in the solution process.
Separable Equations – In this section we solve separable first order differential equations, i.e. differential equations in the form 
N
(
y
)
y
′
=
M
(
x
)
. We will give a derivation of the solution process to this type of differential equation. We’ll also start looking at finding the interval of validity for the solution to a differential equation.
Exact Equations – In this section we will discuss identifying and solving exact differential equations. We will develop a test that can be used to identify exact differential equations and give a detailed explanation of the solution process. We will also do a few more interval of validity problems here as well.
Bernoulli Differential Equations – In this section we solve Bernoulli differential equations, i.e. differential equations in the form 
y
′
+
p
(
t
)
y
=
y
n
. This section will also introduce the idea of using a substitution to help us solve differential equations.
Substitutions – In this section we’ll pick up where the last section left off and take a look at a couple of other substitutions that can be used to solve some differential equations. In particular we will discuss using solutions to solve differential equations of the form 
y
′
=
F
(
y
x
)
 and 
y
′
=
G
(
a
x
+
b
y
)
.
Intervals of Validity – In this section we will give an in depth look at intervals of validity as well as an answer to the existence and uniqueness question for first order differential equations.
Modeling with First Order Differential Equations – In this section we will use first order differential equations to model physical situations. In particular we will look at mixing problems (modeling the amount of a substance dissolved in a liquid and liquid both enters and exits), population problems (modeling a population under a variety of situations in which the population can enter or exit) and falling objects (modeling the velocity of a falling object under the influence of both gravity and air resistance).
Equilibrium Solutions – In this section we will define equilibrium solutions (or equilibrium points) for autonomous differential equations, 
y
′
=
f
(
y
)
. We discuss classifying equilibrium solutions as asymptotically stable, unstable or semi-stable equilibrium solutions.
Euler’s Method – In this section we’ll take a brief look at a fairly simple method for approximating solutions to differential equations. We derive the formulas used by Euler’s Method and give a brief discussion of the errors in the approximations of the solutions.

"""

results = chain.invoke({'text':text})

print(results)
