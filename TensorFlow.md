# Introduction
This guide gets you started programming in the low-level TensorFlow APIs (TensorFlow Core), showing you how to:

+ Mangage your own TensorFlow program ( ) and TensorFlow runtime (), instead of relying on Estimators to manage them.
+ Run TensorFlow operations, using a tf.Session
+ Use high level components (datasets, layers, and feature_columns) in this low level environmnet.
+ Build your own training loop, instead of using the one provided by Estimators.

We recommend using the higher level APIs to build models when possible. Knowing TensorFlow Core is valuable for the following reasons:

+ Experimentation and debugging are both more straight forward when you can use low level TensorFlow operations directly.
+ It gives you a mental model of how things work internally when using the higher level APIs.

# Tensorflow Core Wolkthrough
You might think of TensorFLow Core programs as consisting of two discreate sections:

1. Building the computational graph (a tf.Graph)
2. Running the computational graph (using a tf.Session)

### graph

A __computational graph__ is series of TensorFlow operations arranged into a graph. The graph is composed of two types of objects.

+ Operations (or "ops"): The nodes of the graph. Operations describe calculations that consume and produce tensors.
+ Tensors: The edges in graph. Theses represent the value that will flow through the graph. Most TensorFlow function retur tf.Tensors

> Important: tf.Tensors do not have values, they are just handles to elements in the computation graph.

### Session

to evaluate tensors, instantiate a tf.Session object, informally known as a session. A session encapsulates the state of the TensorFlow runtime, and runs TensorFlow operations.


# Graphs and Sessions

TensorFlow uses a dataflow graph to represent your computation in terms of the dependencies between individual operations. This leads to a low-level programming model in which you first define the dataflow graph, then create a TensorFlow session to run parts of the graph across a set of local and remote devices.

This guide will be most useful if you intend to use the low-level programming model directly. Higher-level APIs such as tf.estimator.Estimator and Keras hide the details of graphs and sessions form the end user, but this guide may also be useful if you want to understand how these APIs are implemented.

[Image]

Dataflow is common programming model for parallel computing. In a dataflow graph, the nodes represent units of computation, and the edges represent the data consumed or produced by a computation.

Dataflow has several advantages that TensorFlow leverages when executing your programs:

+ __Parallelism__ By using explicit edges to represent dependecies between operations, it is easy for the system to identify operations that can execute in parallel.
+ __Distributed execution__ By using explicit edges to represent the values that flow between operations, it is possible for TensorFlow to partition your program across multiple devices (CPUs, GPUs, and TPUs) attached to different machines. TensorFlow inserts the necessary communication and coordination between device.
+ __Compilation__ TensorFlow's XLA compiler can use the information in your dataflow graph to generate faster code, for example, by fusing together adjacent operations.
+ __Portability__ The dataflow graph is a language-independent representation of the code in your model. You can build a dataflow graph in Python, store it in Saved Model, and resotre it in a C++ program for low-latency inference.

### What is a tf.Graph?

A tf.Graph contains two relevant kinds of informations:

+ __Graph structure__ The nodes and edges of the graph, indicating how individual operations are composed together,  but not prescribing how they should be used. The graph structure is like assembly code: inspecting it can convey some useful information, but it does not contain all of the useful context that source code conveys.
+ __Graph collections__ TensorFlow provides a general mechanism for storing collections of medata in a tf.Graph. The tf.add_to_collection function enables you to associate a list of objects with a key(where tf.GraphKey defines some of the standard keys), and tf.get_collection enables you to look up all objects associated with key. Many parts of the TensorFlow libarary use this facility: for example, when you create a tf.Variable, it is added by default to collections representing "global variables" and "trainable variables". When you later come to create a tf.train.Saver or tf.train.Optimizer, the variables in these collections are used as the default arguments.

### Budilding a tf.Graph

Most TensorFlow programs start with a dataflow graph construction phase. In this phase, you invoke TensorFlow API functions that construct new tf.Operation (node) and tf.Tensor (edge) objects and add them to a tf.Graph instance. TensorFlow provides a __default graph__ that is an implicit argument to all API functions in the same context. For example:

+ Calling tf.constant(42.0) creates a single tf.Operation that produces the value 42.0, add it to the default graph, and returns a tf.Tensor that represents the value of the constant.
+ Calling tf.matmul(x, y) creates a single tf.Operation that multiplies the values of tf.Tensor objects x and y, adds it to the default graph, and returs a tf.Tensor that represents the result of the multiplication.
+ Executing v = tf.Variable(0) adds to the graph a tf.Operation that will store a writeable tensor value that persists between tf.Session.run calls. The tf.Variable object wraps this operation, and can be used like a tensor, object wraps this operation, and can be used like a tensor, which will read the current value of the stored value. The tf.Variable object also has methods such as assign and assign_add that create tf.Operation objects that, when executed, update the stored value. (See Variables for more information about variables.)
+ Calling tf.train.Optimizer.minimize will add operations and tensors to the default graph theat calculate gradients, and return a tf.Operation that, when run, will apply those gradients to a set of variables.

More programs rely solely on the default graph. However, see Dealing with multiple graphs for more advanced use cases. High-level APIs such as th.estimator.Estimator API manage the dafalut graph on your behalf, and--for exampel--may create different graphs for training and evaluation.

> Note: Calling most function in the TensorFlow API merely adds operations and tensors to the default graph, but does not perform the actual computation. Instead, you compose these functions until you have a tf.Tensor or tf.Operation that representes the overall computation--such as performing on step of gradient descent--and then pass that object to a tf.Session to perform the computation. See the section "Executing a graph in a tf.Session" for more details.

### Executing a graph in a tf.Session

TensorFlow use the tf.Session class to represent a connection between the client program--typically a Python program, although a similar interface is available in other languages--and the C++ runtime. A tf.Session object provides access to devices in the local machine, and remote devices using the distributed TensorFlow runtime. It also caches information about your tf.Graph so that you can efficiently run the same computation multiple times.

# Operations

| Category                             | Examples                                                  |
|--------------------------------------|-----------------------------------------------------------|
| Element-wise mathematical operations | Add, Sub, Mul, Div, Exp, Log, Greater, Less, Equal...     |
| Array operations                     | Concat, Slice, Split, Constant, Rank, Shape, Shuffle, ... |
| Matrix operations                    | MatMul, MatrixInverse, MatrixDeterminant, ...             |
| Stateful operations                  | Variable, Assign, AssignAdd, ...                          |
| Neural network building blocks       | SoftMax, Sigmoid, ReLU, Convolution2D, MaxPool, ...       |
| Checkpointing operations             | Save, Restore                                             |
| Queue and synchronization operations | Enqueue, Dequeue, MutexAcquire, MutexRelease, ...         |
| Control flow operations              | Merge, Switch, Enter, Leave, NextIteration                |


