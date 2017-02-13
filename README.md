#GSDMM: Short text clustering (Rust)

This project implements the Gibbs sampling algorithm for a Dirichlet Mixture Model of [Yin and Wang 2014](https://pdfs.semanticscholar.org/058a/d0815ce350f0e7538e00868c762be78fe5ef.pdf) for the 
clustering of short text documents. 
Some advantages of this algorithm:
 - It requires only an upper bound `K` on the number of clusters
 - With good parameter selection, the model converges quickly
 - Space efficient and scalable
 
This is a very fast and efficient GSDMM implementation in Rust. See [this project](https://github.com/rwalk/gsdmm) for a simpler but slower Python implementation.

## The Movie Group Process
In their paper, the authors introduce a simple conceptual model for explaining the GSDMM called the Movie Group Process.

Imagine a professor is leading a film class. At the start of the class, the students
are randomly assigned to `K` tables. Before class begins, the students make lists of
their favorite films. The professor repeatedly reads the class role. Each time the student's name is called,
the student must select a new table satisfying one or both of the following conditions:

- The new table has more students than the current table or is empty.
- The new table has students with similar lists of favorite movies.

By following these steps consistently, we might expect that the students eventually arrive at an "optimal" table configuration.

## Usage

The package contains both a library which can be embedded in other projects as well as a standalone command line executable for running the GSDMM process on your data.

To use the executable, you must first build it:
```shell
cargo build --release
```
See the help for additional details:
```shell
./target/release/gsdmm -h
```

The executable should be invoked like this:
```
gsdmm <datafile> <vocabfile> <outprefix> [-k <max_clusters>] [-a <alpha>] [-b <beta>] [-m <maxit>]
```
Each line of the `<datafile>` should be a single short text document with words in the document seperated by single white spaces, and with any desired preprocessing already applied.  The `<vocabfile>` file is list of all valid tokens (again one per line) and preprocessed to align with the words in the `<datafile>`.  Any token not in the `<datafile>` but not in the `<vocabfile>` will be ignored (so you can remove stopwords from the `<datafile>` by simply not including them in the `<vocabfile>`).  The `<outprefix>` is the full path + an output name prefix for the files that will be written by `gsdmm`.  

The parameter `alpha` controls the probability that a document will join an already emptied cluster. Lower values of `alpha`  tend to make convergence faster and prevent emptied clusters from filling again during the iterations.  The parameter `beta` controls the tolerance for diversity in the clustering. Higher values of `beta` allow for more mixing of documents while low values encourage high numbers of clusters with high purity. `k` is an upper bound on the number of clusters.  It should always be chosen to be larger than the number of clusters you think exist in your data. 

The model will compute at most `<maxit>` iterations of the GSDMM file.  If, however, at the end of an iteration no documents have changed clusters, the model is regarded as converged and the algorithm terminates.  This a very strict criteria except in the most trivial examples. Generally, good results can be obtained after 50-100 iterations of the model on real-life data.

### Grades Example
The [examples](example/) folder contains a couple of simple examples to explore the GSDMM model. Let's consider the grades example. The file [grades.txt](example/grades.txt) is a list of documents (one per line).  Each document is a single letter grade recieved by a student.  The clustering problem for this data has an exact solution, where all the letter grades of the same of the same value end up in the same cluster and the number of clusters is equal to the number of possible letter grades.

To run this example:
```
./target/release/gsdmm examples/grades/grades.txt examples/grades/grades_vocab.txt examples/grades_out_ -a 0.3 -b 0.001 -m 1000 -k 10 
```
This will generate three output files:
```
examples/grades/grades_out_cluster_descriptions.txt
examples/grades/grades_out_label_probabilities.csv
examples/grades_out_labels.csv
```
The cluster descriptions file lists the vocabulary for each each cluster, labeled 1-K. The label probabilities file is a `n_docs x K` matrix in `.csv` format giving the probability for each document to appear in the clusters 1-K.  The labels file provides a cluster label for each document and the probability of that label. Each line in the labels file is simply the pair `argmax,max,text` computed from the corresponding line in the label probability file. 

## Practical tips
A few ideas for working with GSDMM:
- Always choose `K` bigger than the expected number of clusters but generally with the same order of magnitude as the expected number.  If the number of clusters remains constant across all iterations, you may not have chosen a large enough value of `K`. Note, however, that large `K` significantly increases the computation time.
- Alpha and beta need to be tuned for each data set and use case. Alpha and beta tend to work in opposite directions and they significantly impact the converence behavior.  It's usually sufficient to start with a subsample of documents to get ballpark estimates for these parameters.
- Monitor the number of clusters and the number of docs transferred at each iteration. Both should die off quickly and then stabilize.  Generally, neither number should increase significantly in subsequent iterations.
