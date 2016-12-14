# Spark machine learning inventory [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

A curated inventory of machine learning methods available on the Apache Spark platform, both in official and third party libraries.

<!-- https://github.com/thlorenz/doctoc -->
<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**

- [Project inventory](#project-inventory)
  - [Machine learning & related libraries](#machine-learning--related-libraries)
    - [Bundled with Spark](#bundled-with-spark)
    - [Third party libraries](#third-party-libraries)
    - [Interfaces](#interfaces)
  - [Notebooks](#notebooks)
  - [Visualization](#visualization)
  - [Others](#others)
- [Task inventory](#task-inventory)
  - [Ensemble learning & parallel modelling](#ensemble-learning--parallel-modelling)
    - [Libraries](#libraries)
    - [Algorithms](#algorithms)
  - [Classification](#classification)
    - [Libraries](#libraries-1)
    - [Algorithms](#algorithms-1)
  - [Clustering](#clustering)
    - [Libraries](#libraries-2)
    - [Algorithms](#algorithms-2)
  - [Deep learning](#deep-learning)
    - [Libraries](#libraries-3)
  - [Feature selection & dimensionality reduction](#feature-selection--dimensionality-reduction)
    - [Libraries](#libraries-4)
    - [Algorithms](#algorithms-3)
  - [Graph computations](#graph-computations)
    - [Libraries](#libraries-5)
  - [Linear algebra](#linear-algebra)
    - [Libraries](#libraries-6)
  - [Matrix factorization & recommender systems](#matrix-factorization--recommender-systems)
    - [Libraries](#libraries-7)
    - [Algorithms](#algorithms-4)
  - [Natural language processing](#natural-language-processing)
    - [Libraries](#libraries-8)
    - [Algorithms](#algorithms-5)
  - [Optimization & hyperparameter search](#optimization--hyperparameter-search)
    - [Libraries](#libraries-9)
    - [Algorithms](#algorithms-6)
  - [Regression](#regression)
    - [Libraries](#libraries-10)
    - [Algorithms](#algorithms-7)
  - [Statistics](#statistics)
  - [Tensor decompositions](#tensor-decompositions)
    - [Libraries](#libraries-11)
    - [Algorithms](#algorithms-8)
  - [Time series](#time-series)
    - [Libraries](#libraries-12)
    - [Algorithms](#algorithms-9)
- [Practical info](#practical-info)
  - [License](#license)
  - [Contributing](#contributing)
  - [Acknowledgments](#acknowledgments)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

<!-- START OF MAIN CONTENT -->


# Project inventory

## Machine learning & related libraries

### Bundled with Spark

- [GraphX](http://spark.apache.org/graphx/) - Apache Spark's API for graphs and graph-parallel computation
- [MLlib](http://spark.apache.org/docs/latest/ml-guide.html) - Apache Spark's built in machine learning library

### Third party libraries

- [Aerosolve](http://airbnb.io/aerosolve/) - A machine learning package built for humans
- [AMIDST](http://www.amidsttoolbox.com/) - probabilistic machine learning
- [CoCoA](https://github.com/gingsmith/cocoa) - communication-efficient distributed coordinate ascent
- [Deeplearning4j](https://deeplearning4j.org/spark.html) - Deeplearning4j on Spark
- [DissolveStruct](http://dalab.github.io/dissolve-struct/) - Distributed Solver for Structured Prediction
- [DistML](https://github.com/intel-machine-learning/DistML) - DistML provide a supplement to mllib to support model-parallel on Spark
- [Elephas](http://maxpumperla.github.io/elephas/) - Distributed Deep learning with Keras & Spark
- [Generalized K-means clustering](https://github.com/derrickburns/generalized-kmeans-clustering) - generalizes the Spark MLLIB Batch and Streaming K-Means clusterers in every practical way
- [KeystoneML](http://keystone-ml.org/) - KeystoneML is a software framework, written in Scala, from the UC Berkeley AMPLab designed to simplify the construction of large scale, end-to-end, machine learning pipelines with Apache Spark
- [MLbase](http://www.mlbase.org/) - MLbase is a platform addressing implementing and consuming Machine Learning at scale
- [ml-matrix](https://github.com/amplab/ml-matrix) - distributed matrix library
- [revrand](https://github.com/NICTA/revrand) - A library of scalable Bayesian generalised linear models with fancy features
- [spark-ts](https://github.com/sryza/spark-timeseries) - Time series for Spark
- [Sparkling Water](http://www.h2o.ai/sparkling-water/) - H2O + Apache Spark
- [Splash](http://zhangyuc.github.io/splash/) - a general framework for parallelizing stochastic learning algorithms on multi-node clusters
- [Spectral LDA on Spark](https://github.com/FurongHuang/spectrallda-tensorspark) -  implements a spectral (third order tensor decomposition) learning method for learning LDA topic model on Spark
- [StreamDM](http://huawei-noah.github.io/streamDM/) - Data Mining for Spark Streaming
- [Thunder](http://thunder-project.org/) - scalable image and time series analysis
- [Zen](https://github.com/cloudml/zen) - aims to provide the largest scale and the most efficient machine learning platform on top of Spark, including but not limited to logistic regression, latent dirichilet allocation, factorization machines and DNN

### Interfaces

- [CaffeOnSpark](https://github.com/yahoo/CaffeOnSpark) - CaffeOnSpark brings deep learning to Hadoop and Spark clusters
- [Elephas](http://maxpumperla.github.io/elephas/) - Distributed Deep learning with Keras & Spark
- [Spark CoreNLP](https://github.com/databricks/spark-corenlp) - CoreNLP wrapper for Spark
- [Spark Highcharts](https://github.com/knockdata/spark-highcharts) - Support Highcharts in Apache Zeppelin
- [Sparkling Water](http://www.h2o.ai/sparkling-water/) - H2O + Apache Spark
- [sparklyr](http://spark.rstudio.com/mllib.html) - sparklyr provides R bindings to Spark’s distributed machine learning library
- [Sparkit-learn](https://github.com/lensacom/sparkit-learn) - PySpark + Scikit-learn = Sparkit-learn
- [Spark-TFOCS](https://github.com/databricks/spark-tfocs) - port of TFOCS: Templates for First-Order Conic Solvers (cvxr.com/tfocs)
- [Hivemall-Spark](https://github.com/maropu/hivemall-spark) - A Hivemall wrapper for Spark
- [Spark PMML exporter validator](https://github.com/selvinsource/spark-pmml-exporter-validator) - Using JPMML Evaluator to validate the PMML models exported from Spark

## Notebooks

- [Beaker](http://beakernotebook.com/) - The data scientist's laboratory
- [Spark Notebook](http://spark-notebook.io/) - Interactive and Reactive Data Science using Scala and Spark
- [sparknotebook](https://github.com/hohonuuli/sparknotebook) - running Apache Spark using Scala in ipython notebook
- [Apache Zeppelin](https://zeppelin.apache.org/) - A web-based notebook that enables interactive data analytics

## Visualization

- [Plotly](https://plot.ly/python/apache-spark/) - Spark Dataframes with Plotly
- [Spark Highcharts](https://github.com/knockdata/spark-highcharts) - Support Highcharts in Apache Zeppelin
- [Spark ML streaming](https://github.com/freeman-lab/spark-ml-streaming) - Visualize streaming machine learning in Spark
- [Vegas](https://github.com/vegas-viz/Vegas) - The missing MatPlotLib for Scala + Spark

## Others

- [Apache Toree](https://toree.apache.org/) - Gateway to Apache Spark
- [Distributed DataFrame](http://ddf.io/) - Simplify Analytics on Disparate Data Sources via a Uniform API Across Engines
- [Apache Metron](http://metron.incubator.apache.org/) - real-time Big Data security
- [PipelineIO](http://pipeline.io/) - Extend ML Pipelines to Serve Production Users
- [Spark Jobserver](https://github.com/spark-jobserver/spark-jobserver) - REST job server for Apache Spark
- [Spark PMML exporter validator](https://github.com/selvinsource/spark-pmml-exporter-validator) - Using JPMML Evaluator to validate the PMML models exported from Spark
- [Spark-Ucores](https://gitlab.com/mora/spark-ucores) - Spark for Unconventional Cores
- [Twitter stream ML](https://github.com/giorgioinf/twitter-stream-mlhttps://github.com/giorgioinf/twitter-stream-ml) - Machine Learning over Twitter's stream. Using Apache Spark, Web Server and Lightning Graph server.
- [Velox](https://github.com/amplab/velox-modelserver) - a system for serving machine learning predictions


<!-- ALGORITHM INVENTORY -->


# Task inventory

- [MLlib](http://spark.apache.org/docs/latest/ml-guide.html) - Apache Spark's built in machine learning library

## Ensemble learning & parallel modelling

### Libraries

- [DistML](https://github.com/intel-machine-learning/DistML) - DistML provide a supplement to mllib to support model-parallel on Spark
- [Elephas](http://maxpumperla.github.io/elephas/) - Distributed Deep learning with Keras & Spark
- [spark-FM-parallelISGD](https://github.com/blebreton/spark-FM-parallelSGD) - Implementation of Factorization Machines on Spark using parallel stochastic gradient descent
- [SparkBoost](https://github.com/tizfa/sparkboost) - A distributed implementation of AdaBoost.MH and MP-Boost using Apache Spark
- [StreamDM](http://huawei-noah.github.io/streamDM/) - Data Mining for Spark Streaming

### Algorithms

- **Adaboost**: [SparkBoost](https://github.com/tizfa/sparkboost)
- **Bagging**: [StreamDM](http://huawei-noah.github.io/streamDM/)

## Classification

### Libraries

- [MLlib](http://spark.apache.org/docs/latest/ml-guide.html) - Apache Spark's built in machine learning library
- [DissolveStruct](http://dalab.github.io/dissolve-struct/) - Distributed Solver for Structured Prediction
- [Spark kNN graphs](https://github.com/tdebatty/spark-knn-graphs) - Spark algorithms for building k-nn graphs
- [Spark-libFM](https://github.com/zhengruifeng/spark-libFM) - implementation of Factorization Machines
- [Sparkling Ferns](https://github.com/CeON/sparkling-ferns) - Implementation of Random Ferns for Apache Spark
- [StreamDM](http://huawei-noah.github.io/streamDM/) - Data Mining for Spark Streaming

### Algorithms

- **Decision Tree**: [MLlib](http://spark.apache.org/docs/latest/ml-guide.html)
- **Factorization Machines**: [spark-FM-parallelISGD](https://github.com/blebreton/spark-FM-parallelSGD), [Spark-libFM](https://github.com/zhengruifeng/spark-libFM)
- **Hoeffding Decision Trees**: [StreamDM](http://huawei-noah.github.io/streamDM/)
- **Gradient-boosted trees**: [MLlib](http://spark.apache.org/docs/latest/ml-guide.html)
- **Linear Discriminant Analysis (LDA)**:
- **Logistic Regression**: [MLlib](http://spark.apache.org/docs/latest/ml-guide.html), [StreamDM](http://huawei-noah.github.io/streamDM/)
- **Multilayer Perceptron**: [MLlib](http://spark.apache.org/docs/latest/ml-guide.html)
- **Naive Bayes**: [MLlib](http://spark.apache.org/docs/latest/ml-guide.html), [StreamDM](http://huawei-noah.github.io/streamDM/)
- **Perceptron**: [StreamDM](http://huawei-noah.github.io/streamDM/)
- **Random Forest**: [MLlib](http://spark.apache.org/docs/latest/ml-guide.html)
- **Support Vector Machine (SVM)**: [MLlib](http://spark.apache.org/docs/latest/ml-guide.html), [StreamDM](http://huawei-noah.github.io/streamDM/)


## Clustering

### Libraries

- [MLlib](http://spark.apache.org/docs/latest/ml-guide.html) - Apache Spark's built in machine learning library
- [Bisecting K-means](https://github.com/yu-iskw/bisecting-kmeans) - implementation of Bisecting KMeans Clustering which is a kind of Hierarchical Clustering algorithm
- [Generalized K-means clustering](https://github.com/derrickburns/generalized-kmeans-clustering) - generalizes the Spark MLLIB Batch and Streaming K-Means clusterers in every practical way
- [Patchwork](https://github.com/crim-ca/patchwork) - Highly Scalable Grid-Density Clustering Algorithm for Spark MLLib
- [spark-tsne](https://github.com/saurfang/spark-tsne) - Distributed t-SNE via Apache Spark
- [StreamDM](http://huawei-noah.github.io/streamDM/) - Data Mining for Spark Streaming

### Algorithms

- **CluStream**: [StreamDM](http://huawei-noah.github.io/streamDM/)
- **Bisecting K-means**: [MLlib](http://spark.apache.org/docs/latest/ml-guide.html), [Bisecting K-means](https://github.com/yu-iskw/bisecting-kmeans)
- **Gaussian Mixture Model (GMM)**: [MLlib](http://spark.apache.org/docs/latest/ml-guide.html)
- **Hierarchical clustering**: [MLlib](http://spark.apache.org/docs/latest/ml-guide.html), [Bisecting K-means](https://github.com/yu-iskw/bisecting-kmeans)
- **K-Means**: [MLlib](http://spark.apache.org/docs/latest/ml-guide.html), [Bisecting K-means](https://github.com/yu-iskw/bisecting-kmeans), [Generalized K-means clustering](https://github.com/derrickburns/generalized-kmeans-clustering)
- **Latent Dirichlet Allocation (LDA)**: [MLlib](http://spark.apache.org/docs/latest/ml-guide.html) 
- **Power Iteration Clustering (PIC)**: [MLlib](http://spark.apache.org/docs/latest/ml-guide.html)
- **StreamKM++**: [StreamDM](http://huawei-noah.github.io/streamDM/)
- **t-SNE**: [spark-tsne](https://github.com/saurfang/spark-tsne)


## Deep Learning

### Libraries

- [CaffeOnSpark](https://github.com/yahoo/CaffeOnSpark) - CaffeOnSpark brings deep learning to Hadoop and Spark clusters
- [Deeplearning4j](https://deeplearning4j.org/spark.html) - Deeplearning4j on Spark
- [DeepSpark](https://github.com/nearbydelta/deepspark) - A neural network library which uses Spark RDD instances
- [Elephas](http://maxpumperla.github.io/elephas/) - Distributed Deep learning with Keras & Spark
- [Sparkling Water](http://www.h2o.ai/sparkling-water/) - H2O + Apache Spark

## Data Transformation, Feature Selection & Dimensionality Reduction

### Libraries

- [MLlib](http://spark.apache.org/docs/latest/ml-guide.html) - Apache Spark's built in machine learning library
- [Modelmatrix](https://github.com/collectivemedia/modelmatrix) - Sparse feature extraction with Spark
- [Spark Infotheoretic Feature Selection](https://github.com/sramirez/spark-infotheoretic-feature-selection) - generic implementation of greedy Information Theoretic Feature Selection (FS) methods
- [Spark MLDP discetization](https://github.com/sramirez/spark-MDLP-discretization) -  implementation of Fayyad's discretizer based on Minimum Description Length Principle (MDLP)
- [spark-tsne](https://github.com/saurfang/spark-tsne) - Distributed t-SNE via Apache Spark

### Algorithms

- **Chi-Squared feature selection**: [MLlib](http://spark.apache.org/docs/latest/ml-guide.html)
- **Information theoretic**: [Spark Infotheoretic Feature Selection](https://github.com/sramirez/spark-infotheoretic-feature-selection)
- **PCA**: [MLlib](http://spark.apache.org/docs/latest/ml-guide.html)
- **MLDP discretization**: [Spark MLDP discetization](https://github.com/sramirez/spark-MDLP-discretization)
- **TF-IDF**: [MLlib](http://spark.apache.org/docs/latest/ml-guide.html)
- **t-SNE**: [spark-tsne](https://github.com/saurfang/spark-tsne)
- **Word2Vec**: [MLlib](http://spark.apache.org/docs/latest/ml-guide.html)

## Graph computations

### Libraries

- [GraphX](http://spark.apache.org/graphx/) - Apache Spark's API for graphs and graph-parallel computation

- [Spark kNN graphs](https://github.com/tdebatty/spark-knn-graphs) - Spark algorithms for building k-nn graphs
- [SparklingGraph](https://sparkling-graph.github.io/) - large scale, distributed graph processing made easy


## Itemset mining, frequent pattern mining & association rules

- **FP-Growth**: [MLlib](http://spark.apache.org/docs/latest/ml-guide.html)
- **PrefixSpan**: [MLlib](http://spark.apache.org/docs/latest/ml-guide.html)

## Linear algebra

### Libraries

- [lazy-linalg](https://github.com/brkyvz/lazy-linalg) - A package full of linear algebra operators for Apache Spark MLlib's linalg package
- [ml-matrix](https://github.com/amplab/ml-matrix) - distributed matrix library

### Algorithms

- **Singular Value Decomposition (SVD)**:
- **Principal Component Analysis (PCA)**: [MLlib](http://spark.apache.org/docs/latest/ml-guide.html)

## Matrix factorization & recommender systems

### Libraries

- [MLlib](http://spark.apache.org/docs/latest/ml-guide.html) - Apache Spark's built in machine learning library

- [spark-FM-parallelISGD](https://github.com/blebreton/spark-FM-parallelSGD) - Implementation of Factorization Machines on Spark using parallel stochastic gradient descent
- [Spark-libFM](https://github.com/zhengruifeng/spark-libFM) - implementation of Factorization Machines
- [Streaming Matrix Factorization](https://github.com/brkyvz/streaming-matrix-factorization) - Distributed Streaming Matrix Factorization implemented on Spark for Recommendation Systems

### Algorithms

- **Collaborative filtering**: [MLlib](http://spark.apache.org/docs/latest/ml-guide.html)
- **Factorization Machines**: [spark-FM-parallelISGD](https://github.com/blebreton/spark-FM-parallelSGD), [Spark-libFM](https://github.com/zhengruifeng/spark-libFM)
- **Matrix factorization**: [Streaming Matrix Factorization](https://github.com/brkyvz/streaming-matrix-factorization)

## Natural language processing

### Libraries

- [Spark CoreNLP](https://github.com/databricks/spark-corenlp) - CoreNLP wrapper for Spark
- [Spectral LDA on Spark](https://github.com/FurongHuang/spectrallda-tensorspark) -  implements a spectral (third order tensor decomposition) learning method for learning LDA topic model on Spark
- [TopicModelling](https://github.com/intel-analytics/TopicModeling) - Topic Modeling on Apache Spark

### Algorithms

- **Coreference resolution**: [Spark CoreNLP](https://github.com/databricks/spark-corenlp)
- **Latent Dirichlet Allocation (LDA)**: [MLlib](http://spark.apache.org/docs/latest/ml-guide.html) 
- **Named Entity Recognition (NER)**: [Spark CoreNLP](https://github.com/databricks/spark-corenlp)
- **Open information extraction**: [Spark CoreNLP](https://github.com/databricks/spark-corenlp)
- **Part-of-speech (POS) tagging**: [Spark CoreNLP](https://github.com/databricks/spark-corenlp)
- **Sentiment analysis**: [Spark CoreNLP](https://github.com/databricks/spark-corenlp)
- **Topic Modelling**: [Spectral LDA on Spark](https://github.com/FurongHuang/spectrallda-tensorspark), [TopicModelling](https://github.com/intel-analytics/TopicModeling)

## Optimization & hyperparameter search

### Libraries

- [MLlib](http://spark.apache.org/docs/latest/ml-guide.html) - Apache Spark's built in machine learning library

- [Elephas](http://maxpumperla.github.io/elephas/) - Distributed Deep learning with Keras & Spark
- [Spark-TFOCS](https://github.com/databricks/spark-tfocs) - port of TFOCS: Templates for First-Order Conic Solvers (cvxr.com/tfocs)

### Algorithms

- **Alternating Least Squares (ALS)**: [MLlib](http://spark.apache.org/docs/latest/ml-guide.html)
- **First-Order Conic solvers**: [Spark-TFOCS](https://github.com/databricks/spark-tfocs)
- **Gradient descent**: [MLlib](http://spark.apache.org/docs/latest/ml-guide.html)
- **Grid Search**: [MLlib](http://spark.apache.org/docs/latest/ml-guide.html)
- **Iteratively Reweighted Least Squares (IRLS)**: [MLlib](http://spark.apache.org/docs/latest/ml-guide.html)
- **Limited-memory BFGS (L-BFGS)**: [MLlib](http://spark.apache.org/docs/latest/ml-guide.html)
- **Normal equation solver**: [MLlib](http://spark.apache.org/docs/latest/ml-guide.html)
- **Stochastic gradient descent (SGD)**: [MLlib](http://spark.apache.org/docs/latest/ml-guide.html)
- **Tree of Parzen estimators (TPE -- hyperopt)**: [Elephas](http://maxpumperla.github.io/elephas/) - Distributed Deep learning with Keras & Spark

## Regression

### Libraries

- [MLlib](http://spark.apache.org/docs/latest/ml-guide.html) - Apache Spark's built in machine learning library
- [revrand](https://github.com/NICTA/revrand) - A library of scalable Bayesian generalised linear models with fancy features
- [StreamDM](http://huawei-noah.github.io/streamDM/) - Data Mining for Spark Streaming

### Algorithms

- **Bayesian generalised linear models**: [revrand](https://github.com/NICTA/revrand)
- **Decision tree regression**: [MLlib](http://spark.apache.org/docs/latest/ml-guide.html)
- **Generalized linear regression**: [MLlib](http://spark.apache.org/docs/latest/ml-guide.html)
- **Gradient-boosted tree regression**: [MLlib](http://spark.apache.org/docs/latest/ml-guide.html)
- **Isotonic regression**: [MLlib](http://spark.apache.org/docs/latest/ml-guide.html)
- **Linear regression**: [MLlib](http://spark.apache.org/docs/latest/ml-guide.html), [StreamDM](http://huawei-noah.github.io/streamDM/)
- **Linear least squares**: [MLlib](http://spark.apache.org/docs/latest/ml-guide.html)
- **Random forest regression**: [MLlib](http://spark.apache.org/docs/latest/ml-guide.html)
- **Ridge regression**: [MLlib](http://spark.apache.org/docs/latest/ml-guide.html)
- **Survival regression**: [MLlib](http://spark.apache.org/docs/latest/ml-guide.html)
- **Support Vector Machine (SVM)**: [MLlib](http://spark.apache.org/docs/latest/ml-guide.html)

## Statistics

- **Hypothesis testing**: [MLlib](http://spark.apache.org/docs/latest/ml-guide.html)
- **Kernel density estimation**: [MLlib](http://spark.apache.org/docs/latest/ml-guide.html)

## Tensor decompositions

### Libraries

- [Spectral LDA on Spark](https://github.com/FurongHuang/spectrallda-tensorspark) -  implements a spectral (third order tensor decomposition) learning method for learning LDA topic model on Spark

### Algorithms

- **Spectral LDA**: [Spectral LDA on Spark](https://github.com/FurongHuang/spectrallda-tensorspark)


## Time series

### Libraries

- [spark-ts](https://github.com/sryza/spark-timeseries) - Time series for Spark
- [Thunder](http://thunder-project.org/) - scalable image and time series analysis

### Algorithms


<!-- END OF MAIN CONTENT -->

# Practical info

## License

[![CC0](http://i.creativecommons.org/p/zero/1.0/88x31.png)](http://creativecommons.org/publicdomain/zero/1.0/)

## Contributing

Please, read the [Contribution Guidelines](https://github.com/claesenm/spark-ml-inventory/blob/master/CONTRIBUTING.md) before submitting your suggestion.

To add content, feel free to [open an issue](https://github.com/claesenm/spark-ml-inventory/issues) or [create a pull request](https://github.com/claesenm/spark-ml-inventory/pulls).

## Acknowledgments

This inventory is inspired by [mfornos’ inventory of awesome microservices](https://github.com/mfornos/awesome-microservices).

Table of contents generated with [DocToc](https://github.com/thlorenz/doctoc).

