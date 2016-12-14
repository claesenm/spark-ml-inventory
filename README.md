# Spark machine learning inventory [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

A curated inventory of machine learning methods available on the Apache Spark platform, both in official and third party libraries.

This inventory is inspired by [mfornos’ inventory of awesome microservices](https://github.com/mfornos/awesome-microservices).

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
- [Algorithm inventory](#algorithm-inventory)
  - [Classification](#classification)
  - [Clustering](#clustering)
  - [Deep learning](#deep-learning)
  - [Graph computations](#graph-computations)
  - [Matrix factorization](#matrix-factorization)
  - [Natural language processing](#natural-language-processing)
  - [Regression](#regression)
  - [Statistics](#statistics)
  - [Tensor decompositions](#tensor-decompositions)
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
- [CoCoA](https://github.com/gingsmith/cocoa) - communication-efficient distributed coordinate ascent
- [Deeplearning4j](https://deeplearning4j.org/spark.html) - Deeplearning4j on Spark
- [DissolveStruct](http://dalab.github.io/dissolve-struct/) - Distributed Solver for Structured Prediction
- [Elephas](http://maxpumperla.github.io/elephas/) - Distributed Deep learning with Keras & Spark
- [Generalized K-means clustering](https://github.com/derrickburns/generalized-kmeans-clustering) - generalizes the Spark MLLIB Batch and Streaming K-Means clusterers in every practical way
- [KeystoneML](http://keystone-ml.org/) - KeystoneML is a software framework, written in Scala, from the UC Berkeley AMPLab designed to simplify the construction of large scale, end-to-end, machine learning pipelines with Apache Spark
- [MLbase](http://www.mlbase.org/) - MLbase is a platform addressing implementing and consuming Machine Learning at scale
- [ml-matrix](https://github.com/amplab/ml-matrix) - distributed matrix library
- [Sparkling Water](http://www.h2o.ai/sparkling-water/) - H2O + Apache Spark
- [Splash](http://zhangyuc.github.io/splash/) - a general framework for parallelizing stochastic learning algorithms on multi-node clusters
- [Spectral LDA on Spark](https://github.com/FurongHuang/spectrallda-tensorspark) -  implements a spectral (third order tensor decomposition) learning method for learning LDA topic model on Spark
- [Thunder](http://thunder-project.org/) - scalable image and time series analysis
- [Zen](https://github.com/cloudml/zen) - aims to provide the largest scale and the most efficient machine learning platform on top of Spark, including but not limited to logistic regression, latent dirichilet allocation, factorization machines and DNN

### Interfaces

- [CaffeOnSpark](https://github.com/yahoo/CaffeOnSpark) - CaffeOnSpark brings deep learning to Hadoop and Spark clusters
- [Elephas](http://maxpumperla.github.io/elephas/) - Distributed Deep learning with Keras & Spark
- [Sparkling Water](http://www.h2o.ai/sparkling-water/) - H2O + Apache Spark
- [Spark CoreNLP](https://github.com/databricks/spark-corenlp) - CoreNLP wrapper for Spark
- [sparkplyr](http://spark.rstudio.com/mllib.html) - sparklyr provides R bindings to Spark’s distributed machine learning library
- [Sparkit-learn](https://github.com/lensacom/sparkit-learn) - PySpark + Scikit-learn = Sparkit-learn
- [Spark-TFOCS](https://github.com/databricks/spark-tfocs) - port of TFOCS: Templates for First-Order Conic Solvers (cvxr.com/tfocs)
- [Hivemall-Spark](https://github.com/maropu/hivemall-spark) - A Hivemall wrapper for Spark
- [Spark PMML exporter validator](https://github.com/selvinsource/spark-pmml-exporter-validator) - Using JPMML Evaluator to validate the PMML models exported from Spark

## Notebooks

- [Beaker](http://beakernotebook.com/) - The data scientist's laboratory
- [Spark Notebook](http://spark-notebook.io/) - Interactive and Reactive Data Science using Scala and Spark
- [Apache Zeppelin](https://zeppelin.apache.org/) - A web-based notebook that enables interactive data analytics

## Visualization

- [Spark ML streaming](https://github.com/freeman-lab/spark-ml-streaming) - Visualize streaming machine learning in Spark
- [Vegas](https://github.com/vegas-viz/Vegas) - The missing MatPlotLib for Scala + Spark

## Others

- [Apache Toree](https://toree.apache.org/) - Gateway to Apache Spark
- [Distributed DataFrame](http://ddf.io/) - Simplify Analytics on Disparate Data Sources via a Uniform API Across Engines
- [PipelineIO](http://pipeline.io/) - Extend ML Pipelines to Serve Production Users
- [Spark Jobserver](https://github.com/spark-jobserver/spark-jobserver) - REST job server for Apache Spark
- [Spark PMML exporter validator](https://github.com/selvinsource/spark-pmml-exporter-validator) - Using JPMML Evaluator to validate the PMML models exported from Spark
- [Twitter stream ML](https://github.com/giorgioinf/twitter-stream-mlhttps://github.com/giorgioinf/twitter-stream-ml) - Machine Learning over Twitter's stream. Using Apache Spark, Web Server and Lightning Graph server.
- [Velox](https://github.com/amplab/velox-modelserver) - a system for serving machine learning predictions


<!-- ALGORITHM INVENTORY -->


# Task inventory

- [MLlib](http://spark.apache.org/docs/latest/ml-guide.html) - Apache Spark's built in machine learning library

## Ensemble learning

### Libraries

- [SparkBoost](https://github.com/tizfa/sparkboost) - A distributed implementation of AdaBoost.MH and MP-Boost using Apache Spark

## Classification

### Libraries

- [DissolveStruct](http://dalab.github.io/dissolve-struct/) - Distributed Solver for Structured Prediction
- [Spark kNN graphs](https://github.com/tdebatty/spark-knn-graphs) - Spark algorithms for building k-nn graphs
- [Spark-libFM](https://github.com/zhengruifeng/spark-libFM) - implementation of Factorization Machines
- [Sparkling Ferns](https://github.com/CeON/sparkling-ferns) - Implementation of Random Ferns for Apache Spark

## Clustering

### Libraries

- [Bisecting K-means](https://github.com/yu-iskw/bisecting-kmeans) - implementation of Bisecting KMeans Clustering which is a kind of Hierarchical Clustering algorithm
- [Generalized K-means clustering](https://github.com/derrickburns/generalized-kmeans-clustering) - generalizes the Spark MLLIB Batch and Streaming K-Means clusterers in every practical way
- [Patchwork](https://github.com/crim-ca/patchwork) - Highly Scalable Grid-Density Clustering Algorithm for Spark MLLib
- [Spark TSNE](https://github.com/saurfang/spark-tsne) - Distributed t-SNE via Apache Spark

## Deep learning

### Libraries

- [CaffeOnSpark](https://github.com/yahoo/CaffeOnSpark) - CaffeOnSpark brings deep learning to Hadoop and Spark clusters
- [Deeplearning4j](https://deeplearning4j.org/spark.html) - Deeplearning4j on Spark
- [DeepSpark](https://github.com/nearbydelta/deepspark) - A neural network library which uses Spark RDD instances
- [Elephas](http://maxpumperla.github.io/elephas/) - Distributed Deep learning with Keras & Spark
- [Sparkling Water](http://www.h2o.ai/sparkling-water/) - H2O + Apache Spark

## Feature selection & dimensionality reduction

### Libraries

- [Modelmatrix](https://github.com/collectivemedia/modelmatrix) - Sparse feature extraction with Spark
- [Spark Infotheoretic Feature Selection](https://github.com/sramirez/spark-infotheoretic-feature-selection) - generic implementation of greedy Information Theoretic Feature Selection (FS) methods
- [Spark MLDP discetization](https://github.com/sramirez/spark-MDLP-discretization) -  implementation of Fayyad's discretizer based on Minimum Description Length Principle (MDLP)
- [Spark TSNE](https://github.com/saurfang/spark-tsne) - Distributed t-SNE via Apache Spark

## Graph computations

### Libraries

- [GraphX](http://spark.apache.org/graphx/) - Apache Spark's API for graphs and graph-parallel computation

- [Spark kNN graphs](https://github.com/tdebatty/spark-knn-graphs) - Spark algorithms for building k-nn graphs

## Linear algebra

### Libraries

- [lazy-linalg](https://github.com/brkyvz/lazy-linalg) - A package full of linear algebra operators for Apache Spark MLlib's linalg package
- [ml-matrix](https://github.com/amplab/ml-matrix) - distributed matrix library

## Matrix factorization & recommender systems

### Libraries

- [Spark-libFM](https://github.com/zhengruifeng/spark-libFM) - implementation of Factorization Machines
- [Streaming Matrix Factorization](https://github.com/brkyvz/streaming-matrix-factorization) - Distributed Streaming Matrix Factorization implemented on Spark for Recommendation Systems

## Natural language processing

### Libraries

- [Spark CoreNLP](https://github.com/databricks/spark-corenlp) - CoreNLP wrapper for Spark
- [TopicModelling](https://github.com/intel-analytics/TopicModeling) - Topic Modeling on Apache Spark

### Algorithms

- **Coreference resolution**: [Spark CoreNLP](https://github.com/databricks/spark-corenlp)
- **Named Entity Recognition (NER)**: [Spark CoreNLP](https://github.com/databricks/spark-corenlp)
- **Open information extraction**: [Spark CoreNLP](https://github.com/databricks/spark-corenlp)
- **Part-of-speech (POS) tagging**: [Spark CoreNLP](https://github.com/databricks/spark-corenlp)
- **Sentiment analysis**: [Spark CoreNLP](https://github.com/databricks/spark-corenlp)
- **Topic Modelling**: [TopicModelling](https://github.com/intel-analytics/TopicModeling)

## Optimization

### Libraries

- [Spark-TFOCS](https://github.com/databricks/spark-tfocs) - port of TFOCS: Templates for First-Order Conic Solvers (cvxr.com/tfocs)

### Algorithms

- **First-Order Conic solvers**: [Spark-TFOCS](https://github.com/databricks/spark-tfocs)

## Regression

### Libraries

### Algorithms

## Statistics

### Libraries

### Algorithms

## Tensor decompositions

### Libraries

- [Spectral LDA on Spark](https://github.com/FurongHuang/spectrallda-tensorspark) -  implements a spectral (third order tensor decomposition) learning method for learning LDA topic model on Spark

### Algorithms

- **Spectral LDA**: [Spectral LDA on Spark](https://github.com/FurongHuang/spectrallda-tensorspark)

<!-- END OF MAIN CONTENT -->

# Practical info

## License

[![CC0](http://i.creativecommons.org/p/zero/1.0/88x31.png)](http://creativecommons.org/publicdomain/zero/1.0/)

## Contributing

Please, read the [Contribution Guidelines](https://github.com/claesenm/spark-ml-inventory/blob/master/CONTRIBUTING.md) before submitting your suggestion.

To add content, feel free to [open an issue](https://github.com/claesenm/spark-ml-inventory/issues) or [create a pull request](https://github.com/claesenm/spark-ml-inventory/pulls).

## Acknowledgments

Table of contents generated with [DocToc](https://github.com/thlorenz/doctoc)
