###Simple Machine Learning Pipeline

_About_
A simple class `Pipeline`, with helper functions to facilitate exploratory machine learning

_Why_

* Want a way to quickly explore __small__ datasets like those typically found in medical labs
* Prefer to use Julia instead of other languages
* Inspired by [scikit-learn's Pipeline class][7] and [Grid Seach][8]



__Warning: Not Optimized__

* This code is not a registered package
  * Install command:
  * Pkg.clone("https://github.com/JOfTheAncientGermanSpear/MLPipe.jl.git
")
* If you find bugs or have optimization suggestions, please feel free to contribute a fix or let us know.


##How To Use
View [Iris Example notebook](https://github.com/JOfTheAncientGermanSpear/MLPipe.jl/blob/master/Iris_Example.ipynb)

View [Parallel Example](https://github.com/JOfTheAncientGermanSpear/MLPipe.jl/blob/master/src/parallelExample.jl)


##Required Packages

* [DataFrames][1]
* [Distributions][2]
* [HypothesisTests][3]
* [Lazy][4]
* [Gadfly][5]
* [PyPlot][6]


[1]: https://github.com/JuliaStats/DataFrames.jl
[2]: https://github.com/JuliaStats/Distributions.jl
[3]: https://github.com/JuliaStats/HypothesisTests.jl
[4]: https://github.com/MikeInnes/Lazy.jl
[5]: https://github.com/dcjones/Gadfly.jl
[6]: http://github.com/stevengj/PyPlot.jl
[7]: http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
[8]: http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html
