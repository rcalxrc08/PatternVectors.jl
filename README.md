# PatternVectors

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://rcalxrc08.github.io/PatternVectors.jl/)
[![Build Status](https://github.com/rcalxrc08/PatternVectors.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/rcalxrc08/PatternVectors.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![codecov](https://codecov.io/gh/rcalxrc08/PatternVectors.jl/graph/badge.svg?token=GqpkxPrlXg)](https://codecov.io/gh/rcalxrc08/PatternVectors.jl)

##### This is a Julia package containing some useful functionality to generate Arrays with specific patterns.
The current patterns allow you to build:
- zero values vectors: convenient representation for arrays of the form:
    ```Julia
    [0,0,0,0,0,0...]
    ```
- single values vectors: convenient representation for arrays of the form:
    ```Julia
    [a,a,a,a,a,a...]
    ```
- alternate values vectors: convenient representation for arrays of the form:
    ```Julia
    [a,b,a,b,a,b...]
    ```
- alternate padded values vectors: convenient representation for arrays of the form:
    ```Julia
    [x,a,b,a,b,a,b...,y]
    ```

The module is standalone.

## How to Install
To install the package simply type on the Julia REPL the following:
```Julia
Pkg.clone("https://github.com/rcalxrc08/PatternVectors.jl")
```
## How to Test
After the installation, to test the package type on the Julia REPL the following:
```julia
Pkg.test("PatternVectors")
```
## Example of Usage
```julia
#Import the Package
using PatternVectors
pattern_alt=EvenOddPattern(0.2,2.3)
x=PatternVector(10,pattern_alt)
y=randn(10)
pattern_altpad=PaddedEvenOddPattern(0.2,-2.0,4.0,2.3)
z=PatternVector(10,pattern_altpad)
@show @. sin(x)
@show @. sin(x)+exp(z)
@show @. sin(x)*y
@show @. sin(x)*y+z
```
