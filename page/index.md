<!-- =============================
     ABOUT
    ============================== -->
\begin{section}{title="About this Package", name="About"}
\lead{PatternVectors.jl is a Julia package containing some useful array representation for peculiar one dimensional arrays patterns.}
It currently contains the following pattern types:
- ZeroPattern: convenient representation for vectors of the form:
	`````
	[0,0,0,0,0,0...]
	`````
	
- FillPattern: convenient representation for vectors of the form:
	`````
	[a,a,a,a,a,a...]
	`````

- InitialValuePattern: convenient representation for vectors of the form:
	`````
	[a,b,b,b,b,b...]
	`````

- FinalValuePattern: convenient representation for vectors of the form:
	`````
	[a,a,a,a,a,a...,b]
	`````

- PaddedFillPattern: convenient representation for vectors of the form:
	`````
	[x,a,a,a,a,a...,y]
	`````

- EvenOddPattern: convenient representation for vectors of the form:
	`````
	[a,b,a,b,a,b...]
	`````
	
- PaddedEvenOddPattern: convenient representation for vectors of the form:
    `````
    [x,a,b,a,b,a,b...,y]
    `````

Defining a new pattern is easy and fully supported by the library.
The module is standalone.
\end{section}
\begin{section}{title="Getting Started", name="Usage"}
**_EvenOddPattern_**

To build a EvenOddPattern one needs to provide:
* the value for odd indices
* the value for even indices
The various values must be of the same type and the length must be greater than one.
The way to build an PatternVector with such pattern is the following:
```julia:example_build_pattern_vector
using PatternVectors
value_odd=0.2
value_even=2.3
pattern=EvenOddPattern(value_odd,value_even)
length_av=7
x_av=PatternVector(length_av,pattern)
```
**_PaddedEvenOddPattern_**

To build a PaddedEvenOddPattern one needs to provide: 
* the initial value
* the value for even indices
* the value for odd indices
* the final value
The various values must be of the same type and the length must be greater than three.
The way to build an PatternVector with such pattern is the following:
```julia:example_build_pattern_padded_vector
using PatternVectors
initial_value=0.2
value_odd=-0.2
value_even=2.3
final_value=1.3
length_av=7
x_av=PatternVector(length_av,PaddedEvenOddPattern(initial_value,value_even,value_odd,final_value))
```

**_Operation on Pattern Vectors_**

The following applies:
* PatternVector is closed under getindex for range of integers.
```julia:example_unary_operations_1
using PatternVectors
apv=PatternVector(70,PaddedEvenOddPattern(0.2,-2.0,4.0,2.3))
z_small=apv[1:7:50]
z_small
```
* Any scalar unary function applied directly to PatternVectors will produce an array of the same type.
```julia:example_unary_operations_2
using PatternVectors
pattern=EvenOddPattern(0.2,-2.0)
av=PatternVector(10,pattern)
z_sin=@. sin(av)
z_sin
```
**_Operation between Pattern Vectors_**

The following applies:
* Binary scalar functions between PatternVectors will produce a PatternVector.
* Binary scalar functions between PatternVector and **any** other type deriving from AbstractArray will produce an array of the other type.

```julia:example_binary_operations
using PatternVectors
x=PatternVector(10,EvenOddPattern(0.2,2.3))
y=randn(10)
z=PatternVector(10,PaddedEvenOddPattern(0.2,-2.0,4.0,2.3))
@. sin(x)*y+z
```
\end{section}
\begin{section}{title="Performances Comparison", name="Performances"}
Here the common usages of the package are tested.

**_Simple multiplication_**
```julia:performance_test_multiplication
using PatternVectors, BenchmarkTools
n=10_000
x=PatternVector(n,EvenOddPattern(0.2,2.3))
y=randn(n)
x_c=collect(x)
@btime @. $x*$y;
@btime @. $x_c*$y;
```

**_Flipping sign based on index and sum_**
```julia:performance_test_pattern_vector
using PatternVectors, BenchmarkTools, LinearAlgebra
n=10_000
function f_std_scalar(f_x)
	N=length(f_x)
	sum_z=zero(eltype(f_x))
	for i in 1:N
		w=ifelse(isodd(i),1,-1)
		sum_z+=@views @inbounds w*f_x[i]
	end
	return sum_z
end

function f_std_scalar_2(f_x)
	N=length(f_x)
	sum_z=zero(eltype(f_x))
	for i in 1:N
		if(isodd(i))
			sum_z+=@views @inbounds f_x[i]
		else
			sum_z-=@views @inbounds f_x[i]
		end
	end
	return sum_z
end

function f_std_vec(f_x)
	N=length(f_x)
	idx=1:N
	W=@. ifelse(isodd(idx),1,-1)
	return sum(W.*f_x)
end

function f_std_vec_linear_algebra(f_x)
	N=length(f_x)
	idx=1:N
	W=@. ifelse(isodd(idx),1,-1)
	return dot(W,f_x)
end

function f_apv(f_x)
	N=length(f_x)
	W=PatternVector(n,EvenOddPattern(1,-1))
	return sum(W.*f_x)
end
x=randn(n)
f_x=@. sin(x)+x*cos(x)
@btime f_std_scalar($f_x);
@btime f_std_scalar_2($f_x);
@btime f_std_vec($f_x);
@btime f_std_vec_linear_algebra($f_x);
@btime f_apv($f_x);
```
**_Simpson Integration_**
```julia:performance_test_pattern_padded_vector
using PatternVectors, BenchmarkTools,LinearAlgebra
n2=10_000
function f_simpson_std_scalar(f_x)
	N=length(f_x)
	sum_z=zero(eltype(f_x))
	for i in 1:N
		w=ifelse(i==1,1/3,ifelse(i==N,1/3,4/3))
		sum_z+=@views @inbounds w*f_x[i]
	end
	return sum_z
end

function f_simpson_std_vec(f_x)
	N=length(f_x)
	idx=1:N
	W=@. ifelse(idx==1,1/3,ifelse(idx==N,1/3,4/3))
	return sum(W.*f_x)
end

function f_simpson_std_vec_linear_algebra(f_x)
	N=length(f_x)
	idx=1:N
	W=@. ifelse(idx==1,1/3,ifelse(idx==N,1/3,4/3))
	return dot(W,f_x)
end

function f_simpson_apv(f_x)
	N=length(f_x)
	W=PatternVector(N,PaddedEvenOddPattern(1/3,4/3,4/3,1/3))
	return sum(W.*f_x)
end

function f_simpson_apv_linear_algebra(f_x)
	N=length(f_x)
	W=PatternVector(N,PaddedEvenOddPattern(1/3,4/3,4/3,1/3))
	return dot(W,f_x)
end

x2=randn(n2)
f_x2=@. sin(x2)+x2*cos(x2)
@btime f_simpson_std_scalar($f_x2);
@btime f_simpson_std_vec($f_x2);
@btime f_simpson_std_vec_linear_algebra($f_x2);
@btime f_simpson_apv($f_x2);
@btime f_simpson_apv_linear_algebra($f_x2);
```
To be noticed the performance improvements thanks to the usage of PatternVector, and to be noticed that the first function proposed is not compatible with the CUDA.jl stack.

\end{section}

\begin{section}{title="GPU Support", name="GPU"}
The main objective of this library is to write code without iterating on indices for both cpu and gpu support.
The main class is PatternVector which, unlike the standard arrays in julia, it is an immutable object, hence we can send it to gpu without issues,
and being in most of the cases (up to the client) a lightweight object, this sending operation is efficient.
\end{section}

\begin{section}{title="New Patterns Definition", name="New Patterns"}
In most of the cases the provided patterns suffice the applications. 
In case you need a new pattern, the following steps are needed:
- Define a container (namely MyNewPattern{T}) inheriting from AbstractPattern{T} where T is the data type you are going to store in the array.
- Define a constructor for it.
- Implement the interface:
  - pattern\_minimum\_size: define the minimum size for your type
  - getindex\_pattern: define the logic to extract scalars
  - getindex\_pattern_range: define the logic to extract ranges
  - materialize\_pattern: define the way to compute broadcasted functions with the new pattern


In case you need to mix your pattern with other existing patterns, you will have to specify we is the more general pattern able to store both:
- determine\_mixed\_pattern(::Type{T}, ::Type{V}) where {T <: MyNewPattern{M}, V <: MyOldPattern{N}} where {M, N} = MyWinningPattern{promote\_type(M, N)}

And you are ready to go!

In case you want to use your new pattern in AD applications you will have to provide two additional implementations:
- ChainRulesCore.rrule(::Type{MyNewPattern}, args...): the rrule for the constructor of your newly defined pattern.
- pattern\_to\_vector\_pullback: the rrule to convert from pattern to array.

For more details have a look at the already implemented types.

\end{section}