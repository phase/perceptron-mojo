from Benchmark import Benchmark
from DType import DType
from Intrinsics import strided_load
from List import VariadicList
from Math import div_ceil, min
from Memory import memset_zero
from Object import object, Attr
from Pointer import DTypePointer
from Random import rand, random_f64
from TargetInfo import dtype_sizeof, dtype_simd_width

alias type = F32
alias dtype = DType.f32
#@register_passable
struct Matrix:
    var data: DTypePointer[dtype]
    var rows: Int
    var cols: Int
    
    # non-@register_passable version
    fn __init__(self&, rows: Int, cols: Int):
        self.data = DTypePointer[dtype].alloc(rows * cols)
        memset_zero(self.data, rows * cols)
        self.rows = rows
        self.cols = cols
     
    fn __moveinit__(self&, owned existing: Self):
        self.data = existing.data
        self.rows = existing.rows
        self.cols = existing.cols
    
    # @register_passable version
    #fn __init__(rows: Int, cols: Int) -> Self:
    #    let data = DTypePointer[dtype].alloc(rows * cols)
    #    memset_zero(data, rows * cols)
    #    return Self {
    #        data: data,
    #        rows: rows,
    #        cols: cols
    #    }
    
    # @register_passable version
    #fn __copyinit__(existing: Self) -> Self:
    #    let data = DTypePointer[dtype].alloc(existing.rows * existing.cols)
    #    let self = Self {
    #        data: data,
    #        rows: existing.rows,
    #        cols: existing.cols
    #    }
    #    for y in range(self.rows):
    #        for x in range(self.cols):
    #            self.store[1](y, x, existing[y,x])
    #    return self
    
    # non-@register_passable version
    fn __copyinit__(self&, existing: Self):
        self.rows = existing.rows
        self.cols = existing.cols
        self.data = DTypePointer[dtype].alloc(self.rows * self.cols)
        for y in range(self.rows):
            for x in range(self.cols):
                self.store[1](y, x, existing[y,x])
    
    fn __del__(owned self):
        self.data.free()
    
    @always_inline
    fn length(self&) -> Int:
        return self.rows * self.cols
    
    fn zero(self&):
        memset_zero(self.data, self.length())
    
    fn randomize(self&):
        rand(self.data, self.length())
    
    fn reshape(self&, rows: Int, cols: Int):
        self.rows = rows
        self.cols = cols
    
    @always_inline
    fn __getitem__(self, y: Int, x: Int) -> type:
        return self.load[1](y, x)
    
    @always_inline
    fn load[nelts: Int](self, y: Int, x: Int) -> SIMD[dtype, nelts]:
        return self.data.simd_load[nelts](y * self.cols + x)
    
    @always_inline
    fn load_tr[nelts: Int](self, y: Int, x: Int) -> SIMD[dtype, nelts]:
        return strided_load[nelts, dtype](self.data + y * self.cols + x, self.cols)
    
    @always_inline
    fn __setitem__(self, y: Int, x: Int, val: type):
        return self.store[1](y, x, val)
    
    @always_inline
    fn store[nelts: Int](self, y: Int, x: Int, val: SIMD[dtype, nelts]):
        self.data.simd_store[nelts](y * self.cols + x, val)


@always_inline
def benchmark[func: fn(Matrix, Matrix, Matrix) -> None]
    (M : Int, N : Int, K : Int):
    var C = Matrix(M, N)
    var A = Matrix(M, K)
    A.randomize()
    var B = Matrix(M, N)
    B.randomize()
    
    @always_inline
    @parameter
    fn test_fn():
        _ = func(C, A, B)
    
    let secs = F64(Benchmark().run[test_fn]()) / 1_000_000_000

    # HACK: prevents matrices from being destroyed
    _ = A.data
    _ = B.data
    _ = C.data
    
    let gflops = ((2*M*N*K)/secs) / 1e9
    print(gflops, "GFLOP/s")

# C = A * B
fn matmul_naive(C: Matrix, A: Matrix, B: Matrix):
    for m in range(C.rows):
        for n in range(C.cols):
            for k in range(A.cols):
                C[m, n] += A[m, k] * B[k, n]

# SIMD vector width
alias nelts = dtype_simd_width[dtype]()
fn matmul_vectorized_0(C: Matrix, A: Matrix, B: Matrix):
    for m in range(C.rows):
        for n in range(C.cols):
            for kv in range(0, A.cols, nelts):
                C[m, n] += (A.load[nelts](m, kv) * B.load_tr[nelts](kv, n)).reduce_add()
            for k in range(nelts*(A.cols//nelts), A.cols):
                C[m, n] += A[m, k] * B[k, n]

from Functional import vectorize
fn matmul_vectorized_1(C: Matrix, A: Matrix, B: Matrix):
    for m in range(C.rows):
        for n in range(C.cols):
            @parameter
            fn dot[nelts: Int](k: Int):
                C[m, n] += (A.load[nelts](m, k) * B.load_tr[nelts](k, n)).reduce_add()
            vectorize[nelts, dot](A.cols)

from Functional import parallelize
fn matmul_parallelized(C: Matrix, A: Matrix, B: Matrix):
    @parameter
    fn calc_row(m: Int):
        for n in range(C.cols):
            @parameter
            fn dots[nelts: Int](k: Int):
                C[m, n] += (A.load[nelts](m, k) * B.load_tr[nelts](k, n)).reduce_add()
            vectorize[nelts, dots](A.cols)
    parallelize[calc_row](C.rows)


from Functional import Static2DTileUnitFunc as Tile2DFunc
fn tile[f: Tile2DFunc, tile_x: Int, tile_y: Int](end_x: Int, end_y: Int):
    for y in range(0, end_y, tile_y):
        for x in range(0, end_x, tile_x):
            f[tile_x, tile_y](x, y)

fn matmul_tiled_parallelized(C: Matrix, A: Matrix, B: Matrix):
    @parameter
    fn calc_row(m: Int):
        @parameter
        fn calc_tile[tile_x: Int, tile_y: Int](x: Int, y: Int):
            for n in range(y, y + tile_y):
                @parameter
                fn dot[nelts: Int](k: Int):
                    C[m, n] += (A.load[nelts](m, k + x) * B.load_tr[nelts](k + x, n)).reduce_add()
                vectorize[nelts, dot](tile_x)
            
        alias tile_size = 4
        tile[calc_tile, nelts * tile_size, nelts * tile_size](C.cols, A.cols)
    parallelize[calc_row](C.rows)

from Functional import vectorize_unroll
fn matmul_tiled_unrolled_parallelized(C: Matrix, A: Matrix, B: Matrix):
    @parameter
    fn calc_row(m: Int):
        @parameter
        fn calc_tile[tile_x: Int, tile_y: Int](x: Int, y: Int):
            for n in range(y, y + tile_y):
                @parameter
                fn dot[nelts: Int](k: Int):
                    C[m, n] += (A.load[nelts](m, k + x) * B.load_tr[nelts](k + x, n)).reduce_add()
                vectorize_unroll[nelts, tile_x//nelts, dot](tile_x)
                
        alias tile_size = 4
        tile[calc_tile, nelts * tile_size, nelts * tile_size](A.cols, C.cols)
    parallelize[calc_row](C.rows)


from Autotune import autotune, search
from Time import now
from Pointer import Pointer

alias matmul_fn = fn(Matrix, Matrix, Matrix) -> None

# The optimal tile factor is highly hardware dependent.
# Mojo can automatically select the best tile factor using "autotuning".
@adaptive
fn matmul_autotune_impl(C: Matrix, A: Matrix, B: Matrix):
    @parameter
    fn calc_row(m: Int):
        @parameter
        fn calc_tile[tile_x: Int, tile_y: Int](x: Int, y: Int):
            for n in range(y, y + tile_y):
                @parameter
                fn dot[nelts: Int](k: Int):
                    C[m,n] += (A.load[nelts](m,k+x) * B.load_tr[nelts](k+x,y)).reduce_add()
                vectorize_unroll[nelts, tile_x // nelts, dot](tile_x)

        # use autotune instead of hardcoding 4        
        alias tile_size = autotune(1, 2, 4, 8, 16, 32, 64)
        tile[calc_tile, nelts * tile_size, nelts * tile_size](A.cols, C.cols)

    parallelize[calc_row](C.rows)
        
# ^ That generates multiple "candidates" for the matmul function.
# Mojo needs an evaluator function to asses each candidate.
fn matmul_evaluator(funcs: Pointer[matmul_fn], size: Int) -> Int:
    print("matmul_evaluator, number of candidates: ", size)
    let eval_begin: Int = now()
    
    let M = 512
    let N = 512
    let K = 512
    print("Optimizing for size:", M, "x", N, "x", K)
    
    var best_idx: Int = -1
    var best_time: Int = -1
    
    var C = Matrix(M, N)
    var A = Matrix(M, K)
    var B = Matrix(K, N)
    let Cptr = Pointer[Matrix].address_of(C).address
    let Aptr = Pointer[Matrix].address_of(A).address
    let Bptr = Pointer[Matrix].address_of(B).address
    
    for f_idx in range(size):
        let func = funcs.load(f_idx)
        
        @always_inline
        @parameter
        fn wrapper():
            func(C, A, B)
        
        let cur_time = Benchmark(1, 100_000, 500_000_000, 1000_000_000).run[wrapper]()
        if best_idx < 0 or best_time > cur_time:
            best_idx = f_idx
            best_time = cur_time
    
    let eval_end: Int = now()
    _ = A.data
    _ = B.data
    _ = C.data
    print("Time spent in matmul_evaluator, ms:", (eval_end - eval_begin) // 1000000)
    print("Best candidate idx:", best_idx)
    return best_idx

fn matmul_autotune(C: Matrix, A: Matrix, B: Matrix):
    alias best_impl: matmul_fn
    search[
        matmul_fn,
        VariadicList(matmul_autotune_impl.__adaptive_set),
        matmul_evaluator -> best_impl
    ]()
    return best_impl(C, A, B)

fn transpose_naive(mat: Matrix) -> Matrix:
    let new_mat = Matrix(mat.cols, mat.rows)
    for m in range(mat.rows):
        for n in range(mat.cols):
            new_mat[n, m] = mat[m, n]
    return new_mat

def test_transpose():
    print("testing transpose")
    var mat = Matrix(3, 4)
    mat.randomize()
    
    print("current mat:")
    for m in range(mat.rows):
        for n in range(mat.cols):
            print("[",m,",",n,"]: ", mat[m, n])
    let new_mat = transpose_naive(mat)
    
    print("new mat:")
    for m in range(new_mat.rows):
        for n in range(new_mat.cols):
            print("[",m,",",n,"]: ", new_mat[m, n])

#test_transpose()

# TODO look at the matrix mult impls to speed this up with vectorizing / tiling
alias elementwise_func = fn(F32) -> F32

fn elementwise_map[func: elementwise_func](mat: Matrix) -> Matrix:
    let new_mat = Matrix(mat.rows, mat.cols)
    for m in range(mat.rows):
        for n in range(mat.cols):
            new_mat[m, n] = func(mat[m, n])
    return new_mat

alias elementwise_merge_func = fn(F32, F32) -> F32
fn elementwise_merge[func: elementwise_merge_func](a: Matrix, b: Matrix) -> Matrix:
    let new_mat = Matrix(a.rows, a.cols)
    for m in range(a.rows):
        for n in range(a.cols):
            new_mat[m, n] = func(a[m, n], b[m, n])
    return new_mat

@always_inline
fn times2(a: F32) -> F32:
    return a * 2

@always_inline
fn negate(a: F32) -> F32:
    return -a

@always_inline
fn add1(a: F32) -> F32:
    return a + 1

fn div(a: F32, b: F32) -> F32:
    return a / b

fn sub(a: F32, b: F32) -> F32:
    return a - b

@always_inline
fn square(a: F32) -> F32:
    return a * a

from Math import exp, log
alias matrix_exp = elementwise_map[exp[1, DType.f32]]
alias matrix_log = elementwise_map[log[1, DType.f32]]
alias matrix_neg = elementwise_map[negate]
alias matrix_add1 = elementwise_map[add1]
alias matrix_div = elementwise_merge[div]
alias matrix_sub = elementwise_merge[sub]
alias matrix_square = elementwise_map[square]

# todo use generic impl above
fn matrix_scale(s: F32, mat: Matrix) -> Matrix:
    let new_mat = Matrix(mat.rows, mat.cols)
    for m in range(mat.rows):
        for n in range(mat.cols):
            new_mat[m, n] = s * mat[m, n]
    return new_mat

fn leaky_relu(a: F32) -> F32:
    if (a < 0):
        return a * 0.01
    else:
        return a
alias matrix_leaky_relu = elementwise_map[leaky_relu]


@always_inline
fn matrix_mult(a: Matrix, b: Matrix) -> Matrix:
    let result = Matrix(a.rows, b.cols)
    #matmul_tiled_unrolled_parallelized(result, a, b)
    matmul_naive(result, a, b)
    return result


def test_map():
    print("testing transpose")
    var mat = Matrix(3, 4)
    mat.randomize()

    print("current mat:")
    for m in range(mat.rows):
        for n in range(mat.cols):
            print("[",m,",",n,"]: ", mat[m, n])

    let new_mat = matrix_leaky_relu(matrix_log(mat))

    print("new mat:")
    for m in range(new_mat.rows):
        for n in range(new_mat.cols):
            print("[",m,",",n,"]: ", new_mat[m, n])
#test_map()


from IO import print_no_newline
fn print_matrix(matrix&: Matrix):
    for m in range(matrix.rows):
        print_no_newline("[")
        for n in range(matrix.cols):
            print_no_newline(matrix[m,n])
            if n < matrix.cols - 1:
                print_no_newline(" ")
        print("]")


struct MatrixTuple:
    var a: Matrix
    var b: Matrix

    fn __init__(self&, a: Matrix, b: Matrix):
        self.a = a
        self.b = b

from StaticTuple import StaticTuple
struct LayerNode:
    var features: Matrix
    var weights: Matrix

    fn __init__(self&, features: Matrix, weights: Matrix):
        self.features = features
        self.weights = weights
    
    fn forward(self&) -> Matrix:
        return matrix_mult(self.features, self.weights)

    fn gradient(self&, grad: Matrix) -> MatrixTuple:
        let a = matrix_mult(grad, transpose_naive(self.weights))
        let b = matrix_mult(transpose_naive(self.features), grad)
        return MatrixTuple(a, b)


struct ReluNode:
    var input: Matrix

    fn __init__(self&, input: Matrix):
        self.input = input
    
    fn forward(self&) -> Matrix:
        return matrix_leaky_relu(self.input)

    fn gradient(self&, grad: Matrix) -> Matrix:
        return matrix_leaky_relu(grad)


struct LogisticLossNode:
    var labels: Matrix
    var predictions: Matrix

    fn __init__(self&, labels: Matrix, predictions: Matrix):
        self.labels = labels
        self.predictions = predictions

    fn forward(self&) -> Matrix:
        let x = matrix_mult(matrix_neg(self.labels), self.predictions)
        return matrix_log(matrix_add1(matrix_exp(x)))

    fn gradient(self&) -> Matrix:
        let denom = matrix_add1(matrix_exp(matrix_mult(self.labels, self.predictions)))
        return matrix_div(matrix_neg(self.labels), denom)

struct SingleLayerPerceptron:
    fn __init__(self&):
        pass

    # Trains a set of weights to 
    # X: feature matrix  (observations, features)
    # y: labels          (observations, 1)
    fn train(self&, X: Matrix, y: Matrix):
        let step_size = 0.01

        # random init weights
        var weights = Matrix(X.cols, y.cols)
        weights.randomize()
        # transform the rand()'s 0 - 1 range to -0.01 - 0.01
        for m in range(weights.rows):
            for n in range(weights.cols):
                weights[m, n] = (weights[m, n] - 0.5) / 50
        
        for epoch in range(300):
            # forward propogation
            var layer = LayerNode(X, weights)
            var relu = ReluNode(layer.forward())
            let pred = relu.forward()
            var loss = LogisticLossNode(y, pred)
            
            # check mean squared error
            let label_diff = matrix_square(matrix_sub(y, pred))
            var mse: F32 = 0
            for m in range(label_diff.rows):
                for n in range(label_diff.cols):
                    mse += label_diff[m, n]
            mse /= y.rows
            if (epoch % 10 == 0):
                print("loss:", mse)

            # backward propogation
            let loss_grad = loss.gradient()
            let relu_grad = relu.gradient(loss_grad)
            let layer_grad = layer.gradient(relu_grad)
            let weights_grad = layer_grad.b
            weights = matrix_sub(weights, matrix_scale(step_size, weights_grad))

# transform the rand()'s 0 - 1 range to -0.01 - 0.01
@always_inline
fn shrink(v: F32) -> F32:
    return (v - 0.5) / 50

def test_mlp():
    var mlp = SingleLayerPerceptron()
    let rows = 50
    let X = Matrix(rows, 2)
    let y = Matrix(rows, 1)
    
    # each row gets 3 noise values for the 2 inputs and 1 output
    let noise = DTypePointer[DType.f32].alloc(rows * 3)
    rand[DType.f32](noise, rows * 3)

    # XOR inputs
    var a = 0
    var b = 0
    for m in range(rows):
        # XOR output that will be the label
        let c = a ^ b
        let n1 = shrink(noise.simd_load[1](m * 3))
        let n2 = shrink(noise.simd_load[1](m * 3 + 1))
        let n3 = shrink(noise.simd_load[1](m * 3 + 2))
        
        # record the inputs and output, and add noise
        X[m, 0] = a + n1
        X[m, 1] = b + n2
        y[m, 0] = c + n3
        
        # increment a & b so we hit every xor combination
        # 0 ^ 0 = 0
        # 1 ^ 0 = 1
        # 0 ^ 1 = 1
        # 1 ^ 1 = 0
        a = (a + 1) % 2
        if a == 0:
            b = (b + 1) % 2
    mlp.train(X, y)

test_mlp()

