using Test
import tensor as ten

using ITensors

#あとは各数値の計算をしてみるテストをする

@testset "contraction" begin
    # インデックス定義（2値）
    i = Index(2, "i")
    j = Index(2, "j")
    k = Index(2, "k")
    l = Index(2, "l")
    m = Index(2, "m")
    n = Index(2, "n")

    K_const = -1.0

    A = ten.four_leg_tensor_def(i, j, k, l, K_const)
    B = ten.four_leg_tensor_def(j, i, m, n, K_const)

    C = A * B

    @test dim(C) == 16
    @test dim(inds(C)[1]) == 2
    @test dim(inds(C)[2]) == 2
    @test dim(inds(C)[3]) == 2
    @test dim(inds(C)[4]) == 2
end

@testset "three-leg tensor" begin
    # インデックス定義（2値）
    i = Index(2, "i")
    j = Index(2, "j")
    k = Index(2, "k")
    l = Index(2, "l")
    m = Index(2, "m")
    n = Index(2, "n")

    K_const = -1.0

    A = ten.three_leg_tensor_def(i, j, k, l, K_const)
    B = ten.three_leg_tensor_def(j, i, m, n, K_const)

    C = A * B

    @test dim(C) == 4
    @test dim(inds(C)[1]) == 2
    @test dim(inds(C)[2]) == 2
end

@testset "corner-matrix" begin
    # インデックス定義（2値）
    i = Index(2, "i")
    j = Index(2, "j")
    k = Index(2, "k")
    l = Index(2, "l")
    m = Index(2, "m")
    n = Index(2, "n")

    K_const = -1.0

    A = ten.two_leg_tensor_def(i, j, k, l, K_const)
    B = ten.two_leg_tensor_def(j, i, m, n, K_const)

    C = A * B

    @test dim(C) == 1
end

@testset "value of corner-matrix" begin
    #次元
    L = 2

    # インデックス定義（2値）
    i = Index(2^L, "i")
    j = Index(2^L, "j")
    k = Index(2, "k")
    l = Index(2, "l")

    K_const = -1.0

    A = ten.two_leg_tensor_def(i, j, k, l, K_const)

    Amat = zeros(Float64, 2^L, 2^L)

    for a in 1:2
        for b in 1:2
            for c in 1:2^L
                for d in 1:2^L
                    e = a - 1
                    f = b - 1
                    g = c - 1
                    h = d - 1
                    Amat[c, d] += ten.four_leg_tensor_cal(e, f, g, h, K_const)
                end
            end
        end
    end

    for c in 1:2^L
        for d in 1:2^L
            @test abs(Amat[c, d] - A[c, d]) < 1e-13
        end
    end
end

@testset "value of three-leg matrix" begin
    #次元
    L = 2

    # インデックス定義（2値）
    i = Index(2^L, "i")
    j = Index(2, "j")
    k = Index(2^L, "k")
    l = Index(2, "l")

    K_const = -1.0

    A = ten.three_leg_tensor_def(i, j, k, l, K_const)

    Amat = zeros(Float64, 2^L, 2, 2^L)

    for a in 1:2
        for b in 1:2^L
            for c in 1:2
                for d in 1:2^L
                    e = a - 1
                    f = b - 1
                    g = c - 1
                    h = d - 1
                    Amat[b, c, d] += ten.four_leg_tensor_cal(e, f, g, h, K_const)
                end
            end
        end
    end

    for b in 1:2^L
        for c in 1:2
            for d in 1:2^L
                @test abs(Amat[b, c, d] - A[b, c, d]) < 1e-13
            end
        end
    end
end

@testset "expand corner-matrix" begin

end
