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

    K_const = 1.0

    A = ten.four_leg_tensor_def(i, j, k, l, K_const)
    B = ten.four_leg_tensor_def(j, i, m, n, K_const)

    C = A * B

    @test dim(C) == 16
    @test dim(inds(C)[1]) == 2
    @test dim(inds(C)[2]) == 2
    @test dim(inds(C)[3]) == 2
    @test dim(inds(C)[4]) == 2
end

@testset "three-foot tensor" begin
    # インデックス定義（2値）
    i = Index(2, "i")
    j = Index(2, "j")
    k = Index(2, "k")
    l = Index(2, "l")
    m = Index(2, "m")
    n = Index(2, "n")

    K_const = 1.0

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

    K_const = 1.0

    A = ten.two_leg_tensor_def(i, j, k, l, K_const)
    B = ten.two_leg_tensor_def(j, i, m, n, K_const)

    C = A * B

    @test dim(C) == 1
end
