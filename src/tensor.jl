module tensor
using ITensors

#4脚テンソルの中身を計算する。a,b,c,dは0or1 Kは逆温度を含む定数
function four_leg_tensor_cal(a::Int64, b::Int64, c::Int64, d::Int64, K::Float64)
    # 0 or 1 のビットを ±1 に変換し、隣接ペアの積を計算
    s1 = 2a - 1
    s2 = 2b - 1
    s3 = 2c - 1
    s4 = 2d - 1
    exponent = K * (s1 * s2 + s2 * s3 + s3 * s4 + s4 * s1)
    return exp(exponent)
end

#4脚テンソルの定義
function four_leg_tensor_def(i::Index{Int64}, j::Index{Int64}, k::Index{Int64}, l::Index{Int64}, K_const::Float64)
    W = ITensor(i, j, k, l)
    for a in 0:1, b in 0:1, c in 0:1, d in 0:1
        val = four_leg_tensor_cal(a, b, c, d, K_const)
        W[i=>a+1, j=>b+1, k=>c+1, l=>d+1] = val
    end
    return W
end

#3脚テンソルの定義
function three_leg_tensor_def(i::Index{Int64}, j::Index{Int64}, k::Index{Int64}, l::Index{Int64}, K_const::Float64)
    P = ITensor(i, j, k, l)
    for a in 0:dim(i)-1, b in 0:1, c in 0:dim(k)-1, d in 0:1
        val = four_leg_tensor_cal(a, b, c, d, K_const)
        P[i=>a+1, j=>b+1, k=>c+1, l=>d+1] = val
    end

    # c1 と c2 に関して縮約
    δ_l = delta(l)  # l に関する単位テンソル

    return P * δ_l
end

#2脚テンソルの定義
function two_leg_tensor_def(i::Index{Int64}, j::Index{Int64}, k::Index{Int64}, l::Index{Int64}, K_const::Float64)
    C = ITensor(i, j, k, l)
    for a in 0:dim(i)-1, b in 0:dim(j)-1, c in 0:1, d in 0:1
        val = four_leg_tensor_cal(a, b, c, d, K_const)
        C[i=>a+1, j=>b+1, k=>c+1, l=>d+1] = val
    end

    # c1 と c2 に関して縮約
    δ_k = delta(k)  # k に関する単位テンソル
    δ_l = delta(l)  # l に関する単位テンソル

    return C * δ_k * δ_l
end

#3脚テンソル(横)の拡大
function Expand_PR(PR::ITensor, W::ITensor, α::Index{Int64}, k::Index{Int64}, ξ::Index{Int64}, i::Index{Int64})
    #拡大
    PR2 = PR * W

    #文字の結合
    αk = combiner(α, k)
    ξi = combiner(ξ, i)

    PR2_ext = αk * PR2
    ξ = Index(dim(uniqueinds(αk)[1]), "ξ")
    replaceinds!(PR2_ext, uniqueinds(αk)[1] => ξ)

    PR2_ext = PR2_ext * ξi
    α = Index(dim(uniqueinds(ξi)[1]), "α")
    replaceinds!(PR2_ext, uniqueinds(ξi)[1] => α)

    return PR2_ext, ξ, α, ξi
end

#3脚テンソル(縦)の拡大
function Expand_PC(PC::ITensor, W::ITensor, β::Index{Int64}, l::Index{Int64}, η::Index{Int64}, j::Index{Int64})
    PC2 = PC * W
    βl = combiner(β, l)
    ηj = combiner(η, j)

    PC2_ext = βl * PC2
    η = Index(dim(uniqueinds(βl)[1]), "η")
    replaceinds!(PC2_ext, uniqueinds(βl)[1] => η)

    PC2_ext = PC2_ext * ηj
    β = Index(dim(uniqueinds(ηj)[1]), "β")
    replaceinds!(PC2_ext, uniqueinds(ηj)[1] => β)

    return PC2_ext, η, β, ηj
end

#角転送行列の拡大
function Expand_C(PR::ITensor, C::ITensor, PC::ITensor, W::ITensor, ξi::ITensor, ηj::ITensor, α::Index{Int64}, β::Index{Int64})
    #拡大
    C2 = PR * C * PC * W

    #文字の結合
    C2_ext = ξi * C2 * ηj

    replaceinds!(C2_ext, uniqueinds(ξi)[1] => α)
    replaceinds!(C2_ext, uniqueinds(ηj)[1] => β)

    return C2_ext
end






end # module tensor
