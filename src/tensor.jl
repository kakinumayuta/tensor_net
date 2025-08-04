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
function Expand_PR(PR::ITensor, W::ITensor, α::Index{Int64}, k::Index{Int64}, ξ::Index{Int64}, i::Index{Int64}, j::Index{Int64}, l::Index{Int64})
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

    replaceinds!(PR2_ext, j => l)

    return PR2_ext, ξ, α, ξi
end

#3脚テンソル(縦)の拡大
function Expand_PC(PC::ITensor, W::ITensor, β::Index{Int64}, l::Index{Int64}, η::Index{Int64}, j::Index{Int64}, i::Index{Int64}, k::Index{Int64})
    PC2 = PC * W
    βl = combiner(β, l)
    ηj = combiner(η, j)

    PC2_ext = βl * PC2
    η = Index(dim(uniqueinds(βl)[1]), "η")
    replaceinds!(PC2_ext, uniqueinds(βl)[1] => η)

    PC2_ext = PC2_ext * ηj
    β = Index(dim(uniqueinds(ηj)[1]), "β")
    replaceinds!(PC2_ext, uniqueinds(ηj)[1] => β)

    replaceinds!(PC2_ext, i => k)

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

#角転送行列の対角化
function Diagonal_C_matrix(C2::ITensor, α::Index{Int64}, β::Index{Int64})
    # 固有値分解
    D, U = eigen(C2, α, β)

    # インデックスの取り出し
    dl = uniqueind(D, U)
    dr = commonind(D, U)

    # 正しい方法でUlを作成（置き換えは個別に！）
    Ul = replaceinds(U, (β => α, dr => dl))

    return D, U, Ul
end

#対角行列を絶対値の大きい順にソートする
function Sort_Diagonal(D::ITensor)
    Dnew = copy(D)

    #対角要素の並び替え
    for i in 1:size(D)[1]
        for j in (i+1):size(D)[2]
            if abs(real(Dnew[i, i])) < abs(real(Dnew[j, j]))
                tmp = Dnew[i, i]
                Dnew[i, i] = Dnew[j, j]
                Dnew[j, j] = tmp
            end
        end
    end

    return Dnew
end

#対角行列を絶対値が大きい順にソートした時のインデックスを返す
function Sort_idx(D::ITensor, i::Index{Int64}, j::Index{Int64})
    # 対角要素を取り出す
    diag_elements = Float64[]
    for i in 1:size(D)[1]
        push!(diag_elements, D[i, i])
    end

    # 絶対値でソートしたインデックスを取得
    sorted_indices = sortperm(abs.(diag_elements), rev=true)

    return sorted_indices
end

#固有値をソートし並べ替え、制限した後の対角行列を返す
function Restrict_Diagonal(D::ITensor, χ::Int64, a1::Index{Int64}, a2::Index{Int64})
    sortedD = Sort_Diagonal(D)
    tmp1 = Index(size(D)[1], "t1")
    tmp2 = Index(size(D)[2], "t2")
    idx = Sort_idx(D::ITensor, tmp1::Index{Int64}, tmp2::Index{Int64})

    #制限の必要がなければ返す
    if size(D)[1] < χ
        return sortedD, idx
    end

    resD = ITensor(a1', a2)

    for i in 1:χ
        resD[i, i] = sortedD[i, i]
    end

    return resD, idx
end

#固有値をソートし並べ替えた時のインデックスで並び替え、制限した後の対角化行列を返す
function Restrict_EigenvecsU(U::ITensor, χ::Int64, a1::Index{Int64}, a2::Index{Int64}, idx::Vector{Int64})
    #並び替え
    newU = copy(U)

    for i in 1:size(U)[1]
        cou = 1
        for j in idx
            newU[i, cou] = U[i, j]
            cou += 1
        end
    end

    #制限の必要がなければ返す
    if size(U)[1] < χ
        return newU
    end

    #大きさを制限
    resU = ITensor(a1, a2)
    for i in 1:size(U)[1]
        for j in 1:χ
            resU[i, j] = newU[i, j]
        end
    end

    return resU
end

#固有値をソートし並べ替えた時のインデックスで並び替え、制限した後の対角化行列を返す(左)
function Restrict_EigenvecsUl(Ul::ITensor, χ::Int64, a1::Index, a2::Index, idx::Vector{Int64})
    #並び替え
    newUl = copy(Ul)

    for i in 1:size(Ul)[1]
        cou = 1
        for j in idx
            newUl[i, cou] = Ul[i, j]
            cou += 1
        end
    end

    #制限の必要がなければ返す
    if size(Ul)[1] < χ
        return Ul
    end

    #大きさを制限
    resUl = ITensor(a2', a1)
    for i in 1:χ
        for j in 1:size(Ul)[2]
            resUl[i, j] = newUl[i, j]
        end
    end

    return resUl
end

#自発磁化
function Self_Magnetization(G::ITensor, W::ITensor, O::Float64)
    #自発磁化
    a = 0.0
    for i in 0:1
        for j in 0:1
            for k in 0:1
                for l in 0:1
                    a += 1 / O * G[i+1, j+1, k+1, l+1] * (2 * i - 1) * W[i+1, j+1, k+1, l+1]
                end
            end
        end
    end

    return a
end

#スピン相関
function Spin_Correlation(G::ITensor, W::ITensor, O::Float64)
    #スピン相関
    b = 0.0
    for i in 0:1
        for j in 0:1
            for k in 0:1
                for l in 0:1
                    b += 1 / O * G[i+1, j+1, k+1, l+1] * (2 * i - 1) * (2 * j - 1) * W[i+1, j+1, k+1, l+1]
                end
            end
        end
    end

    return b
end

function Expand(PR::ITensor, PC::ITensor, W::ITensor, α::Index{Int64}, β::Index{Int64}, ξ::Index{Int64}, η::Index{Int64}, i::Index{Int64}, j::Index{Int64}, k::Index{Int64}, l::Index{Int64})
    #3脚テンソル(横)の拡大
    PR2, ξ, α, ξi = Expand_PR(PR, W, α, k, ξ, i, j, l)

    #3脚テンソル(縦)の拡大
    PC2, η, β, ηj = Expand_PC(PC, W, β, l, η, j, i, k)

    #角転送行列の拡大
    C2 = Expand_C(PR, C, PC, W, ξi, ηj, α, β)

    return PR2, PC2, C2, ξ, α, η, β, C2
end


end # module tensor
