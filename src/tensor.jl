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






end # module tensor
