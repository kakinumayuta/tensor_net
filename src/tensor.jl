module tensor

# 関数定義(a,b,c,dは0or1 Kは逆温度を含む定数)
function four_leg_tensor_cal(a::Int64, b::Int64, c::Int64, d::Int64, K::Float64)
    # 0 or 1 のビットを ±1 に変換し、隣接ペアの積を計算
    s1 = 2a - 1
    s2 = 2b - 1
    s3 = 2c - 1
    s4 = 2d - 1
    exponent = K * (s1*s2 + s2*s3 + s3*s4 + s4*s1)
    return exp(exponent)
end






end # module tensor
