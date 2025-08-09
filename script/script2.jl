using ITensors
import tensor as ten

#Index
#大きな足
α = Index(2, "α")
ξ = Index(2, "ξ")
β = Index(2, "β")
η = Index(2, "η")

#小さな足
i = Index(2, "i")
j = Index(2, "j")
k = Index(2, "k")
l = Index(2, "l")

#固定端の場合⇒1、自由端の場合⇒2
c1 = Index(2, "c1")
c2 = Index(2, "c2")

#βJ
K_const = -0.5

#圧縮後のサイズ
χ_num = 2


#4脚テンソルの定義
W = ten.four_leg_tensor_def(i, j, k, l, K_const)

#2脚テンソルの定義
C = ten.two_leg_tensor_def(α, β, c1, c2, K_const)

#3脚テンソルの定義
PR = ten.three_leg_tensor_def(α, ξ, l, c1, K_const)

#3脚テンソルの定義
PC = ten.three_leg_tensor_def(β, η, k, c1, K_const)

#ベース
PRbf = PR
PCbf = PC
Cbf = C

#回数
n = 1

#拡大
PRaf, PCaf, Caf, newξ, α, newη, β = ten.Expand(PRbf, PCbf, W, Cbf, α, β, ξ, η, i, j, k, l)

@show PCaf
#圧縮
resD, resD2, resD3, resD4, resU, resU2, resU3, resU4, resUl, resUl2, resUl3, resUl4, resPRL, resPRR, resPCU, resPCD = ten.Compression_and_Copy(Caf, PRaf, PCaf, α, β, newξ, newη, i, j, k, l, χ_num, n)

#観測
A, B = ten.Observation(W, resD, resPRL, resD2, resPCU, resD3, resPRR, resD4, resPCD)

println("自発磁化=$A")
println("スピン相関=$B")

#テンソルの更新
PRbf = PRaf
PCbf = PCaf
Cbf = Caf
