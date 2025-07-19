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
c3 = Index(2, "c3")
c4 = Index(2, "c4")

K_const = 1.0


#4脚テンソルの定義
W = four_leg_tensor_def(i, j, k, l, K_const)

#2脚テンソルの定義
C = two_leg_tensor_def(α, β, c1, c2, K_const)

#3脚テンソルの定義
PR = three_leg_tensor_def(α, ξ, l, c3, K_const)#3脚テンソル(縦)

#3脚テンソルの定義
PC = three_leg_tensor_def(β, η, k, c4, K_const)

C2=PR*C*PC*W
