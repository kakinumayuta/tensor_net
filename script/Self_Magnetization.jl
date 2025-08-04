using ITensors
import tensor as ten

#Index
#大きな足
α = Index(2, "α")
ξ = Index(2, "ξ")
β = Index(2, "β")
η = Index(2, "η")

#コピーのための足の定義
λ = Index(4, "λ")
γ = Index(4, "γ")
ϵ = Index(4, "ϵ")
θ = Index(4, "θ")

#小さな足
i = Index(2, "i")
j = Index(2, "j")
k = Index(2, "k")
l = Index(2, "l")

#固定端の場合⇒1、自由端の場合⇒2
c1 = Index(2, "c1")
c2 = Index(2, "c2")

K_const = -0.5


#4脚テンソルの定義
W = ten.four_leg_tensor_def(i, j, k, l, K_const)

#2脚テンソルの定義
C = ten.two_leg_tensor_def(α, β, c1, c2, K_const)

#3脚テンソルの定義
PR = ten.three_leg_tensor_def(α, ξ, l, c1, K_const)

#3脚テンソルの定義
PC = ten.three_leg_tensor_def(β, η, k, c1, K_const)

PR2 = PR
PC2 = PC
C2 = C


for i in 1:5
    #拡大
    PR2, PC2, C2, ξ, α, η, β, C2 = ten.Expand(PR, PC, W, α, β, ξ, η, i, j, k, l)

    #対角化
    D, U, Ul = ten.Diagonal_C_matrix(C2, α, β)

    #固有値の制限
    χ_num = 2
    χ = Index(χ_num, "χ")
    ν = Index(χ_num, "ν")
    μ = Index(χ_num, "μ")
    σ = Index(χ_num, "σ")

    #圧縮過程
    resD, idx = ten.Restrict_Diagonal(D, χ_num, χ, χ)
    resU = ten.Restrict_EigenvecsU(U, χ_num, β, χ, idx)
    resUl = ten.Restrict_EigenvecsUl(Ul, χ_num, α, χ, idx)

    #固有ベクトル(右)のコピー
    resU2 = replaceinds(resU, (χ => ν, β => ξ))
    resU3 = replaceinds(resU, (χ => μ, β => ϵ))
    resU4 = replaceinds(resU, (χ => σ, β => θ))

    #固有ベクトル(左)のコピー
    resUl2 = replaceinds(resUl, (χ' => ν', α => λ))
    resUl3 = replaceinds(resUl, (χ' => μ', α => γ))
    resUl4 = replaceinds(resUl, (χ' => σ', α => η))

    #対角行列のコピー
    resD2 = replaceinds(resD, (χ => ν, χ' => ν'))
    resD3 = replaceinds(resD, (χ => μ, χ' => μ'))
    resD4 = replaceinds(resD, (χ => σ, χ' => σ'))

    #3脚テンソルのコピー
    PRL = copy(PR2)
    PCD = copy(PC2)
    PCU = replaceinds(PCD, (β => ϵ, η => λ, k => i))
    PRR = replaceinds(PRL, (α => γ, ξ => θ, l => j))

    #3脚テンソルの圧縮
    resPRL = resUl * PRL * resU2
    resPCU = resUl2 * PCU * resU3
    resPRR = resUl3 * PRR * resU4
    resPCD = resUl4 * PCD * resU

    #環境テンソル
    G = resD * resPRL * resD2 * resPCU * resD3 * resPRR * resD4 * resPCD

    #分配関数
    O = G * W

    #自発磁化
    A = ten.Self_Magnetization(G, W, real(O[])) |> display
    B = ten.Spin_Correlation(G, W, real(O[])) |> display

    PR = PR2
    PC = PC2
    C = C2
end
