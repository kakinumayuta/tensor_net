using ITensors
using Plots
import tensor as ten

#Indexの定義
α, β, ξ, η, i, j, k, l, c1, c2 = ten.Index_def()

#βJ
K_const = -0.01

#試行回数
n = 10

#圧縮後のサイズ
χ_num = 4

#格子サイズを格納する配列
Lvec = Int64[]

#物理量を格納する配列
SMvec = ComplexF64[]
SCvec = ComplexF64[]


#4脚テンソルの定義
W = ten.four_leg_tensor_def(i, j, k, l, K_const)

#2脚テンソルの定義
C = ten.two_leg_tensor_def(α, β, c1, c2, K_const)

#3脚テンソルの定義
PR = ten.three_leg_tensor_def(α, ξ, l, c1, K_const)

#3脚テンソルの定義
PC = ten.three_leg_tensor_def(β, η, k, c1, K_const)


#対角化
D, U, Ul = ten.Diagonal_C_matrix(C, α, β)

#圧縮(圧縮する必要のない場合はそのままのサイズで帰ってくる)
resD, resD2, resD3, resD4, resU, resU2, resU3, resU4, resUl, resUl2, resUl3, resUl4, resPRL, resPRR, resPCU, resPCD = ten.Compression_and_Copy(D, U, Ul, PR, PC, α, β, ξ, η, i, j, k, l, χ_num, 0)


#観測
SM, SC = ten.Observation(W, resD, resPRL, resD2, resPCU, resD3, resPRR, resD4, resPCD)

#物理量の格納
push!(SMvec, SM)
push!(SCvec, SC)

#格子のサイズ
Lsize = 2 ^ (0 + 1) + 1

#格子サイズの格納
push!(Lvec, Lsize)

#出力
println("$Lsize×$Lsize")
println("Self Magnetization=$SM")
println("Spin Correlation=$SC")

#拡大前の処理
global PRbf = PR
global PCbf = PC
global Cbf = C

global αbf = α
global βbf = β
global ξbf = ξ
global ηbf = η


for L in 1:n
    #拡大
    PRaf, PCaf, Caf, ξaf, αaf, ηaf, βaf = ten.Expand(PRbf, PCbf, W, Cbf, αbf, βbf, ξbf, ηbf, i, j, k, l)


    #対角化
    D, U, Ul = ten.Diagonal_C_matrix(Caf, αaf, βaf)

    #圧縮(圧縮する必要のない場合はそのままのサイズで帰ってくる)
    resD, resD2, resD3, resD4, resU, resU2, resU3, resU4, resUl, resUl2, resUl3, resUl4, resPRL, resPRR, resPCU, resPCD = ten.Compression_and_Copy(D, U, Ul, PRaf, PCaf, αaf, βaf, ξaf, ηaf, i, j, k, l, χ_num, L)


    #観測
    SM, SC = ten.Observation(W, resD, resPRL, resD2, resPCU, resD3, resPRR, resD4, resPCD)

    #物理量の格納
    push!(SMvec, SM)
    push!(SCvec, SC)

    #格子のサイズ
    Lsize = 2 ^ (L + 1) + 1

    #格子サイズの格納
    push!(Lvec, Lsize)

    #出力
    println("$Lsize×$Lsize")
    println("Self Magnetization=$SM")
    println("Spin Correlation=$SC")

    #テンソルの更新
    global PRbf = PRaf
    global PCbf = PCaf
    global Cbf = Caf

    #Indexの更新
    global αbf = αaf
    global βbf = βaf
    global ξbf = ξaf
    global ηbf = ηaf
end

#グラフの出力
fig = plot(Lvec, real(SMvec))
savefig(fig, "Self_Magnetization.png")
fig = plot(Lvec, real(SCvec))
savefig(fig, "Spin_Correlation.png")
