%%-------------------------------------------------------------------------
% 作者：       赵敏琨         学号：2018302068     
% 日期：       2021年5月
% 说明：       基于正态分布的最小错误率贝叶斯分类器
% 软件版本：   MATLAB R2018a
%%-------------------------------------------------------------------------
%% 主函数:基于正态分布的最小错误率贝叶斯分类器
function [P_po, fg_array] = Gauss_BayesClassifier(train_set, test_set, P_pr, flag)
%%基于最大似然的统计参数估计、最小错误率的贝叶斯分类器
% P_pr：行向量，每列代表第（列数）类的先验概率
% N：样本个数； d：特征维数； C：输入类数
[N, d, C] = size(train_set);

%%基于正态分布的类条件概率密度函数参数估计
% 用极大似然法估计均值和协方差矩阵
u = mean(train_set);
SIG_Temp = 0;
for k = 1:C     %极大似然估计协方差前系数为1/N
    SIG(:,:,k) = (N-1)/N.*cov(train_set(:,:,k));
    if flag == 1        
        SIG_Temp = SIG_Temp + SIG(:,:,k);
    end
end
if flag == 1    %两个特征相关时对协方差矩阵的处理
    for k = 1:C
        SIG(:,:,k) = SIG_Temp;
    end
end

% 正态分布的类条件概率密度函数，见ClassConDensity子函数

%%最小错误率Bayes分类器
% 计算各类的后验概率 测试集样本x
N_test = size(test_set,1);
P_po = zeros(N_test, C);
for n = 1:N_test
    x = test_set(n,1:d);
    % 每次计算一个样本的后验概率
    P_po(n, :) = P_pr .* ClassConDensity(d, C, u, SIG, x);    
end

% 对数决策面方程
syms x1 x2
X = [x1 x2];
fg_array = P_pr .* ClassConDensity(d, C, u, SIG, X);
fg_array = log(fg_array);

end

%% 内嵌函数：计算类条件概率密度
function P_ccd = ClassConDensity(d, C, u, SIG, x)
%%计算类条件概率密度函数
% N：样本个数； d：特征维数； C：输入类别数； 
% u：估计所得均值（每页是d维行向量，页号即为类别号）； 
% SIG：估计所得协方差矩阵（每页是一个2矩阵，页号即为类别号）； 
% x：输入特征（d维行向量） 
% 公式中x、u为列向量，转成列向量
x = x';
for k = 1:C     %共C列，每列代表第k类（第1类为女f，第2类为男m）
    P_ccd(1,k) = ((2*pi)^(d/2)*(det(SIG(:,:,k)))^0.5)^(-1)*exp(-0.5*(x-u(:,:,k)')'*SIG(:,:,k)^(-1)*(x-u(:,:,k)'));  
end
end