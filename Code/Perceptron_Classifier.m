%%-------------------------------------------------------------------------
% 作者：       赵敏琨         学号：2018302068     
% 日期：       2021年5月
% 说明：       基于单感知机的分类器
% 软件版本：   MATLAB R2018a
%%-------------------------------------------------------------------------
%% 单感知机固定增量法求判别函数
clc, clear, close all

while 1
    z_C = input('选择数据集（1：作业题，2：自生成）：') ;
    switch z_C
        case 1
            % 作业题样本集
            z1 = [0 0; 0 1];
            z2 = [1 0; 1 1];
            N = 2;
            break
        case 2
            % 生成样本集
            N = 5;  %类样本个数
            mu1 = [-1.05 -1.05];
            sigma1 = [1 0.5; 0.5 2];
            R = chol(sigma1);
            z1 = repmat(mu1,N,1) + randn(N,2)*R;  %第一类
            mu2 = [1.05 1.05];
            sigma2 = [1 0.5; 0.5 1];
            R = chol(sigma2);
            z2 = repmat(mu2,N,1) + randn(N,2)*R;  %第二类
            break
        otherwise
            disp('非法，请重新输入')
    end
end

for k = 1:N
    % 样本矩阵 
    X(:,k) = z1(k,:)';
    X(:,k+N) = z2(k,:)';
    % 增广矩阵（Augmented matrix）
    X_aug(:,k) = [X(:,k); 1];
    X_aug(:,k+N) = [X(:,k+N); 1];
end
      
% 初始迭代权向量和步长
W(:,1) = [1 1 1]';
rho = 1; 
m = 1;
k = 1;

plot(X(1,1:N),X(2,1:N),'ro'), hold on
plot(X(1,N+1:N+N),X(2,N+1:N+N),'b*'), hold on
syms x1 x2
name_legend = {'类1' '类2'};

AutoSTOP = 0;
while 1    
    % 迭代，错分类样本修正权值
    while m <= 2*N
        if m <= N && W(:,end)'*X_aug(:,m) <= 0
            W(:,end+1) = W(:,end) + rho * X_aug(:,m);
        end
        if m > N && W(:,end)'*X_aug(:,m) >= 0
            W(:,end+1) = W(:,end) - rho * X_aug(:,m);
        end
        m = m + 1;
    end
    if AutoSTOP == W(:,end)
        break
    end
    AutoSTOP = W(:,end);  %本轮迭代权值等于上一轮迭代权值停止
    if m > 2*N
        m = 1;
    end
    fg = [x1 x2 1] * AutoSTOP;  %决策面方程
    fimplicit(fg == 0)
    name_DB = ['第' num2str(k) '轮决策面'];
    name_legend = [name_legend name_DB];
    legend(name_legend, 'Location', 'best')
    hold on
    ManualSTOP = input('按0停止，回车继续：');
    if ManualSTOP == 0
        break
    end
    k = k + 1;
end
