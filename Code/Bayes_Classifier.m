%%-------------------------------------------------------------------------
% 作者：       赵敏琨
% 日期：       2021年5月
% 说明：       贝叶斯分类器作业题
% 注意：       分节运行
% 软件版本：   MATLAB R2018a
%%-------------------------------------------------------------------------
%% 第①题，协方差矩阵不相等
clc, clear, close all
w1.X1 = [1 1 2]';       %类w1,特征X1、第一列
w1.X2 = [1 0 -1]';      %类w1,特征X2、第二列
w2.X1 = [-1 -1 -2]';    %类w2,特征X1、第一列
w2.X2 = [1 0 -1]';      %类w2,特征X2、第二列
% 计算均值
w1.x1_bar = mean(w1.X1);
w1.x2_bar = mean(w1.X2);
w2.x1_bar = mean(w2.X1);
w2.x2_bar = mean(w2.X2);
w1.X_bar = [w1.x1_bar w1.x2_bar]';
w2.X_bar = [w2.x1_bar w2.x2_bar]';
% 计算协方差矩阵
w1.SIG = cov(w1.X1, w1.X2);
w2.SIG = cov(w2.X1, w2.X2);
% 计算先验概率（由样本个数计算的频率）
P_w1 = size(w1.X1,1) / (size(w1.X1,1) + size(w2.X1,1));
P_w2 = size(w2.X1,1) / (size(w1.X1,1) + size(w2.X1,1));
% 构造判别函数
syms x1 x2
x = [x1; x2];
fg1 = -0.5 * (x - w1.X_bar)' * w1.SIG^(-1) * (x - w1.X_bar) ...
    - 0.5 * log(det(w1.SIG)) + log(P_w1);
fg2 = -0.5 * (x - w2.X_bar)' * w2.SIG^(-1) * (x - w2.X_bar) ...
    - 0.5 * log(det(w2.SIG)) + log(P_w2);
fg = fg2 - fg1;
% 画图
fg_plothandle = fimplicit(fg == 0, 'b -', 'Linewidth', 1);  %根据方程画分界线（曲线所有部分）
Temp12X = fg_plothandle.XData;
Temp12Y = fg_plothandle.YData;
plot(Temp12X(1:116), Temp12Y(1:116), 'b -', 'Linewidth', 1), hold on  %需手动调节显示范围（决策线）
plot(w1.X1, w1.X2, 'r o', 'Markersize', 8), hold on
plot(w2.X1, w2.X2, 'r *', 'Markersize', 8), hold on
axis([-2.2 2.2 -2.2 Temp12Y(119)])

% 决策
plot(2, 0, 'm s', 'Markersize', 8), hold on
legend({'w1-w2 决策面','类w1','类w2','样本点'},'Location','best')
if subs(fg, [x1 x2], [2 0]) < 0     %手动输入样本点
    disp('样本点属于w1类')
else
    disp('样本点属于w2类')
end

%% 第①题，假设协方差矩阵相等
clc, clear, close all
w1.X1 = [1 1 2]';       %类w1,特征X1、第一列
w1.X2 = [1 0 -1]';      %类w1,特征X2、第二列
w2.X1 = [-1 -1 -2]';    %类w2,特征X1、第一列
w2.X2 = [1 0 -1]';      %类w2,特征X2、第二列
% 计算均值
w1.x1_bar = mean(w1.X1);
w1.x2_bar = mean(w1.X2);
w2.x1_bar = mean(w2.X1);
w2.x2_bar = mean(w2.X2);
w1.X_bar = [w1.x1_bar w1.x2_bar]';
w2.X_bar = [w2.x1_bar w2.x2_bar]';
% 计算协方差矩阵与相等的协方差
w1.SIG = cov(w1.X1, w1.X2);
w2.SIG = cov(w2.X1, w2.X2);
SIG = w1.SIG + w2.SIG;
% 计算先验概率（由样本个数计算的频率）
P_w1 = size(w1.X1,1) / (size(w1.X1,1) + size(w2.X1,1));
P_w2 = size(w2.X1,1) / (size(w1.X1,1) + size(w2.X1,1));
% 构造判别函数
syms x1 x2
x = [x1; x2];
fg1 = (SIG^(-1) * w1.X_bar)' * x ...
    - 0.5 * w1.X_bar' * SIG^(-1) * w1.X_bar + log(P_w1);
fg2 = (SIG^(-1) * w2.X_bar)' * x ...
    - 0.5 * w2.X_bar' * SIG^(-1) * w2.X_bar + log(P_w2);
fg = fg2 - fg1;
% 画图
fg_plothandle = fimplicit(fg == 0, 'b -', 'Linewidth', 1);  %根据方程画分界线（曲线所有部分）
Temp12X = fg_plothandle.XData;
Temp12Y = fg_plothandle.YData;
plot(Temp12X(1:end), Temp12Y(1:end), 'b -', 'Linewidth', 1), hold on  %需手动调节显示范围（决策线）
plot(w1.X1, w1.X2, 'r o', 'Markersize', 8), hold on
plot(w2.X1, w2.X2, 'r *', 'Markersize', 8), hold on
axis([-2.2 2.2 -2.2 2.2])

% 决策
plot(2, 0, 'm s', 'Markersize', 8), hold on
legend({'w1-w2 决策面','类w1','类w2','样本点'},'Location','best')
if subs(fg, [x1 x2], [2 0]) < 0     %手动输入样本点
    disp('样本点属于w1类')
else
    disp('样本点属于w2类')
end

%% 第②题，协方差矩阵不相等
clc, clear, close all
w1.X1 = [0 2 1]';       %类w1,特征X1、第一列
w1.X2 = [0 1 0]';       %类w1,特征X2、第二列
w2.X1 = [-1 -2 -2]';    %类w2,特征X1、第一列
w2.X2 = [1 0 -1]';      %类w2,特征X2、第二列
w3.X1 = [0 0 1]';       %类w3,特征X1、第一列
w3.X2 = [-2 -1 -2]';    %类w3,特征X2、第二列
% 计算均值
w1.x1_bar = mean(w1.X1);
w1.x2_bar = mean(w1.X2);
w2.x1_bar = mean(w2.X1);
w2.x2_bar = mean(w2.X2);
w3.x1_bar = mean(w3.X1);
w3.x2_bar = mean(w3.X2);
w1.X_bar = [w1.x1_bar w1.x2_bar]';
w2.X_bar = [w2.x1_bar w2.x2_bar]';
w3.X_bar = [w3.x1_bar w3.x2_bar]';
% 计算协方差矩阵
w1.SIG = cov(w1.X1, w1.X2);
w2.SIG = cov(w2.X1, w2.X2);
w3.SIG = cov(w3.X1, w3.X2);
% 计算先验概率（由样本个数计算的频率）
P_w1 = size(w1.X1,1) / (size(w1.X1,1) + size(w2.X1,1) + size(w3.X1,1));
P_w2 = size(w2.X1,1) / (size(w1.X1,1) + size(w2.X1,1) + size(w3.X1,1));
P_w3 = size(w3.X1,1) / (size(w1.X1,1) + size(w2.X1,1) + size(w3.X1,1));
% 构造判别函数
syms x1 x2
x = [x1; x2];
fg1 = x' * (-0.5) * w1.SIG^(-1) * x + (w1.SIG^(-1) * w1.X_bar)' * x ...
    - 0.5 * w1.X_bar' * w1.SIG^(-1) * w1.X_bar - 0.5 * log(det(w1.SIG)) + log(P_w1);
fg2 = x' * (-0.5) * w2.SIG^(-1) * x + (w2.SIG^(-1) * w2.X_bar)' * x ...
    - 0.5 * w2.X_bar' * w2.SIG^(-1) * w2.X_bar - 0.5 * log(det(w2.SIG)) + log(P_w2);
fg3 = x' * (-0.5) * w3.SIG^(-1) * x + (w3.SIG^(-1) * w3.X_bar)' * x ...
    - 0.5 * w3.X_bar' * w3.SIG^(-1) * w3.X_bar - 0.5 * log(det(w3.SIG)) + log(P_w3);
% 画图
fg12_plothandle = fimplicit(fg1 == fg2, 'g -', 'Linewidth', 1); %根据方程画分界线（曲线所有部分）
Temp12X = fg12_plothandle.XData;
Temp12Y = fg12_plothandle.YData;
fg23_plothandle = fimplicit(fg2 == fg3, 'm -', 'Linewidth', 1); %根据方程画分界线（曲线所有部分）
Temp23X = fg23_plothandle.XData;
Temp23Y = fg23_plothandle.YData;
fg31_plothandle = fimplicit(fg3 == fg1, 'b -', 'Linewidth', 1); %根据方程画分界线（曲线所有部分）
Temp31X = fg31_plothandle.XData;
Temp31Y = fg31_plothandle.YData;
close
plot(Temp12X(97:293), Temp12Y(97:293), 'g -', 'Linewidth', 1), hold on  %需手动调节显示范围（决策线）
plot(Temp23X(1:53), Temp23Y(1:53), 'm -', 'Linewidth', 1), hold on  %需手动调节显示范围（决策线）
plot(Temp31X(185:304), Temp31Y(185:304), 'b -', 'Linewidth', 1), hold on  %需手动调节显示范围（决策线）
plot(w1.X1, w1.X2, 'r o', 'Markersize', 8), hold on
plot(w2.X1, w2.X2, 'r *', 'Markersize', 8), hold on
plot(w3.X1, w3.X2, 'r s', 'Markersize', 8), hold on
axis([-2.2 2.2 -4 2.2])

% 决策
plot(-2, 2, 'm s', 'Markersize', 8), hold on
legend({'w1-w2 决策面','w2-w3 决策面','w3-w1 决策面','类w1','类w2','类w3','样本点'},'Location','best')
if subs(fg1-fg2, [x1 x2], [-2 2]) > 0     %手动输入样本点
    if subs(fg3-fg1, [x1 x2], [-2 2]) < 0     %手动输入样本点
        disp('样本点属于w1类')
    else
        disp('样本点属于w3类')
    end
else
    if subs(fg2-fg3, [x1 x2], [-2 2]) > 0     %手动输入样本点
        disp('样本点属于w2类')
    else
        disp('样本点属于w3类')
    end
end

%% 第②题，假设协方差矩阵相等
clc, clear, close all
w1.X1 = [0 2 1]';       %类w1,特征X1、第一列
w1.X2 = [0 1 0]';       %类w1,特征X2、第二列
w2.X1 = [-1 -2 -2]';    %类w2,特征X1、第一列
w2.X2 = [1 0 -1]';      %类w2,特征X2、第二列
w3.X1 = [0 0 1]';       %类w3,特征X1、第一列
w3.X2 = [-2 -1 -2]';    %类w3,特征X2、第二列
% 计算均值
w1.x1_bar = mean(w1.X1);
w1.x2_bar = mean(w1.X2);
w2.x1_bar = mean(w2.X1);
w2.x2_bar = mean(w2.X2);
w3.x1_bar = mean(w3.X1);
w3.x2_bar = mean(w3.X2);
w1.X_bar = [w1.x1_bar w1.x2_bar]';
w2.X_bar = [w2.x1_bar w2.x2_bar]';
w3.X_bar = [w3.x1_bar w3.x2_bar]';
% 计算协方差矩阵与相等的协方差
w1.SIG = cov(w1.X1, w1.X2);
w2.SIG = cov(w2.X1, w2.X2);
w3.SIG = cov(w3.X1, w3.X2);
SIG = w1.SIG + w2.SIG + w3.SIG;
% 计算先验概率（由样本个数计算的频率）
P_w1 = size(w1.X1,1) / (size(w1.X1,1) + size(w2.X1,1) + size(w3.X1,1));
P_w2 = size(w2.X1,1) / (size(w1.X1,1) + size(w2.X1,1) + size(w3.X1,1));
P_w3 = size(w3.X1,1) / (size(w1.X1,1) + size(w2.X1,1) + size(w3.X1,1));
% 构造判别函数
syms x1 x2
x = [x1; x2];
fg1 = x' * (-0.5) * SIG^(-1) * x + (SIG^(-1) * w1.X_bar)' * x ...
    - 0.5 * w1.X_bar' * SIG^(-1) * w1.X_bar - 0.5 * log(det(SIG)) + log(P_w1);
fg2 = x' * (-0.5) * SIG^(-1) * x + (SIG^(-1) * w2.X_bar)' * x ...
    - 0.5 * w2.X_bar' * SIG^(-1) * w2.X_bar - 0.5 * log(det(SIG)) + log(P_w2);
fg3 = x' * (-0.5) * SIG^(-1) * x + (SIG^(-1) * w3.X_bar)' * x ...
    - 0.5 * w3.X_bar' * SIG^(-1) * w3.X_bar - 0.5 * log(det(SIG)) + log(P_w3);
% 画图
fg12_plothandle = fimplicit(fg1 == fg2, 'g -', 'Linewidth', 1); %根据方程画分界线（曲线所有部分）
Temp12X = fg12_plothandle.XData;
Temp12Y = fg12_plothandle.YData;
fg23_plothandle = fimplicit(fg2 == fg3, 'm -', 'Linewidth', 1); %根据方程画分界线（曲线所有部分）
Temp23X = fg23_plothandle.XData;
Temp23Y = fg23_plothandle.YData;
fg31_plothandle = fimplicit(fg3 == fg1, 'b -', 'Linewidth', 1); %根据方程画分界线（曲线所有部分）
Temp31X = fg31_plothandle.XData;
Temp31Y = fg31_plothandle.YData;
close
plot(Temp12X(1:179), Temp12Y(1:179), 'g -', 'Linewidth', 1), hold on  %需手动调节显示范围（决策线）
plot(Temp23X(1:122), Temp23Y(1:122), 'm -', 'Linewidth', 1), hold on  %需手动调节显示范围（决策线）
plot(Temp31X(1:174), Temp31Y(1:174), 'b -', 'Linewidth', 1), hold on  %需手动调节显示范围（决策线）
plot(w1.X1, w1.X2, 'r o', 'Markersize', 8), hold on
plot(w2.X1, w2.X2, 'r *', 'Markersize', 8), hold on
plot(w3.X1, w3.X2, 'r s', 'Markersize', 8), hold on
axis([-3 2.2 -3 2.2])

% 决策
plot(-2, 2, 'm s', 'Markersize', 8), hold on
legend({'w1-w2 决策面','w2-w3 决策面','w3-w1 决策面','类w1','类w2','类w3','样本点'},'Location','best')
if subs(fg1-fg2, [x1 x2], [-2 2]) > 0     %手动输入样本点
    if subs(fg3-fg1, [x1 x2], [-2 2]) < 0     %手动输入样本点
        disp('样本点属于w1类')
    else
        disp('样本点属于w3类')
    end
else
    if subs(fg2-fg3, [x1 x2], [-2 2]) > 0     %手动输入样本点
        disp('样本点属于w2类')
    else
        disp('样本点属于w3类')
    end
end
