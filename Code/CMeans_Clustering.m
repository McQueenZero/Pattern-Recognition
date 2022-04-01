%%-------------------------------------------------------------------------
% 作者：       赵敏琨
% 日期：       2021年6月
% 说明：       C均值法聚类练习
% 软件版本：   MATLAB R2018a
%%-------------------------------------------------------------------------
%% 初始化
clc
clear
close all
while 1
    X_C = input('选择数据集（1：例题，2：自生成）：') ;
    switch X_C
        case 1
            X = [0 0; 1 0; 0 1; 1 1; 2 1; 1 2; 2 2; 3 2; 6 6;
                7 6; 8 6; 6 7; 7 7; 8 7; 9 7; 7 8; 8 8; 9 8; 
                8 9; 9 9];
            break
        case 2
            % 生成样本集
            N = 50;  %类样本个数
            mu1 = [-1 -1];
            sigma1 = [1 0.5; 0.5 2];
            R = chol(sigma1);
            z1 = repmat(mu1,N,1) + randn(N,2)*R;  %第一类
            mu2 = [1 1];
            sigma2 = [1 0.5; 0.5 1];
            R = chol(sigma2);
            z2 = repmat(mu2,N,1) + randn(N,2)*R;  %第二类
            X = [z1; z2];
            break
        otherwise
            disp('非法，请重新输入')
    end
end

X1.c = X(1, :); X2.c = X(2, :);   %center
C1 = Inf;  C2 = Inf;  %center record

%% C均值法
while 1
    X1.d = []; X2.d = [];  %data
    for ii = 1:size(X, 1)
        if norm(X(ii,:)-X1.c) < norm(X(ii,:)-X2.c)
            X1.d = [X1.d; X(ii, :)];
        else
            X2.d = [X2.d; X(ii, :)];
        end
    end
    X1.c = mean(X1.d);  X2.c = mean(X2.d);  %更新聚类中心
    disp(['类1中心:[' num2str(X1.c(1)) ' ' num2str(X1.c(2)) ']' ...
        ' 类2中心:[' num2str(X2.c(1)) ' ' num2str(X2.c(2)) ']'])
    if ~norm(C1-X1.c) && ~norm(C2-X2.c)  %1轮迭代中心不变
        break
    end
    C1 = X1.c; C2 = X2.c;  %记录聚类中心
    
%     close all
%     figure
%     plot(X1.d(:,1), X1.d(:,2), 'ro'), hold on
%     plot(X2.d(:,1), X2.d(:,2), 'b*'), hold on
end
figure
plot(X1.d(:,1), X1.d(:,2), 'ro'), hold on
plot(X2.d(:,1), X2.d(:,2), 'b*'), hold on
% for k = 1:size(X, 1)
%    text(X(k,1), X(k,2), num2str(k)); 
% end
% legend({'类1', '类2'}, 'Location', 'best')

% 凸包问题：画聚类包络线
dt1=delaunayTriangulation(X1.d(:,1), X1.d(:,2));
k1 = convexHull(dt1);
plot(X1.d(k1,1),X1.d(k1,2),'r');
dt2=delaunayTriangulation(X2.d(:,1), X2.d(:,2));
k2 = convexHull(dt2);
plot(X2.d(k2,1),X2.d(k2,2),'b');
