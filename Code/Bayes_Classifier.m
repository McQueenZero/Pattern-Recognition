%%-------------------------------------------------------------------------
% ���ߣ�       ������
% ���ڣ�       2021��5��
% ˵����       ��Ҷ˹��������ҵ��
% ע�⣺       �ֽ�����
% ����汾��   MATLAB R2018a
%%-------------------------------------------------------------------------
%% �ڢ��⣬Э����������
clc, clear, close all
w1.X1 = [1 1 2]';       %��w1,����X1����һ��
w1.X2 = [1 0 -1]';      %��w1,����X2���ڶ���
w2.X1 = [-1 -1 -2]';    %��w2,����X1����һ��
w2.X2 = [1 0 -1]';      %��w2,����X2���ڶ���
% �����ֵ
w1.x1_bar = mean(w1.X1);
w1.x2_bar = mean(w1.X2);
w2.x1_bar = mean(w2.X1);
w2.x2_bar = mean(w2.X2);
w1.X_bar = [w1.x1_bar w1.x2_bar]';
w2.X_bar = [w2.x1_bar w2.x2_bar]';
% ����Э�������
w1.SIG = cov(w1.X1, w1.X2);
w2.SIG = cov(w2.X1, w2.X2);
% ����������ʣ����������������Ƶ�ʣ�
P_w1 = size(w1.X1,1) / (size(w1.X1,1) + size(w2.X1,1));
P_w2 = size(w2.X1,1) / (size(w1.X1,1) + size(w2.X1,1));
% �����б���
syms x1 x2
x = [x1; x2];
fg1 = -0.5 * (x - w1.X_bar)' * w1.SIG^(-1) * (x - w1.X_bar) ...
    - 0.5 * log(det(w1.SIG)) + log(P_w1);
fg2 = -0.5 * (x - w2.X_bar)' * w2.SIG^(-1) * (x - w2.X_bar) ...
    - 0.5 * log(det(w2.SIG)) + log(P_w2);
fg = fg2 - fg1;
% ��ͼ
fg_plothandle = fimplicit(fg == 0, 'b -', 'Linewidth', 1);  %���ݷ��̻��ֽ��ߣ��������в��֣�
Temp12X = fg_plothandle.XData;
Temp12Y = fg_plothandle.YData;
plot(Temp12X(1:116), Temp12Y(1:116), 'b -', 'Linewidth', 1), hold on  %���ֶ�������ʾ��Χ�������ߣ�
plot(w1.X1, w1.X2, 'r o', 'Markersize', 8), hold on
plot(w2.X1, w2.X2, 'r *', 'Markersize', 8), hold on
axis([-2.2 2.2 -2.2 Temp12Y(119)])

% ����
plot(2, 0, 'm s', 'Markersize', 8), hold on
legend({'w1-w2 ������','��w1','��w2','������'},'Location','best')
if subs(fg, [x1 x2], [2 0]) < 0     %�ֶ�����������
    disp('����������w1��')
else
    disp('����������w2��')
end

%% �ڢ��⣬����Э����������
clc, clear, close all
w1.X1 = [1 1 2]';       %��w1,����X1����һ��
w1.X2 = [1 0 -1]';      %��w1,����X2���ڶ���
w2.X1 = [-1 -1 -2]';    %��w2,����X1����һ��
w2.X2 = [1 0 -1]';      %��w2,����X2���ڶ���
% �����ֵ
w1.x1_bar = mean(w1.X1);
w1.x2_bar = mean(w1.X2);
w2.x1_bar = mean(w2.X1);
w2.x2_bar = mean(w2.X2);
w1.X_bar = [w1.x1_bar w1.x2_bar]';
w2.X_bar = [w2.x1_bar w2.x2_bar]';
% ����Э�����������ȵ�Э����
w1.SIG = cov(w1.X1, w1.X2);
w2.SIG = cov(w2.X1, w2.X2);
SIG = w1.SIG + w2.SIG;
% ����������ʣ����������������Ƶ�ʣ�
P_w1 = size(w1.X1,1) / (size(w1.X1,1) + size(w2.X1,1));
P_w2 = size(w2.X1,1) / (size(w1.X1,1) + size(w2.X1,1));
% �����б���
syms x1 x2
x = [x1; x2];
fg1 = (SIG^(-1) * w1.X_bar)' * x ...
    - 0.5 * w1.X_bar' * SIG^(-1) * w1.X_bar + log(P_w1);
fg2 = (SIG^(-1) * w2.X_bar)' * x ...
    - 0.5 * w2.X_bar' * SIG^(-1) * w2.X_bar + log(P_w2);
fg = fg2 - fg1;
% ��ͼ
fg_plothandle = fimplicit(fg == 0, 'b -', 'Linewidth', 1);  %���ݷ��̻��ֽ��ߣ��������в��֣�
Temp12X = fg_plothandle.XData;
Temp12Y = fg_plothandle.YData;
plot(Temp12X(1:end), Temp12Y(1:end), 'b -', 'Linewidth', 1), hold on  %���ֶ�������ʾ��Χ�������ߣ�
plot(w1.X1, w1.X2, 'r o', 'Markersize', 8), hold on
plot(w2.X1, w2.X2, 'r *', 'Markersize', 8), hold on
axis([-2.2 2.2 -2.2 2.2])

% ����
plot(2, 0, 'm s', 'Markersize', 8), hold on
legend({'w1-w2 ������','��w1','��w2','������'},'Location','best')
if subs(fg, [x1 x2], [2 0]) < 0     %�ֶ�����������
    disp('����������w1��')
else
    disp('����������w2��')
end

%% �ڢ��⣬Э����������
clc, clear, close all
w1.X1 = [0 2 1]';       %��w1,����X1����һ��
w1.X2 = [0 1 0]';       %��w1,����X2���ڶ���
w2.X1 = [-1 -2 -2]';    %��w2,����X1����һ��
w2.X2 = [1 0 -1]';      %��w2,����X2���ڶ���
w3.X1 = [0 0 1]';       %��w3,����X1����һ��
w3.X2 = [-2 -1 -2]';    %��w3,����X2���ڶ���
% �����ֵ
w1.x1_bar = mean(w1.X1);
w1.x2_bar = mean(w1.X2);
w2.x1_bar = mean(w2.X1);
w2.x2_bar = mean(w2.X2);
w3.x1_bar = mean(w3.X1);
w3.x2_bar = mean(w3.X2);
w1.X_bar = [w1.x1_bar w1.x2_bar]';
w2.X_bar = [w2.x1_bar w2.x2_bar]';
w3.X_bar = [w3.x1_bar w3.x2_bar]';
% ����Э�������
w1.SIG = cov(w1.X1, w1.X2);
w2.SIG = cov(w2.X1, w2.X2);
w3.SIG = cov(w3.X1, w3.X2);
% ����������ʣ����������������Ƶ�ʣ�
P_w1 = size(w1.X1,1) / (size(w1.X1,1) + size(w2.X1,1) + size(w3.X1,1));
P_w2 = size(w2.X1,1) / (size(w1.X1,1) + size(w2.X1,1) + size(w3.X1,1));
P_w3 = size(w3.X1,1) / (size(w1.X1,1) + size(w2.X1,1) + size(w3.X1,1));
% �����б���
syms x1 x2
x = [x1; x2];
fg1 = x' * (-0.5) * w1.SIG^(-1) * x + (w1.SIG^(-1) * w1.X_bar)' * x ...
    - 0.5 * w1.X_bar' * w1.SIG^(-1) * w1.X_bar - 0.5 * log(det(w1.SIG)) + log(P_w1);
fg2 = x' * (-0.5) * w2.SIG^(-1) * x + (w2.SIG^(-1) * w2.X_bar)' * x ...
    - 0.5 * w2.X_bar' * w2.SIG^(-1) * w2.X_bar - 0.5 * log(det(w2.SIG)) + log(P_w2);
fg3 = x' * (-0.5) * w3.SIG^(-1) * x + (w3.SIG^(-1) * w3.X_bar)' * x ...
    - 0.5 * w3.X_bar' * w3.SIG^(-1) * w3.X_bar - 0.5 * log(det(w3.SIG)) + log(P_w3);
% ��ͼ
fg12_plothandle = fimplicit(fg1 == fg2, 'g -', 'Linewidth', 1); %���ݷ��̻��ֽ��ߣ��������в��֣�
Temp12X = fg12_plothandle.XData;
Temp12Y = fg12_plothandle.YData;
fg23_plothandle = fimplicit(fg2 == fg3, 'm -', 'Linewidth', 1); %���ݷ��̻��ֽ��ߣ��������в��֣�
Temp23X = fg23_plothandle.XData;
Temp23Y = fg23_plothandle.YData;
fg31_plothandle = fimplicit(fg3 == fg1, 'b -', 'Linewidth', 1); %���ݷ��̻��ֽ��ߣ��������в��֣�
Temp31X = fg31_plothandle.XData;
Temp31Y = fg31_plothandle.YData;
close
plot(Temp12X(97:293), Temp12Y(97:293), 'g -', 'Linewidth', 1), hold on  %���ֶ�������ʾ��Χ�������ߣ�
plot(Temp23X(1:53), Temp23Y(1:53), 'm -', 'Linewidth', 1), hold on  %���ֶ�������ʾ��Χ�������ߣ�
plot(Temp31X(185:304), Temp31Y(185:304), 'b -', 'Linewidth', 1), hold on  %���ֶ�������ʾ��Χ�������ߣ�
plot(w1.X1, w1.X2, 'r o', 'Markersize', 8), hold on
plot(w2.X1, w2.X2, 'r *', 'Markersize', 8), hold on
plot(w3.X1, w3.X2, 'r s', 'Markersize', 8), hold on
axis([-2.2 2.2 -4 2.2])

% ����
plot(-2, 2, 'm s', 'Markersize', 8), hold on
legend({'w1-w2 ������','w2-w3 ������','w3-w1 ������','��w1','��w2','��w3','������'},'Location','best')
if subs(fg1-fg2, [x1 x2], [-2 2]) > 0     %�ֶ�����������
    if subs(fg3-fg1, [x1 x2], [-2 2]) < 0     %�ֶ�����������
        disp('����������w1��')
    else
        disp('����������w3��')
    end
else
    if subs(fg2-fg3, [x1 x2], [-2 2]) > 0     %�ֶ�����������
        disp('����������w2��')
    else
        disp('����������w3��')
    end
end

%% �ڢ��⣬����Э����������
clc, clear, close all
w1.X1 = [0 2 1]';       %��w1,����X1����һ��
w1.X2 = [0 1 0]';       %��w1,����X2���ڶ���
w2.X1 = [-1 -2 -2]';    %��w2,����X1����һ��
w2.X2 = [1 0 -1]';      %��w2,����X2���ڶ���
w3.X1 = [0 0 1]';       %��w3,����X1����һ��
w3.X2 = [-2 -1 -2]';    %��w3,����X2���ڶ���
% �����ֵ
w1.x1_bar = mean(w1.X1);
w1.x2_bar = mean(w1.X2);
w2.x1_bar = mean(w2.X1);
w2.x2_bar = mean(w2.X2);
w3.x1_bar = mean(w3.X1);
w3.x2_bar = mean(w3.X2);
w1.X_bar = [w1.x1_bar w1.x2_bar]';
w2.X_bar = [w2.x1_bar w2.x2_bar]';
w3.X_bar = [w3.x1_bar w3.x2_bar]';
% ����Э�����������ȵ�Э����
w1.SIG = cov(w1.X1, w1.X2);
w2.SIG = cov(w2.X1, w2.X2);
w3.SIG = cov(w3.X1, w3.X2);
SIG = w1.SIG + w2.SIG + w3.SIG;
% ����������ʣ����������������Ƶ�ʣ�
P_w1 = size(w1.X1,1) / (size(w1.X1,1) + size(w2.X1,1) + size(w3.X1,1));
P_w2 = size(w2.X1,1) / (size(w1.X1,1) + size(w2.X1,1) + size(w3.X1,1));
P_w3 = size(w3.X1,1) / (size(w1.X1,1) + size(w2.X1,1) + size(w3.X1,1));
% �����б���
syms x1 x2
x = [x1; x2];
fg1 = x' * (-0.5) * SIG^(-1) * x + (SIG^(-1) * w1.X_bar)' * x ...
    - 0.5 * w1.X_bar' * SIG^(-1) * w1.X_bar - 0.5 * log(det(SIG)) + log(P_w1);
fg2 = x' * (-0.5) * SIG^(-1) * x + (SIG^(-1) * w2.X_bar)' * x ...
    - 0.5 * w2.X_bar' * SIG^(-1) * w2.X_bar - 0.5 * log(det(SIG)) + log(P_w2);
fg3 = x' * (-0.5) * SIG^(-1) * x + (SIG^(-1) * w3.X_bar)' * x ...
    - 0.5 * w3.X_bar' * SIG^(-1) * w3.X_bar - 0.5 * log(det(SIG)) + log(P_w3);
% ��ͼ
fg12_plothandle = fimplicit(fg1 == fg2, 'g -', 'Linewidth', 1); %���ݷ��̻��ֽ��ߣ��������в��֣�
Temp12X = fg12_plothandle.XData;
Temp12Y = fg12_plothandle.YData;
fg23_plothandle = fimplicit(fg2 == fg3, 'm -', 'Linewidth', 1); %���ݷ��̻��ֽ��ߣ��������в��֣�
Temp23X = fg23_plothandle.XData;
Temp23Y = fg23_plothandle.YData;
fg31_plothandle = fimplicit(fg3 == fg1, 'b -', 'Linewidth', 1); %���ݷ��̻��ֽ��ߣ��������в��֣�
Temp31X = fg31_plothandle.XData;
Temp31Y = fg31_plothandle.YData;
close
plot(Temp12X(1:179), Temp12Y(1:179), 'g -', 'Linewidth', 1), hold on  %���ֶ�������ʾ��Χ�������ߣ�
plot(Temp23X(1:122), Temp23Y(1:122), 'm -', 'Linewidth', 1), hold on  %���ֶ�������ʾ��Χ�������ߣ�
plot(Temp31X(1:174), Temp31Y(1:174), 'b -', 'Linewidth', 1), hold on  %���ֶ�������ʾ��Χ�������ߣ�
plot(w1.X1, w1.X2, 'r o', 'Markersize', 8), hold on
plot(w2.X1, w2.X2, 'r *', 'Markersize', 8), hold on
plot(w3.X1, w3.X2, 'r s', 'Markersize', 8), hold on
axis([-3 2.2 -3 2.2])

% ����
plot(-2, 2, 'm s', 'Markersize', 8), hold on
legend({'w1-w2 ������','w2-w3 ������','w3-w1 ������','��w1','��w2','��w3','������'},'Location','best')
if subs(fg1-fg2, [x1 x2], [-2 2]) > 0     %�ֶ�����������
    if subs(fg3-fg1, [x1 x2], [-2 2]) < 0     %�ֶ�����������
        disp('����������w1��')
    else
        disp('����������w3��')
    end
else
    if subs(fg2-fg3, [x1 x2], [-2 2]) > 0     %�ֶ�����������
        disp('����������w2��')
    else
        disp('����������w3��')
    end
end
