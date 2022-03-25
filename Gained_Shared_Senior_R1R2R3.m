function [R1, R2, R3] = Gained_Shared_Senior_R1R2R3(indBest)


pop_size = length(indBest);

R1=indBest(1:round(pop_size*0.1));  % 四舍五入取出Best people(适应度排序前100p%的个体的编号)
R1rand = ceil(length(R1) * rand(pop_size, 1));  % 向上取整 产生NP个值在1~0.1NP的列向量
R1 = R1(R1rand);  % R1为NP维，每个值都是Best people的编号

R2=indBest(round(pop_size*0.1)+1:round(pop_size*0.9));  % Better people
R2rand = ceil(length(R2) * rand(pop_size, 1));
R2 = R2(R2rand);

R3=indBest(round(pop_size*0.9)+1:end);  % 后100p%为Worst people
R3rand = ceil(length(R3) * rand(pop_size, 1));
R3 = R3(R3rand);% R3为NP维，每个值都是Worst people的编号

end
