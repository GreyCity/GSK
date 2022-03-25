function [R1, R2, R3] = Gained_Shared_Junior_R1R2R3(indBest)


pop_size = length(indBest);
R0=1:pop_size;
R1=[];  % R1为NP维，左邻居
R2=[];  % R2为NP维，右邻居
R3=[];  % R3为NP维，随机非同号

for i=1:pop_size
    % 循环NP次，为每个个体找邻居
    ind=find(indBest==i);  % ind是indBest的标号(标号的标号)
    if(ind==1)% best的邻居是右边的两个
    R1(i)=indBest(2);
    R2(i)=indBest(3);
    elseif(ind==pop_size)% worst的邻居是左边的两个
    R1(i)=indBest(pop_size-2);
    R2(i)=indBest(pop_size-1);
    else  % Common 邻居是左右各一个
    R1(i)=indBest(ind-1);
    R2(i)=indBest(ind+1);
    end
end

R3 = floor(rand(1, pop_size) * pop_size) + 1;  % 向下取整，生成NP个范围在1~NP的随机整数

for i = 1 : 99999999
    pos = ((R3 == R2) | (R3 == R1) | (R3 == R0));  % 检查下标是否相同，|为逻辑“或”按位操作，pos为相同下标位置
    if sum(pos)==0
        break;
    else % regenerate r2 if it is equal to r0 or r1  % 如果 R3 等于R0,R1,R2则重新产生。
        R3(pos) = floor(rand(1, sum(pos)) * pop_size) + 1;  % 相同的小标处重新产生
    end
    if i > 1000 % this has never happened so far  % 非酋检测器，产生了1000次还没得到满意的R3时触发error。
        error('Can not genrate R3 in 1000 iterations');
    end
end

end
