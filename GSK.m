%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Gaining-Sharing Knowledge Based Algorithm for Solving Optimization
%%Problems: A Novel Nature-Inspired Algorithm
%%Authors: Ali Wagdy Mohamed, Anas A. Hadi , Ali Khater Mohamed
%%注释编写：GreyCity
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc;
clear all;

format long;  % 保留16位小数
Alg_Name='GSK';  % 算法名称GSK
n_problems=30;  % 问题数量，CEC2017提出的30个问题
ConvDisp=1;
Run_No=51;  % 循环51轮

for problem_size = [10 30 50 100]  % 问题规模，决策变量的数量(D)
    % 第一层循环，每个问题分别求解决策变量数为10,30,50,100。
    
    max_nfes = 10000 * problem_size;  % The maximum number of function evaluations(MaxFES): 10000*D(CEC2017 终止条件)
    rand('seed', sum(100 * clock));  % 基于时间生成随机种子
    val_2_reach = 10^(-8);  % 误差界限:10^-8(CEC2017 终止条件)
    max_region = 100.0;  % 搜寻范围_上界
    min_region = -100.0;  % 搜寻范围_下界
    lu = [-100 * ones(1, problem_size); 100 * ones(1, problem_size)];  % lu = [min_region ...; max_region ...]，size = 2 * D。
    fhd=@cec17_func;  % 调用cec17_func的函数
    analysis= zeros(30,6);  % 用于保存数据分析 size = n_problems * 6个数据分析指标
    
    for func = 1 : n_problems
        % 第二层循环，分别计算30个问题。以第一个问题为例func=1。
        
        optimum = func * 100.0;  % 真实全局最优解
        % Record the best results
        outcome = [];
        fprintf('\n-------------------------------------------------------\n')
        fprintf('Function = %d, Dimension size = %d\n', func, problem_size)
        dim1=[];
        dim2=[];
        
        for run_id = 1 : Run_No
            % 第三层循环，迭代51次。
            
            bsf_error_val=[];  % 误差
            run_funcvals = [];  % 函数值
            pop_size = 100;  % 种群个体数(论文中的NP)
            G_Max=fix(max_nfes/pop_size);  % 最大种群代数(论文中的GEN) G_MAX = 10^4*D/NP=1000 规定的操作步数最大种群代数
            
            % Initialize the main population
            popold = repmat(lu(1, :), pop_size, 1) + rand(pop_size, problem_size) .* (repmat(lu(2, :) - lu(1, :), pop_size, 1));
            % 初始化种群，生成 NP*D矩阵，值为下界 + rand * (上界 - 下届)
            pop = popold; % the old population becomes the current population
            
            fitness = fhd(pop',func);  % 计算适应度等于目标函数的值
            fitness = fitness';  % size = NP*1
            
            nfes = 0;  % 操作计数器
            bsf_fit_var = 1e+300;  % 一个很大的数
            
            %%%%%%%%%%%%%%%%%%%%%%%% for out%%%%%%%%%%%%%%%%%
            for i = 1 : pop_size
                % 循环NP次，找出适应度值最小(优)值
                nfes = nfes + 1;
                % if nfes > max_nfes; exit(1); end
                if nfes > max_nfes; break; end
                if fitness(i) < bsf_fit_var
                    bsf_fit_var = fitness(i);  % 史上最优值
                end
                run_funcvals = [run_funcvals;bsf_fit_var];  % 记小本本消耗fes次数
            end
            
            %%%%%%%%%%%%%%%%%%%%%%%% Parameter settings%%%%%%%%%%
            KF=0.5;% Knowledge Factor
            KR=0.9;%Knowledge Ratio
            K=10*ones(pop_size,1);%Knowledge Rate  ！！！这个K是一个size = NP*1，value = 10的矩阵
            
            g=0;  % 当前代数初始化为0
            % main loop
            while nfes < max_nfes  
                g=g+1;
                D_Gained_Shared_Junior=ceil((problem_size)*(1-g/G_Max).^K);  % 初级码长 size = NP*1 value = D*(1-g/G_Max)^K
                D_Gained_Shared_Senior=problem_size-D_Gained_Shared_Junior;  % 高级码长
                pop = popold; % the old population becomes the current population
                
                [valBest, indBest] = sort(fitness, 'ascend');  % valBest是适应度递增排序，indBest是下标(!不改变fitness的顺序)
                [Rg1, Rg2, Rg3] = Gained_Shared_Junior_R1R2R3(indBest);  % Rg1左邻居 Rg2右邻居 Rg3随机
                
                [R1, R2, R3] = Gained_Shared_Senior_R1R2R3(indBest);  % R1:Best people R2:Better people R3:Worst people
                R01=1:pop_size;  %R01:排序前下标(自然顺序)
                Gained_Shared_Junior=zeros(pop_size, problem_size);  % 初级交叉 NP*D
                ind1=fitness(R01)>fitness(Rg3);  % 随机选出的个体适应度比自然顺序的适应度好的下标
                
                if(sum(ind1)>0)  % 对这些个体进行初级交叉：xnew_ij = xold_ij + KF*[(x_i-1 - x_i+1) + (x_r - x_i)]
                    Gained_Shared_Junior (ind1,:)= pop(ind1,:) + KF*ones(sum(ind1), problem_size) .* (pop(Rg1(ind1),:) - pop(Rg2(ind1),:)+pop(Rg3(ind1), :)-pop(ind1,:)) ;
                end
                ind1=~ind1;  % 反选
                if(sum(ind1)>0)  % 其余个体初级交叉：xnew_ij = xold_ij + KF*[(x_i-1 - x_i+1) + (x_i - x_r)]
                    Gained_Shared_Junior(ind1,:) = pop(ind1,:) + KF*ones(sum(ind1), problem_size) .* (pop(Rg1(ind1),:) - pop(Rg2(ind1),:)+pop(ind1,:)-pop(Rg3(ind1), :)) ;
                end
                R0=1:pop_size;  %R01:排序前下标(自然顺序)
                Gained_Shared_Senior=zeros(pop_size, problem_size);  % 高级交叉 size = NP*D
                ind=fitness(R0)>fitness(R2);% 随机Better People适应度比自然顺序的适应度好的下标
                if(sum(ind)>0)  % 对这些个体进行高级交叉：xnew_ij = xold_ij + KF*[(x_i-1 - x_i+1) + (x_m - x_i)]
                    Gained_Shared_Senior(ind,:) = pop(ind,:) + KF*ones(sum(ind), problem_size) .* (pop(R1(ind),:) - pop(ind,:) + pop(R2(ind),:) - pop(R3(ind), :)) ;
                end
                ind=~ind;  % 反选
                if(sum(ind)>0)  % 其余个体高级交叉：xnew_ij = xold_ij + KF*[(x_i-1 - x_i+1) + (x_i - x_m)]
                    Gained_Shared_Senior(ind,:) = pop(ind,:) + KF*ones(sum(ind), problem_size) .* (pop(R1(ind),:) - pop(R2(ind),:) + pop(ind,:) - pop(R3(ind), :)) ;
                end
                Gained_Shared_Junior = boundConstraint(Gained_Shared_Junior, pop, lu);  % 边界处理
                Gained_Shared_Senior = boundConstraint(Gained_Shared_Senior, pop, lu);
                
                % D_Gained_Shared_Junior为初级码长size = NP*1 value = D*(1-g/G_Max).^K
                D_Gained_Shared_Junior_mask=rand(pop_size, problem_size)<=(D_Gained_Shared_Junior(:, ones(1, problem_size))./problem_size);   % 初级码位置 rand<= D_junior/D (初级码占比) 相当于在NP*D中以初级码率选择
                D_Gained_Shared_Senior_mask=~D_Gained_Shared_Junior_mask;  % 高级码位置
                
                D_Gained_Shared_Junior_rand_mask=rand(pop_size, problem_size)<=KR*ones(pop_size, problem_size);  % 概率小于KR进行初级交叉
                D_Gained_Shared_Junior_mask=and(D_Gained_Shared_Junior_mask,D_Gained_Shared_Junior_rand_mask);  % “与”逻辑得到要执行初级交叉的位置
                
                D_Gained_Shared_Senior_rand_mask=rand(pop_size, problem_size)<=KR*ones(pop_size, problem_size);
                D_Gained_Shared_Senior_mask=and(D_Gained_Shared_Senior_mask,D_Gained_Shared_Senior_rand_mask);  % 要执行高级交叉的位置
                ui=pop;
                
                ui(D_Gained_Shared_Junior_mask) = Gained_Shared_Junior(D_Gained_Shared_Junior_mask);  % 要执行初级交叉的位置进行初级交叉
                ui(D_Gained_Shared_Senior_mask) = Gained_Shared_Senior(D_Gained_Shared_Senior_mask);  % 要执行高级交叉的位置进行高级交叉

                
                children_fitness = fhd(ui', func);
                children_fitness = children_fitness';  % 交叉后的适应度
                
                for i = 1 : pop_size
                    nfes = nfes + 1;
                    if nfes > max_nfes; break; end
                    if children_fitness(i) < bsf_fit_var
                        bsf_fit_var = children_fitness(i); % 更新史上最优值
                        bsf_solution = ui(i, :);  % 记录史上最优解
                    end
                    run_funcvals = [run_funcvals;bsf_fit_var]; % 最优值记在小本本上，最多可记录max_nfes次

                end
                
                [fitness, Child_is_better_index] = min([fitness, children_fitness], [], 2);  % 拼接old-new适应度，按行取最小值，和取最小值的位置(第几列)
                
                popold = pop;
                popold(Child_is_better_index == 2, :) = ui(Child_is_better_index == 2, :);  % new优于old进入种群
                           
               % fprintf('NFES:%d, bsf_fit:%1.6e,pop_Size:%d,D_Gained_Shared_Junior:%2.2e,D_Gained_Shared_Senior:%2.2e\n', nfes,bsf_fit_var,pop_size,problem_size*sum(sum(D_Gained_Shared_Junior))/(pop_size*problem_size),problem_size*sum(sum(D_Gained_Shared_Senior))/(pop_size*problem_size))
  
            end % end while loop  种群迭代了G_max代
            
            bsf_error_val = bsf_fit_var - optimum;  % 计算误差
            if bsf_error_val < val_2_reach
                bsf_error_val = 0;
            end         
            
            fprintf('%d th run, best-so-far error value = %1.8e\n', run_id , bsf_error_val)  % 报告
            outcome = [outcome bsf_error_val];  % 记录误差
            
            % plot convergence figures
            if (ConvDisp)  % if(1)
                run_funcvals=run_funcvals-optimum;
                run_funcvals=run_funcvals';
                dim1(run_id,:)=1:length(run_funcvals);  % 操作次数=max_nfes
                dim2(run_id,:)=log10(run_funcvals);  % 适应度取对数
            end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%
        end %% end 1 run
        
        % save ststiatical output in analysis file%%%%
        analysis(func,1)=min(outcome);  % 数据分析：最小值、中位数、最大值、平均值、标准差、找到中位数的时候
        analysis(func,2)=median(outcome);
        analysis(func,3)=max(outcome);
        analysis(func,4)=mean(outcome);
        analysis(func,5)=std(outcome);
        median_figure=find(outcome== median(outcome));
        analysis(func,6)=median_figure(1);
        
        file_name=sprintf('Results\\%s_CEC2017_Problem#%s_problem_size#%s',Alg_Name,int2str(func),int2str(problem_size));  % 保存文件为GSK_CEC2017_Problem#1_problem_size#10.mat
        save(file_name,'outcome');  % 记录误差
        % print statistical output and save convergence figures%%%
        fprintf('%e\n',min(outcome));  % 打印数据分析
        fprintf('%e\n',median(outcome));
        fprintf('%e\n',mean(outcome));
        fprintf('%e\n',max(outcome));
        fprintf('%e\n',std(outcome));
        dim11=dim1(median_figure,:);
        dim22=dim2(median_figure,:);
        file_name=sprintf('Figures\\Figure_Problem#%s_Run#%s',int2str(func),int2str(median_figure));  % 保存文件为Figure_Problem#1_Run#%1 2 ... 51.mat
        save(file_name,'dim1','dim2');  % 保存适应度
    end %% end 1 function run
    
    file_name=sprintf('Results\\analysis_%s_CEC2017_problem_size#%s',Alg_Name,int2str(problem_size));  % 保存文件为analysis_GSK_CEC2017_problem_size#10.mat
    save(file_name,'analysis');  % 保存数据分析
end %% end all function runs in all dimensions
