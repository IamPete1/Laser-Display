clc
clear 
close all

% import a image file and convert to a .wav for laser display
image = imread('MH_test3.jpg');
%image = imread('star.jpg');
%image = imread('star2.png');
%image = imread('cloud2.png');
%image = imread('rec.png');

figure
subplot(3,2,1)
hold all
imshow(image)
title('Input Image')

% convert to grayscale 
I = rgb2gray(image);

subplot(3,2,2)
hold all
imshow(I)
title('Converted to grayscale')

% do a contor plot of the image
% use a single level to start with
%figure;
%C = imcontour(I,1);
C = contourc(double(I),1);

% split out to individual contours
index = 1;
i = 1;
while index < size(C,2)
    contour(i).level = C(1,index); %#ok<SAGROW>
    
    % read in the cords
    contour(i).x = C(1,index+(1:C(2,index))); %#ok<SAGROW>
    contour(i).y = C(2,index+(1:C(2,index)))*-1; %#ok<SAGROW>
           
    i = i + 1;
    index = index + C(2,index) + 1;
end

max_size = max(size(image,1),size(image,2));
for n = 1:numel(contour)
    % move and scale between +- 1, maintain Aspect ratio
    % take size from original image
    contour(n).x = (contour(n).x / (0.5*max_size)) - 1;
    contour(n).y = (contour(n).y / (0.5*max_size)) + 1;

    contour(n) = simplify_contour(contour(n));

    % caculate the length, maybe also throw out by lenght, this is sort of
    % draw time i guess
    contour(n).length = sum(sqrt(diff(contour(n).x).^2 + diff(contour(n).y).^2));
    contour(n).size = numel(contour(n).x);
    
    % start and end points of the contour
    contour(n).start = [contour(n).x(1),contour(n).y(1)];
    contour(n).end = [contour(n).x(end),contour(n).y(end)];
    
    contour(n).closed = all(contour(n).start == contour(n).end);
end

% remove any that have short lenght
contour([contour.length] < 10^-2) = [];

% plot individual contors
subplot(3,2,[3,4,5,6])
title('Scalled and edges found')
hold all
xlim([-1,1])
ylim([-1,1])
axis equal
for n = 1:numel(contour)
    %plot(contour(n).x,contour(n).y,'-')
    
    if contour(n).closed
        plot(contour(n).x,contour(n).y,'r-*')
    else
        plot(contour(n).x,contour(n).y,'b-*')
    end
end

if any(~[contour.closed])
    warning('Not all contours are closed')
end

% two-opt tsp
[sorted, reversed] = two_opt(cell2mat({contour.start}'),cell2mat({contour.end}'),cell2mat({contour.length}'));

if all([contour.closed])
   % use GA to optimise set-TSP
   [sorted, contour] = GA_set_TSP(contour,sorted);
   reversed = zeros(numel(contour),1);
end

% make sure we have used all the contors, but only onece each
if numel(unique(sorted)) ~= numel(sorted) || numel(sorted) ~= numel(contour)
    error('sorting cock up')
end

% convert into single X Y data set
x_data = zeros(numel([contour.x]),1);
y_data = zeros(numel([contour.x]),1);
index = 1;
line_disp = true(sum([contour.size]),1);
for n = 1:numel(contour)
    contour_size = numel([contour(sorted(n)).x]);
    
    if ~reversed(n)
        x_data(index+(0:contour_size-1)) = contour(sorted(n)).x;
        y_data(index+(0:contour_size-1)) = contour(sorted(n)).y;
    else
        x_data(index+(0:contour_size-1)) = fliplr(contour(sorted(n)).x);
        y_data(index+(0:contour_size-1)) = fliplr(contour(sorted(n)).y);
    end
    
    if n ~= numel(contour)
        index = index + contour_size;
        line_disp(index-1,1) = false;
    end
end

% work out the total cost to check tsp maths
%cost = sum(sqrt(diff([x_data;x_data(1)]).^2 + diff([y_data;y_data(1)]).^2))

% resample to give more points on display bit
line_length = sqrt(diff(x_data).^2 + diff(y_data).^2);
time_step = min(line_length)/1;
line_length(~line_disp) = min(line_length);
time = [0;cumsum(line_length)];
resamp_time = 0:time_step:time(end);
x_data = interp1(time,x_data,resamp_time');
y_data = interp1(time,y_data,resamp_time');

% plot joined up contors
figure
subplot(2,4,[1,2])

hold all
plot(x_data)
ylabel('X data')
xlim([1,numel(x_data)])

subplot(2,4,[5,6])

hold all
plot(y_data)
ylabel('Y data')
xlim([1,numel(x_data)])

subplot(2,4,[3,4,7,8])
hold all
plot([x_data;x_data(1)],[y_data;y_data(1)],'*-')
xlim([-1,1])
ylim([-1,1])
axis equal
title('Single line')


plot_time = 1/2;

%Sampling Frequency
Fs = 192000; 

num_repeat = 500;
file_name = 'test';

% basic resample to get correct number of display hz
disp_hz = 60;

% how many samples do we get to draw the whole thing in?
total_samples = Fs * (1/disp_hz);

if numel(x_data) > total_samples
    orig_time = 1:numel(x_data);
    
    new_time = linspace(1,numel(x_data),total_samples);
    
    x_resample2 = interp1(orig_time,x_data',new_time)';
    y_resample2 = interp1(orig_time,y_data',new_time)';
else
    x_resample2 = x_data;
    y_resample2 = y_data;
end

audiowrite([file_name,'_resampled2.wav'],repmat([x_resample2,y_resample2],num_repeat,1),round(Fs))

function [Contour_tour, reversed] = two_opt(starts, ends, lengths)

points = [starts; ends;];
points_contor = [(1:size(starts,1))';(1:size(ends,1))'];
contour_cost = sum(lengths);

tour_length = size(points,1);

% build a cost matrix
cost_mat = inf(tour_length);
for i = 1:tour_length
    for j = 1:tour_length
        if i == j
            continue
        end        
        if points_contor(i) == points_contor(j)
             cost_mat(i,j) = lengths(points_contor(i));
             continue
        end
        cost_mat(i,j) = sqrt( (points(i,1) - points(j,1))^2 + (points(i,2) - points(j,2))^2 );
    end
end

% work out the cost
cost = 0;
for i = 1:tour_length -1 
    cost = cost + cost_mat(i,i+1);
end
cost = cost + cost_mat(end,1);

fprintf('Start Cost %g\n',cost-contour_cost)

% Greedy Start
tour = ones(tour_length,1);
i = 1;
while i < tour_length   
    
    index = find(points_contor == points_contor(tour(i)));
    index(index == tour(i)) = [];
    
    i = i + 1;
    tour(i) = index;
    
    if i < tour_length
        costs = cost_mat( tour(i),:);
        costs(unique(tour)) = inf;
        
        [~,index] = min(costs);
        i = i + 1;
        tour(i) = index;
    end
end


% work out the cost
cost = 0;
for i = 1:tour_length -1 
    cost = cost + cost_mat(tour(i),tour(i+1));
end
cost = cost + cost_mat(tour(end),tour(1));

fprintf('Greedy Cost %g\n',cost-contour_cost)

max_iter = 1500;
for iter = 1:max_iter
    % try all swaps
    improved = false;
    for i = 1:2:tour_length
        for j = i+1:2:tour_length
           
            start_bit = tour(1:i-1);
            middle_bit = tour(i:j);
            end_bit = tour(j+1:end);
            
            test_route = [start_bit; flipud(middle_bit); end_bit;];
            
            test_cost = 0;
            for k = 1:tour_length -1
                test_cost = test_cost + cost_mat(test_route(k),test_route(k+1));
            end
            test_cost = test_cost + cost_mat(test_route(end),test_route(1));
            
            
            if test_cost < cost
                tour = test_route;
                cost = test_cost;
                improved = true;
                break;
            end
        end
        if improved
            break
        end
    end
    if ~improved
        break;
    end
end

if iter == max_iter
    warning('Two-Opt Max itter')
end

fprintf('Two-opt Cost %g\n',cost-contour_cost)

% sort back into contor order
tour = reshape(tour,[2,numel(tour)/2])';
Contour_tour = points_contor(tour);

if any(Contour_tour(:,1) ~= Contour_tour(:,2))
    error('cock up')
end
    

Contour_tour = Contour_tour(:,1);
reversed = tour(:,1) > tour(:,2);

end

function contour = simplify_contour(contour)

% figure
% hold all
% title(sprintf('before %i points',numel(contour.x)))
% plot(contour.x,contour.y,'*-')
% axis equal

i = 1;
while i < numel(contour.y) - 2
%   scatter(contour.x(i),contour.y(i),'r')
%   scatter(contour.x(i+1),contour.y(i+1),'g')
%   scatter(contour.x(i+2),contour.y(i+2),'k')
    
    dy1 = contour.y(i+1) - contour.y(i);
    dx1 = contour.x(i+1) - contour.x(i);
    line_length = sqrt(dy1.^2 + dx1.^2);

    ang1 = atan2(dy1,dx1);
    
    dy2 = contour.y(i+2) - contour.y(i);
    dx2 = contour.x(i+2) - contour.x(i);
    
    ang2 = atan2(dy2,dx2);

    change = ang1 - ang2;
    
    dist = sin(change) * line_length;
    
    
    if abs(dist) < 0.0005
        contour.x(i+1) = [];
        contour.y(i+1) = [];
    else
        i = i + 1;
    end
end

% figure
% hold all
% title(sprintf('after %i points',numel(contour.x)))
% plot(contour.x,contour.y,'*-')
% axis equal

end

function [sorted, contour] = GA_set_TSP(contour,two_opt)

num_contour = numel(contour);
contour_size = [contour.size] - 1;
%contour_cost = sum([contour.length]);
max_stall = 150;
max_iter = 5000;

pop_size = 1000;
keep_pop = max(round(pop_size*0.1),1);

% generate intial population
pop_tour = zeros(pop_size,num_contour);
for i = 1:pop_size
    pop_tour(i,:) = two_opt;    
end
pop_start_stop = zeros(pop_size,num_contour);
for i = 1:num_contour
    pop_start_stop(:,i) = randi([0,contour_size(i)-1],pop_size,1);
end

% build cost matrix
num_points = sum(contour_size);
points = zeros(num_points,3);
contour_point_start = zeros(1,num_contour);
index = 1;
for i = 1:num_contour
    contour_point_start(i) = index;
    points(index:index+contour_size(i)-1,[1,2]) = [contour(i).x(1:end-1);contour(i).y(1:end-1)]';
    points(index:index+contour_size(i)-1,3) = i;
    
    index = index + contour_size(i);
end
cost_mat = inf(num_points);
for i = 1:num_points
   for j = 1:num_points
      if points(i,3) == points(j,3)
          %continue
      end      
      cost_mat(i,j) = sqrt(sum( ( points(i,[1,2]) - points(j,[1,2]) ).^2 ));
   end    
end


index_mat = repmat([1:pop_size]',1,num_contour);
cost_size = size(cost_mat);

% itterate the GA
cost = zeros(pop_size,1);
best_cost = inf;
stall = 0;
iter = 0;
while iter < max_iter && stall < max_stall
    iter = iter + 1;
    
    % evaluate the population
    index = sub2ind(size(pop_start_stop),index_mat,pop_tour);    
    cost_mat_points = contour_point_start(pop_tour) + pop_start_stop(index);
    cost_mat_points1 = cost_mat_points(:,[2:end,1]);
    index = sub2ind(cost_size,cost_mat_points,cost_mat_points1);
    cost = sum(cost_mat(index),2);

    [~, best_index] = sort(cost);
    
    if best_cost == cost(best_index(1))
        stall = stall + 1;
    else
        stall = 0;
    end
    best_cost = cost(best_index(1));
    
    %fprintf('%g\n',best_cost);
    
    % set whole population to the best and mutate
    for j = 1:pop_size
        if j == best_index(1)
            % leave the best unchanged
            continue
        end
        % pick one of the top
        index = randi(keep_pop);
        pop_tour(j,:) = pop_tour(best_index(index),:);
        pop_start_stop(j,:) = pop_start_stop(best_index(index),:);

        % mutate
        type = randi(6);
        switch type
            case 1
                % flip, like two opt
                index = sort(randi(num_contour,1,2));                
                pop_tour(j,index(1):index(2)) = fliplr(pop_tour(j,index(1):index(2)));                
                
            case 2
                % swap two points
                index = sort(randi(num_contour,1,2));
                pop_tour(j,index) = pop_tour(j,fliplr(index)); 
                
            case 3
                % slide along
                index = sort(randi(num_contour,1,2));
                pop_tour(j,index(1):index(2)) = pop_tour(j,[index(1)+1:index(2),index(1)]);
                
                
            case 4
                % pick new start stop point for one
                index = randi(num_contour);
                pop_start_stop(j,index) = randi([0,contour_size(index)-1]); 
                
            case 5
                % pick new start stop point for two
                for l = 1:2
                    index = randi(num_contour);
                    pop_start_stop(j,index) = randi([0,contour_size(index)-1]); 
                end
                
            case 6
                % pick new start stop point for three
                for l = 1:3                    
                    index = randi(num_contour);
                    pop_start_stop(j,index) = randi([0,contour_size(index)-1]);                    
                end
        end
        
        
    end
    
end

if iter == max_iter
    warning('GA Max itter')
end

fprintf('GA Cost %g\n',cost(best_index(1)))


sorted = pop_tour(best_index(1),:);
start_stop = pop_start_stop(best_index(1),:)+1;

% update the contors
for i = 1:num_contour
    % remove the duplicate point
    x = contour(i).x(1:end-1);
    y = contour(i).y(1:end-1);
    
    contour(i).x = x([start_stop(i):end,1:start_stop(i)]);
    contour(i).y = y([start_stop(i):end,1:start_stop(i)]);

    % update start stop
    contour(i).start = [contour(i).x(1),contour(i).y(1)];
    contour(i).end = [contour(i).x(end),contour(i).y(end)];    
end

end

    
